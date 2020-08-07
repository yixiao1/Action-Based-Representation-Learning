import glob
import logging
import math
import os
import sys

#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.misc import imresize
import torch
from torchvision.utils import save_image
import cv2

from coilutils.drive_utils import checkpoint_parse_configuration_file
from configs import g_conf, merge_with_yaml
from network import CoILModel
from input.coil_dataset import convert_scenario_name_number, encode_directions
from logger import coil_logger

# We  need to add two things here to the python path,

from cexp.agents.agent import Agent
from agents.navigation.local_planner import RoadOption

import carla


def location_to_gps(location, lat_ref=42.0, lon_ref= 2.0):
    """
    Convert from world coordinates to GPS coordinates
    :param lat_ref: latitude reference for the current map
    :param lon_ref: longitude reference for the current map
    :param location: location to translate
    :return: dictionary with lat, lon and height
    """
    EARTH_RADIUS_EQUA = 6378137.0   # pylint: disable=invalid-name
    scale = math.cos(lat_ref * math.pi / 180.0)
    mx = scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    my = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
    mx += location.x
    my += location.y

    lon = mx * 180.0 / (math.pi * EARTH_RADIUS_EQUA * scale)
    lat = 360.0 * math.atan(math.exp(my / (EARTH_RADIUS_EQUA * scale))) / math.pi - 90.0
    z = location.z

    return {'lat': lat, 'lon': lon, 'z': z}

#def distance_vehicle(waypoint, vehicle_position):
#
#    wp_gps = location_to_gps(waypoint.location)
#    dx = wp_gps['lat'] - vehicle_position[0]
#    dy = wp_gps['lon'] - vehicle_position[1]

#    return math.sqrt(dx * dx + dy * dy)

def distance_vehicle(waypoint, vehicle_position):

    #wp_gps = location_to_gps(waypoint.location, lat_ref=lat_ref, lon_ref=lon_ref)
    dx = waypoint.location.x - vehicle_position.x
    dy = waypoint.location.y - vehicle_position.y

    return math.sqrt(dx * dx + dy * dy)

class CoILBaselineCEXP(Agent):

    def setup(self, path_to_config_file):

        yaml_conf, checkpoint_number, agent_name, encoder_params = checkpoint_parse_configuration_file(path_to_config_file)

        # Take the checkpoint name and load it
        if encoder_params is not None:
            self.checkpoint = torch.load(os.path.join('/', os.path.join(*os.path.realpath(__file__).split('/')[:-2]),
                                                 '_logs',
                                                 yaml_conf.split('/')[-2], yaml_conf.split('/')[-1].split('.')[-2] + '_' + str(encoder_params['encoder_checkpoint'])
                                                 , 'checkpoints', str(checkpoint_number) + '.pth'))

            # Once the ENCODER_MODEL_CONFIGURATION was defined, we use the pre-trained encoder model to extract bottleneck Z and drive the E-t-E agent
            self.encoder_checkpoint = torch.load(os.path.join('/', os.path.join(*os.path.realpath(__file__).split('/')[:-2]),
                                                     '_logs', encoder_params['encoder_folder'], encoder_params['encoder_exp'],
                                                     'checkpoints', str(encoder_params['encoder_checkpoint']) + '.pth'))

            self.encoder_model = CoILModel(g_conf.ENCODER_MODEL_TYPE, g_conf.ENCODER_MODEL_CONFIGURATION)
            self.encoder_model.load_state_dict(self.encoder_checkpoint['state_dict'])
            self.encoder_model.cuda()
            self.encoder_model.eval()

        else:
            self.checkpoint = torch.load(os.path.join('/', os.path.join(*os.path.realpath(__file__).split('/')[:-2]),
                                             '_logs',
                                             yaml_conf.split('/')[-2], yaml_conf.split('/')[-1].split('.')[-2]
                                             , 'checkpoints', str(checkpoint_number) + '.pth'))


        # do the merge here
        # TODO THE MERGE IS REQUIRED DEPENDING ON THE SITUATION
        g_conf.immutable(False)
        merge_with_yaml(os.path.join('/', os.path.join(*os.path.realpath(__file__).split('/')[:-2]),
                                     yaml_conf), encoder_params)

        self._model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION, g_conf.ENCODER_MODEL_CONFIGURATION)
        self.first_iter = True
        logging.info("Setup Model")
        # Load the model and prepare set it for evaluation
        self._model.load_state_dict(self.checkpoint['state_dict'])
        self._model.cuda()
        self._model.eval()
        self.latest_image = None
        self.latest_image_tensor = None
        # We add more time to the curve commands
        self._expand_command_front = 5
        self._expand_command_back = 3

        # TODO: Merge with Felipe's code
        self._msn = None
        self._lat_ref = 0
        self._lon_ref = 0
        # Check the agent name
        self._name = agent_name

        self.count = 0

    def sensors(self):
        sensors = [{'type': 'sensor.camera.rgb',
                   'x': 2.0, 'y': 0.0,
                    'z': 1.40, 'roll': 0.0,
                    'pitch': -15.0, 'yaw': 0.0,
                    'width': 800, 'height': 600,
                    'fov': 100,
                    'id': 'rgb'},
                   {'type': 'sensor.can_bus',
                    'reading_frequency': 25,
                    'id': 'can_bus'
                    },
                   {'type': 'sensor.other.gnss',
                    'x': 0.7, 'y': -0.4, 'z': 1.60,
                    'id': 'GPS'}
                   ]

        return sensors

    """
    def make_state(self, exp):
        # state is divided in three parts, the speed, the angle_error, the high level command
        # Get the closest waypoint

        #waypoint, _ = self._get_current_wp_direction(exp._ego_actor.get_transform().location, exp._route)
        #norm, angle = compute_magnitude_angle(waypoint.location, exp._ego_actor.get_transform().location,
        #                                      exp._ego_actor.get_transform().rotation.yaw)
        #return np.array([_get_forward_speed(exp._ego_actor) / 12.0,  # Normalize to by dividing by 12
        #                 angle / 180.0])
        self._global_plan = exp._route

        input_data = exp._sensor_interface.get_data()
        # TODO this should be capilarized
        #if 'scenario' in g_conf.MEASUREMENTS_INPUTS:
        #    scenario = convert_scenario_name_number(exp._environment_data['exp_measurements'])
        #    input_data.update({'scenario': scenario})
        return input_data
    """

    def make_state(self, exp):
        """
        This function also do the necessary processing of the state for the run step function
        :param exp:
        :return:
        """

        self._global_plan = exp._route

        # we also need to get the latitute longitude ref
        # TODO this needs to be adpated for a CARLA challenge submission
        self._lat_ref = exp._lat_ref
        self._lon_ref = exp._lon_ref

        input_data = exp._sensor_interface.get_data()
        self._vehicle_pos = exp._ego_actor.get_transform().location
        # TODO this should be capilarized



        input_data.update({'sensor_input': self._process_sensors(input_data['rgb'][1])})
        if self._msn is not None:
            input_data.update({'scenario': self._msn(input_data['sensor_input'])})

        #if 'scenario' in g_conf.MEASUREMENTS_INPUTS:
        #    scenario = convert_scenario_name_number(exp._environment_data['exp_measurements'])
        #    print (" SCENARIO NUMBER ", scenario)
        #    input_data.update({'scenario': scenario})

        return input_data

    def run_step(self, input_data):

        # Get the current directions for following the route

        directions = self._get_current_direction(self._vehicle_pos)
        logging.debug(" Current direction %f ", directions)

        # Take the forward speed and normalize it for it to go from 0-1
        network_input = input_data['can_bus'][1]['speed'] / g_conf.SPEED_FACTOR
        network_input = torch.cuda.FloatTensor([network_input]).unsqueeze(0)

        # TODO remove ifs
        #if 'scenario' in g_conf.MEASUREMENTS_INPUTS:
        #    network_input = torch.cat((torch.cuda.FloatTensor([input_data['scenario']]),
        #                               network_input), 1)

        # Compute the forward pass processing the sensors got from CARLA.
        # TODO we start with an if but we can build a class hierarquical !

        if g_conf.MODEL_TYPE in ['coil-icra', 'coil-icra-KLD', 'separate-supervised']:
            directions_tensor = torch.cuda.LongTensor([directions])
            #print("       Directions ", int(directions))
            if False:
                save_path = os.path.join('temp','ete_baseline')
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_image(input_data['sensor_input'], os.path.join(save_path, 'run_input_' + str(self.count).zfill(5) + ".png"))
                self.count += 1
            model_outputs = self._model.forward_branch(input_data['sensor_input'],
                                                       network_input,
                                                       directions_tensor)

        elif g_conf.MODEL_TYPE in ['coil-icra-VAE']:
            directions_tensor = torch.cuda.LongTensor([directions])
            if g_conf.ENCODER_MODEL_TYPE in ['VAE']:
                if g_conf.LABELS_SUPERVISED:
                    input = torch.cat((input_data['sensor_input'], torch.zeros(1, 1, 88, 200).cuda()), dim = 1)
                    recon_x, mu, _, z = self.encoder_model(input)
                else:
                    recon_x, mu, _, z = self.encoder_model(input_data['sensor_input'])

            elif g_conf.ENCODER_MODEL_TYPE in ['Affordances']:
                mu, _ = self.encoder_model(input_data['sensor_input'])

            if False:
                save_path = os.path.join('temp','affordances_upperbound')
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                if g_conf.LABELS_SUPERVISED:
                    save_image(input_data['sensor_input'], os.path.join(save_path, 'run_input_' + str(self.count).zfill(5) + ".png"))
                    split = torch.split(torch.squeeze(recon_x, dim=1), [3, 1], dim=1)
                    save_image(split[0], os.path.join(save_path, 'run_recon_rgb_' + str(self.count).zfill(5) + ".png"))
                    save_image(split[1], os.path.join(save_path, 'run_recon_labels_' + str(self.count).zfill(5) + ".png"))
                else:
                    save_image(input_data['sensor_input'], os.path.join(save_path, 'run_input_' + str(self.count).zfill(5) + ".png"))
                    #save_image(recon_x, os.path.join(save_path, 'run_recon_' + str(self.count).zfill(5) + ".png"))
                self.count += 1

            model_outputs = self._model.forward_branch(mu, network_input, directions_tensor)

            #print(' frame', self.count)
            #print(' direction', directions_tensor)
            #print(' branch output', model_outputs)


        elif g_conf.MODEL_TYPE in ['separate-supervised-NoSpeed', 'coil-icra-NoSpeed']:
            directions_tensor = torch.cuda.LongTensor([directions])
            if False:
                save_path = os.path.join('temp', 'ETE_resnet34_6')
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_image(input_data['sensor_input'],os.path.join(save_path, 'run_input_' + str(self.count).zfill(5) + ".png"))
                self.count += 1
            model_outputs = self._model.forward_branch(input_data['sensor_input'], directions_tensor)


        else:
            directions_tensor = torch.cuda.FloatTensor(encode_directions(directions))
            model_outputs = self._model.forward(self._process_sensors(input_data['rgb'][1]),
                                                network_input,
                                                directions_tensor)[0]

        steer, throttle, brake = self._process_model_outputs(model_outputs[0])
        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        logging.debug("Output %f %f %f " % (control.steer,control.throttle, control.brake))

        if self.first_iter:
            coil_logger.add_message('Iterating', {"Checkpoint": self.checkpoint['iteration'],
                                                  'Agent': str(steer)},
                                    self.checkpoint['iteration'])
        # There is the posibility to replace some of the predictions with oracle predictions.
        self.first_iter = False

        #print(['steer: ', control.steer, 'throttle: ', control.throttle, 'brake: ', control.brake])

        return control

    def get_attentions(self, layers=None):
        """
        Returns
            The activations obtained from the first layers of the latest iteration.

        """
        if layers is None:
            layers = [0, 1, 2]
        if self.latest_image_tensor is None:
            raise ValueError('No step was ran yet. '
                             'No image to compute the activations, Try Running ')
        all_layers = self._model.get_perception_layers(self.latest_image_tensor)
        cmap = plt.get_cmap('inferno')
        attentions = []
        for layer in layers:
            y = all_layers[layer]
            att = torch.abs(y).mean(1)[0].data.cpu().numpy()
            att = att / att.max()
            att = cmap(att)
            att = np.delete(att, 3, 2)
            attentions.append(imresize(att, [88, 200]))
        return attentions

    def _process_sensors(self, sensor):
        sensor = sensor[:, :, 0:3]  # BGRA->BRG drop alpha channel
        sensor = sensor[g_conf.IMAGE_CUT[0]:g_conf.IMAGE_CUT[1], :, :]  # crop
        sensor = scipy.misc.imresize(sensor, (g_conf.SENSORS['rgb_central'][1], g_conf.SENSORS['rgb_central'][2]))
        self.latest_image = sensor

        sensor = np.swapaxes(sensor, 0, 1)
        sensor = np.transpose(sensor, (2, 1, 0))
        sensor = torch.from_numpy(sensor / 255.0).type(torch.FloatTensor).cuda()
        image_input = sensor.unsqueeze(0)
        self.latest_image_tensor = image_input

        return image_input

    def _get_current_direction(self, vehicle_position):

        #print("      number of waypoints in global plan:", len(self._global_plan))

        # for the current position and orientation try to get the closest one from the waypoints
        closest_id = 0
        min_distance = 100000
        for index in range(len(self._global_plan)):

            waypoint = self._global_plan[index][0]

            computed_distance = distance_vehicle(waypoint, vehicle_position)
            if computed_distance < min_distance:
                min_distance = computed_distance
                closest_id = index

        #print("      closest waypoint", closest_id)
        logging.debug("Closest waypoint {} dist {}".format(closest_id, min_distance))
        direction = self._global_plan[closest_id][1]

        if direction == RoadOption.LEFT:
            direction = 3.0
        elif direction == RoadOption.RIGHT:
            direction = 4.0
        elif direction == RoadOption.STRAIGHT:
            direction = 5.0
        else:
            direction = 2.0

        return direction

    def _process_model_outputs(self, outputs):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        steer, throttle, brake = outputs[0], outputs[1], outputs[2]
        if brake < 0.05:
            brake = 0.0

        if throttle > brake:
            brake = 0.0

        return steer, throttle, brake

    def _expand_commands(self, topological_plan):
        """ The idea is to make the intersection indications to last longer"""

        # O(2*N) algorithm , probably it is possible to do in O(N) with queues.

        # Get the index where curves start and end
        curves_start_end = []
        inside = False
        start = -1
        current_curve = RoadOption.LANEFOLLOW
        for index in range(len(topological_plan)):

            command = topological_plan[index][1]
            if command != RoadOption.LANEFOLLOW and not inside:
                inside = True
                start = index
                current_curve = command

            if command == RoadOption.LANEFOLLOW and inside:
                inside = False
                # End now is the index.
                curves_start_end.append([start, index, current_curve])
                if start == -1:
                    raise ValueError("End of curve without start")

                start = -1

        for start_end_index_command in curves_start_end:
            start_index = start_end_index_command[0]
            end_index = start_end_index_command[1]
            command = start_end_index_command[2]

            # Add the backwards curves ( Before the begginning)
            for index in range(1, self._expand_command_front + 1):
                changed_index = start_index - index
                if changed_index > 0:
                    topological_plan[changed_index] = (topological_plan[changed_index][0], command)

            # add the onnes after the end
            for index in range(0, self._expand_command_back):
                changed_index = end_index + index
                if changed_index < len(topological_plan):
                    topological_plan[changed_index] = (topological_plan[changed_index][0], command)

        return topological_plan
