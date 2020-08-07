import logging
from enum import Enum
from configs import g_conf, merge_with_yaml
import carla
import torch
import math
import numpy as np
import os
import scipy
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from scipy.misc import imresize
from drive.affordances import  get_driving_affordances
from drive.local_planner import LocalPlanner
from network import CoILModel, EncoderModel
from coilutils.drive_utils import checkpoint_parse_configuration_file

# TODO make a sub class for a non learnable agent

"""
    Interface for the CARLA basic npc agent.
"""

def save_attentions(images, all_layers, iteration, folder_name, layers=None, save_input=False, big_size=False):

    # Plot the attentions that are computed by the network directly here.
    # maybe take the intermediate layers and compute the attentions  ??
    if layers is None:
        layers = [0, 1, 2]


    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    if save_input:
        # We save the images directly as a source of comparison
        if not os.path.exists(os.path.join(folder_name, 'inputs')):
            os.mkdir(os.path.join(folder_name, 'inputs'))

        for i in range(images.shape[0]):
            save_image(images[i], os.path.join(folder_name, 'inputs', str(iteration).zfill(8) + '.png' ))


    # We save now the attention maps for the layers
    cmap = plt.get_cmap('inferno')
    for layer in layers:

        if not os.path.exists(os.path.join(folder_name, 'layer' + str(layer))):
            os.mkdir(os.path.join(folder_name, 'layer' + str(layer)))

        y = all_layers[layer]      #shape [1, 64, 22, 50]
        att = torch.abs(y).mean(1)[0].data.cpu().numpy()      #shape [22, 50]
        att = att / att.max()                   #shape [22, 50]
        if big_size:
            att = scipy.misc.imresize(att, [395, 800])
        else:
            att = scipy.misc.imresize(att, [88, 200])
        scipy.misc.imsave(os.path.join(folder_name, 'layer' + str(layer), str(iteration).zfill(8) + '.png' ), cmap(att))

def encode_directions(directions):
    if directions == 2.0:
        return [1, 0, 0, 0]
    elif directions == 3.0:
        return [0, 1, 0, 0]
    elif directions == 4.0:
        return [0, 0, 1, 0]
    elif directions == 5.0:
        return [0, 0, 0, 1]
    else:
        raise ValueError("Unexpcted direction identified %s" % str(directions))


def get_forward_speed(vehicle):
    """ Convert the vehicle transform directly to forward speed """

    velocity = vehicle.get_velocity()
    transform = vehicle.get_transform()
    vel_np = np.array([velocity.x, velocity.y, velocity.z])
    pitch = np.deg2rad(transform.rotation.pitch)
    yaw = np.deg2rad(transform.rotation.yaw)
    orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    speed = np.dot(vel_np, orientation)
    return speed

def compute_relative_angle(vehicle, waypoint):
    vehicle_transform = vehicle.get_transform()
    v_begin = vehicle_transform.location
    v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                     y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

    v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
    w_vec = np.array([waypoint.transform.location.x -
                      v_begin.x, waypoint.transform.location.y -
                      v_begin.y, 0.0])

    relative_angle = math.acos(np.clip(np.dot(w_vec, v_vec) /
                             (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))
    _cross = np.cross(v_vec, w_vec)
    if _cross[2] < 0:
        relative_angle *= -1.0

    if np.isnan(relative_angle):
        relative_angle = 0.0

    return relative_angle


class AgentState(Enum):
    """
    AGENT_STATE represents the possible states of a roaming agent
    """
    NAVIGATING = 1
    BLOCKED_BY_VEHICLE = 2
    BLOCKED_RED_LIGHT = 3
    BLOCKED_BY_PEDESTRIAN = 4


class AffordancesAgent(object):

    def __init__(self, path_to_config_file):
        # params for now it is not used but we might want to use this to set


        self.setup(path_to_config_file)
        self.save_attentions = False

    def setup(self, path_to_config_file):
        self._agent = None
        self.route_assigned = False
        self.count = 0

        exp_dir = os.path.join('/', os.path.join(*path_to_config_file.split('/')[:-1]))

        yaml_conf, checkpoint_number, agent_name, encoder_params = checkpoint_parse_configuration_file(path_to_config_file)

        if encoder_params == "None":
            encoder_params = None

        g_conf.immutable(False)
        merge_with_yaml(os.path.join('/', os.path.join(*path_to_config_file.split('/')[:-4]), yaml_conf), encoder_params)

        if g_conf.MODEL_TYPE in ['one-step-affordances']:
            # one step training, no need to retrain FC layers, we just get the output of encoder model as prediciton
            self._model = EncoderModel(g_conf.ENCODER_MODEL_TYPE, g_conf.ENCODER_MODEL_CONFIGURATION)
            self.checkpoint = torch.load(os.path.join(exp_dir, 'checkpoints', str(checkpoint_number) + '.pth'))
            print("Affordances Model ", str(checkpoint_number) + '.pth', "loaded from ", os.path.join(exp_dir, 'checkpoints'))
            self._model.load_state_dict(self.checkpoint['state_dict'])
            self._model.cuda()
            self._model.eval()


        elif g_conf.MODEL_TYPE in ['separate-affordances']:
            if encoder_params is not None:
                self.encoder_model = EncoderModel(g_conf.ENCODER_MODEL_TYPE, g_conf.ENCODER_MODEL_CONFIGURATION)
                self.encoder_model.cuda()
                # Here we load the pre-trained encoder (not fine-tunned)
                if g_conf.FREEZE_ENCODER:
                    encoder_checkpoint = torch.load(
                    os.path.join(os.path.join('/', os.path.join(*path_to_config_file.split('/')[:-4])), '_logs',
                                 encoder_params['encoder_folder'], encoder_params['encoder_exp'], 'checkpoints',
                                     str(encoder_params['encoder_checkpoint']) + '.pth'))
                    print("Encoder model ", str(encoder_params['encoder_checkpoint']), "loaded from ",
                          os.path.join('_logs', encoder_params['encoder_folder'], encoder_params['encoder_exp'],
                                       'checkpoints'))
                    self.encoder_model.load_state_dict(encoder_checkpoint['state_dict'])
                    self.encoder_model.eval()
                    for param_ in self.encoder_model.parameters():
                        param_.requires_grad = False

                else:
                    encoder_checkpoint = torch.load(os.path.join(exp_dir, 'checkpoints', str(checkpoint_number) + '_encoder.pth'))
                    print("FINE TUNNED encoder model ", str(checkpoint_number) + '_encoder.pth', "loaded from ",
                          os.path.join(exp_dir, 'checkpoints'))
                    self.encoder_model.load_state_dict(encoder_checkpoint['state_dict'])
                    self.encoder_model.eval()
                    for param_ in self.encoder_model.parameters():
                        param_.requires_grad = False
            else:
                raise RuntimeError('encoder_params can not be None in MODEL_TYPE --> separate-affordances')

            self._model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION, g_conf.ENCODER_MODEL_CONFIGURATION)
            self.checkpoint = torch.load(os.path.join(exp_dir, 'checkpoints', str(checkpoint_number) + '.pth'))
            print(
            "Affordances Model ", str(checkpoint_number) + '.pth', "loaded from ", os.path.join(exp_dir, 'checkpoints'))
            self._model.load_state_dict(self.checkpoint['state_dict'])
            self._model.cuda()
            self._model.eval()



    def get_sensors_dict(self):
        """
        The agent sets the sensors that it is going to use. That has to be
        set into the environment for it to produce this data.
        """
        sensors_dict = [{'type': 'sensor.camera.rgb',
                'x': 2.0, 'y': 0.0,
                'z': 1.40, 'roll': 0.0,
                'pitch': -15.0, 'yaw': 0.0,
                'width': 800, 'height': 600,
                'fov': 100,
                'id': 'rgb_central'}
                        ]

        return sensors_dict
    # TODO we set the sensors here directly.
    def sensors(self):
        return self._sensors_dict

    def get_state(self, exp_list, target_speed = 20.0):
        """
            Based on the exp object it makes all the affordances.
        :param exp:
        :return:
        """
        exp = exp_list[0]
        self._vehicle = exp._ego_actor

        if self._agent is None:
            self._agent = True
            self._state = AgentState.NAVIGATING
            args_lateral_dict = {
                'K_P': 1,
                'K_D': 0.02,
                'K_I': 0,
                'dt': 1.0 / 20.0}
            self._local_planner = LocalPlanner(
                self._vehicle, opt_dict={'target_speed': target_speed,
                                         'lateral_control_dict': args_lateral_dict})
            self._hop_resolution = 2.0
            self._path_seperation_hop = 2
            self._path_seperation_threshold = 0.5
            self._grp = None

        if not self.route_assigned:
            plan = []
            for transform, road_option in exp._route:
                wp = exp._ego_actor.get_world().get_map().get_waypoint(transform.location)
                plan.append((wp, road_option))

            self._local_planner.set_global_plan(plan)
            self.route_assigned = True

        input_data = exp._sensor_interface.get_data()
        input_data = self._process_sensors(input_data['rgb_central'][1])    #torch.Size([1, 3, 88, 200]

        if g_conf.MODEL_TYPE in ['one-step-affordances']:
            c_output, r_output, layers= self._model.forward_outputs(input_data.cuda(),
                                                             torch.cuda.FloatTensor([exp._forward_speed/g_conf.SPEED_FACTOR]).unsqueeze(0),
                                                             torch.cuda.FloatTensor(encode_directions(exp._directions)).unsqueeze(0))
        elif g_conf.MODEL_TYPE in ['separate-affordances']:
            if g_conf.ENCODER_MODEL_TYPE in ['action_prediction', 'stdim' ,'ETEDIM',
                                                         'FIMBC', 'one-step-affordances']:
                e, layers = self.encoder_model.forward_encoder(input_data.cuda(),
                                                             torch.cuda.FloatTensor([exp._forward_speed/g_conf.SPEED_FACTOR]).unsqueeze(0),
                                                             torch.cuda.FloatTensor(encode_directions(exp._directions)).unsqueeze(0))
                c_output, r_output = self._model.forward_test(e)
            elif g_conf.ENCODER_MODEL_TYPE in ['ETE', 'ETE_inverse_model', 'forward',
                                                           'ETE_stdim']:
                e, layers = self.encoder_model.forward_encoder(input_data.cuda(),
                                                            torch.cuda.FloatTensor([exp._forward_speed/g_conf.SPEED_FACTOR]).unsqueeze(0),
                                                            torch.cuda.FloatTensor(encode_directions(exp._directions)).unsqueeze(0))
                c_output, r_output = self._model.forward_test(e)

        if self.save_attentions:
            exp_params = exp._exp_params
            attentions_full_path = os.path.join(os.environ["SRL_DATASET_PATH"], exp_params['package_name'], exp_params['env_name'],
                                                str(exp_params['env_number'])+'_'+ exp._agent_name, str(exp_params['exp_number']))
            save_attentions(input_data.cuda(), layers, self.count, attentions_full_path, save_input=False, big_size=False)

        self.count += 1


        affordances = {}

        output_relative_angle = torch.squeeze(r_output[0]).cpu().detach().numpy() * 1.0

        is_pedestrian_hazard = False
        if c_output[0][0, 0] < c_output[0][0, 1]:
            is_pedestrian_hazard = True

        is_red_tl_hazard = False
        if c_output[1][0, 0] < c_output[1][0, 1]:
            is_red_tl_hazard = True

        is_vehicle_hazard = False
        if (c_output[2][0, 0] < c_output[2][0, 1]):
            is_vehicle_hazard = True

        affordances.update({'is_pedestrian_hazard': is_pedestrian_hazard})
        affordances.update({'is_red_tl_hazard': is_red_tl_hazard})
        affordances.update({'is_vehicle_hazard':is_vehicle_hazard})
        affordances.update({'relative_angle': output_relative_angle})
        # Now we consider all target speed to be 20.0
        affordances.update({'target_speed': target_speed})

        #affordances.update({'GT_is_pedestrian_hazard': })
        #affordances.update({'GT_is_red_tl_hazard': })
        #affordances.update({'GT_is_vehicle_hazard': })
        gt_relative_angle = compute_relative_angle(self._vehicle, self._local_planner.get_target_waypoint())
        affordances.update({'GT_relative_angle': gt_relative_angle})
        affordances.update(
            {'ERROR_relative_angle': output_relative_angle - gt_relative_angle})

        return affordances


    def make_reward(self, exp):
        # Just basically return None since the reward is not used for a non

        return None

    def step(self, affordances):
        hazard_detected = False
        is_vehicle_hazard = affordances['is_vehicle_hazard']
        is_red_tl_hazard = affordances['is_red_tl_hazard']
        is_pedestrian_hazard = affordances['is_pedestrian_hazard']
        relative_angle = affordances['relative_angle']
        target_speed = affordances['target_speed']
        # once we meet a speed limit sign, the target speed changes

        #if target_speed != self._local_planner._target_speed:
        #    self._local_planner.set_speed(target_speed)
        #forward_speed = affordances['forward_speed']
        
        if is_vehicle_hazard:
            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True
        
        if is_red_tl_hazard:
            self._state = AgentState.BLOCKED_RED_LIGHT
            hazard_detected = True

        if is_pedestrian_hazard:
            self._state = AgentState.BLOCKED_BY_PEDESTRIAN
            hazard_detected = True

        if hazard_detected:
            control = self.emergency_stop()
        
        else:
            self._state = AgentState.NAVIGATING
            control = self._local_planner.run_step(relative_angle, target_speed)
            
        logging.debug("Output %f %f %f " % (control.steer,control.throttle, control.brake))

        return control


    def reinforce(self, rewards):
        """
        This agent cannot learn so there is no reinforce
        """
        pass

    def reset(self):
        print (" Correctly reseted the agent")
        self.route_assigned = False
        self._agent = None
        self.count = 0


    def emergency_stop(self):
        """
        Send an emergency stop command to the vehicle
        :return:
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False

        return control

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
