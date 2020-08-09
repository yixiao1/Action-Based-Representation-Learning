import os
import glob
import traceback
import collections
import sys
import math
import copy
import json
import random
import gc
import numpy as np

import torch
import cv2
from torchvision.utils import save_image
from torchvision import transforms

from torch.utils.data import Dataset

from . import splitter
from . import data_parser

# TODO: Warning, maybe this does not need to be included everywhere.
from configs import g_conf

from coilutils.general import sort_nicely

from cexp.cexp import CEXP
from cexp.env.scenario_identification import identify_scenario
from cexp.env.environment import NoDataGenerated



def join_classes(labels_image, classes_join):
    compressed_labels_image = np.copy(labels_image)
    for key, value in classes_join.items():
        compressed_labels_image[np.where(labels_image == int(key))] = value

    return compressed_labels_image

def parse_remove_configuration(configuration):
    """
    Turns the configuration line of sliptting into a name and a set of params.
    """

    if configuration is None:
        return "None", None
    print('conf', configuration)
    conf_dict = collections.OrderedDict(configuration)

    name = 'remove'
    for key in conf_dict.keys():
        if key != 'weights' and key != 'boost':
            name += '_'
            name += key

    return name, conf_dict

LANE_FOLLOW_DISTANCE = 25.0

def convert_scenario_name_number(measurements):
    scenario = identify_scenario(measurements['distance_intersection'], measurements['road_angle'])

    #print('scenario', scenario)

    if scenario == 'S0_lane_following':
        return [1, 0, 0, 0]
    elif scenario == 'S1_lane_following_curve':
        return [0, 1, 0, 0]
    elif scenario == 'S2_before_intersection':
        return [0, 0, 1, 0]
    elif scenario == 'S3_intersection':
        return [0, 0, 0, 1]
    else:
        raise ValueError("Unexpcted scenario identified %s" % scenario)

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


def check_size(image_filename, size):
    img = cv2.imread(image_filename, cv2.IMREAD_COLOR)
    return img.shape[0] == size[1] and img.shape[1] == size[2]


def get_episode_weather(episode):
    with open(os.path.join(episode, 'metadata.json')) as f:
        metadata = json.load(f)
    return int(metadata['weather'])


class CoILDataset(Dataset):
    """ The conditional imitation learning dataset"""

    def __init__(self, transform=None, preload_name=None,
                 process_type = None, vd_json_file_path = None):

        # We add to the preload name all the remove labels
        if g_conf.REMOVE is not None and g_conf.REMOVE is not "None":
            name, self._remove_params = parse_remove_configuration(g_conf.REMOVE)
            self.preload_name = preload_name + '_' + name
            self._check_remove_function = getattr(splitter, name)
        else:
            self._check_remove_function = lambda _, __: False
            self._remove_params = []
            self.preload_name = preload_name

        if self.preload_name is not None and os.path.exists(
                os.path.join('_preloads', self.preload_name + '.npy')):
            print(" Loading from NPY: ", self.preload_name + '.npy')
            self.sensor_data_names, self.measurements = np.load(
                os.path.join('_preloads', self.preload_name + '.npy'), allow_pickle=True)

            for key in self.sensor_data_names.keys():
                print( '   ======> '+ key +' images: ', len(self.sensor_data_names[key]))
            print('   ======> measurements:', len(self.measurements))

        else:
            self.sensor_data_names, self.measurements = self._pre_load_image_folders(process_type,
                                                                                     vd_json_file_path)
            for key in self.sensor_data_names.keys():
                print( '   ======> '+ key+' images: ', len(self.sensor_data_names[key]))

            print('   ======> measurements:', len(self.measurements))


        self.transform = transform

        self.batch_read_number = 0

    def __len__(self):
        return len(self.measurements)

    def __getitem__(self, index):
        """
        Get item function used by the dataset loader
        returns all the measurements with the desired image.

        Args:
            index:

        Returns:

        """
        try:
            measurements = self.measurements[index].copy()
            # Here we convert the measurements from the daset to float tensors.
            for k, v in measurements.items():
                try:
                    if k == 'directions':
                        v = encode_directions(v)
                    v = torch.from_numpy(np.asarray([v, ]))
                    measurements[k] = v.float()

                except:
                    pass
                    #print (measurements)

            for sensor_name in self.sensor_data_names.keys():
                if g_conf.ENCODER_MODEL_TYPE in ['forward', 'action_prediction', 'stdim', 'ETE_inverse_model'] \
                        and g_conf.PROCESS_NAME in ['train_encoder']:
                    img = cv2.imread(self.sensor_data_names[sensor_name][index], cv2.IMREAD_COLOR)
                    ti = random.choice(g_conf.POSITIVE_CONSECUTIVE_THR)

                    if g_conf.DATA_USED == 'all':
                        # this means the image is from central camera
                        if index % 3 == 0:
                            img_i = cv2.imread(self.sensor_data_names[sensor_name][
                                                   index + ti * 3],
                                               cv2.IMREAD_COLOR)

                            if g_conf.ENCODER_MODEL_TYPE in ['forward', 'action_prediction', 'stdim', 'ETE_inverse_model']:
                                measurements_i = self.measurements[index + ti * 3].copy()
                                for k, v in measurements_i.items():
                                    try:
                                        if k == 'directions':
                                            v = encode_directions(v)
                                        v = torch.from_numpy(np.asarray([v, ]))
                                        measurements[k] = [measurements[k], v.float()]
                                    except:
                                        pass

                        # this means the image is from lateral cameras
                        else:
                            img_i = cv2.imread(self.sensor_data_names[sensor_name][
                                                   index + ti * 3 - (index%3)],
                                               cv2.IMREAD_COLOR)

                            if g_conf.ENCODER_MODEL_TYPE in ['forward', 'action_prediction', 'stdim', 'ETE_inverse_model']:
                                measurements_i = self.measurements[index + ti * 3 - (index % 3)].copy()
                                for k, v in measurements_i.items():
                                    try:
                                        if k == 'directions':
                                            v = encode_directions(v)
                                        v = torch.from_numpy(np.asarray([v, ]))
                                        measurements[k] = [measurements[k], v.float()]
                                    except:
                                        pass

                    elif g_conf.DATA_USED == 'central':
                        img_i = cv2.imread(
                            self.sensor_data_names[sensor_name][index + ti],
                            cv2.IMREAD_COLOR)

                        if g_conf.ENCODER_MODEL_TYPE in ['forward', 'action_prediction', 'stdim', 'ETE_inverse_model']:
                            measurements_i = self.measurements[index + ti].copy()
                            for k, v in measurements_i.items():
                                try:
                                    if k == 'directions':
                                        v = encode_directions(v)
                                    v = torch.from_numpy(np.asarray([v, ]))
                                    measurements[k] = [measurements[k], v.float()]
                                except:
                                    pass

                    else:
                        raise RuntimeError("Haven't implement yet for this kind of g_conf.DATA_USED")

                    if sensor_name == 'rgb':
                        img = cv2.cvtColor(img,
                                           cv2.COLOR_BGR2RGB)  # cv2.imread changes the RGB channel to BGR, we need to convert them back to RGB
                        img_i = cv2.cvtColor(img_i,
                                             cv2.COLOR_BGR2RGB)  # cv2.imread changes the RGB channel to BGR, we need to convert them back to RGB

                        # Apply the image transformation
                        if self.transform is not None:
                            boost = 1
                            img = self.transform(self.batch_read_number * boost, img)
                            img_i = self.transform(self.batch_read_number * boost, img_i)
                        else:
                            img = img.transpose(2, 0, 1)
                            img_i = img_i.transpose(2, 0, 1)

                        img = img.astype(np.float)
                        img = torch.from_numpy(img).type(torch.FloatTensor)
                        img = img / 255.

                        img_i = img_i.astype(np.float)
                        img_i = torch.from_numpy(img_i).type(torch.FloatTensor)
                        img_i = img_i / 255.

                    elif sensor_name == 'labels':
                        # Apply the image transformation

                        if self.transform is not None:
                            boost = 1
                            img = self.transform(self.batch_read_number * boost, img)
                            img_i = self.transform(self.batch_read_number * boost, img_i)
                        else:
                            img = img.transpose(2, 0, 1)
                            img_i = img_i.transpose(2, 0, 1)

                        img = img[2, :, :]
                        img_i = img_i[2, :, :]

                        if g_conf.LABELS_CLASSES != 13:
                            img = join_classes(img, g_conf.JOIN_CLASSES)
                            img_i = join_classes(img_i, g_conf.JOIN_CLASSES)

                        img = img.astype(np.float)
                        img = torch.from_numpy(img).type(torch.FloatTensor)
                        img = img / (g_conf.LABELS_CLASSES - 1)

                        img_i = img_i.astype(np.float)
                        img_i = torch.from_numpy(img_i).type(torch.FloatTensor)
                        img_i = img_i / (g_conf.LABELS_CLASSES - 1)

                    measurements[sensor_name] = [img, img_i]

                else:
                    img = cv2.imread(self.sensor_data_names[sensor_name][index], cv2.IMREAD_COLOR)

                    if sensor_name == 'rgb':
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2.imread changes the RGB channel to BGR, we need to convert them back to RGB

                        # Apply the image transformation

                        if self.transform is not None:
                            boost = 1
                            img = self.transform(self.batch_read_number * boost, img)
                        else:
                            img = img.transpose(2, 0, 1)

                        img = img.astype(np.float)
                        img = torch.from_numpy(img).type(torch.FloatTensor)
                        img = img / 255.

                    elif sensor_name == 'labels':
                        # Apply the image transformation

                        if self.transform is not None:
                            boost = 1
                            img = self.transform(self.batch_read_number * boost, img)
                        else:
                            img = img.transpose(2, 0, 1)

                        img = img[2,:,:]

                        if g_conf.LABELS_CLASSES != 13:
                            img= join_classes(img, g_conf.JOIN_CLASSES)

                        img = img.astype(np.float)
                        img = torch.from_numpy(img).type(torch.FloatTensor)
                        img = img / (g_conf.LABELS_CLASSES - 1)

                    measurements[sensor_name] = img

            self.batch_read_number += 1

        except AttributeError:
            traceback.print_exc()
            print ("Blank IMAGE")
            measurements = self.measurements[0].copy()
            for k, v in measurements.items():
                v = torch.from_numpy(np.asarray([v, ]))
                measurements[k] = v.float()
            measurements['steer'] = 0.0
            measurements['throttle'] = 0.0
            measurements['brake'] = 0.0
            measurements['rgb_central'] = np.zeros(3, 88, 200)
        except IndexError:
            traceback.print_exc()
            print ("INDEX  ERROR")

            return self.__getitem__(index + random.randint(0,11)*random.choice((-1, 1)))

        return measurements

    def is_measurement_partof_experiment(self, measurement_data):

        # If the measurement data is not removable is because it is part of this experiment dataa
        return not self._check_remove_function(measurement_data, self._remove_params)


    def _add_data_point(self,  float_dicts, data_point, camera_angle):

        """
        Add a data point to the vector that is the full dataset
        :param float_dicts:
        :param data_point:
        :param camera_angle: the augmentation angle to be applyed to the steering.
        :return:
        """

        # we augment the steering if the camera angle is != 0
        new_data_copy = copy.deepcopy(data_point)
        if camera_angle != 0:
            new_data_copy['measurements']['steer'] = self.augment_steering(camera_angle,
                                                    new_data_copy['measurements']['steer'],
                                                    new_data_copy['measurements']['forward_speed'] * 3.6)

            new_data_copy['measurements']['relative_angle'] = self.augment_relative_angle(camera_angle,
                                                    new_data_copy['measurements']['relative_angle'])


        new_data_copy['measurements']['forward_speed'] = \
            data_point['measurements']['forward_speed'] / g_conf.SPEED_FACTOR

        for key in new_data_copy['measurements']:
            if key in ['is_pedestrian_hazard', 'is_red_tl_hazard', 'is_vehicle_hazard']:
                new_data_copy['measurements'][key] = int(data_point['measurements'][key])

        float_dicts.append(new_data_copy['measurements'])

        del new_data_copy


    def _pre_load_image_folders(self, process_type, vd_json_file_path):
        """
        We preload a dataset compleetely and pre process if necessary by using the
        C-EX interface.
        Args
            the path for the dataset

        Returns
            sensor data names: it is a vector with n dimensions being one for each sensor modality
            for instance, rgb only dataset will have a single vector with all the image names.
            float_data: all the wanted float data is loaded inside a vector, that is a vector
            of dictionaries.

        """
        sensor_data_names = {}

        for sensor in g_conf.SENSORS.keys():
            sensor_data_names[sensor.split("_")[0]] = []

        if process_type in ['validation']:
            if vd_json_file_path is not None:
                jsonfile = [vd_json_file_path]   # The validation dataset file full path.
            else:
                raise RuntimeError("You need to define the validation json file path when you call CoILDataset")

        else:
            jsonfile = g_conf.EXPERIENCE_FILE


        # We check one image at least to see if matches the size expected by the network
        checked_image = True
        float_dicts = []


        for json in jsonfile:
            env_batch = CEXP(json, params=None, execute_all=True, ignore_previous_execution=True)
            # Here we start the server without docker
            env_batch.start(no_server=True, agent_name='Agent')  # no carla server mode.
            # count, we count the environments that are read

            for env in env_batch:
                # it can be personalized to return different types of data.
                print("Environment Name: ", env)
                try:
                    env_data = env.get_data()  # returns a basically a way to read all the data properly
                except NoDataGenerated:
                    print("No data generate for episode ", env)
                else:
                    for exp in env_data:
                        print("    Exp: ", exp[1])
                        for batch in exp[0]:
                            print("      Batch: ", batch[1], " of len ", len(batch[0]))
                            for data_point in batch[0]:
                                # We delete some non floatable cases
                                del data_point['measurements']['ego_actor']
                                del data_point['measurements']['opponents']
                                del data_point['measurements']['lane']
                                del data_point['measurements']['hand_brake']
                                del data_point['measurements']['reverse']
                                del data_point['measurements']['walkers']
                                del data_point['measurements']['steer_noise']
                                del data_point['measurements']['brake_noise']
                                del data_point['measurements']['throttle_noise']
                                del data_point['measurements']['closest_vehicle_distance']
                                del data_point['measurements']['closest_red_tl_distance']
                                del data_point['measurements']['closest_pedestrian_distance']

                                self._add_data_point(float_dicts, data_point, 0)  # Some alteration

                                if g_conf.DATA_USED == 'all':
                                    self._add_data_point(float_dicts, data_point, -30)  # Some alteration
                                    self._add_data_point(float_dicts, data_point, 30)

                                for sensor in g_conf.SENSORS.keys():
                                    if not checked_image:
                                        # print(data_point[sensor], g_conf.SENSORS[sensor])
                                        if not check_size(data_point[sensor], g_conf.SENSORS[sensor]):
                                            raise RuntimeError("Unexpected image size for the network")
                                        checked_image = True
                                    # TODO launch meaningful exception if not found sensor name

                                    sensor_data_names[sensor.split('_')[0]].append(data_point[sensor])

                                    if g_conf.DATA_USED == 'all':
                                        sensor_data_names[sensor.split('_')[0]].append(
                                            data_point[sensor.split('_')[0] + '_left'])
                                        sensor_data_names[sensor.split('_')[0]].append(
                                            data_point[sensor.split('_')[0] + '_right'])

        # Make the path to save the pre loaded datasets
        if not os.path.exists('_preloads'):
            os.mkdir('_preloads')
        # If there is a name we saved the preloaded data
        if self.preload_name is not None:
            np.save(os.path.join('_preloads', self.preload_name), [sensor_data_names, float_dicts])

        return sensor_data_names, float_dicts


    def augment_directions(self, directions):

        if directions == 2.0:
            if random.randint(0, 100) < 20:
                directions = random.choice([3.0, 4.0, 5.0])

        return directions


    def augment_relative_angle(self, camera_angle, relative_angle):
        """
            augment for the lateral cameras relative angle
                Args:
                    camera_angle: the angle of the camera
                    relative_angle: the central camera relative angle

                Returns:
                    the augmented relative angle

                """

        relative_angle = np.clip(relative_angle +
                                 g_conf.AUGMENT_RELATIVE_ANGLE*np.deg2rad(camera_angle),
                                 -g_conf.AUGMENT_RA_CLIP, g_conf.AUGMENT_RA_CLIP)

        return relative_angle


    def augment_steering(self, camera_angle, steer, speed):
        """
            Apply the steering physical equation to augment for the lateral cameras steering
        Args:
            camera_angle: the angle of the camera
            steer: the central steering
            speed: the speed that the car is going

        Returns:
            the augmented steering

        """
        time_use = 1.0
        car_length = 6.0
        #old_steer = steer

        pos = camera_angle > 0.0
        neg = camera_angle <= 0.0
        # You should use the absolute value of speed
        speed = math.fabs(speed)
        rad_camera_angle = math.radians(math.fabs(camera_angle))
        val = g_conf.AUGMENT_LATERAL_STEERINGS * (
            math.atan((rad_camera_angle * car_length) / (time_use * speed + 0.05))) / 3.1415
        steer -= pos * min(val, 0.3)
        steer += neg * min(val, 0.3)

        steer = min(1.0, max(-1.0, steer))
        return steer


    def controls_position(self):
        return np.where(self.meta_data[:, 0] == b'control')[0][0]


    """
        Methods to interact with the dataset attributes that are used for training.
    """
    # TODO should be static and maybe a single method

    def extract_targets(self, data):
        """
        Method used to get to know which positions from the dataset are the targets
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """

        # here we have two frames' measurement, we pick up the latter one at time t+1
        if g_conf.ENCODER_MODEL_TYPE in ['forward', 'action_prediction', 'stdim', 'ETE_inverse_model']:
            targets_twoframes_vec = []
            for i in range(2):
                targets_vec = []
                for target_name in g_conf.TARGETS:
                    targets_vec.append(data[target_name][i])
                targets_twoframes_vec.append(torch.cat(targets_vec, 1))

            return targets_twoframes_vec

        else:
            targets_vec = []
            for target_name in g_conf.TARGETS:
                targets_vec.append(data[target_name])

            return torch.cat(targets_vec, 1)

    def extract_affordances_targets(self, data, type = None):
        """
        Method used to get to know which positions from the dataset are the targets
        for this experiments
        Args:
            data: the set of all float data got from the dataset
            type: for affordances, there are two types: classification, regression

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        targets_vec = []
        if type is not None:
            for target_name in g_conf.AFFORDANCES_TARGETS[type]:
                targets_vec.append(data[target_name])
        else:
            for target_name in g_conf.AFFORDANCES_TARGETS:
                targets_vec.append(data[target_name])

        return torch.cat(targets_vec, 1)

    def extract_aux_targets(self, data):
        """
        Method used to get to know which positions from the dataset are the targets
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        targets_vec = []
        for target_name in g_conf.TARGETS_AUX:
            targets_vec.append(data[target_name])

        return torch.cat(targets_vec, 1)

    def extract_commands(self, data):
        """
        Method used to get to know which positions from the dataset are the targets
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        if g_conf.ENCODER_MODEL_TYPE in ['forward', 'action_prediction', 'stdim', 'ETE_inverse_model'] \
                and g_conf.PROCESS_NAME in ['train_encoder']:
            input_twoframes_vec = []
            for i in range(2):
                inputs_vec = []
                for input_name in g_conf.COMMANDS:
                    inputs_vec.append(data[input_name][i])
                input_twoframes_vec.append(torch.squeeze(torch.cat(inputs_vec, 1).cuda()))

            return input_twoframes_vec

        else:
            inputs_vec = []
            for input_name in g_conf.COMMANDS:
                if len(data[input_name].size()) > 2:
                    inputs_vec.append(torch.squeeze(data[input_name]))
                else:
                    inputs_vec.append(data[input_name])

            cat_input = torch.cat(inputs_vec, 1)
            return cat_input

    def extract_inputs(self, data):
        """
        Method used to get to know which positions from the dataset are the inputs
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """

        # here we have two frames' measurement, we pick up the latter one at time t+1
        if g_conf.ENCODER_MODEL_TYPE in ['forward', 'action_prediction', 'stdim', 'ETE_inverse_model'] \
                and g_conf.PROCESS_NAME in ['train_encoder']:
            input_twoframes_vec = []
            for i in range(2):
                inputs_vec = []
                for input_name in g_conf.INPUTS:
                    inputs_vec.append(data[input_name][i])
                input_twoframes_vec.append(torch.cat(inputs_vec, 1).cuda())
            return input_twoframes_vec

        else:
            inputs_vec = []
            for input_name in g_conf.INPUTS:
                if len(data[input_name].size()) > 2:
                    inputs_vec.append(torch.squeeze(data[input_name]))
                else:
                    inputs_vec.append(data[input_name])

            cat_input = torch.cat(inputs_vec, 1)
            return cat_input

    def extract_intentions(self, data):
        """
        Method used to get to know which positions from the dataset are the inputs
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        inputs_vec = []
        for input_name in g_conf.INTENTIONS:
            inputs_vec.append(data[input_name])

        return torch.cat(inputs_vec, 1)

    def action_class(self, data):
        """
        Method used to divide actions (ie steer, throttle or brake) in several classes according to given ranges
        Args:
            data: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """

        # here we have two frames' measurement, we pick up the latter one at time t+1

        if g_conf.ENCODER_MODEL_TYPE in ['forward', 'action_prediction', 'ETE_inverse_model']:
            targets_twoframes_vec = []
            for i in range(2):
                targets_vec = []
                for target_name in g_conf.TARGETS:
                    target_c_f = torch.full(data[target_name][i].shape, 0)
                    # for two classes, we take the first part as 0 and the rest 1
                    if len(g_conf.ACTION_CLASS_RANGE[target_name]) == 1:
                        target_c = torch.where(
                            data[target_name][i] < g_conf.ACTION_CLASS_RANGE[target_name][0],
                            torch.full(data[target_name][i].shape, 0),
                            torch.full(data[target_name][i].shape, 1))
                        targets_vec.append(target_c)

                    else:
                        for bin_id in range(len(g_conf.ACTION_CLASS_RANGE[target_name]) + 1):
                            if bin_id == 0:
                                continue

                            elif bin_id == len(g_conf.ACTION_CLASS_RANGE[target_name]):
                                target_c = torch.where(
                                    data[target_name][i] >= g_conf.ACTION_CLASS_RANGE[target_name][-1],
                                    torch.full(data[target_name][i].shape, bin_id),
                                    torch.full(data[target_name][i].shape, 0))

                            else:
                                target_c_1 = torch.where(
                                    data[target_name][i] >= g_conf.ACTION_CLASS_RANGE[target_name][bin_id - 1],
                                    torch.full(data[target_name][i].shape, bin_id),
                                    torch.full(data[target_name][i].shape, 0))
                                target_c_2 = torch.where(
                                    data[target_name][i] < g_conf.ACTION_CLASS_RANGE[target_name][bin_id],
                                    torch.full(data[target_name][i].shape, bin_id),
                                    torch.full(data[target_name][i].shape, 0))
                                target_c = (target_c_1 == target_c_2).type(torch.FloatTensor) * bin_id

                            target_c_f = target_c_f + target_c
                        targets_vec.append(target_c_f)
                targets_twoframes_vec.append(torch.cat(targets_vec, 1))

            return targets_twoframes_vec
