#!/usr/bin/env python
import sys

import logging
import os
import json
import numpy as np
import glob
import re
import shutil


import time
from cexp.cexp import CEXP
from cexp.env.scenario_identification import identify_scenario
from cexp.env.environment import NoDataGenerated
#import seaborn as sns
from other.screen_manager import ScreenManager
import argparse
import scipy.ndimage
from skimage import io
import numpy as np


import  subprocess

############################
##########################

# get the speed
def orientation_vector(measurement_data):
    pitch = np.deg2rad(measurement_data['orientation'][0])
    yaw = np.deg2rad(measurement_data['orientation'][0])
    orientation = np.array([np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), np.sin(pitch)])
    return orientation

def forward_speed(measurement_data):
    vel_np = np.array(measurement_data['velocity'])
    speed = np.dot(vel_np, orientation_vector(measurement_data))

    return speed

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def make_video(folder_path, output_name):

    # ffmpeg -f image2 -r 1/5 -i image%05d.png -vcodec mpeg4 -y movie.mp4

    # '-r', '1/5'
    subprocess.call(['ffmpeg', '-f', 'image2', '-i', os.path.join(folder_path, 'image%05d.png') ,
                     '-vcodec', 'mpeg4', '-y', output_name + '.mp4'])

    shutil.rmtree(folder_path)
    os.mkdir(folder_path)


# ***** main loop *****
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Path viewer')
    # parser.add_argument('model', type=str, help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('-pt', '--path', default="")

    parser.add_argument(
        '--episodes',
        nargs='+',
        dest='episodes',
        type=str,
        default='all'
    )
    parser.add_argument(
        '--agent-name',
        help=' the json configuration file name',
        default=None
    )

    parser.add_argument(
        '-s', '--step_size',
        type=int,
        default=1
    )
    parser.add_argument(
        '--dataset',
        help=' the json configuration file name',
        default=None
    )
    parser.add_argument(
        '--make-videos',
        help=' make videos from episodes',
        action='store_true'
    )

    args = parser.parse_args()
    path = args.path

    # make the temporary folder for images if we make videos
    if args.make_videos:
        if not os.path.exists('_tmp_img'):
            os.mkdir('_tmp_img')

    first_time = True
    count = 0
    step_size = args.step_size

    # Start a screen to show everything. The way we work is that we do IMAGES x Sensor.
    # But maybe a more arbitrary configuration may be useful
    screen = None
    # We keep the three camera configuration with central well

    central_camera_name = 'rgb_central'
    left_camera_name = 'rgb_central'
    right_camera_name = 'rgb_central'

    # A single loop being made
    jsonfile = args.dataset
    # Dictionary with the necessary params related to the execution not the model itself.
    params = {'save_dataset': True,
              'save_sensors': True,
              'docker_name': 'carlalatest:latest',
              'gpu': 0,
              'batch_size': 1,
              'remove_wrong_data': False,
              'non_rendering_mode': False,
              'carla_recording': True
              }
    # TODO for now batch size is one
    number_of_iterations = 123

    # this could be joined
    # THe experience is built, the files necessary
    env_batch = CEXP(jsonfile, params, execute_all=True, ignore_previous_execution=True)
    # Here some docker was set
    env_batch.start(no_server=True, agent_name=args.agent_name)  # no carla server mode.
    # count, we count the environments that are read
    for env in env_batch:
        steer_vec = []
        throttle_vec = []
        brake_vec = []
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
                    print("      Batch: ", batch[1])
                    step = 0  # Add the size
                    count_images = 0
                    while step < len(batch[0]):

                        data_point = batch[0][step]
                        rgb_center = io.imread(data_point[central_camera_name])[:,:,:3]
                        rgb_left = io.imread(data_point[left_camera_name])[:,:,:3]
                        rgb_right = io.imread(data_point[right_camera_name])[:,:,:3]

                        if screen is None:
                            screen = ScreenManager()
                            screen.start_screen([rgb_center.shape[1], rgb_center.shape[0]], [3, 1],
                                                1, no_display=True)

                        status = {'speed': forward_speed(data_point['measurements']['ego_actor']),
                                  'directions': 2.0,
                                  'distance_intersection': data_point['measurements']['distance_intersection'],
                                  'road_angle': data_point['measurements']['road_angle'],
                                  'scenario': identify_scenario(data_point['measurements']['distance_intersection'],
                                                                data_point['measurements']['road_angle'])
                                  }

                        # if we make video we set something for output image
                        if args.make_videos:
                            output_image = os.path.join('_tmp_img',
                                           'image' + str(count_images).zfill(5) + '.png')
                        else:
                            output_image = None

                        screen.plot_camera_steer(rgb_left, screen_position=[0, 0])
                        screen.plot_camera_steer(rgb_center, control=None,
                                                 screen_position=[1, 0], status=status)
                        screen.plot_camera_steer(rgb_right, screen_position=[2, 0],
                                                 output_img=output_image)
                        step += step_size
                        count_images += 1
                    if args.make_videos:
                        make_video('_tmp_img', 'env_'+ env._environment_name +'exp_'+str(exp[1])
                                                +'_batch_' + str(batch[1]))


            print("################################")


    # TODO add a more detailed summary for the enves that were collected.

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    
    
    #screen.plot_camera_steer(rgb_center, control=[data_point['measurements']['steer'],
    #                                          data_point['measurements']['throttle'],
    #                                          data_point['measurements']['brake']],
    #                     screen_position=[1, 0], status=status)