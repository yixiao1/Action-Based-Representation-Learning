import traceback
import argparse
import time
import logging
import os
import glob
import json
import multiprocessing
import subprocess

from cexp.agents.RANDOMAgent import RANDOMAgent
from cexp.agents.NPCAgent import NPCAgent
from cexp.cexp import CEXP
import sys


# TODO I have a problem with respect to where to put files

# THE IDEA IS TO RUN EXPERIENCES IN MULTI GPU MODE SUCH AS
def collect_data(json_file, params, eliminated_environments,
                 collector_id, noise=False, randomAgent=False):
    # The idea is that the agent class should be completely independent

    # TODO this has to go to a separate file and to be merged with package
    if randomAgent:
        agent = RANDOMAgent(
            sensors_dict = [{'type': 'sensor.camera.rgb',
                    'x': 2.0, 'y': 0.0,
                    'z': 1.40, 'roll': 0.0,
                    'pitch': -15.0, 'yaw': 0.0,
                    'width': 800, 'height': 600,
                    'fov': 100,
                    'id': 'rgb_central'},



                   {'type': 'sensor.camera.rgb',
                    'x': 2.0, 'y': 0.0,
                    'z': 1.40, 'roll': 0.0,
                    'pitch': -15.0, 'yaw': -30.0,
                    'width': 800, 'height': 600,
                    'fov': 100,
                    'id': 'rgb_left'},



                   {'type': 'sensor.camera.rgb',
                    'x': 2.0, 'y': 0.0,
                    'z': 1.40, 'roll': 0.0,
                    'pitch': -15.0, 'yaw': 30.0,
                    'width': 800, 'height': 600,
                    'fov': 100,
                    'id': 'rgb_right'},

                    {'type': 'sensor.can_bus',
                     'reading_frequency': 25,
                     'id': 'can_bus'
                     },

                    {'type': 'sensor.other.gnss',
                     'x': 0.7, 'y': -0.4, 'z': 1.60,
                     'id': 'GPS'}

                   ], noise=noise)
    else:
        agent = NPCAgent(
            sensors_dict=[{'type': 'sensor.camera.rgb',
                           'x': 2.0, 'y': 0.0,
                           'z': 1.40, 'roll': 0.0,
                           'pitch': -15.0, 'yaw': 0.0,
                           'width': 800, 'height': 600,
                           'fov': 100,
                           'id': 'rgb_central'},

                          {'type': 'sensor.camera.rgb',
                           'x': 2.0, 'y': 0.0,
                           'z': 1.40, 'roll': 0.0,
                           'pitch': -15.0, 'yaw': -30.0,
                           'width': 800, 'height': 600,
                           'fov': 100,
                           'id': 'rgb_left'},

                          {'type': 'sensor.camera.rgb',
                           'x': 2.0, 'y': 0.0,
                           'z': 1.40, 'roll': 0.0,
                           'pitch': -15.0, 'yaw': 30.0,
                           'width': 800, 'height': 600,
                           'fov': 100,
                           'id': 'rgb_right'},

                          {'type': 'sensor.can_bus',
                           'reading_frequency': 25,
                           'id': 'can_bus'
                           },

                          {'type': 'sensor.other.gnss',
                           'x': 0.7, 'y': -0.4, 'z': 1.60,
                           'id': 'GPS'}

                          ], noise=noise)

    # this could be joined
    env_batch = CEXP(json_file, params=params, execute_all=True,
                     eliminated_environments=eliminated_environments)
    # THe experience is built, the files necessary
    # to load CARLA and the scenarios are made
    with open(json_file, 'r') as f:
        json_dict = json.loads(f.read())
        package_name = json_dict['package_name']

    # Here some docker was set
    if randomAgent:
        if not noise:
            env_batch.start(agent_name='Random')
        else:
            env_batch.start(agent_name='Random_noise')
    else:
        if not noise:
            env_batch.start(agent_name='Multi')
        else:
            env_batch.start(agent_name='Multi_noise')

    for env in env_batch:
        try:
            # The policy selected to run this experience vector (The class basically) This policy can also learn, just
            # by taking the output from the experience.
            # I need a mechanism to test the rewards so I can test the policy gradient strategy
            print (" Collector ", collector_id, " Collecting for ", env)
            states, rewards = agent.unroll(env)
            agent.reinforce(rewards)
        except KeyboardInterrupt:
            env.stop()
            break
        except:
            traceback.print_exc()
            # Just try again
            agent.reset()
            env.stop()
            print(" ENVIRONMENT BROKE trying again.")

    env_batch.cleanup()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


def execute_collector(json_file, params, eliminated_environments,
                      collector_id, noise=False, randomAgent=False):
    p = multiprocessing.Process(target=collect_data,
                                args=(json_file, params,
                                      eliminated_environments, collector_id, noise, randomAgent))
    p.start()


def get_eliminated_environments(json_file, start_position, end_position):

    """
    List all the episodes BUT the range between start end position.
    """
    with open(json_file, 'r') as f:
        json_dict = json.loads(f.read())

    count = 0
    eliminated_environments_list = []
    for env_name in json_dict['envs'].keys():
        if count < start_position or count >= end_position:
            eliminated_environments_list.append(env_name)
        count += 1
    return eliminated_environments_list

def test_eliminated_environments_division():

    pass


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Release Data Collectors')
    argparser.add_argument(
        '-n', '--number_collectors',
        default=1,
        type=int,
        help=' the number of collectors used')
    # TODO add some general repetition
    argparser.add_argument(
        '-b', '--batch_size',
        default=1,
        type=int,
        help=' the batch size for the execution')
    argparser.add_argument(
        '-s', '--start_episode',
        default=0,
        type=int,
        help=' the first episode')
    argparser.add_argument(
        '-d', '--delete-wrong',
        action="store_true",
        help=' the first episode')
    argparser.add_argument(
        '-j', '--json-config',
        help=' full path to the json configuration file',
        required=True)
    argparser.add_argument(
        '-ct', '--container-name',
        dest='container_name',
        default='carlalatest:latest',
        help='The name of the docker container used to collect data',
        required=True)
    argparser.add_argument(
        '-ge',
        nargs='+',
        dest='eliminated_gpus',
        type=str)
    argparser.add_argument(
        '-r', '--resize-images',
        action="store_true",
        help=' resize images once the episode finished')
    argparser.add_argument(
        '-o', '--add-noise',
        action="store_true",
        help=' adding noise during data collection')
    argparser.add_argument(
        '-ra', '--random-agent',
        action="store_true",
        help='using random agent during data collection')

    args = argparser.parse_args()
    print(os.path.realpath(__file__).split('/')[:-1])

    json_file = args.json_config

    with open(json_file, 'r') as f:
        json_dict = json.loads(f.read())

    environments_per_collector = len(json_dict['envs'])/args.number_collectors
    if environments_per_collector < 1.0:
        raise ValueError(" Too many collectors")
    if not environments_per_collector.is_integer():
        raise ValueError(" Number of Collectors must divide the number of envs %d " % len(json_dict['envs']))

    # Set GPUS to eliminate.
    # we get all the gpu (STANDARD 10, make variable)
    gpu_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # we eliminate the ones not used
    if args.eliminated_gpus is not None:
        for el in args.eliminated_gpus:
            del gpu_list[gpu_list.index(el)]

    print (" FINAL LIST", gpu_list)
    for i in range(args.number_collectors):
        gpu = gpu_list[i % len(gpu_list)]
        print (" GPU ", gpu)
        # A single loop being made
        # Dictionary with the necessary params related to the execution not the model itself.
        params = {'save_dataset': True,
                  'save_sensors': True,
                  'save_trajectories': False,
                  'resize_images': args.resize_images,
                  'docker_name': args.container_name,
                  'gpu': gpu,
                  'batch_size': 1,
                  'remove_wrong_data': args.delete_wrong,
                  'non_rendering_mode': False,
                  'carla_recording': False,
                  'trajectories_directory': os.path.join('database', args.json_config + '_trajectories')
                  }

        if i == args.number_collectors-1 and not environments_per_collector.is_integer():
            extra_env = 1
        else:
            extra_env = 0

        # we list all the possible environments
        eliminated_environments = get_eliminated_environments(json_file,
                                                              int(environments_per_collector) * (i),
                                                              int(environments_per_collector) * (i+1)
                                                              + extra_env)

        print (" Collector ", i, "Start ",  int(environments_per_collector) * (i),
               "End ", int(environments_per_collector) * (i+1) + extra_env)

        execute_collector(json_file, params, eliminated_environments, i, noise=args.add_noise, randomAgent=args.random_agent)