
import time
import random
import logging
import sys
import traceback
import shutil

from cexp.env.scenario_identification import distance_to_intersection, identify_scenario
from cexp.env.server_manager import start_test_server, check_test_server

from cexp.cexp import CEXP
from cexp.benchmark import benchmark, check_benchmarked_environments
from cexp.agents.NPCAgent import NPCAgent

import carla
import os


JSONFILE = 'database/sample_benchmark.json'
environments_dict_base = [
    'WetSunset_route00024',
    'SoftRainSunset_route00000',
    'WetNoon_route00024'
]

params = {'save_dataset': True,
          'docker_name': 'carlalatest:latest',
          'gpu': 5,
          'batch_size': 1,
          'remove_wrong_data': False,
          'non_rendering_mode': False,
          'carla_recording': False  # TODO testing
          }

agent = NPCAgent()
AGENT_NAME = 'NPCAgent'
# The episodes to be checked must always be sequential

def check_folder(env_name, number_episodes):

    """ Check if the folder contain the expected number of episodes
        and if they are complete.
    """

    path = os.path.join(os.environ["SRL_DATASET_PATH"], 'sample_benchmark', env_name)
    # List number of folders check if match expected

    environments_count = 0
    for filename in os.listdir(path):
        try:
            int_filename = int(filename)
            environments_count += 1
        except:
            pass

    assert environments_count == number_episodes


def check_dataset(number_episode_dics):

    """ Check if each of  folder contain the expected number of episodes """

    for env_name in number_episode_dics.keys():

        check_folder(env_name, number_episode_dics[env_name])


def check_benchmark_file(benchmark_name , expected_episodes):
    benchmark_dict = check_benchmarked_environments(JSONFILE, benchmark_name)
    print (" Produced this dict")
    print (benchmark_dict)
    benchmarked_episodes = 0

    for env_benchmarked in benchmark_dict.keys():

        benchmarked_episodes += len(benchmark_dict[env_benchmarked])


    return benchmarked_episodes


# TEST 1 Create the entire dataset and them check if the folder has one experiment per environment
def test_1_collect():

    # Collect the full dataset sequential
    # Expected one episode per

    env_batch = CEXP(JSONFILE, params, iterations_to_execute=10, sequential=True, port=6666)

    env_batch.start()
    for env in env_batch:
        try:
            # The policy selected to run this experience vector (The class basically) This policy can also learn, just
            # by taking the output from the experience.
            # I need a mechanism to test the rewards so I can test the policy gradient strategy
            _, _ = agent.unroll(env)

        except KeyboardInterrupt:
            env.stop()
            break
        except:
            traceback.print_exc()
            # Just try again
            env.stop()
            print(" ENVIRONMENT BROKE trying again.")

    test_dict = {}
    for env in environments_dict_base:
        test_dict.update({env:1})
    check_dataset(test_dict)


def test_2_collect():
    # Do four runs of
    # Collect the full dataset sequential
    # Expected one episode per

    env_batch = CEXP(JSONFILE, params, iterations_to_execute=3, sequential=True, port=6666)

    env_batch.start()
    for env in env_batch:
        try:
            # The policy selected to run this experience vector (The class basically) This policy can also learn, just
            # by taking the output from the experience.
            # I need a mechanism to test the rewards so I can test the policy gradient strategy
            _, _ = agent.unroll(env)

        except KeyboardInterrupt:
            env.stop()
            break
        except:
            traceback.print_exc()
            # Just try again
            env.stop()
            print(" ENVIRONMENT BROKE trying again.")

    print ("REDOING ")

    del env_batch
    # Redo all
    env_batch = CEXP(JSONFILE, params, iterations_to_execute=3, sequential=True, port=6666)

    env_batch.start()
    for env in env_batch:
        try:
            # The policy selected to run this experience vector (The class basically) This policy can also learn, just
            # by taking the output from the experience.
            # I need a mechanism to test the rewards so I can test the policy gradient strategy
            _, _ = agent.unroll(env)

        except KeyboardInterrupt:
            env.stop()
            break
        except:
            traceback.print_exc()
            # Just try again
            env.stop()
            print(" ENVIRONMENT BROKE trying again.")

    test_dict = {}
    for env in environments_dict_base:
        test_dict.update({env: 2})


    #check_dataset(test_dict)

    del env_batch

    print ( " NOW IT SHOULD HAVE NOT EXPS")

    env_batch = CEXP(JSONFILE, params, execute_all=True, sequential=False, port=6666)
    env_batch.start()

    for env in env_batch:
        print (env)


# TEST 3  Actually completing and doing all the episodes

if __name__ == '__main__':
    # PORT 6666 is the default port for testing server

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    if not check_test_server(6666):
        print (" WAITING FOR DOCKER TO BE STARTED")
        start_test_server(6666, gpu=5)

    #client = carla.Client('localhost', 6666)
    #client.set_timeout(45.0)
    #world = client.load_world('Town01')
    if os.path.exists(os.path.join(os.environ["SRL_DATASET_PATH"], 'sample_benchmark')):
        shutil.rmtree(os.path.join(os.environ["SRL_DATASET_PATH"], 'sample_benchmark'))

    #test_distance_intersection_speed(world)
    # The idea is that the agent class should be completely independent
    #test_1_collect()
    # Auto Cleanup
    test_2_collect()
    # this could be joined
    # THe experience is built, the files necessary

