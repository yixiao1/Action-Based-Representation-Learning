
import time
import random
import json
import logging
import sys
import shutil
import traceback

from cexp.env.scenario_identification import distance_to_intersection, identify_scenario
from cexp.env.server_manager import start_test_server, check_test_server

from cexp.cexp import CEXP
from cexp.benchmark import benchmark, check_benchmarked_environments, read_benchmark_summary_metric
from cexp.agents.NPCAgent import NPCAgent

import carla
import os


JSONFILE = 'database/sample_benchmark2.json'
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

    path = os.path.join(os.environ["SRL_DATASET_PATH"], 'sample_benchmark2', env_name)
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

    print ("Benchmarked ", benchmarked_episodes, " episodes")
    return benchmarked_episodes



def summarize_benchmark(benchmark_name, agent_name, checkpoint):


    final_dictionary = {
        'episode_completion': 'episodes_completion',
        'result': 'episodes_fully_completed'
    }

    input_metrics = {
        'episode_completion': 0,
        'result': 0
    }

    agent_checkpoint_name = agent_name

    # go on each of the folders that you can find inside.

    with open(benchmark_name, 'r') as f:
        json_file = json.loads(f.read())

    for env_name in json_file['envs'].keys():

        path = os.path.join(os.environ["SRL_DATASET_PATH"],  json_file['package_name'], env_name,
                            agent_checkpoint_name + '_benchmark_summary.csv')
        print (" PATH ", path)
        if not os.path.exists(path):
            raise ValueError("Trying to get summary of unfinished benchmark")

        results = read_benchmark_summary_metric(path)
        print ( " REUSLTS  ")
        print (results)

        for metric in input_metrics.keys():
            try:
                final_dictionary[metric] = sum(results[metric]) / len(json_file['envs'])
            except KeyError:  # To overcomme the bug on reading files csv
                final_dictionary[metric] = sum(results[metric[:-1]]) / len(json_file['envs'])

    outfile_name = os.path.join(os.environ["SRL_DATASET_PATH"], json_file['package_name'],
                                benchmark_name.split('.')[-2].split('/')[-1] + '.csv')
    csv_outfile = open(outfile_name, 'w')

    csv_outfile.write("%s,%s,%s\n"
                      % ('step', 'episodes_completion', 'episodes_fully_completed'))
    csv_outfile.write("%f" % (0))

    for metric in final_dictionary.keys():

        csv_outfile.write(",%f" % (final_dictionary[metric]))

    csv_outfile.write("\n")

    csv_outfile.close()

# TEST a simple run of a benchmark

def test_1_benchmark():
    # Benchmark the full dataset, test the output file
    benchmark(JSONFILE, None, "5", 'cexp/agents/NPCAgent.py', None, port=4444)
    summarize_benchmark(JSONFILE, 'NPCAgent', '')


# TEST 2 Squential benchmark, run one episode fail and continue




# TEST 3  Random adding and many problems

if __name__ == '__main__':
    # PORT 6666 is the default port for testing server


    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    if not check_test_server(4444):
        print (" WAITING FOR DOCKER TO BE STARTED")
        start_test_server(4444)



    #if os.path.exists(os.path.join(os.environ["SRL_DATASET_PATH"], 'sample_benchmark2')):
    #    shutil.rmtree(os.path.join(os.environ["SRL_DATASET_PATH"], 'sample_benchmark2'))


    #test_distance_intersection_speed(world)
    # The idea is that the agent class should be completely independent
    #test_1_collect()
    # Auto Cleanup
    test_1_benchmark()
    # this could be joined
    # THe experience is built, the files necessary

    #test_2_benchmark()
