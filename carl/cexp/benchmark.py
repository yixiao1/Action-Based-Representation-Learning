import os
import logging
import json
import sys
import importlib
import shutil
import numpy as np
import traceback

from cexp.cexp import CEXP
from cexp.env.datatools.data_parser import read_benchmark_summary, read_benchmark_summary_metric

# TODO ADD the posibility to configure what goes in and what goes out ( OUput format)
###


def parse_results_summary(summary):

    result_dictionary = {
        'episodes_completion': summary['score_route'],
        'result': float(summary['result'] == 'SUCCESS'),
        'infractions_score': summary['score_penalty'],
        'number_red_lights': summary['number_red_lights'],
        'total_number_traffic_lights': summary['total_number_traffic_lights']
    }

    return result_dictionary


def check_benchmarked_environments(json_filename, agent_checkpoint_name):

    """ return a dict with each environment that has a vector of dicts of results

        The len of each environment is the number of times this environment has been benchmarked.
     """

    benchmarked_environments = {}

    with open(json_filename, 'r') as f:
        json_file = json.loads(f.read())

    if not os.path.exists(os.path.join(os.environ["SRL_DATASET_PATH"], json_file['package_name'])):
        return {}  # return empty dictionary no case was benchmarked

    for env_name in json_file['envs'].keys():
        path = os.path.join(os.environ["SRL_DATASET_PATH"],  json_file['package_name'], env_name,
                            agent_checkpoint_name + '_benchmark_summary.csv')
        if os.path.exists(path):
            env_summary, _ = read_benchmark_summary(path)
            benchmarked_environments.update({env_name: env_summary})

    return benchmarked_environments

def check_benchmark_finished(json_filename, agent_checkpoint_name):

    with open(json_filename, 'r') as f:
        json_file = json.loads(f.read())

    if not os.path.exists(os.path.join(os.environ["SRL_DATASET_PATH"], json_file['package_name'])):
        print (" PACKAGE DOES NOT EXIST")
        return False  # return empty dictionary no case was benchmarked

    for env_name in json_file['envs'].keys():
        path = os.path.join(os.environ["SRL_DATASET_PATH"],  json_file['package_name'], env_name,
                            agent_checkpoint_name + '_benchmark_summary.csv')
        if not os.path.exists(path):
            print ( " PATH ", path, "does not exist")
            return False

        else:
            data_matrix = np.loadtxt(open(path, "rb"), delimiter=",", skiprows=1)
            if len(data_matrix) == 0:
                return False

    return True


def check_benchmarked_episodes_metric(json_filename, agent_checkpoint_name):

    """ return a dict with each metric from the header and
        The len of each environment is the number of times this environment has been benchmarked.
     """

    benchmarked_metric_dict = {}

    with open(json_filename, 'r') as f:
        json_file = json.loads(f.read())

    if not os.path.exists(os.path.join(os.environ["SRL_DATASET_PATH"], json_file['package_name'])):
        return {}  # return empty dictionary no case was benchmarked

    for env_name in json_file['envs'].keys():
        path = os.path.join(os.environ["SRL_DATASET_PATH"],  json_file['package_name'], env_name,
                            agent_checkpoint_name + '_benchmark_summary.csv')

        if os.path.exists(path):
            benchmark_env_results, header = read_benchmark_summary(path)

            if not benchmarked_metric_dict:
                # This is the first iteration, we use it to take the header.
                for key in header[1:]:
                    benchmarked_metric_dict.update({key:[]})

            for rep_key in benchmark_env_results.keys():
                for info, key in zip(benchmark_env_results[rep_key], header[1:]):
                    benchmarked_metric_dict[key].append(info)

    return benchmarked_metric_dict




def add_summary(environment_name, summary, json_filename, agent_checkpoint_name):
    """
    Add summary file, if it exist writte another repetition.
    :param environment_name:
    :param summary:
    :param json_filename:
    :param agent_checkpoint_name:
    :return:
    """
    # The rep is now zero, but if the benchmark already started we change that
    repetition_number = 0

    with open(json_filename, 'r') as f:
        json_file = json.loads(f.read())
    # if it doesnt exist we add the file, this is how we are writting.
    filename = os.path.join(os.environ["SRL_DATASET_PATH"], json_file['package_name'],
                            environment_name, agent_checkpoint_name + '_benchmark_summary.csv')
    set_of_metrics = ['episodes_completion', 'result', 'infractions_score',
                      'number_red_lights', 'total_number_traffic_lights']

    if not os.path.exists(filename):
        csv_outfile = open(filename, 'w')
        csv_outfile.write("%s,%s,%s,%s,%s,%s\n"
                          % ('rep', 'episodes_completion', 'result', 'infractions_score',
                             'number_red_lights', 'total_number_traffic_lights'))

        csv_outfile.close()

    else:
        # Check the summary to get the repetition number
        summary_exps = check_benchmarked_environments(json_filename, agent_checkpoint_name)

        env_experiments = summary_exps[environment_name]

        repetition_number = len(env_experiments.keys())

    logging.debug("added summary for " + agent_checkpoint_name + '_benchmark_summary.csv')
    # parse the summary for this episode
    results = parse_results_summary(summary)

    csv_outfile = open(filename, 'a')
    csv_outfile.write("%f" % float(repetition_number) )
    for metric_result in set_of_metrics:

        csv_outfile.write(",%f" % results[metric_result])

    csv_outfile.write("\n")
    csv_outfile.close()




def benchmark(benchmark_name, docker_image, gpu, agent_class_path, agent_params_path,
              batch_size=1, save_dataset=False, port=None, save_sensors=False,
              agent_checkpoint_name=None, save_trajectories=False, direct_read=False):

    """
    Computes the benchmark for a given json file containing a certain number of experiences.

    :param benchmark_name: the name of the json file used for the benchmark
    :param docker_image: the docker image that is going to be created to perform the bench
    :param gpu: the gpu number to be used
    :param agent_class_path: the pointer to the agent that is going to be benchmarked
    :param agent_params_path: the pointer to the params file of the agent
    :param batch_size: number of repetions ( Simultaneous ) NOT IMPLEMENTED
    :param number_repetions: number of repetitions necessary
    :param save_dataset: if you are saving the data when benchmarking
    :param port: the port, in this case expect the docker to not be initialized
    :return:
    """

    module_name = os.path.basename(agent_class_path).split('.')[0]
    sys.path.insert(0, os.path.dirname(agent_class_path))
    agent_module = importlib.import_module(module_name)
    if agent_checkpoint_name is None:
        agent_checkpoint_name = agent_module.__name__

    params = {'save_dataset': save_dataset,
              'save_sensors': save_sensors,
              'docker_name': docker_image,
              'save_trajectories': save_trajectories,
              'gpu': gpu,
              'batch_size': batch_size,
              'remove_wrong_data': False,
              'non_rendering_mode': False,  # This could be added as a parameter
              'carla_recording': False,
              'direct_read': direct_read
              }
    env_batch = None
    # this could be joined
    while True:
        try:
            # We reattempt in case of failure of the benchmark
            env_batch = CEXP(benchmark_name, params, execute_all=True, port=port)
            # to load CARLA and the scenarios are made
            # Here some docker was set
            env_batch.start(agent_name=agent_checkpoint_name)
            # take the path to the class and instantiate an agent
            agent = getattr(agent_module, agent_module.__name__)(agent_params_path)
            # if there is no name for the checkpoint we set it as the agent module name
            for env in env_batch:
                try:
                    print(" =======> Driving in environment: ", env)
                    _, _ = agent.unroll(env)
                    # Just execute the environment. For this case the rewards doesnt matter.

                    summary = env.get_info()['summary']
                    logging.debug("Finished episode got summary ")
                    # Add partial summary to allow continuation
                    add_summary(env._environment_name, summary,
                                benchmark_name, agent_checkpoint_name)

                except KeyboardInterrupt:
                    env.stop()
                    break
                except Exception as e:
                    traceback.print_exc()
                    # By any exception you have to delete the environment generated data
                    env.eliminate_data()
                    # And you have to try again so we retry everything and rebuild the CEXP
                    env.stop()
                    # TODO maybe keep the docker
                    raise e

            del env_batch
            break
        except KeyboardInterrupt:
            del env_batch
            break
        except:
            traceback.print_exc()




def benchmark_cleanup(package_name, agent_checkpoint_name):

    shutil.rmtree(os.environ["SRL_DATASET_PATH"], package_name,
                  agent_checkpoint_name)
