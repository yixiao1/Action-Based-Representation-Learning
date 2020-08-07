
import traceback

import sys
import logging

import os
import time
import subprocess
import socket
import json
import signal

from contextlib import closing

from logger import coil_logger
from configs import g_conf, merge_with_yaml, set_type_of_process
from coilutils.checkpoint_schedule import  maximun_checkpoint_reach, get_next_checkpoint,\
    is_next_checkpoint_ready, get_latest_evaluated_checkpoint, validation_stale_point
#from coilutils.general import compute_average_std_separatetasks,  write_header_control_summary,\
#     write_data_point_control_summary, camelcase_to_snakecase, unique
#from plotter.plot_on_map import plot_episodes_tracks
from cexp.benchmark import benchmark, check_benchmarked_episodes_metric, check_benchmark_finished
from coilutils.drive_utils import write_summary_csv

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

# TODO we keep the same strategy

def cexp_benchmark(benchmark_conf_path, checkpoint_number, gpu, params, exp_batch, exp_alias, encoder_params):
    # experiment_set, exp_batch, exp_alias, control_filename, task_list):
    # TODO we need more ways of sumarizing such as joining tasks, etc that probably goes to a single file

    # We create a name to log this agent

    if encoder_params is not None:
        print(' =======> Running ETE model with ENCODER')
        agent_checkpoint_name = str(exp_batch) + '_' + str(exp_alias + '_' + str(encoder_params['encoder_checkpoint'])) + '_' + str(checkpoint_number)
        with open(os.path.join('_logs', exp_batch, exp_alias + '_' + str(encoder_params['encoder_checkpoint']), 'config.json'), 'w') as fo:
            json_input_dict = {
                "yaml": "configs/" + exp_batch + "/" + exp_alias + '.yaml',
                "checkpoint": checkpoint_number,
                "agent_name": agent_checkpoint_name,
                "encoder_params": encoder_params
            }
            fo.write(json.dumps(json_input_dict, sort_keys=True, indent=4))
        print ("            START DOCKER: ", params['docker'])

        # We use the baseline CEXP directly here it compute any missing episode for this benchmark
        benchmark(benchmark_conf_path, params['docker'], gpu, 'drive/CoILBaselineCEXP.py',
                  agent_params_path=os.path.join('_logs', exp_batch, exp_alias + '_' + str(encoder_params['encoder_checkpoint']),'config.json'),
                  batch_size=1, save_dataset=False, save_sensors=False,
                  agent_checkpoint_name=agent_checkpoint_name)

        benchmark_name = benchmark_conf_path.split('/')[-1].split('.')[0]
        file_base_out = os.path.join('_logs', exp_batch, exp_alias + '_' + str(encoder_params['encoder_checkpoint']),
                                     g_conf.PROCESS_NAME + '_csv',
                                     benchmark_name + '.csv')

        summary_data_file = os.path.join('_logs', exp_batch, exp_alias + '_' + str(encoder_params['encoder_checkpoint']),
                                     g_conf.PROCESS_NAME + '_csv',
                                     benchmark_name + '_summary.csv')

        summary_data = check_benchmarked_episodes_metric(benchmark_conf_path, agent_checkpoint_name)

        if check_benchmark_finished(benchmark_conf_path, agent_checkpoint_name):
            write_summary_csv(file_base_out, agent_checkpoint_name, summary_data, summary_data_file)

    else:
        print(' =======> Running ETE model without ENCODER')
        agent_checkpoint_name = str(exp_batch) + '_' + str(exp_alias) + '_' + str(checkpoint_number)
        # We write the params file
        with open(os.path.join('_logs', exp_batch, exp_alias, 'config.json'),'w') as fo:
            json_input_dict = {
                "yaml": "configs/" + exp_batch + "/" + exp_alias + '.yaml',
                "checkpoint": checkpoint_number,
                "agent_name": agent_checkpoint_name,
                "encoder_params": None
            }
            fo.write(json.dumps(json_input_dict, sort_keys=True, indent=4))
        print ("            START DOCKER: ", params['docker'])

        # We use the baseline CEXP directly here it compute any missing episode for this benchmark
        benchmark(benchmark_conf_path, params['docker'], gpu, 'drive/CoILBaselineCEXP.py',
                                 agent_params_path = os.path.join('_logs', exp_batch, exp_alias, 'config.json'),
                                 batch_size=1, save_dataset=False, save_sensors=False,
                                 agent_checkpoint_name=agent_checkpoint_name)

        benchmark_name = benchmark_conf_path.split('/')[-1].split('.')[0]
        file_base_out = os.path.join('_logs', exp_batch, exp_alias,
                                     g_conf.PROCESS_NAME + '_csv',
                                     benchmark_name + '.csv')

        summary_data_file = os.path.join('_logs', exp_batch, exp_alias,
                                         g_conf.PROCESS_NAME + '_csv',
                                         benchmark_name + '_summary.csv')

        summary_data = check_benchmarked_episodes_metric(benchmark_conf_path, agent_checkpoint_name)
        if check_benchmark_finished(benchmark_conf_path, agent_checkpoint_name):

            write_summary_csv(file_base_out, agent_checkpoint_name, summary_data, summary_data_file)


def execute(gpu, exp_batch, exp_alias, benchmark_conf_path, params, encoder_params):
    """
    Main loop function. Executes driving benchmarks the specified iterations.
    Args:
        gpu:
        exp_batch:
        exp_alias:
        benchmark_name: The full path for the json file that is the specification of the benchmark to be used
        params:

    Returns:

    """

    try:
        print("Running ", __file__, " On GPU ", gpu, "of experiment name ", exp_alias)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        if not os.path.exists('_output_logs'):
            os.mkdir('_output_logs')

        merge_with_yaml(os.path.join('configs', exp_batch, exp_alias + '.yaml'), encoder_params)

        set_type_of_process('drive', benchmark_conf_path)

        if params['suppress_output']:
            sys.stdout = open(os.path.join('_output_logs',
                              g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out"),
                              "a", buffering=1)
            sys.stderr = open(os.path.join('_output_logs',
                              exp_alias + '_err_'+g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out"),
                              "a", buffering=1)

        coil_logger.add_message('Loading', {'Poses': benchmark_conf_path})
        benchmark_name = benchmark_conf_path.split('/')[-1].split('.')[0]

        """
            #####
            Preparing the output files that will contain the driving summary
            #####
        """

        latest = get_latest_evaluated_checkpoint(benchmark_name)
        if latest is not None:
            print(" =====> Finished ETE checkpoint until: ", int(latest))
        else:
            print(" =====> None of ETE checkpoints has yet been started ")

        """ 
            ######
            Run a single driving benchmark specified by the checkpoint were validation is stale
            ######
        """

        if g_conf.FINISH_ON_VALIDATION_STALE is not None:

            while validation_stale_point(g_conf.FINISH_ON_VALIDATION_STALE) is None:
                time.sleep(0.1)

            validation_state_iteration = validation_stale_point(g_conf.FINISH_ON_VALIDATION_STALE)
            cexp_benchmark(benchmark_conf_path, validation_state_iteration,
                           gpu, params, exp_batch, exp_alias)

        else:
            """
            #####
            Main Loop , Run a benchmark for each specified checkpoint on the "Test Configuration"
            #####
            """
            while not maximun_checkpoint_reach(latest, g_conf.TEST_SCHEDULE):
                try:
                    # Get the correct checkpoint
                    # We check it for some task name, all of then are ready at the same time
                    if is_next_checkpoint_ready(g_conf.TEST_SCHEDULE, benchmark_name):

                        next_checkpoint = get_next_checkpoint(g_conf.TEST_SCHEDULE, benchmark_name)
                        print(" =====> Now is going to drive ETE checkpoint: ", next_checkpoint)
                        cexp_benchmark(benchmark_conf_path, next_checkpoint,
                                       gpu, params, exp_batch, exp_alias, encoder_params)

                    else:
                        time.sleep(0.1)

                except KeyboardInterrupt:
                    traceback.print_exc()
                    coil_logger.add_message('Error', {'Message': 'Killed By User'})
                    break

                except:
                    traceback.print_exc()
                    coil_logger.add_message('Error', {'Message': 'Something happened'})
                    break


            coil_logger.add_message('Finished', {})

    except KeyboardInterrupt:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        coil_logger.add_message('Error', {'Message': 'Killed By User'})

    except:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        coil_logger.add_message('Error', {'Message': 'Something happened'})

    os.kill(os.getpid(), signal.SIGTERM)




