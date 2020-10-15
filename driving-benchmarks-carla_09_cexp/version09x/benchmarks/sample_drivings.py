
import json
import os
import sys
import importlib



def perform(docker, gpu, agent, config, port, agent_name, non_rendering_mode,
            save_trajectories, small=False, make_videos=False):

    """

    :param docker:
    :param gpu:
    :param agent:
    :param config:
    :param port:
    :param agent_name:
    :param non_rendering_mode:
    :param save_trajectories:
    :param small: If small is set to true it means that the only new town conditions will be
                  used
    :return:
    """
    # Perform the benchmark

    #conditions = ['training']
    conditions = ['newtown']

    #tasks = ['regular', 'dense', 'empty']
    tasks = ['dense']
    #tasks = ['empty']
    #tasks = ['regular']

    towns = {'training': 'Town01',
             'newweather': 'Town01',
             'newtown': 'Town02',
             'newweathertown': 'Town02',
             'trainingsw': 'Town01',
             'newtownsw': 'Town02'}

    module_name = os.path.basename(agent).split('.')[0]
    sys.path.insert(0, os.path.dirname(agent))
    print ( "HANG ON IMPORT")
    agent_module = importlib.import_module(module_name)
    if agent_name is None:
        agent_name = agent_module.__name__


    from version09x.benchmark import execute_benchmark

    for c in conditions:
        for t in tasks:
            file_name = os.path.join('sample_drivings', 'nocrash_' + c + '_' + t + '_' + towns[c] + '_samples.json')
            if is_generated(os.path.join('version09x/descriptions', file_name)):
                execute_benchmark(file_name,
                              docker, gpu, agent_module, config, port=port,
                              agent_name=agent_name,
                              non_rendering_mode=non_rendering_mode,
                              save_trajectories=save_trajectories,
                              make_videos=make_videos)

            else:
                print('YOU NEED TO GENERATE JSON FILE FIRST !!')




def is_generated(benchmark_name):

    if os.path.exists(benchmark_name):
        return True

    else:
        return False

