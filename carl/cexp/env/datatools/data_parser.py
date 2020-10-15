import glob
import os
import json
import numpy as np

from cexp.env.utils.general import sort_nicely



def read_benchmark_summary(benchmark_csv):
    """
        Make a dict of the benchmark csv were the keys are the environment names

    :param benchmark_csv:
    :return:
    """

    # If the file does not exist, return None,None, to point out that data is missing
    if not os.path.exists(benchmark_csv):
        return None, None

    f = open(benchmark_csv, "r")
    header = f.readline()
    header = header.split(',')
    header[-1] = header[-1][:-2]
    f.close()

    data_matrix = np.loadtxt(open(benchmark_csv, "rb"), delimiter=",", skiprows=1)
    control_results_dic = {}
    count = 0

    if len(data_matrix) == 0:
        return None, None
    if len(data_matrix.shape) == 1:
        data_matrix = np.expand_dims(data_matrix, axis=0)

    for env_name in data_matrix[:, 0]:

        control_results_dic.update({env_name: data_matrix[count, 1:]})
        count += 1

    return control_results_dic, header


def read_benchmark_summary_metric(benchmark_csv):
    """
        Make a dict of the benchmark csv were the keys are the environment names

    :param benchmark_csv:
    :return:
    """

    f = open(benchmark_csv, "rU")
    header = f.readline()
    header = header.split(',')
    header[-1] = header[-1][:-2]
    f.close()

    data_matrix = np.loadtxt(benchmark_csv, delimiter=",", skiprows=1)
    summary_dict = {}

    if len(data_matrix) == 0:
        return None

    if len(data_matrix.shape) == 1:
        data_matrix = np.expand_dims(data_matrix, axis=0)

    count = 0
    for _ in header:
        summary_dict.update({header[count]: data_matrix[:, count]})
        count += 1

    return summary_dict

""" Parse the data that was already written """


def get_number_executions(agent_name, environments_path):
    """
    List all the environments that

    :param path:
    :return:
    """

    number_executions = {}
    envs_list = glob.glob(os.path.join(environments_path, '*'))
    for env in envs_list:
        env_name = env.split('/')[-1]
        results, _ = read_benchmark_summary(os.path.join(env,
                                                         agent_name + '_benchmark_summary.csv')
                                            )
        if results is None:
            number_executions.update({env_name: 0})
        else:
            number_executions.update({env_name: len(results)})

        #for file in os.listdir(env):
        #    env_exec_name = os.path.join(env, file)
        #    print("     file ", '_'.join(file.split('_')[1:]))
        #    if os.path.isdir(env_exec_name) and '_'.join(file.split('_')[1:]) == agent_name:
        #        # it exist but it needs to have a summary !


    # We should reduce the fact that we have the metadata
    return number_executions


def parse_measurements(measurement):
    with open(measurement) as f:
        measurement_data = json.load(f)
    return measurement_data

def parse_scenarios(measurement):
    with open(measurement) as f:
        scenario_data = json.load(f)
    return scenario_data

def parse_environment(path, metadata_dict, read_sensors=True, agent_name=''):
    """

    :param path:
    :param metadata_dict:
    :param read_sensors: if we are going to read the sensors described on the environment
                         description
    :return:
    """

    # We start on the root folder, We want to list all the episodes
    experience_list = glob.glob(os.path.join(path, '[0-9]_' + agent_name + '*'))

    sensors_types = metadata_dict['sensors']

    # TODO probably add more metadata
    # the experience number
    exp_vec = []

    print ( "READ SENSORS ", read_sensors)
    print ("expected sensor types ", sensors_types)
    for exp in experience_list:

        batch_list = glob.glob(os.path.join(exp, '[0-9]'))
        batch_vec = []
        for batch in batch_list:
            if 'summary.json' not in os.listdir(batch):
                print (" Episode not finished skiping...")  #TODO this is a debug message on my logging system YET TO BE MADE
                continue

            measurements_list = glob.glob(os.path.join(batch, 'measurement*'))
            sort_nicely(measurements_list)
            ## Written scenario list.

            scenario_list = glob.glob(os.path.join(batch, 'scenario*'))
            sort_nicely(scenario_list)
            sensors_lists = {}

            if read_sensors:
                for sensor in sensors_types:
                    sensor_l = glob.glob(os.path.join(batch, sensor['id'] + '*'))
                    sort_nicely(sensor_l)
                    sensors_lists.update({sensor['id']: sensor_l})
            #print (sensors_types)


            data_point_vec = []
            print ( "This ", len(measurements_list), " Measurements ")
            for i in range(len(measurements_list)):
                data_point = {}
                data_point.update({'measurements': parse_measurements(measurements_list[i])})
                if i < len(scenario_list):
                    data_point.update({'scenario': parse_scenarios(scenario_list[i])['scenario']})
                if read_sensors:
                    for sensor in sensors_types:
                        # TODO can bus and GPS are not implemented a sensors yet
                        if sensor['id'] == 'can_bus' or sensor['id'] == 'GPS':
                            continue

                        data_point.update({sensor['id']: sensors_lists[sensor['id']][i]})

                data_point_vec.append(data_point)
            batch_vec.append((data_point_vec, batch.split('/')[-1]))

        # It is a tuple with the data and the data folder name
        exp_vec.append((batch_vec, exp.split('/')[-1]))

    return exp_vec


