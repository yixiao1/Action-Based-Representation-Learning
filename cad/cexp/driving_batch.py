import json
import carla
import random
import collections
import os
import logging

from cexp.env.utils.general import sort_nicely_dict
from cexp.env.server_manager import ServerManagerDocker, find_free_port, check_test_server
from cexp.env.environment import Environment
import cexp.env.utils.route_configuration_parser as parser


# Consume the environments based on how many times they were already executed
# Consume de environmens ignoring if they were executed already
# Do an execution eliminating part of the environments used.


class DrivingBatch(object):
    """
    THE main CEXP module.
    It contains the instanced env files that can be iterated to have instanced environments.
    """

    _default_params = {'save_dataset': False,
                       'save_sensors': False,
                       'save_opponents': False,
                       'save_walkers': False,
                       'make_videos': False,
                       'docker_name': None,
                       'save_affordances': True,
                       'gpu': 0,
                       'batch_size': 1,
                       'remove_wrong_data': False,
                       'non_rendering_mode': False,
                       'carla_recording': True,
                       'direct_read': False
                      }

    def __init__(self, jsonfile, params=None, port=None,
                 ignore_previous_execution=False,
                 eliminated_environments=None):
        """
        The initialization should receive the following parameters
        :param jsonfile:
        :param params:
        :param iterations_to_execute:
        :param sequential:
        :param eliminated_envs: list of the environments that are not going to be used
        :param port:
        """

        if params is None:
            self._params = DrivingBatch._default_params
        else:
            self._params = {}
            for key, value in DrivingBatch._default_params.items():
                if key in params.keys():  # If it exist you add  it from the params
                    self._params.update({key: params[key]})
                else:  # if tit is not the case you use default
                    self._params.update({key: value})

        # THIS IS ALWAYS 1 for now
        self._batch_size = self._params['batch_size']  # How many CARLAs are going to be ran.
        # Create a carla server description here, params set which kind like docker or straight.
        self._environment_batch = []
        for i in range(self._batch_size):
            self._environment_batch.append(ServerManagerDocker(self._params))

        # We get the folder where the jsonfile is located.
        print (jsonfile.split('/')[:-1])
        self._jsonfile_path = os.path.join(*jsonfile.split('/')[:-1])
        # Read the json file being
        with open(jsonfile, 'r') as f:
            self._json = json.loads(f.read())
        # The timeout for waiting for the server to start.
        self.client_timeout = 45.0
        # The os environment file
        if "SRL_DATASET_PATH" not in os.environ and self._params['save_dataset']:
            raise ValueError("SRL DATASET not defined, set the place where the dataset is going to be saved")

        # uninitialized environments vector
        self._environments = None
        self._client_vec = None
        # set a fixed port to be looked into
        self._port = port
        # add eliminated environments
        if eliminated_environments is None:
            self._eliminated_environments = {}
        else:
            self._eliminated_environments = eliminated_environments
        # setting to ignore all the previous experiments when executing envs
        self.ignore_previous_execution = ignore_previous_execution


    def start(self, no_server=False, agent_name=None):
        """
        Sstart the carla servers and configure the environments associated with the execution.
        :param no_server:
        :param agent_name: the name of an agent to check for previous executions.
        :return:
        """
        # TODO: this setup is hardcoded for Batch_size == 1
        # TODO add here several server starts into a for
        # TODO for i in range(self._batch_size)
        logging.debug("Starting the CEXP System !")
        if agent_name is not None and not self.ignore_previous_execution:
            Environment.check_for_executions(agent_name, self._json['package_name'])

        if no_server:
            self._client_vec = []
        else:
            if self._port is None:
                # Starting the carla simulators
                for env in self._environment_batch:
                    free_port = find_free_port()
                    env.reset(port=free_port)
            else:
                # We convert it to integers
                self._port = int(self._port)
                if not check_test_server(self._port):
                    logging.debug("No Server online starting one !")
                    self._environment_batch[0].reset(port=self._port)
                free_port = self._port  # This is just a test mode where CARLA is already up.
            # setup world and client assuming that the CARLA server is up and running
            self._client_vec = [carla.Client('localhost', free_port)]
            self._client_vec[0].set_timeout(self.client_timeout)
            #print(self._client_vec[0].get_client_version())
            #print(self._client_vec[0].get_server_version())

        # Create the configuration dictionary of the exp batch to pass to all environments
        env_params = {
            'batch_size': self._batch_size,
            'make_videos': self._params['make_videos'],
            'save_dataset': self._params['save_dataset'],
            'save_sensors': self._params['save_dataset'] and self._params['save_sensors'],
            'save_opponents': self._params['save_opponents'],  #
            'package_name': self._json['package_name'],
            'save_walkers': self._params['save_walkers'],
            'remove_wrong_data': self._params['remove_wrong_data'],
            'non_rendering_mode': self._params['non_rendering_mode'],
            'carla_recording': self._params['carla_recording'],
            'direct_read': self._params['direct_read'],
            'agent_name': agent_name,
            'debug': False  # DEBUG SHOULD BE SET
        }

        # We instantiate environments here using the recently connected client
        self._environments = {}
        parserd_exp_dict = parser.parse_exp_vec(self._jsonfile_path, collections.OrderedDict(
                                                sort_nicely_dict(self._json['envs'].items())))

        # For all the environments on the file.
        for env_name in self._json['envs'].keys():
            # We have the options to eliminate some events from execution.
            if env_name in self._eliminated_environments:
                continue
            # Check the random seed, if using a global one or local for each case
            env_random_seed = self._check_random_seed(parserd_exp_dict, env_name)
            # Add the seconds per meter metric !
            if 'seconds_per_meter' not in parserd_exp_dict[env_name]:
                if 'default_seconds_per_meter' in self._json:
                    parserd_exp_dict[env_name].update(
                        {'seconds_per_meter': self._json['default_seconds_per_meter']})

                else:
                    parserd_exp_dict[env_name].update(
                        {'seconds_per_meter': 0.8})
            # Instance an _environments.
            env = Environment(env_name, self._client_vec, parserd_exp_dict[env_name],
                              env_params, env_random_seed)
            # add the additional sensors ( The ones not provided by the policy )
            self._environments.update({env_name: env})

    def _check_random_seed(self, parserd_exp_dict, env_name):
        if 'random_seed' in parserd_exp_dict[env_name]:
            env_random_seed = parserd_exp_dict[env_name]['random_seed']
        elif 'default_random_seed' in self._json:
            env_random_seed = self._json['default_random_seed']
        else:
            env_random_seed = int(random.random() * 100000)
        return env_random_seed

    def _add_timeout(self, parserd_exp_dict, env_name):
        if 'seconds_per_meter' not in parserd_exp_dict[env_name]:
            if 'default_seconds_per_meter' in self._json:
                parserd_exp_dict[env_name].update(
                    {'seconds_per_meter': self._json['default_seconds_per_meter']})

            else:
                parserd_exp_dict[env_name].update(
                    {'seconds_per_meter': 0.8})


    def _get_execution_list(self):
        """
        We compute the environments that have already been executed. If they were
        not set to ignore you will not execute an environment again.


        :return: a list with the non-executed environments
        """

        # This strategy of execution takes into consideration the env repetition
        #  and execute a certain number of times.from
        # The environment itself is able to tell when the repetition is already made.
        execution_list = []
        for env_name in self._environments.keys():
            # The default is
            repetitions = 1
            if "repetitions" in self._json['envs'][env_name]:
                repetitions = self._json['envs'][env_name]['repetitions']

            if env_name in Environment.number_of_executions.keys():
                repetitions_rem = max(0, repetitions -\
                                      Environment.number_of_executions[env_name])
                execution_list += [self._environments[env_name]] * repetitions_rem

            else:
                # We add all the repetitions to the execution list
                execution_list += [self._environments[env_name]] * repetitions

        return execution_list

    def __iter__(self):
        """
        The iterator for the CEXP attempts to execute every experiment on the json file.
        If the ignore previous execution is set it will reexecute previously executed experiments.
        Otherwise it just execute the missing ones.
        :return:
        """
        if self._environments is None:
            raise ValueError("You are trying to iterate over an not started driving "
                             "object, run the start method ")

        return iter(self._get_execution_list())

    def __del__(self):
        self.cleanup()
        Environment.number_of_executions = {}

    def __len__(self):
        return len(self._get_execution_list())

    def cleanup(self):

        if len(self._client_vec) > 0 and self._port is None:  # we test if it is actually running
            self._environment_batch[0].stop()

    def get_eliminated_environments(json_file, ):

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