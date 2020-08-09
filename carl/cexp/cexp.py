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

# I see now 4 execution, consumption modes:
# TODO separate what creates the environment list from the actual environmental runner class
# Execute a certain number of iterations sequentially or randomly
# Consume the environments based on how many times they were already executed
# Consume de environmens ignoring if they were executed already
# Do an execution eliminating part of the environments used.


# TODO the only execution mode is execute all. Things are controlled always on json
# TODO randomness does not exist.


class CEXP(object):
    """
    THE main CEXP module.
    It contains the instanced env files that can be iterated to have instanced environments to get
    """

    _default_params = {'save_dataset': False,
                       'save_sensors': False,
                       'save_trajectories': False,
                       'save_walkers': False,
                       'make_videos': False,
                       'resize_images':False,
                       'save_opponents': False,
                       'save_opp_trajectories': False,
                       'docker_name': None,
                       'gpu': 0,
                       'batch_size': 1,
                       'remove_wrong_data': False,
                       'non_rendering_mode': False,
                       'carla_recording': True,
                       'direct_read': False,
                       'trajectories_directory': 'database'
                      }

    def __init__(self, jsonfile, params=None, iterations_to_execute=0, sequential=False,
                 port=None, execute_all=False, ignore_previous_execution=False,
                 eliminated_environments=None):
        # TODO data consumption and benchmarking differentiation is important.
        """

        :param jsonfile:
        :param params:
        :param iterations_to_execute:
        :param sequential:
        :param eliminated_envs: list of the environments that are not going to be used
        :param port:
        """


        if params is None:
            self._params = CEXP._default_params
        else:
            self._params = {}
            for key, value in CEXP._default_params.items():
                if key in params.keys():  # If it exist you add  it from the params
                    self._params.update({key: params[key]})
                else:  # if tit is not the case you use default
                    self._params.update({key: value})

        # Todo thuis goes out with the merge
        if 'save_sensors' not in self._params:
            self._params.update({'save_sensors': self._params['save_dataset']})

        self._batch_size = self._params['batch_size']  # How many CARLAs are going to be ran.
        # Create a carla server description here, params set which kind like docker or straight.
        self._environment_batch = []
        for i in range(self._batch_size):
            self._environment_batch.append(ServerManagerDocker(self._params))

        # We get the folder where the jsonfile is located.

        self._jsonfile_path = os.path.join( '/', *jsonfile.split('/')[:-1])

        # Executing
        self._execute_all = execute_all
        # Read the json file being
        with open(jsonfile, 'r') as f:
            self._json = json.loads(f.read())
        # The timeout for waiting for the server to start.
        self.client_timeout = 25.0
        # The os environment file
        if "SRL_DATASET_PATH" not in os.environ and self._params['save_dataset']:
            raise ValueError("SRL DATASET not defined, "
                             "set the place where the dataset is going to be saved")

        # uninitialized environments vector
        self._environments = None
        # Starting the number of iterations that are going to be ran.
        self._iterations_to_execute = iterations_to_execute
        self._client_vec = None
        # if the loading of environments will be sequential or random.
        self._sequential = sequential
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
            logging.debug(" Connecting to the free port client")
            self._client_vec = [carla.Client('localhost', free_port)]
            self._client_vec[0].set_timeout(self.client_timeout)

        # Create the configuration dictionary of the exp batch to pass to all environments
        env_params = {
            'batch_size': self._batch_size,
            'make_videos': self._params['make_videos'],
            'resize_images': self._params['resize_images'],
            'save_dataset': self._params['save_dataset'],
            'save_sensors': self._params['save_dataset'] and self._params['save_sensors'],
            'save_opponents': self._params['save_opponents'],
            'package_name': self._json['package_name'],
            'save_walkers': self._params['save_walkers'],
            'save_trajectories': self._params['save_trajectories'],
            'remove_wrong_data': self._params['remove_wrong_data'],
            'non_rendering_mode': self._params['non_rendering_mode'],
            'carla_recording': self._params['carla_recording'],
            'direct_read': self._params['direct_read'],
            'agent_name': agent_name,
            'trajectories_directory': self._params['trajectories_directory'],
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
            # Instance an _environments.
            env = Environment(env_name, self._client_vec, parserd_exp_dict[env_name], env_params)
            # add the additional sensors ( The ones not provided by the policy )
            self._environments.update({env_name: env})

    def __iter__(self):
        if self._environments is None:
            raise ValueError("You are trying to iterate over an not started cexp "
                             "object, run the start method ")
        # This strategy of execution takes into consideration the env repetition
        #  and execute a certain number of times.from
        # The environment itself is able to tell when the repetition is already made.
        if self._execute_all:
            execution_list = []
            # TODO not working on execute all mode.
            #print ("EXECUTIONS")
            #print (Environment.number_of_executions)        {'ClearSunset_route00022': 0, 'ClearSunset_route00023': 1,'ClearSunset_route00020': 0,...}
            for env_name in self._environments.keys():
                repetitions = 1
                # TODO check necessity
                #  We check the remaining necessary executions for each of the environments
                #if "repetitions" not in self._json['envs'][env_name] and not self.ignore_previous_execution:
                #    raise ValueError(" Setting to execute all but repetition information is not  on the json file")

                if "repetitions" in self._json['envs'][env_name]:
                    repetitions = self._json['envs'][env_name]['repetitions']

                #print (" Env name ", env_name)
                if env_name in Environment.number_of_executions.keys():
                    repetitions_rem = max(0, repetitions -\
                                      Environment.number_of_executions[env_name])
                    execution_list += [self._environments[env_name]] * repetitions_rem

                else:
                    # We add all the repetitions to the execution list
                    execution_list += [self._environments[env_name]] * repetitions

            return iter(execution_list)
        # These two modes ignore the repetitions parameter and just keep executing.
        elif self._sequential:
            return iter([self._environments[list(self._environments.keys())[i % len(self._environments)]]
                         for i in range(self._iterations_to_execute)])
        else:
            return iter([self._environments[random.choice(list(self._environments.keys()))] for _ in range(self._iterations_to_execute)])

    def __len__(self):
        return self._iterations_to_execute

    def __del__(self):
        self.cleanup()
        Environment.number_of_executions = {}

    def cleanup(self):

        if len(self._client_vec) > 0 and self._port is None:  # we test if it is actually running
            self._environment_batch[0].stop()
