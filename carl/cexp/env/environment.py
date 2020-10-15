import json
import logging
import os
import shutil

from carl.cexp.env.experience import Experience
from carl.cexp.env.datatools.map_drawer import draw_pedestrians, draw_opp_trajectories, \
                                          draw_trajectories
import carl.cexp.env.datatools.data_parser as parser

# The scenarios should not have this triggering thing they can however. add some scenario editor ??


# define the exception for non existent data
class NoDataGenerated(Exception):
   """Base class for other exceptions"""
   pass
"""
The environment class encapsulates the experience all the scenarios that the policy is going to execute
as well as a communication channel with the CARLA servers.
It also can have additional sensors that are environment related not policy related.
"""

# TODO you should only report an episode in case of crash.


class Environment(object):
    # We keep track here the number of times this class was executed.
    number_of_executions = {}

    def __init__(self, name, client_vec, env_config, env_params):

        # The ignore previous param is to avoid searching for executed iterations
        # TODO this requires more testing
        # TODO this may create expendable execution files

        # We keep these configuration files so we can reset the environment
        self._env_config = env_config
        self._env_params = env_params
        self._batch_size = env_params['batch_size']
        # if the data is going to be saved for this environment
        self._save_data = env_params['save_dataset']
        # the name of this experience object
        self._environment_name = name
        # We have already a connection object to a CARLA server
        self._client_vec = client_vec
        # The route is already specified
        self._route = env_config['route']
        # An experience is associate with a certain town name
        #  ( THat is also associated with scenarios and a route)
        self._town_name = env_config['town_name']
        # Thee scenarios that are going to be associated with this route.
        self._scenarios = env_config['scenarios']
        # All the sensors that are going to be spawned, a vector of dictionaries
        self._sensor_desc_vec = []
        # The vehicle car model that is going to be spawned
        self._vehicle_model = env_config['vehicle_model']
        # the list of all experiences to be instantiated at the start
        self._exp_list = []
        # the name of the package this env is into
        self._package_name = env_params['package_name']
        logging.debug("Instantiated Environment %s" % self._environment_name)
        # the trajectories can be written on a map. That is very helpful
        self._save_trajectories = env_params['save_trajectories']
        # functions defined by the policy to compute the
        # adequate state and rewards based on CARLA data
        self.StateFunction = None
        self.RewardFunction = None
        # The information dictionary to be queried by the environment users.
        self._env_exec_info = []
        # We set an agent to a previous executed agent.
        self._last_executing_agent = env_params['agent_name']
        self.trajectories_directory = env_params['trajectories_directory']


    @staticmethod
    def check_for_executions(agent_name, package_name):
        """
            with this function we check for executions for the environment
        :return:
        """

        if not Environment.number_of_executions:
            if "SRL_DATASET_PATH" not in os.environ:
                raise ValueError("SRL_DATASET_PATH not defined,"
                                 " set the place where the dataset was saved before")
            Environment.number_of_executions = \
                parser.get_number_executions(agent_name, os.path.join(
                                                            os.environ["SRL_DATASET_PATH"],
                                                            package_name))

    def __str__(self):
        return self._environment_name

    def _cleanup(self):
        """
        Remove and destroy all actors
        """
        # make the exp vec empty
        for exp in self._exp_list:
            exp.cleanup()
        self._exp_list = []
        # we remove all the sensors everytime. No sensor addition on building time
        self._sensor_desc_vec = []

    def _record(self):
        """
            record the results summary and set this as an executed example

        """
        # get all the exps to get the summary
        self._env_exec_info = []
        for exp in self._exp_list:
            exp.record()
            # TODO experimental
            #exp.cleanup()
            self._env_exec_info.append(exp.get_summary())

        if self._save_trajectories:
            self.draw_trajectory(self.trajectories_directory)

        if self._environment_name in Environment.number_of_executions:
            Environment.number_of_executions[self._environment_name] += 1
        else:
            raise ValueError("Cleaning up non created environment")

    def stop(self):
        self._cleanup()

    def set_sensors(self, sensors):
        if not isinstance(sensors, list):
            raise ValueError(" Sensors added to the environment should be a list of dictionaries")

        self._sensor_desc_vec += sensors

    def reset(self, StateFunction=None, RewardFunction=None, agent_name=None):
        """
        Reset the environment, when reseting it is necessary to define
        the function that will provide the reward at every step and
        also the function that will provide the state to the user on every step

        THe optional agent name can be also set for data saving purposes.

        :param StateFunction:
        :param RewardFunction:
        :param agent_name:
        :return:
        """
        # if reward or state functions are not we basically return the
        if RewardFunction is None:
            RewardFunction = (lambda x: None)
        if StateFunction is None:
            StateFunction = (lambda x: None)

        # save the last executing agent name. This is to be used for logging purposes
        if agent_name is not None:
            self._last_executing_agent = agent_name
        # create the environment
        if self._environment_name not in Environment.number_of_executions:
            Environment.number_of_executions.update({self._environment_name: 0})

        if len(self._exp_list) > 0:
            self.stop()

        # set the state and reward functions to be used on this episode
        self.StateFunction = StateFunction
        self.RewardFunction = RewardFunction

        for i in range(self._batch_size):
            exp_params = {
                'env_name': self._environment_name,
                'package_name': self._package_name,
                'town_name': self._town_name,
                'weather_profile': self._env_config['weather_profile'],
                'env_number': Environment.number_of_executions[self._environment_name],
                'exp_number': i,
                'save_data': self._save_data,
                'make_videos': self._env_params['make_videos'],
                'resize_images': self._env_params['resize_images'],
                'save_sensors': self._env_params['save_sensors'],
                'save_opponents': self._env_params['save_opponents'],
                'save_walkers': self._env_params['save_walkers'],
                'non_rendering_mode': self._env_params['non_rendering_mode'],
                'carla_recording': self._env_params['carla_recording'],
                'remove_wrong_data': self._env_params['remove_wrong_data'],
                'debug': self._env_params['debug']
            }
            self._exp_list.append(Experience(self._client_vec[i], self._vehicle_model, self._route,
                                             self._sensor_desc_vec, self._scenarios, exp_params,
                                             self._last_executing_agent))
        # if it is the first time we execute this env
        if self._save_data and self._environment_name in Environment.number_of_executions:
            # we use one of the experiments to build the metadata
            self._exp_list[0]._writer.save_metadata(self, self._exp_list[0]._instanced_sensors)

        for exp in self._exp_list:
            exp.tick_scenarios()
        # We tick the scenarios to get them started

        logging.debug("Started Environment %s" % self._environment_name)

        return StateFunction(self._exp_list), \
                 RewardFunction(self._exp_list)

    def get_data(self):
        # Each environment can have a reference datapoint,
        # where the data is already collected. That can go
        # Directly to the json where the data is collected.
        # This is the package that is where the data is saved.
        # It is always save in the SRL path
        root_path = os.path.join(os.environ["SRL_DATASET_PATH"], self._package_name,
                                 self._environment_name)
        # If the metadata does not exist the environment does not have a reference data.
        if not os.path.exists(os.path.join(root_path, 'metadata.json')):
            raise NoDataGenerated("The data is not generated yet")

        # Read the metadata telling the sensors that exist
        with open(os.path.join(root_path, 'metadata.json'), 'r') as f:
            metadata_dict = json.loads(f.read())

        full_episode_data_dict = parser.parse_environment(root_path, metadata_dict)

        return full_episode_data_dict

    # TODO does remove data make sense ?
    def remove_data(self):
        """
            Remove all data from this specific environment
        """
        root_path = os.path.join(os.environ["SRL_DATASET_PATH"], self._package_name,
                                 self._environment_name)
        # If the metadata does not exist the environment does not have a reference data.
        if not os.path.exists(os.path.join(root_path, 'metadata.json')):
            raise NoDataGenerated("The data is not generated yet")

        shutil.rmtree(root_path)

    def _is_running(self):
        """
            We use the running experiments to check if the route is still running
        """
        for exp in self._exp_list:
            if exp.is_running():  # If any exp is still running then this environment is still on.
                return True
        # if no exp is running then the environment is already done
        return False

    def run_step(self, control_vec, affordances):
        """
        Run an step on the simulation using the agent control
        :param control_vec:
        :return:
        """

        control = control_vec[0][0]
        control_noise = control_vec[0][1]

        # If we don't receive a list we can force it
        if not isinstance(control_vec, list):
            control_vec = [control_vec]

        # Run the loop for all the experiments on the batch.
        # update all scenarios
        # TODO there is no loop when using multibatch
        exp = self._exp_list[0]
        # We save the normal one and apply the noise control.
        _ = exp.tick_scenarios_control(control)
        exp.apply_control(control_noise)
        exp.tick_world()
        exp.save_experience(affordances[0])

        return self.StateFunction(self._exp_list), \
                    self.RewardFunction(self._exp_list)

    def get_info(self):

        """
            Returns the current information about the executions being held
            as well as if this env is running or not.
        """

        info = {
            'summary': None
        }

        # If the environment is still running there is no summary yet
        if self._is_running():
            info.update({'status': 'Running'})
        else:
            # If it is not running we basically try to record what is happening.
            self._record()
            info.update({'status': 'Finished'})
            info['summary'] = self._env_exec_info[0]

        return info


    def draw_pedestrians(self, steps):
        draw_pedestrians(self._last_executing_agent, self.get_data(), self._environment_name,
                         self._exp_list[0].world, steps)

    def draw_opp_trajectories(self, directory):
        pass

    def draw_trajectory(self, directory):
        # Draw the trajectory made on the last instance of this enviroment (experience)
        if not os.path.exists(directory):
            os.mkdir(directory)

        draw_trajectories(directory,
                          self.get_data(),
                          self._last_executing_agent + '_' + self._package_name + '_' + self._environment_name,
                          self._exp_list[0].world,
                          self._exp_list[0]._route,
                          direct_read=self._env_params['direct_read'])






