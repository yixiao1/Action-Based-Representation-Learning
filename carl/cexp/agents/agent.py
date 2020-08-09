"""
The agent class is an interface to run experiences, the actual policy must inherit from agent in order to
execute. It should implement the run_step function
"""
import logging

class Agent(object):

    def __init__(self, path_to_conf_file=None):
        # agent's initialization

        self._name = 'Agent'  # just in case the name is not used.
        self.setup(path_to_conf_file)


    def setup(self, path_to_config):
        """
        overwritte
        :param path_to_config:
        :return:
        """
        pass
    def run_step(self, input_data):
        """
        Execute one step of navigation. Must be implemented
        :return: control
        """
        pass

    # TODO TRY A SIMPLE THREAD FOR EXECUTION HERE

    def _run_step_batch(self, input_data_vec):
        # TODO ELIMINATE this For make the inference inside and code the angent to make it
        controls_vec = []
        for input_data in input_data_vec:
            controls_vec.append(self.run_step(input_data))

        return controls_vec

    def make_reward(self, exp):
        """
        Return the reward for a given step. Must be implemented by some inherited class
        :param measurements:
        :param sensors:
        :param scenarios:
        :return:
        """
        pass

    def _make_reward_batch(self, exp_vec):
        reward_vec = [None] * len(exp_vec)
        count = 0
        for exp in exp_vec:
            if exp.is_running():
                reward_vec[count] = self.make_reward(exp)
            count += 1
        return reward_vec

    def make_state(self, exp):
        """
        for a given step of the run return the current relevant state for
        :param measurements:
        :param sensors:
        :param scenarios:
        :return:
        """
        pass

    def _make_state_batch(self, exp_vec):
        state_vec = []
        for exp in exp_vec:
            if exp.is_running():
                state_vec.append(self.make_state(exp))
        return state_vec

    def sensors(self):
        sensors_vec = []
        return sensors_vec

    def reinforce(self, rewards):
        # Should contain the  weight update algorithm if the agent uses it.

        pass


    def reset(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        pass

    def add_value(self, value_batch, value_vec):
        for value in value_vec:
            if value is None:
                return
        if len(value_vec) > 0:
            for i in range(len(value_batch)):
                value_batch[i].append(value_vec[i])

    def unroll(self, environment):
        """
         unroll a full episode for the agent. This produces an state and reward vectors
         that are defined by the agent, that can be used directly for learning.
        """
        # You reset the agent before you start any unroll process
        self.reset()
        # You reset the scenario with and pass the make reward functions
        #  that are going to be used on the training.
        environment.set_sensors(self.sensors())
        state, reward = environment.reset(self._make_state_batch, self._make_reward_batch,
                                          self._name)

        # print("what is state?", state)      # affordances.py -> get_driving_affordances(exp)

        # Start the rewards and state vectors used
        reward_batch = [[]] * environment._batch_size
        state_batch = [[]] * environment._batch_size

        while environment._is_running():

            controls = self._run_step_batch(state)
            # With this the experience runner also unroll all the scenarios
            # Experiment on the batch.

            # we need to save these affordancs to measurement files
            state, reward = environment.run_step(controls, state)

            # TODO check the posible sizes mismatches here
            self.add_value(reward_batch, reward)
            self.add_value(state_batch, state)


        environment._record()
        environment.stop()

        return state_batch, reward_batch