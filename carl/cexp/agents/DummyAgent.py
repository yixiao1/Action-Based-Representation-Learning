
from cexp.agents.agent import Agent
import carla

class DummyAgent(Agent):


    def make_reward(self, measurements, sensors, scenarios):
        """
        Return the reward for a given step. Must be implemented by some inherited class
        :param measurements:
        :param sensors:
        :param scenarios:
        :return:
        """
        return None

    def make_state(self, measurements, sensors, scenarios):
        """
        for a given run.Must be implemented by some inherited class
        :param measurements:
        :param sensors:
        :param scenarios:
        :return:
        """

        return None

    def run_step(self, input_data):
        print("=====================>")
        for key, val in input_data.items():
            if hasattr(val[1], 'shape'):
                shape = val[1].shape
                print("[{} -- {:06d}] with shape {}".format(key, val[0], shape))
            else:
                print("[{} -- {:06d}] ".format(key, val[0]))
        print("<=====================")

        # DO SOMETHING SMART

        # RETURN CONTROL
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 1.0
        control.brake = 0.0
        control.hand_brake = False

        return control
