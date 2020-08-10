"""
The agent class is an interface to run experiences, the actual policy must inherit from agent in order to
execute. It should implement the run_step function
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
#import pytrees
import carla


from cexp.agents.agent import Agent

from agents.navigation.local_planner import RoadOption



# Hyperparameters
learning_rate = 0.001
gamma = 0.99

def compute_magnitude_angle(target_location, current_location, orientation):
    """
    Compute relative angle and distance between a target_location and a current_location

    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :return: a tuple composed by the distance to the object and the angle between both objects
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / (norm_target+0.000001)))

    return (norm_target, d_angle)


def distance_vehicle(waypoint, vehicle_position):

    dx = waypoint.location.x - vehicle_position.x
    dy = waypoint.location.y - vehicle_position.y

    return math.sqrt(dx * dx + dy * dy)


def _get_forward_speed(vehicle):
    """ Convert the vehicle transform directly to forward speed """

    velocity = vehicle.get_velocity()
    transform = vehicle.get_transform()
    vel_np = np.array([velocity.x, velocity.y, velocity.z])
    pitch = np.deg2rad(transform.rotation.pitch)
    yaw = np.deg2rad(transform.rotation.yaw)
    orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    speed = np.dot(vel_np, orientation)
    return speed


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = 2   # Basically is an error with respect to the central waypoint in angle, and the current speed
        self.action_space = 7   # For the actions you can throttle, brake, steer left, steer right, do nothing.

        self.l1 = nn.Linear(self.state_space, 128, bias=True)
        self.l2 = nn.Linear(128, 256, bias=True)
        self.l3 = nn.Linear(256, self.action_space, bias=True)

        self.gamma = gamma

        # Episode policy and reward history
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l3,
            nn.Softmax(dim=-1)
        )
        return model(x)


class PGAgent(Agent):

    def setup(self, config_file_path):
        # TODO this should actually point to a configuration file
        checkpoint_number = config_file_path
        self._policy = Policy()
        if checkpoint_number is not None:
            checkpoint = torch.load(checkpoint_number)
            self._policy.load_state_dict(checkpoint['state_dict'])
        self._optimizer = optim.Adam(self._policy.parameters(), lr=learning_rate)
        self._iteration = 0
        self._episode = 0

    def run_step(self, state):
        # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
        state = torch.from_numpy(state).type(torch.FloatTensor)
        action = self._policy(Variable(state))
        c = Categorical(action)
        try:
            action = c.sample()

        except RuntimeError as r:
            import traceback
            traceback.print_exc()
            print("input state", state)
            print(action)
            print(c)
            raise r

        # previous_state = state
        #    action = previous_action
        # Add log probability of our chosen action to our history
        self._iteration += 1
        if self._policy.policy_history.nelement() != 0:
            self._policy.policy_history = torch.cat([self._policy.policy_history, c.log_prob(action).unsqueeze(0)])
        else:
            self._policy.policy_history = (c.log_prob(action).unsqueeze(0))

        control = carla.VehicleControl()

        # We can only have one action, so action will be something like this.
        # 0 -- Nothing
        # 1 -- Throttle
        # 2 -- Brake
        # 3 -- Steer Left
        # 4 -- Throttle Steer Left
        # 5 -- Steer Right
        # 6 -- Throttle Steer Right

        if action == 1:
            control.throttle = 0.7
        elif action == 2:
            control.brake = 1.0
        elif action == 3:
            control.steer = -0.5
        elif action == 4:
            control.steer = -0.5
            control.throttle = 0.7
        elif action == 5:
            control.steer = 0.5
        elif action == 6:
            control.steer = 0.5
            control.throttle = 0.7

        return control

    def make_reward(self, exp):
        """
        Basic reward that basically returns 1.0 for when the agent is alive and zero otherwise.
        :return: 1.0
        """

        return 1.0

    def make_state(self, exp):
        # state is divided in three parts, the speed, the angle_error, the high level command
        # Get the closest waypoint
        waypoint, _ = self._get_current_wp_direction(exp._ego_actor.get_transform().location, exp._route)
        norm, angle = compute_magnitude_angle(waypoint.location, exp._ego_actor.get_transform().location,
                                              exp._ego_actor.get_transform().rotation.yaw)

        return np.array([_get_forward_speed(exp._ego_actor) / 12.0,  # Normalize to by dividing by 12
                         angle / 180.0])


    def _get_current_wp_direction(self, vehicle_position, route):

        # for the current position and orientation try to get the closest one from the waypoints
        closest_id = 0
        closest_waypoint = None
        min_distance = 100000
        for index in range(len(route)):

            waypoint = route[index][0]

            computed_distance = distance_vehicle(waypoint, vehicle_position)
            if computed_distance < min_distance:
                min_distance = computed_distance
                closest_id = index
                closest_waypoint = waypoint

        direction = route[closest_id][1]
        if direction == RoadOption.LEFT:
            direction = 3.0
        elif direction == RoadOption.RIGHT:
            direction = 4.0
        elif direction == RoadOption.STRAIGHT:
            direction = 5.0
        else:
            direction = 2.0

        return closest_waypoint, direction


    # TODO reinforce was changed to update

    def update(self, reward_batch):

        for rewards in reward_batch:
            # Should contain the  weight update algorithm if the agent uses it.
            R = 0
            # running_reward = (10 * 0.99) + (self._iteration * 0.01)
            # Discount future rewards back to the present using gamma
            discount_rewards = []
            for r in rewards[::-1]:
                R = r + self._policy.gamma * R
                discount_rewards.insert(0, R)

            # Scale rewards
            discount_rewards = torch.FloatTensor(discount_rewards)

            discount_rewards = (discount_rewards - discount_rewards.mean()) /\
                               (discount_rewards.std() + 0.000001)
            # TODO THIS IS CLEARLY WRONG NEED TO FILL AND MAKE A UNIQUE NUMPY HERE
            # Calculate loss

            loss = (torch.sum(torch.mul(self._policy.policy_history[0:len(discount_rewards)], Variable(discount_rewards)).mul(-1), -1))


        # Update network weights
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # Save and initialize episode history counters
        self._policy.loss_history.append(loss.data.item())
        self._policy.reward_history.append(np.sum(reward_batch[0]))
        self._policy.policy_history = Variable(torch.Tensor())


    def reset(self):
        """
        Destroy (clean-up) the agent objects that are use on CARLA
        :return:
        """
        self._episode += 1
        if self._episode % 100 == 0:
            state = {
                'iteration': self._episode,
                'state_dict': self._policy.state_dict()
            }
            print ("Saved")
            torch.save(state, str(self._episode) + '.pth')
        pass


