import numpy as np
import math
import carla

def is_vehicle_hazard(ego, map, vehicle_list, proximity_threshold):
    """
            Check if a given vehicle is an obstacle in our way. To this end we take
            into account the road and lane the target vehicle is on and run a
            geometry test to check if the target vehicle is under a certain distance
            in front of our ego vehicle.
            WARNING: This method is an approximation that could fail for very large
             vehicles, which center is actually on a different lane but their
             extension falls within the ego vehicle lane.
            :param vehicle_list: list of potential obstacle to check
            :return: a tuple given by (bool_flag, vehicle), where
                     - bool_flag is True if there is a vehicle ahead blocking us
                       and False otherwise
                     - vehicle is the blocker object itself
            """

    ego_vehicle_location = ego.get_location()
    ego_vehicle_waypoint = map.get_waypoint(ego_vehicle_location)

    for target_vehicle in vehicle_list:
        # do not account for the ego vehicle
        if target_vehicle.id == ego.id:
            continue

        # if the object is not in our lane it's not an obstacle
        target_vehicle_waypoint = map.get_waypoint(target_vehicle.get_location())
        if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
            continue

        loc = target_vehicle.get_location()
        if is_within_distance_ahead(loc, ego_vehicle_location,
                                    ego.get_transform().rotation.yaw,
                                    proximity_threshold):
            return (True, target_vehicle)

    return (False, None)


def is_light_red_europe_style(ego, map, lights_list, proximity_threshold):
    """
    This method is specialized to check European style traffic lights.
    :param lights_list: list containing TrafficLight objects
    :return: a tuple given by (bool_flag, traffic_light), where
             - bool_flag is True if there is a traffic light in RED
              affecting us and False otherwise
             - traffic_light is the object itself or None if there is no
               red traffic light affecting us
    """
    ego_vehicle_location = ego.get_location()
    ego_vehicle_waypoint = map.get_waypoint(ego_vehicle_location)

    for traffic_light in lights_list:
        object_waypoint = map.get_waypoint(traffic_light.get_location())
        if object_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                object_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
            continue

        loc = traffic_light.get_location()
        if is_within_distance_ahead(loc, ego_vehicle_location,
                                    ego.get_transform().rotation.yaw,
                                    proximity_threshold):
            if traffic_light.state == carla.TrafficLightState.Red:
                return (True, traffic_light)

    return (False, None)

def is_light_red(ego, map, lights_list, proximity_threshold):
    """
    Method to check if there is a red light affecting us. This version of
    the method is compatible with both European and US style traffic lights.

    :param lights_list: list containing TrafficLight objects
    :return: a tuple given by (bool_flag, traffic_light), where
             - bool_flag is True if there is a traffic light in RED
               affecting us and False otherwise
             - traffic_light is the object itself or None if there is no
               red traffic light affecting us
    """
    if map.name == 'Town01' or map.name == 'Town02':
        return is_light_red_europe_style(ego, map, lights_list, proximity_threshold)
    else:
        raise RuntimeError("Not available yet")


def is_within_distance_ahead(target_location, current_location, orientation, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.

    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)


    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    # If the target is out of the maximum distance we set, we detect it False. But we still get the distance
    if norm_target > max_distance:
        return False


def is_within_distance_ahead(target_location, current_location, orientation, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.

    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return (True, norm_target)

    # If the target is out of the maximum distance we set, we detect it False. But we still get the distance
    if norm_target > max_distance:
        return (False, None)

    else:
        forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
        d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

        if d_angle < 90.0:
            return (True, norm_target)

        else:
            return (False, None)





"""
Aff1 Ego Speed:  must be used in order to properly plan the manouvers
"""

def get_speed():
    """
     Bassically return the vehicle ego-speed
    :return:
    """
    pass

"""
Aff2 Lateral Distance to Centerlane: The position with respect to the lane the vehicle  is in.
"""
def compute_distance_to_centerline(ego_location, closest_wp_location):

    # TODO which one is the closest waypoint on a intersection.

    ego_yaw = ego_location['orientation'][2]
    waypoint_yaw = closest_wp_location['orientation'][2]

    # we fistly make all range to be [0, 360)
    if ego_yaw >= 0.0:
        ego_yaw %= 360.0
    else:
        ego_yaw %= -360.0
        if ego_yaw != 0.0:
            ego_yaw += 360.0

    if waypoint_yaw >= 0.0:
        waypoint_yaw %= 360.0
    else:
        waypoint_yaw %= -360.0
        if waypoint_yaw != 0.0:
            waypoint_yaw += 360.0

    # we need to do some transformations to Cartesian coordinate system
    waypoint_C_yaw = 90.0 - waypoint_yaw
    waypoint_rad = np.deg2rad(waypoint_C_yaw)                          # from degree to radian
    # To define the distance to the centerline, we need to firstly calculate the road tangent, then compute the distance between the agent and the tangent
    waypoint_x = closest_wp_location['position'][0]
    waypoint_y = closest_wp_location['position'][1]
    ego_x = ego_location['position'][0]
    ego_y = ego_location['position'][1]
    # road tangent: y = slope * x + b    ---> slope*x-y+b=0
    slope = math.tan(waypoint_rad)
    b = waypoint_y - slope * waypoint_x

    return abs(slope * ego_x - ego_y + b) / math.sqrt(math.pow(slope,2) + math.pow(-1,2))

"""
Aff3 Relative Angle: The angle with respect to the center of the lane.
"""

def compute_relative_angle(ego_location, closest_wp_location):
    """
    given the location of ego and the closest fordward waypoint on lane, w
    e compute the relative angel between the ego's orientation and lane's orientation
    :return: relative angle
    """

    # TODO intersection is not used we have no sense of the values.

    # We calculate the "Relative Angle" by computing the difference of orientation yaw between
    # closest waypoint and ego
    # Note that the axises and intervals of yaw are different, one is [-180,180], and the other
    # is [0, 360], we need to do some transformations
    ego_yaw = ego_location['orientation'][2]
    waypoint_yaw = closest_wp_location['orientation'][2]

    # we fistly make all range to be [0, 360)
    if ego_yaw >= 0.0:
        ego_yaw %= 360.0
    else:
        ego_yaw %= -360.0
        if ego_yaw != 0.0:
            ego_yaw += 360.0

    if waypoint_yaw >= 0.0:
        waypoint_yaw %=  360.0
    else:
        waypoint_yaw %= -360.0
        if waypoint_yaw != 0.0:
            waypoint_yaw += 360.0

    # we need to do some transformations to Cartesian coordinate system
    waypoint_C_yaw = 90.0 - waypoint_yaw

    if waypoint_C_yaw < 0.0:
        waypoint_C_yaw += 360.0
    ego_C_yaw = 90.0 - ego_yaw
    if ego_C_yaw < 0.0:
        ego_C_yaw += 360.0

    angle_distance = waypoint_C_yaw-ego_C_yaw

    # This is for the case that the waypoint yaw and ego yaw are respectively near to 360.0 or 0.0
    if abs(angle_distance) < 180.0:
        relative_angle = np.deg2rad(angle_distance)
    else:
        if waypoint_C_yaw > ego_C_yaw:
            angle_distance = (360.0-waypoint_C_yaw) + (ego_C_yaw-0.0)
            relative_angle = -np.deg2rad(angle_distance)
        else:
            angle_distance = (waypoint_C_yaw-0.0) + (360.0-ego_C_yaw)
            relative_angle = np.deg2rad(angle_distance)

    return relative_angle

"""
Aff4 Distance Lead Vehicle: The distance to the vehicle in front
"""
# TODO we might want to get distance to all the vechicles


def get_distance_lead_vehicle(vehicle, map, world, max_distance = 50.0):

    # We need to be careful if this is properly done.

    actor_list = world.get_actors()
    vehicle_list = actor_list.filter("*vehicle*")

    ego_vehicle_location = vehicle.get_location()
    ego_vehicle_waypoint = map.get_waypoint(ego_vehicle_location)

    min_distance = max_distance + 10.0
    for target_vehicle in vehicle_list:
        # do not account for the ego vehicle
        if target_vehicle.id == vehicle.id:
            continue

        # if the object is not in our lane it's not an obstacle
        target_vehicle_waypoint = map.get_waypoint(target_vehicle.get_location())
        if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
            continue

        loc = target_vehicle.get_location()

        sign, distance = is_within_distance_ahead(loc, ego_vehicle_location, vehicle.get_transform().rotation.yaw, max_distance)

        if sign:
            if distance < min_distance:
                min_distance = distance

    return min_distance


"""
Aff5 Distance to active traffic light: The closest traffic light.
"""


"""
Aff6 Distance  Visible Pedestrians: Regressing the distance to all pedestrians that are visible.
"""
def distance_to_visible_pedestrians():

    """
        Adding the FOV for detecting the pedestrians
    :return:
    """

    pedestrians_distance = distance_to_all_pedestrians

    #distance_visible_vec
    return distance_visible_vec


"""
    Classification Post Processing
    Receives the affordances and outputs based on some thresholds

"""


def ego_vehicle_moving():
    pass

def properly_oriented_towards():
    pass



"""
Classification
Ego Vehicle is Moving ? 2 Frames
This is important for the ego vehicle to know if it understands the correlation of camera movement with ego-speed and ego-movement.
Is the vehicle properly oriented towards the lane ? 1 Frame
Is the vehicle on the correct lane ? After noisy vehicle dataset.
Is the lead vehicle closer than 10 meters ?
Is there a pedestrian crossing the road that is within 10 meters on the vehicle FOV ?
Is affected by traffic light?
Regression
Ego Speed ? 2 Frame
Knowing the speed is important, we know that already
Relative Angle. 1 Frame
Relative angle with respect to the closest waypoint.
Problems
The closest waypoint might not be a next waypoint
Distance to the center of the road.
Again the closest waypoint is wrong.
It is probably not being considered inside an intersection.
The distribution is wrong.
Distance to front vehicle (lead vehicle)
The one that we are using is fine I think.
Distance to the closest pedestrian.
Distance to the traffic light.
"""
