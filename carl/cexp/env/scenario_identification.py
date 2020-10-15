# TODO Might require optimization since it has to be computed on every iteration

# Functions usefull for scenario identification

import logging
import numpy as np
import math
from agents.tools.misc import vector
from srunner.tools.scenario_helper import get_distance_along_route

def angle_between(orientation_1, orientation_2):
    """
    Compute relative angle and distance between a target_location and a current_location

    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :return: a tuple composed by the distance to the object and the angle between both objects
    """
    #target_vector = np.array([target_location.x - current_location.x,
    #                          target_location.y - current_location.y])

    norm_target = np.linalg.norm(orientation_2)

    d_angle = math.degrees(math.acos(np.dot(orientation_1, orientation_2) / (norm_target+0.000001)))

    return d_angle

def yaw_difference(t1, t2):
    dyaw = t1.rotation.yaw - t2.rotation.yaw

    return math.sqrt(dyaw * dyaw)


def angle_between_transforms(location1, location2, location3):
    v_1 = vector(location1,location2)
    v_2 = vector(location2,location3)

    vec_dots = np.dot(v_1, v_2)
    cos_wp = vec_dots / abs((np.linalg.norm(v_1) * np.linalg.norm(v_2)))
    angle_wp = math.acos(min(1.0, cos_wp))  # COS can't be larger than 1, it can happen due to float imprecision
    return angle_wp


def distance_to_intersection(vehicle, wmap, resolution=0.1):
    # TODO heavy function, takes 70MS this can be reduced.

    # Add a cutting p

    total_distance = 0

    reference_waypoint = wmap.get_waypoint(vehicle.get_transform().location)

    while not reference_waypoint.is_intersection:
        reference_waypoint = reference_waypoint.next(resolution)[0]
        total_distance += resolution

    return total_distance


def get_current_road_angle(vehicle, wmap, resolution=0.05):

    reference_waypoint = wmap.get_waypoint(vehicle.get_transform().location)
    # we go a bit in to the future to identify future curves

    next_waypoint = reference_waypoint.next(resolution)[0]
    #for i in range(10):
    #next_waypoint = next_waypoint.next(resolution)[0]

    yet_another_waypoint = next_waypoint.next(resolution)[0]

    return angle_between_transforms(reference_waypoint.transform.location,
                                    next_waypoint.transform.location,
                                    yet_another_waypoint.transform.location)


def op_vehicle_distance(waypoint, list_close_vehicles):

    # if the waypoint has a vehicle different than the ego one.
    for op_vehicle in list_close_vehicles:
        distance_op = waypoint.get_transform().location.distance(op_vehicle.get_transform().location)
        if distance_op < 2.0:
            return distance_op

    # there is not
    return -1

# Get the route for the next points that the vehicle is going to be.

def get_route_of_next_points(vehicle, route):

    # Basically if the distance get bigger means that the points are getting further
    # away and i found my position on this vector
    distance_previous = 10000

    for point in route:

        distance_result = vehicle.get_transform().location.distance(
                                point[0].location)
        if distance_result > distance_previous:

            return route[(route.index(point)-1):]

        distance_previous = distance_result


    return route

def get_distance_lead_vehicle(vehicle, map, world, max_distance = 50.0):
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

    #print('min_distance', min_distance)
    return min_distance


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


# TODO this is distance to the triggers not actually useful

def get_distance_closest_scenarios(route, list_scenarios, percentage_completed):

    # We take the route starting from the vehicle postion there
    percentage_completed = percentage_completed/100.0
    route_cut = route[int(percentage_completed*len(route)):]

    print ( " PERCENTAGE COMPLETED ", percentage_completed)

    print ( " LEN ROUTE CUT ", len(route_cut))
    # TODO only working for scenarios 3 and 4

    triggers_scenario3 = []
    triggers_scenario4 = []

    for scenario in list_scenarios:
        # We get all the scenario 3 and 4 triggers
        if type(scenario).__name__ == 'DynamicObjectCrossing':
            triggers_scenario3.append(scenario._trigger_location)

        elif type(scenario).__name__ == 'VehicleTurningRight' or type(scenario).__name__ == 'VehicleTurningLeft':
            triggers_scenario4.append(scenario._trigger_location)


    distance_scenario3 = -1
    distance_scenario4 = -1

    print ( "TRIGGERS ")

    print (triggers_scenario3)

    print ( "#####")

    print (triggers_scenario4)

    for trigger in triggers_scenario3:
        distance, found = get_distance_along_route(route_cut, trigger)

        if found == True and distance_scenario3 == -1 or distance < distance_scenario3:
            distance_scenario3 = distance

    for trigger in triggers_scenario4:
        distance, found = get_distance_along_route(route_cut, trigger)

        if found == True and distance_scenario4 == -1 or distance < distance_scenario4:
            distance_scenario4 = distance


    return distance_scenario3, distance_scenario4

def get_distance_closest_crossing_waker(exp):
    # TODO we get the distance of the walker which is crossing the closest
    distance_pedestrian_crossing = -1
    closest_pedestrian_crossing = None
    for scenario in exp._list_scenarios:
        # We get all the scenario 3 and 4 triggers
        if type(scenario).__name__ == 'DynamicObjectCrossing':
            print (" DISTANCE TO OTHERS ")
            # Distance to the other actors
            for actor in scenario.other_actors:
                if actor.is_alive:
                    actor_distance = exp._ego_actor.get_transform().location.distance(
                        actor.get_transform().location)
                    print (actor_distance, " type ", actor.type_id)

                    if 'walker' in actor.type_id:
                        if distance_pedestrian_crossing != -1:
                            if actor_distance < distance_pedestrian_crossing:
                                distance_pedestrian_crossing = actor_distance
                                closest_pedestrian_crossing = actor
                        else:
                            distance_pedestrian_crossing = actor_distance
                            closest_pedestrian_crossing = actor

    return distance_pedestrian_crossing, closest_pedestrian_crossing


def identify_scenario(distance_intersection,
                      distance_lead_vehicle=-1,
                      distance_crossing_walker=-1,
                      thresh_intersection=25.0,
                      thresh_lead_vehicle=25.0,
                      thresh_crossing_walker=10.0

                      ):

    """
    Returns the scenario for this specific point or trajectory

    S0: Lane Following -Straight - S0_lane_following
    S1: Intersection - S1_intersection
    S2: Traffic Light/ before intersection - S2_before_intersection
    S3: Lane Following with a car in front
    S4: Stop for a lead vehicle in front of the intersection ( Or continue
    S5: FOllowing a vehicle inside the intersection.
    S6: Pedestrian crossing leaving from hiden coca cola thing: S6_pedestrian



    # These two params are very important when controlling the training procedure

    ### Future ones to add ( higher complexity)


    S4: Control Loss (TS1) - S4_control_loss
    S5: Pedestrian Crossing (TS3) - S5_pedestrian_crossing
    S6: Bike Crossing (TS4)
    S7: Vehicles crossing on red light (TS7-8-9)
    Complex Towns Scenarios
    S8: Lane change
    S9: Roundabout
    S10: Different kinds of intersections with different angles


    :param exp:
    :return:

    We can have for now
    """

    # TODO for now only for scenarios 0-2

    if distance_crossing_walker != -1 and distance_crossing_walker < thresh_crossing_walker:

        return 'S6_pedestrian'

    else:
        if distance_lead_vehicle == -1 or distance_lead_vehicle > thresh_lead_vehicle:
            # There are no vehicle ahead

            if distance_intersection > thresh_intersection:
                # For now far away from an intersection means that it is a simple lane following
                return 'S0_lane_following'

            elif distance_intersection > 1.0:
                # S2  Check if it is directly affected by the next intersection
                return 'S1_before_intersection'

            else:
                return 'S2_intersection'
        else:
            if distance_intersection > thresh_intersection:
                # For now that means that S4 is being followed
                return 'S3_lead_vehicle_following'

            elif distance_intersection > 1.0:
                # Distance intersection.
                return 'S4_lead_vehicle_before_intersection'

            else:  # Then it is

                return 'S5_lead_vehicle_inside_intersection'


def identify_scenario_2(is_red_tl_hazard=False,
                        is_vehicle_hazard=False,
                        is_pedestrian_hazard = False):

    """
    Returns the scenario for this specific point or trajectory

    S0: Lane Following -Straight - S0_lane_following
    S1: Intersection - S1_intersection
    S2: Traffic Light/ before intersection - S2_before_intersection
    S3: Lane Following with a car in front
    S4: Stop for a lead vehicle in front of the intersection ( Or continue
    S5: FOllowing a vehicle inside the intersection.
    S6: Pedestrian crossing leaving from hiden coca cola thing: S6_pedestrian



    # These two params are very important when controlling the training procedure

    ### Future ones to add ( higher complexity)


    S4: Control Loss (TS1) - S4_control_loss
    S5: Pedestrian Crossing (TS3) - S5_pedestrian_crossing
    S6: Bike Crossing (TS4)
    S7: Vehicles crossing on red light (TS7-8-9)
    Complex Towns Scenarios
    S8: Lane change
    S9: Roundabout
    S10: Different kinds of intersections with different angles


    :param exp:
    :return:

    We can have for now
    """

    # TODO for now only for scenarios 0-2

    if is_vehicle_hazard:
        if is_pedestrian_hazard:
            if is_red_tl_hazard:
                return 'S0_vehicle_pedestrian_redTL'
            else:
                return 'S1_vehicle_pedestrian'
        elif is_red_tl_hazard:
            return 'S2_vehicle_redTL'

        else:
            return 'S3_vehicle'

    if is_pedestrian_hazard:
        if is_red_tl_hazard:
            return 'S4_pedestrian_redTL'

        else:
            return 'S5_pedestrian'

    if is_red_tl_hazard:
        return 'S6_redTL'

    return 'S7_normal_driving'






