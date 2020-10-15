
import json
import os
import math
import carla

import cexp.env.utils.route_configuration_parser as parser


from srunner.challenge.utils.route_manipulation import interpolate_trajectory


def test_exp():
    """
    We build the scenario to check if it is posible to use it. To see if
    the generated experience makes sense
    :return: 
    """

    pass



#TODO the fact that it does not use orientation may create bugs.
# TODO READ THE FULL TRANSFORM ON THE ROUTE READING


def calculate_distance(location , waypoint1):

       dx = float(waypoint1['x']) - location.x
       dy = float(waypoint1['y']) - location.y
       dz = float(waypoint1['z']) - location.z
       dist_position = math.sqrt(dx * dx + dy * dy + dz * dz)

       # dyaw = float(waypoint1['yaw']) - wtransform.rotation.yaw


       #dist_angle = math.sqrt(dyaw * dyaw)

       return dist_position


def find_closest_route_trip(trigger_position, coarse_trajectory):

    # the minimum distance
    minimun_distance = 10000
    # the position of minimum distance
    min_distance_position = 0
    count = 0
    for point in coarse_trajectory:

        distance = calculate_distance(point, trigger_position)
        if distance < minimun_distance:
            min_distance_position = count
            minimun_distance = distance

        count +=  1

    return coarse_trajectory[(min_distance_position-1):(min_distance_position+1)]


# TODO : also add the town01 ??

def scenario_route_generator(routes_file, scenarios_tag, scenario_name, number_of_routes):

    """
    returns exp dictionaries containing a single scenario happening on that route.
    :param routes_file:
    :param scenarios_tag:
    :param scenario_number: The name of the scenario that is going to be used.
    :param number_of_routes:
    :return:
    """

    # TODO I need to know the route points related to the scenario ...
    # retrieve worlds annotations
    world_annotations = parser.parse_annotations_file(scenarios_tag)
    # retrieve routes
    route_descriptions_list = parser.parse_routes_file(routes_file)


    # Connect to some carla here.
    env_vec = []
    count = 0
    client = carla.Client('localhost', 2000)

    client.set_timeout(32)

    for _route_description in route_descriptions_list:

        original_trajectory = _route_description['trajectory']

        world = client.load_world(_route_description['town_name'])
        _, _route_description['trajectory'] = interpolate_trajectory(world, _route_description['trajectory'])

        potential_scenarios_definitions, _ = parser.scan_route_for_scenarios(_route_description,
                                                                             world_annotations)
        # remove the trigger positon clustering  since we are only doing one scenario per time.
        # TODO add multi scenario route posibilities
        scenario_definitions = []
        for trigger in potential_scenarios_definitions:
            scenario_definitions += potential_scenarios_definitions[trigger]


        # filter only the scenario that you want to generate the routes.
        # (Doing more than one would be quicker but i preffer this organization for now)
        filtered_scenario_definitions = []
        for pot_scenario in scenario_definitions:
            if pot_scenario['name'] == scenario_name:
                filtered_scenario_definitions.append(pot_scenario)

        print (" The representations")
        print (original_trajectory, filtered_scenario_definitions)

        # Now we get the correspondent route points for the trigger
        for sce_def in filtered_scenario_definitions:

            route = find_closest_route_trip(sce_def['trigger_position'], original_trajectory)
            env_vec.append(
                {scenario_name + '_' + 'route_' + str(count):
                     {
                         'route': route,
                         'scenarios': {
                             scenario_name: sce_def['trigger_position']
                         },
                         'town_name': _route_description['town_name'],
                         'vehicle_model': "vehicle.lincoln.mkz2017"
                     }

                }

            )
            count += 1

    return env_vec[0:number_of_routes]




if __name__ == '__main__':


    print (scenario_route_generator(
           'experience_database_generator/database/routes/routes_town01.xml',
           'experience_database_generator/database/scenarios/all_towns_traffic_scenarios1_3_4.json',
           'Scenario3',
           10)
    )



