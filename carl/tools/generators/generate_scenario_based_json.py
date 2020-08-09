
import json
import argparse
import logging
import sys
import random
import os
from random import randint

import carla
from cexp.env.server_manager import start_test_server, check_test_server



from cexp.env.utils.route_configuration_parser import convert_waypoint_float, \
        parse_annotations_file, parse_routes_file, scan_route_for_scenarios

from srunner.challenge.utils.route_manipulation import interpolate_trajectory

from agents.navigation.local_planner import RoadOption

def clean_route(route):

    curves_start_end = []
    inside = False
    start = -1
    current_curve = RoadOption.LANEFOLLOW
    index = 0
    while index < len(route):

        command = route[index][1]
        if command != RoadOption.LANEFOLLOW and not inside:
            inside = True
            start = index
            current_curve = command

        if command != current_curve and inside:
            inside = False
            # End now is the index.
            curves_start_end.append([start, index, current_curve])
            if start == -1:
                raise ValueError("End of curve without start")

            start = -1
        else:
            index += 1

    return curves_start_end


def get_scenario_list(world, scenarios_json_path, routes_path, routes_id):

    world_annotations = parse_annotations_file(scenarios_json_path)

    route_descriptions_list = parse_routes_file(routes_path)

    per_route_possible_scenarios = []
    for id in routes_id:
        route = route_descriptions_list[id]

        _, route_interpolated = interpolate_trajectory(world, route['trajectory'])

        position_less_10_percent = int(0.1 * len(route_interpolated))
        possible_scenarios, existent_triggers = scan_route_for_scenarios(route_interpolated[:-position_less_10_percent],
                                                                         world_annotations,
                                                                         world.get_map().name)
        #if not possible_scenarios:
        #    continue
        per_route_possible_scenarios.append(possible_scenarios)


    return route_descriptions_list, per_route_possible_scenarios




def parse_scenario(possible_scenarios, wanted_scenarios):

    scenarios_to_add = {}
    for key in possible_scenarios.keys():  # this iterate under different keys
        scenarios_for_trigger = possible_scenarios[key]
        for scenario in scenarios_for_trigger:
            if scenario['name'] in wanted_scenarios:
                #print (scenario)
                convert_waypoint_float(scenario['trigger_position'])
                name  = scenario['name']
                #del scenario['name']
                scenarios_to_add.update({name: scenario})
                # TODO WARNING JUST ONE SCENARIO FOR TRIGGER... THE FIRST ONE
                break

    return scenarios_to_add


# TODO this is considering the curve situations.
def get_scenario_3(world, route, number_scenario3=1):

    _, route_interpolated = interpolate_trajectory(world, route['trajectory'])
    curves_positions = clean_route(route_interpolated)
    print ( "CURVES ", curves_positions)
    number_added_scenarios = 0

    scenario_vec = []
    if not curves_positions or  curves_positions[0][0] > 20.0:
        transform = route_interpolated[0][0]
        scenario_vec.append({
            "pitch": transform.rotation.pitch,
            "x": transform.location.x,
            "y": transform.location.y,
            "yaw": transform.rotation.yaw,
            "z": transform.location.z
        })
        number_added_scenarios += 1
    print (" ROUTE SIZE ", len (route_interpolated))

    for curve_start_end_type in curves_positions:

        if number_added_scenarios == number_scenario3:
            break
        end_curve = curve_start_end_type[1]
        print (curve_start_end_type)

        # we get a position for scenario 3 just after a curve happens  #

        position_scenario_inroute = end_curve + 1

        print ( " position ", position_scenario_inroute)
        transform = route_interpolated[position_scenario_inroute][0]
        scenario_vec.append({
            "pitch": transform.rotation.pitch,
            "x": transform.location.x,
            "y": transform.location.y,
            "yaw": transform.rotation.yaw,
            "z": transform.location.z
        })

        number_added_scenarios += 1






    return scenario_vec


# TODO it is always the first served scenarios

def generate_json_with_scenarios(world, routes_path,
                                 number_per_route, output_json_name,
                                 routes_id, number_of_vehicles):

    """

    :param world:
    :param scenarios_json_path:
    :param routes_path:
    :param wanted_scenarios:
    :param output_json_name:
    :param number_of_routes: the number of routes used on the generation
    :return:
    """


    route_descriptions_list = parse_routes_file('database/' + routes_path)




    print ("###################")

    weather_sets = {'training': ["ClearNoon",
                                  "WetNoon",
                                  "HardRainNoon",
                                   "ClearSunset"]
                    }
    new_json = {"envs": {},
                "package_name": output_json_name.split('/')[-1].split('.')[0],

                }

    for w_set_name in weather_sets.keys():
        # get the actual set  from th name
        w_set = weather_sets[w_set_name]

        for weather in w_set:

            for id in range(len(routes_id)):  # TODO change this to routes id
                # get the possible scenario for a given ID
                specific_scenarios_for_route = get_scenario_3(world,
                                                              route_descriptions_list[routes_id[id]],
                                                              number_per_route)

                scenarios_all = {
                                'background_activity': {"vehicle.*": number_of_vehicles,
                                                        "walker.*": 0},
                               }

                scenarios_all.update({'Scenario3': specific_scenarios_for_route})

                env_dict = {
                    "route": {
                        "file": routes_path,
                        "id": routes_id[id]
                    },
                    "scenarios": scenarios_all,
                    "town_name": "Town01",
                    "vehicle_model": "vehicle.lincoln.mkz2017",
                    "weather_profile": weather
                }

                new_json["envs"].update({weather + '_route'
                                         + str(id).zfill(5): env_dict})

    filename = output_json_name

    print (new_json)
    with open(filename, 'w') as fo:
        fo.write(json.dumps(new_json, sort_keys=True, indent=4))




if __name__ == '__main__':
    description = ("CARLA AD Challenge evaluation: evaluate your Agent in CARLA scenarios\n")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-t', '--town', default='Town01', help='The town name to be used')

    parser.add_argument('-o', '--output', default='database/dataset_scenarios_l0.json',
                        help='The outputfile json')

    parser.add_argument('-r', '--input-route', default='routes/routes_all.xml',
                        help='The outputfile json')

    parser.add_argument('-j', '--scenarios-json',
                        default='database/scenarios/all_towns_traffic_scenarios1_3_4.json',
                        help='the input json with scnarios')



    arguments = parser.parse_args()

    if not check_test_server(6666):
        start_test_server(6666)
        print (" WAITING FOR DOCKER TO BE STARTED")

    client = carla.Client('localhost', 6666)

    client.set_timeout(30.0)
    world = client.load_world(arguments.town)

    generate_json_with_scenarios(world, arguments.input_route,
                                 number_per_route=2,
                                 output_json_name=arguments.output,
                                 routes_id=range(0, 25),
                                 number_of_vehicles=0)








"""                                 
                                 [12376, 21238, 27834, 30004, 3764, 39467, 21392, 32601,
                                            35717, 58918, 30225, 23829, 16324, 49792, 50803, 51257,
                                            17897, 58683, 24335, 32264, 33929, 24963, 12227, 56750,
                                            39729, 15941, 59713, 14291, 62533, 7445, 40421, 47902,
                                            2903, 63748, 36159, 36462, 55221, 12717, 25422, 17761,
                                            30005, 43935, 660, 36669, 57461, 11807, 16169, 24937,
                                            36252, 20835, 40368, 25428, 7478, 24185, 26449, 51947,
                                            30297, 26218, 5174, 63912, 32822, 50572, 41304, 39563,
                                            21645, 21309, 32335, 9815, 24750, 45193, 64943, 6911,
                                            6595, 61112, 3662, 42229, 7304, 20208, 20702, 50579,
                                            27044, 36161, 45297, 43697, 49660, 36649, 37733, 60071,
                                            48731, 51466, 57571, 35073, 32948, 47784, 15110, 29068,
                                            63268, 37777, 23197, 58013, 60807, 49230, 55442, 36754,
                                            36227, 928, 46797, 44611, 31498, 46841, 9656, 18194,
                                            45692, 26394, 9500, 11713, 27882, 58759, 43671, 13972,
                                            48923, 14015, 56472, 9991, 7692, 6155, 19476, 63425,
                                            60546, 31496, 46087, 26777, 16842, 4755, 7088, 4725,
                                            38732, 21283, 20137, 2866, 62425, 22550, 31440, 31166,
                                            31348, 19952, 38799, 64874, 59985, 58060, 7000, 41964,
                                            48912, 16296, 37366, 12965, 8621, 56522, 45200, 39518,
                                            4046, 61402, 15992, 46204, 31992, 57418, 45061, 54986,
                                            6342, 27121, 62606, 21906, 44788, 11483, 41357, 52817,
                                            108, 30943, 56986, 20732, 54341, 23388, 16677, 13877,
                                            16247, 31152, 55499, 41274, 9467, 13276, 35031, 36223,
                                            5018, 32273, 10238, 14088, 29201, 55680, 28862, 50369])
"""