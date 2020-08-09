
import json
import os
import random

if __name__ == '__main__':

    root_route_file_output = '/datatmp/Experiments/yixiao/carl/database'

    # The sensor information should be on get data


    # For each of the routes to be evaluated.

    # Tows to be generated
    town_sets = {'routes/routes_all_Town02.xml': 'routes'
                 }


    # Weathers to be generated later
    weather_sets = {'training': ["ClearNoon",
                                  "WetNoon",
                                  "HardRainNoon",
                                   "ClearSunset"]
                    }


    new_json = {"envs": {},
                "package_name": 'dataset_dynamic_Town02_l0',

                }

    all_list = list(range(10100))

    for w_set_name in weather_sets.keys():
        # get the actual set  from th name
        w_set = weather_sets[w_set_name]

        for weather in w_set:

            for town_name in town_sets.keys():

                for env_number in range(200):

                    id_num = random.choice(all_list)
                    all_list.remove(id_num)

                    env_dict = {
                        "route": {
                            "file": town_name,
                            "id": id_num
                        },
                        "scenarios": {'background_activity': {"cross_factor": 0.1,
                                                              "vehicle.*": 100,
                                                              "walker.*": 300}
                                      },
                        "town_name": "Town02",
                        "vehicle_model": "vehicle.lincoln.mkz2017",
                        "weather_profile": weather
                    }



                    new_json["envs"].update({weather + '_' + town_sets[town_name] + '_route'
                                             + str(env_number).zfill(5): env_dict})

    filename = os.path.join(root_route_file_output, 'dataset_dynamic_Town02_l0.json')

    with open(filename, 'w') as fo:
        # with open(os.path.join(root_route_file_position, 'all_towns_traffic_scenarios3_4.json'), 'w') as fo:
        fo.write(json.dumps(new_json, sort_keys=True, indent=4))
