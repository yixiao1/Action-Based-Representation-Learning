
import json
import os
from random import randint

if __name__ == '__main__':

    root_route_file_output = '/home/yixiao/carl/database'

    # The sensor information should be on get data
    # For each of the routes to be evaluated.

    # Tows to be generated
    town_sets = {'corl2017/Town01_navigation.xml': 'navigation',
                 'routes/routes_town01.xml': 'challenge'
                 }


    # Weathers to be generated later
    weather_sets = {'training': ["ClearNoon",
                                  "WetNoon",
                                  "HardRainNoon",
                                   "ClearSunset"]
                    }



    name_dict = {'training':{'Town01': 'training'
                             },
                 'new_weather': {'Town01': 'newweather'

                 }
    }

    new_json = {"envs": {},
                "package_name": 'dataset_vehicles_walkers_l0',

                }

    for w_set_name in weather_sets.keys():
        # get the actual set  from th name
        w_set = weather_sets[w_set_name]

        for weather in w_set:

            for town_name in town_sets.keys():

                for env_number in range(25):

                    env_dict = {
                        "repetitions": 1,
                        "route": {
                            "file": town_name,
                            "id": env_number
                        },
                        "scenarios": {
                                      'background_activity': {"vehicle.*": 80,
                                                              "walker.*": 300,
                                                              "cross_factor": 0.01}
                                      },
                        "town_name": "Town01",
                        "vehicle_model": "vehicle.lincoln.mkz2017",
                        "weather_profile": weather
                    }

                    new_json["envs"].update({weather + '_' + town_sets[town_name] + '_route'
                                             + str(env_number).zfill(5): env_dict})

    filename = os.path.join(root_route_file_output, 'dataset_vehicles_walkers_l0.json')

    with open(filename, 'w') as fo:
        fo.write(json.dumps(new_json, sort_keys=True, indent=4))
