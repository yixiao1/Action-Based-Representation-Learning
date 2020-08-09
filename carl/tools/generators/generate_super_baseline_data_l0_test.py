
import json
import os



#TODO AUTOMATIC GENERATION of the type , like selecting the correct datasets


if __name__ == '__main__':

    root_route_file_position = 'database/corl2017'
    root_route_file_output = 'database'
    # root_route_file_position = 'srunner/challenge/'
    #filename_town01 = os.path.join(root_route_file_position, 'Town01_navigation.json')

    # The sensor information should be on get data
    sensors = [{'type': 'sensor.camera.rgb',
                'x': 2.0, 'y': 0.0,
                'z': 1.40, 'roll': 0.0,
                'pitch': -15.0, 'yaw': 0.0,
                'width': 800, 'height': 600,
                'fov': 100,
                'id': 'rgb_central'}
               ]

    # For each of the routes to be evaluated.

    # Tows to be generated
    town_sets = {'Town01_navigation.xml': 'navigation',
                 'Town01_one_curve.xml': 'curve'}


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
                "additional_sensors": sensors,
                "package_name": 'dataset_l0_test'}

    for w_set_name in weather_sets.keys():
        # get the actual set  from th name
        w_set = weather_sets[w_set_name]

        for weather in w_set:

            for town_name in town_sets.keys():

                for env_number in range(25):

                    env_dict = {
                        "route": {
                            "file": town_name,
                            "id": env_number
                        },
                        "scenarios": None,
                        "town_name": "Town01",
                        "vehicle_model": "vehicle.lincoln.mkz2017",
                        "weather_profile": weather
                    }

                    new_json["envs"].update({weather + '_' + town_sets[town_name] + '_route'
                                             + str(env_number).zfill(5): env_dict})

    filename = os.path.join(root_route_file_output, 'dataset_l0_test.json')

    with open(filename, 'w') as fo:
        # with open(os.path.join(root_route_file_position, 'all_towns_traffic_scenarios3_4.json'), 'w') as fo:
        fo.write(json.dumps(new_json, sort_keys=True, indent=4))
