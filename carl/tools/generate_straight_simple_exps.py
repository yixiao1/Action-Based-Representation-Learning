
import json
import os

if __name__ == '__main__':

    root_route_file_position = '/network/home/codevilf/experience_database_generator/database/'
    # root_route_file_position = 'srunner/challenge/'
    filename = os.path.join(root_route_file_position, 'straight_routes.json')

    # For each of the routes to be evaluated.
    new_json = {"envs": {},
                "additional_sensors": [],
                "package_name": "straights"}   # TODO change exps to envs
    for env_number in range(16):

        env_dict = {
            "route": {
                "file": "straight_town01.xml",
                "id": env_number
            },
            "scenarios": {
                "file": "None"
            },
            "town_name": "Town01",
            "vehicle_model": "vehicle.lincoln.mkz2017"
        }
        new_json["envs"].update({'route'+str(env_number):env_dict })

    with open(filename, 'w') as fo:
        # with open(os.path.join(root_route_file_position, 'all_towns_traffic_scenarios3_4.json'), 'w') as fo:
        fo.write(json.dumps(new_json, sort_keys=True, indent=4))

