
import json
import os

if __name__ == '__main__':

    root_route_file_position = '/network/home/codevilf/experience_database_generator/database/'
    # root_route_file_position = 'srunner/challenge/'
    filename = os.path.join(root_route_file_position, 'town01_empty.json')

    # The sensor information should be on get data
    sensors = [{'type': 'sensor.camera.rgb',
                'x': 2.0, 'y': 0.0,
                'z': 1.40, 'roll': 0.0,
                'pitch': -15.0, 'yaw': 0.0,
                'width': 800, 'height': 600,
                'fov': 100,
                'id': 'rgb_central'},
               {'type': 'sensor.camera.depth',
                'x': 2.0, 'y': 0.0,
                'z': 1.40, 'roll': 0.0,
                'pitch': -15.0, 'yaw': 0.0,
                'width': 800, 'height': 600,
                'fov': 100,
                'id': 'depth_central'},
               {'type': 'sensor.camera.semantic_segmentation',
                'x': 2.0, 'y': 0.0,
                'z': 1.40, 'roll': 0.0,
                'pitch': -15.0, 'yaw': 0.0,
                'width': 800, 'height': 600,
                'fov': 100,
                'id': 'labels_central'},
               {'type': 'sensor.camera.rgb',
                'x': 2.0, 'y': 0.0,
                'z': 1.40, 'roll': 0.0,
                'pitch': -15.0, 'yaw': -30.0,
                'width': 800, 'height': 600,
                'fov': 100,
                'id': 'rgb_left'},
               {'type': 'sensor.camera.depth',
                'x': 2.0, 'y': 0.0,
                'z': 1.40, 'roll': 0.0,
                'pitch': -15.0, 'yaw': -30.0,
                'width': 800, 'height': 600,
                'fov': 100,
                'id': 'depth_left'},
               {'type': 'sensor.camera.semantic_segmentation',
                'x': 2.0, 'y': 0.0,
                'z': 1.40, 'roll': 0.0,
                'pitch': -15.0, 'yaw': -30.0,
                'width': 800, 'height': 600,
                'fov': 100,
                'id': 'labels_left'},
               {'type': 'sensor.camera.rgb',
                'x': 2.0, 'y': 0.0,
                'z': 1.40, 'roll': 0.0,
                'pitch': -15.0, 'yaw': 30.0,
                'width': 800, 'height': 600,
                'fov': 100,
                'id': 'rgb_right'},
               {'type': 'sensor.camera.depth',
                'x': 2.0, 'y': 0.0,
                'z': 1.40, 'roll': 0.0,
                'pitch': -15.0, 'yaw': 30.0,
                'width': 800, 'height': 600,
                'fov': 100,
                'id': 'depth_right'},
               {'type': 'sensor.camera.semantic_segmentation',
                'x': 2.0, 'y': 0.0,
                'z': 1.40, 'roll': 0.0,
                'pitch': -15.0, 'yaw': 30.0,
                'width': 800, 'height': 600,
                'fov': 100,
                'id': 'labels_right'}
               ]

    # For each of the routes to be evaluated.
    new_json = {"envs": {},
                "additional_sensors": sensors,
                "package_name": "l0_town01_empty"}
    for env_number in range(32):

        env_dict = {
            "route": {
                "file": "routes_town01.xml",
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
