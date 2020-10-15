
import json
import os

if __name__ == '__main__':

    root_route_file_output = 'database'

    new_json = {"envs": {},
                "package_name": 'dataset_ICML_Town01_l0_empty',

                }

    json_file = '/datatmp/Experiments/yixiao/carl/database/dataset_ICML_Town01_train_20Hours.json'
    with open(json_file) as json_:
        data = json.load(json_)
        for key, item in data["envs"].items():
            if not "navigation" in key:
                item["scenarios"] = {}
                new_json["envs"].update({key+"_empty": item})

    filename = os.path.join('/datatmp/Experiments/yixiao/carl/database/dataset_ICML_Town01_train_5Hours_empty_2.json')

    with open(filename, 'w') as fo:
        # with open(os.path.join(root_route_file_position, 'all_towns_traffic_scenarios3_4.json'), 'w') as fo:
        fo.write(json.dumps(new_json, sort_keys=True, indent=4))
