
import json
import os

if __name__ == '__main__':

    root_route_file_output = 'database'

    new_json = {"envs": {},
                "package_name": 'dataset_ICML_Town01_l0_random',

                }

    json_file = '/datatmp/Experiments/yixiao/carl/database/dataset_dynamic_l0.json'
    with open(json_file) as json_:
        data = json.load(json_)
        for i in range(10):
            for key,item in data["envs"].items():
                if i == 0:
                    new_json["envs"].update({key:item})
                else:
                    new_json["envs"].update({key+"_"+str(i): item})

    filename = os.path.join('/datatmp/Experiments/yixiao/carl/dataset_ICML_Town01_l0_random.json')

    with open(filename, 'w') as fo:
        # with open(os.path.join(root_route_file_position, 'all_towns_traffic_scenarios3_4.json'), 'w') as fo:
        fo.write(json.dumps(new_json, sort_keys=True, indent=4))
