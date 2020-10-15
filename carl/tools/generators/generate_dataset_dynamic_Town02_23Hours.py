
import json
import os
import glob

if __name__ == '__main__':

    dataset_path = '/datatmp/fcodevilla/Datasets/MSN/dataset_dynamic_Town02_l0'
    all_episodes_path_list = glob.glob(os.path.join(dataset_path, '*'))

    print(len(all_episodes_path_list))

    episodes_names = []
    for episode_path in all_episodes_path_list:
        episodes_names.append(episode_path.split('/')[-1])

    root_route_file_output = 'database'

    new_json = {"envs": {},
                "package_name": 'dataset_dynamic_Town02_l0',

                }

    json_file = '/datatmp/Experiments/yixiao/carl/database/dataset_dynamic_Town02_l0.json'

    with open(json_file) as json_:
        data = json.load(json_)
        for key, item in data["envs"].items():
            if key in episodes_names:
                new_json["envs"].update({key: item})


    filename = os.path.join('/datatmp/Experiments/yixiao/carl/database/dataset_dynamic_Town02_23Hours.json')

    with open(filename, 'w') as fo:
        # with open(os.path.join(root_route_file_position, 'all_towns_traffic_scenarios3_4.json'), 'w') as fo:
        fo.write(json.dumps(new_json, sort_keys=True, indent=4))
