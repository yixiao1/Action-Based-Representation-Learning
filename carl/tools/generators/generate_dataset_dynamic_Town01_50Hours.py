
import json
import os
import glob

if __name__ == '__main__':

    dataset_path = '/datatmp/fcodevilla/Datasets/MSN/dataset_dynamic_Town01_50Hours'
    all_episodes_path_list = glob.glob(os.path.join(dataset_path, '*'))

    print(len(all_episodes_path_list))

    episodes_names = []
    for episode_path in all_episodes_path_list:
        episodes_names.append(episode_path.split('/')[-1])

    root_route_file_output = 'database'

    new_json = {"envs": {},
                "package_name": 'dataset_dynamic_Town01_50Hours',

                }

    json_file = '/datatmp/Experiments/yixiao/carl/database/dataset_dynamic_l0.json'
    json_file_navigation = '/datatmp/Experiments/yixiao/carl/database/ECCV2020/dataset_ICML_Town01_l0.json'

    with open(json_file) as json_:
        data = json.load(json_)
        for key, item in data["envs"].items():
            if key in episodes_names:
                new_json["envs"].update({key: item})

            if key+'_1' in episodes_names:
                new_json["envs"].update({key+'_1': item})

    with open(json_file_navigation) as json_:
        data = json.load(json_)
        for key, item in data["envs"].items():
            if key in episodes_names:
                new_json["envs"].update({key: item})

            #else:
            #    print('routes', key)

   # with open(json_file_navigation) as json_2:
   #     data = json.load(json_2)
   #     for episode in episodes_names:
   #         if episode in data['envs'].keys():
   #             for key, item in data["envs"].items():
   #                 new_json["envs"].update({key: item})

   #         else:
   #             print('navigation', episode)

    filename = os.path.join('/datatmp/Experiments/yixiao/carl/dataset_dynamic_Town01_50Hours.json')

    with open(filename, 'w') as fo:
        # with open(os.path.join(root_route_file_position, 'all_towns_traffic_scenarios3_4.json'), 'w') as fo:
        fo.write(json.dumps(new_json, sort_keys=True, indent=4))
