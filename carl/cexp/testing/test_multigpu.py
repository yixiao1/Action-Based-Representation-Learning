import json


import os


JSONFILE = 'database/sample_benchmark.json'
environments_dict_base = [
    'WetSunset_route00024',
    'SoftRainSunset_route00000',
    'WetNoon_route00024'
]

params = {'save_dataset': True,
          'docker_name': 'carlalatest:latest',
          'gpu': 5,
          'batch_size': 1,
          'remove_wrong_data': False,
          'non_rendering_mode': False,
          'carla_recording': False  # TODO testing
          }

AGENT_NAME = 'NPCAgent'
# The episodes to be checked must always be sequential

def get_eliminated_environments(json_file, start_position, end_position):

    """
    List all the episodes BUT the range between start end position.
    """
    with open(json_file, 'r') as f:
        json_dict = json.loads(f.read())

    count = 0
    eliminated_environments_list = []
    for env_name in json_dict['envs'].keys():
        if count < start_position or count >= end_position:
            eliminated_environments_list.append(env_name)
        count += 1
    return eliminated_environments_list




if __name__ == '__main__':
    # PORT 6666 is the default port for testing server

    number_of_collectors = 2
    json_config = 'sample_benchmark.json'

    json_file = os.path.join('database', json_config)

    with open(json_file, 'r') as f:
        json_dict = json.loads(f.read())

    environments_per_collector = len(json_dict['envs'])/number_of_collectors
    if environments_per_collector < 1.0:
        raise ValueError(" Too many collectors")

    print ("All Environments")
    print (json_dict['envs'].keys())


    print (" Environment per Collector ", environments_per_collector)
    # If the division is not perfect

    # TODO there are problems on the subdivisions
    for i in range(number_of_collectors):
        # A single loop being made
        if i == number_of_collectors-1 and not environments_per_collector.is_integer():
            extra_env = 1
        else:
            extra_env = 0
        # we list all the possible environments
        print("Start", int(environments_per_collector) * (i),    "end ",
              int(environments_per_collector) * (i+1) + extra_env)

        eliminated_environments = get_eliminated_environments(json_file,
                                                              int(environments_per_collector) * (i),
                                                              int(environments_per_collector) * (i+1) + extra_env)

        print (" Collector ", i )
        print (" Eliminated ", eliminated_environments)

