import json
import argparse

from cexp.cexp import CEXP
from cexp.env.environment import NoDataGenerated
"""
Measure the number of hours and episodes a dataset has. 
It can also be used to clean empty ones a posteriori 
Cleaning a priori is desired

"""



def get_envs_dict(json_filename):
    """
    Returns an environment dictionary to count how many each env was experimented.
    :return:
    """

    with open(json_filename, 'r') as f:
        json_file = json.loads(f.read())

    envs_dict = {}
    for key  in json_file['envs'].keys():

        envs_dict.update({key:0})

    print ( " Envs Dict ", envs_dict)
    return envs_dict



if __name__ == '__main__':

    description = ("CARLA AD Challenge evaluation: evaluate your Agent in CARLA scenarios\n")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--json', default='database/dataset_l0.json',
                        help='path to the json file')
    parser.add_argument('--remove', action='store_true',
                        help='remove empty ')



    args = parser.parse_args()

    # Dictionary with the necessary params related to the execution not the model itself.
    params = {'save_dataset': True,
              'docker_name': 'carlalatest:latest',
              'gpu': 0,
              'batch_size': 1,
              'remove_wrong_data': False,
              'non_rendering_mode': False,
              'carla_recording': True
              }

    # this could be joined
    # THe experience is built, the files necessary
    env_batch = CEXP(args.json, params, execute_all=True,
                     ignore_previous_execution=True)
    # Here some docker was set
    env_batch.start(no_server=True, agent_name='Agent')  # no carla server mode.
    # count, we count the environments that are read
    env_count_dict = get_envs_dict(args.json)
    # We also check the number of hours
    total_hours = 0
    total_episodes = 0
    for env in env_batch:
        steer_vec = []
        throttle_vec = []
        brake_vec = []
        # it can be personalized to return different types of data.
        print("Environment Name: ", env)
        try:
            env_data = env.get_data()  # returns a basically a way to read all the data properly
        except NoDataGenerated:
            print("No data generate for episode ", env)
        except ValueError:
            print (" Episode with errors on Json")
        else:
            not_empty = False
            for exp in env_data:
                print("    Exp: ", exp[1])
                for batch in exp[0]:
                    print("      Batch: ", batch[1])
                    if len(batch[0]) > 0:
                        not_empty = True
                        total_episodes += 1
                        env_count_dict[env._environment_name] += 1


                    # Now we count the ammount of data we have for this batch
                    total_hours += (len(batch[0])/10)/3600
                    print("         Size: ", len(batch[0]))
                    print ("Total Hours: ", total_hours)

            if not not_empty and args.remove:
                env.remove_data()

    print ("####################")
    print ("DATASET has ", total_episodes, " episodes and ", total_hours, " hours of data")
    print ("####################")
    print (">>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<")
    print ("####################")
    print ("ENVS USED")
    print (env_count_dict)
    print ("####################")