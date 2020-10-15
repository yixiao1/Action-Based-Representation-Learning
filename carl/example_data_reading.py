import logging
from cexp.agents import CEXP
from cexp.agents.NPCAgent import NPCAgent
from cexp.env.environment import NoDataGenerated

###
if __name__ == '__main__':

    # A single loop being made
    json = 'database/town01_empty.json'
    # Dictionary with the necessary params related to the execution not the model itself.
    params = {'save_dataset': True,
              'docker_name': 'carlalatest:latest',
              'gpu': 0,
              'batch_size': 1,
              'remove_wrong_data': False,
              'non_rendering_mode': False,
              'carla_recording': True
              }
    # TODO for now batch size is one
    number_of_iterations = 123
    # The idea is that the agent class should be completely independent
    agent = NPCAgent()
    # this could be joined
    # THe experience is built, the files necessary
    env_batch = CEXP(json, params, number_of_iterations, params['batch_size'], sequential=True)
    # Here some docker was set
    env_batch.start(no_server=True)  # no carla server mode.

    for env in env_batch:
        # it can be personalized to return different types of data.
        print ("recovering ", env)
        try:
            env_data = env.get_data()  # returns a basically a way to read all the data properly
        except NoDataGenerated:
            print (" No data generate for episode ", env)
        else:
            # for now it basically returns a big vector containing all the
            print ("PRINTING DATA POINTS")

            for data_point in env_data:
                print("################################")
                print (data_point)


    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)