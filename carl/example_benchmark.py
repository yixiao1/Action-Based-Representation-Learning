import logging

from cexp.cexp import CEXP
from cexp.agents.NPCAgent import NPCAgent


# This is an example of simple benchmark. Lets see how to add a good summary to it.

# TODO ADD THE REPETION SOMEHOW TO THE GENERAL PARAMETERS

# THE REPETITION CAN BE TRANSLATED AS BATCH SIZE IF POSSIBLE

if __name__ == '__main__':

    # A single loop being made
    json_file = 'database/sample_benchmark.json'
    # Dictionary with the necessary params related to the execution not the model itself.
    params = {'save_dataset': False,
              'docker_name': 'carlalatest:latest',
              'gpu': 0,
              'batch_size': 1,
              'remove_wrong_data': False,
              'non_rendering_mode': False,
              'carla_recording': True
              }
    # TODO for now batch size is one
    number_of_iterations = 10
    # The idea is that the agent class should be completely independent
    agent = NPCAgent()

    env_batch = CEXP(json_file, params, number_of_iterations,
                     params['batch_size'], sequential=True, debug=True)  # THe experience is built, the files necessary
    # to load CARLA and the scenarios are made
    # Here some docker was set
    env_batch.start()

    for env in env_batch:
        states, rewards = agent.unroll(env)
        # if the agent is already un
        summary = env.get_summary()
        #
        print (summary)



    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
