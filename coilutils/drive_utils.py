import json
import os


def checkpoint_parse_configuration_file(filename):

    with open(filename, 'r') as f:
        configuration_dict = json.loads(f.read())

    return configuration_dict['yaml'], configuration_dict['checkpoint'], \
           configuration_dict['agent_name'], configuration_dict['encoder_params']


def summarize_benchmark(summary_data):

    final_dictionary = {}
    # we just get the headers for the final dictionary
    for key in summary_data.keys():
        final_dictionary.update({key:0})

    print (" RECEIVED SUMMARY ", summary_data)

    for metric in summary_data.keys():

        try:
            final_dictionary[metric] = sum(summary_data[metric]) / len(summary_data[metric])
        except KeyError:  # To overcomme the bug on reading files csv
            final_dictionary[metric] = sum(summary_data[metric[:-1]]) / len(summary_data[metric])


    return final_dictionary


def write_summary_csv(out_filename, agent_checkpoint_name, summary_data, results_filename):
    # The used tasks are hardcoded, this need to be improved

    print ("Writting summary")

    if not os.path.exists(results_filename):
        results_outfile = open(results_filename, 'w')
        results_outfile.write("%s" % (summary_data))
        results_outfile.close()

    if not os.path.exists(out_filename):

        csv_outfile = open(out_filename, 'w')

        csv_outfile.write("%s,%s,%s,%s,%s, %s\n"
                          % ('step', 'episodes_completion', 'episodes_fully_completed',
                             'average_score_penalty', 'average_infraction_red_lights', 'average_number_traffic_light'))
        csv_outfile.close()

    summary_dict = summarize_benchmark(summary_data)

    csv_outfile = open(out_filename, 'a')

    csv_outfile.write("%s, %f, %f, %f, %f, %f" % (agent_checkpoint_name.split('_')[-1],
                                              summary_dict['episodes_completion'],
                                              summary_dict['result'],
                                              summary_dict['infractions_score'],
                                              summary_dict['number_red_lights'],
                                              summary_dict['total_number_traffic_light']))

    csv_outfile.write("\n")

    csv_outfile.close()

