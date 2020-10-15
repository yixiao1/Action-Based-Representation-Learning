import os
from srunner.scenariomanager.traffic_events import TrafficEventType
from cexp.env.utils.route_configuration_parser import clean_route
import py_trees



def count_number_traffic_lights(route, route_percentage):


    # Need to get the route world and perecentage complete

    route_position = int(len(route) * route_percentage)

    route_cleaned = clean_route(route[0:route_position])

    # we basically count the number of intersection since " ALL OF THEM" have traffic lights
    return len(route_cleaned)


def get_current_completion(master_scenario):
    list_traffic_events = []
    for node in master_scenario.scenario.test_criteria.children:
        if node.list_traffic_events:
            list_traffic_events.extend(node.list_traffic_events)


    for event in list_traffic_events:
        if event.get_type() == TrafficEventType.ROUTE_COMPLETION:
            if event.get_dict():
                return event.get_dict()['route_completed']
            else:
                return 0


def record_route_statistics_default(master_scenario, exp_name):
    """
      This function is intended to be called from outside and provide
      statistics about the scenario (human-readable, for the CARLA challenge.)
    """

    PENALTY_COLLISION_STATIC = 10
    PENALTY_COLLISION_VEHICLE = 10
    PENALTY_COLLISION_PEDESTRIAN = 30
    PENALTY_TRAFFIC_LIGHT = 10
    PENALTY_WRONG_WAY = 5
    PENALTY_SIDEWALK_INVASION = 5
    PENALTY_STOP = 7

    target_reached = False
    failure = False
    result = "SUCCESS"
    score_composed = 0.0
    score_penalty = 0.0
    score_route = 0.0
    return_message = ""

    if master_scenario.scenario.test_criteria.status == py_trees.common.Status.FAILURE:
        failure = True
        result = "FAILURE"
    if master_scenario.scenario.timeout_node.timeout and not failure:
        result = "TIMEOUT"

    list_traffic_events = []
    for node in master_scenario.scenario.test_criteria.children:
        if node.list_traffic_events:
            list_traffic_events.extend(node.list_traffic_events)

    list_collisions = []
    list_red_lights = []  # Here we can take the number of red lights
    list_wrong_way = []
    list_route_dev = []
    list_sidewalk_inv = []
    list_stop_inf = []
    # analyze all traffic events
    for event in list_traffic_events:
        if event.get_type() == TrafficEventType.COLLISION_STATIC:
            score_penalty += PENALTY_COLLISION_STATIC
            msg = event.get_message()
            if msg:
                list_collisions.append(event.get_message())

        elif event.get_type() == TrafficEventType.COLLISION_VEHICLE:
            score_penalty += PENALTY_COLLISION_VEHICLE
            msg = event.get_message()
            if msg:
                list_collisions.append(event.get_message())

        elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN:
            score_penalty += PENALTY_COLLISION_PEDESTRIAN
            msg = event.get_message()
            if msg:
                list_collisions.append(event.get_message())

        elif event.get_type() == TrafficEventType.TRAFFIC_LIGHT_INFRACTION:
            score_penalty += PENALTY_TRAFFIC_LIGHT
            msg = event.get_message()
            if msg:
                list_red_lights.append(event.get_message())

        elif event.get_type() == TrafficEventType.WRONG_WAY_INFRACTION:
            score_penalty += PENALTY_WRONG_WAY
            msg = event.get_message()
            if msg:
                list_wrong_way.append(event.get_message())

        elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
            msg = event.get_message()
            if msg:
                list_route_dev.append(event.get_message())

        elif event.get_type() == TrafficEventType.ON_SIDEWALK_INFRACTION:
            score_penalty += PENALTY_SIDEWALK_INVASION
            msg = event.get_message()
            if msg:
                list_sidewalk_inv.append(event.get_message())

        elif event.get_type() == TrafficEventType.STOP_INFRACTION:
            score_penalty += PENALTY_STOP
            msg = event.get_message()
            if msg:
                list_stop_inf.append(event.get_message())

        elif event.get_type() == TrafficEventType.ROUTE_COMPLETED:
            score_route = 100.0
            target_reached = True
        elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
            if not target_reached:
                if event.get_dict():
                    score_route = event.get_dict()['route_completed']
                else:
                    score_route = 0

    final_score = max(score_route - score_penalty, 0)

    return_message += "\n=================================="
    return_message += "\n==[{}:{}] [Score = {:.2f} : (route_score={}, infractions=-{})]".format(exp_name, result,
                                                                                                 final_score,
                                                                                                 score_route,
                                                                                                 score_penalty)
    if list_collisions:
        return_message += "\n===== Collisions:"
        for item in list_collisions:
            return_message += "\n========== {}".format(item)

    if list_red_lights:
        return_message += "\n===== Red lights:"
        for item in list_red_lights:
            return_message += "\n========== {}".format(item)

    if list_stop_inf:
        return_message += "\n===== STOP infractions:"
        for item in list_stop_inf:
            return_message += "\n========== {}".format(item)

    if list_wrong_way:
        return_message += "\n===== Wrong way:"
        for item in list_wrong_way:
            return_message += "\n========== {}".format(item)

    if list_sidewalk_inv:
        return_message += "\n===== Sidewalk invasions:"
        for item in list_sidewalk_inv:
            return_message += "\n========== {}".format(item)

    if list_route_dev:
        return_message += "\n===== Route deviation:"
        for item in list_route_dev:
            return_message += "\n========== {}".format(item)

    return_message += "\n=================================="

    current_statistics = {'exp_name': exp_name,
                          'score_composed': final_score,
                          'score_route': score_route,
                          'score_penalty': score_penalty,
                          'number_red_lights': len(list_red_lights),
                          'total_number_traffic_lights': count_number_traffic_lights(
                                                                            master_scenario.route,
                                                                            score_route/100.0),
                          'result': result,
                          'help_text': return_message
                          }

    return current_statistics

# TODO probably unused function.

def export_score(score_vec, file_name):

    """

    Receives a vec of dictionary as well as the configuration json
    :param score_vec:
    :return:
    """
    #TODO mechanism to add more information to this

    # TODO add the exp number ( NUmber of times it was made) and the e

    # TODO ADD ALSO a step , different marks for the experience.

    sum_route_score = 0
    sum_route_completed = 0
    sum_infractions = 0
    sum_final_score = 0
    #'score_composed': 0.0, 'score_route': 100.0, 'score_penalty': 0.0
    for score in score_vec:
        sum_final_score += score['score_composed']
        sum_infractions += score['score_penalty']
        sum_route_completed += float(score['score_route'] > 95.0)*100
        sum_route_score += score['score_route']

    # Number of runs
    number_of_runs = len(score_vec)



    # This function actually depends on the user, for now lets get the overall route score and infraction score.

    filename_csv = os.path.join( file_name.split('.')[0] + '.csv')

    csv_outfile = open(filename_csv, 'w')

    # TODO the header should maybe be written before
    csv_outfile.write("score_composed,score_penalty,score_route_completed,score_route\n")

    csv_outfile.write("%f,%f,%f,%f\n"
                      % (sum_final_score/number_of_runs,
                         sum_infractions/number_of_runs,
                         sum_route_completed/number_of_runs,
                         sum_route_score/number_of_runs)
                      )

    csv_outfile.close()

