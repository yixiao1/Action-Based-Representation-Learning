import time
import random
import traceback

from cexp.env.scenario_identification import distance_to_intersection, identify_scenario
from cexp.env.server_manager import start_test_server, check_test_server

from cexp.cexp import CEXP
from cexp.agents.NPCAgent import NPCAgent

import carla



def test_distance_intersection_speed(world, N=100):
    # I will spawn N vehicles

    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    blueprint = random.choice(blueprint_library.filter('vehicle*'))
    capture = time.time()
    count = 1
    for point in spawn_points[:N]:
        vehicle = world.try_spawn_actor(blueprint, point)
        if vehicle is None:
            continue
        print("Spawn pont %d distance %f " %(count, distance_to_intersection(vehicle, world.get_map(), resolution=0.1)))
        count += 1

    print ("Took  %f  seconds " % ((time.time() - capture)/count))



def test_identification(world, N=100):
    # I will spawn N vehicles

    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    blueprint = random.choice(blueprint_library.filter('vehicle*'))
    capture = time.time()
    count = 1
    for point in spawn_points[:N]:
        vehicle = world.try_spawn_actor(blueprint, point)
        if vehicle is None:
            continue
        print("Spawn pont %d distance %f scenario %s " %(count, distance_to_intersection(vehicle, world.get_map(),
                                                                                         resolution=0.1),
                                                         identify_scenario(vehicle)))
        count += 1

    print ("Took  %f  seconds " % ((time.time() - capture)/count))


if __name__ == '__main__':
    # PORT 6666 is the default port for testing server


    if not check_test_server(6666):
        start_test_server(6666)
        print (" WAITING FOR DOCKER TO BE STARTED")


    client = carla.Client('localhost', 6666)

    world = client.load_world('Town01')

    test_distance_intersection_speed(world)
    test_identification(world)
