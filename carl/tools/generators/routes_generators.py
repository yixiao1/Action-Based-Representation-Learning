import argparse
from cexp.env.server_manager import start_test_server, check_test_server
import carla
import socket
from contextlib import closing
""""
This script generates routes based on the compitation of the start and end scenarios.

"""


def write_routes(ofilename, output_routes, town_name):

    with open(ofilename, 'w+') as fd:
        fd.write("<?xml version=\"1.0\"?>\n")
        fd.write("<routes>\n")
        for idx, route in enumerate(output_routes):
            fd.write("\t<route id=\"{}\" map=\"{}\"> \n".format(idx, town_name))
            for wp in route:
                fd.write("\t\t<waypoint x=\"{}\" y=\"{}\" z=\"{}\"".format(wp.location.x,
                                                                           wp.location.y,
                                                                           wp.location.z))

                fd.write(" pitch=\"{}\" roll=\"{}\" yaw=\"{}\" " "/>\n".format(wp.rotation.pitch,
                                                                               wp.rotation.roll,
                                                                               wp.rotation.yaw))
            fd.write("\t</route>\n")

        fd.write("</routes>\n")




def make_routes(filename, world):
    spawn_points = world.get_map().get_spawn_points()
    routes_vector = []
    for point_a in spawn_points:
        for point_b in spawn_points:
            #print (point_a, point_b)
            if point_a != point_b:
                routes_vector.append([point_a, point_b])
            else:
                print (point_a, point_b)

    write_routes(filename, routes_vector, world.get_map().name)



def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


if __name__ == '__main__':

    description = ("CARLA AD Challenge evaluation: evaluate your Agent in CARLA scenarios\n")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-t', '--town', default='Town01', help='The town name to be used')

    parser.add_argument('-o', '--output', default='routes_test.xml', help='The outputfile route')

    arguments = parser.parse_args()

    free_port = find_free_port()

    if not check_test_server(free_port):
        start_test_server(free_port, gpu=0, docker_name='carlaped')
        print (" WAITING FOR DOCKER TO BE STARTED")


    client = carla.Client('localhost', free_port)
    client.set_timeout(25.0)

    world = client.load_world(arguments.town)

    print('World loaded')

    make_routes(arguments.output, world)

