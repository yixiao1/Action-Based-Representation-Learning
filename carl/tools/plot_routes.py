import os
import sys
import matplotlib.pyplot as plt
import logging
import argparse

from cexp.env.datatools.map_drawer import draw_map, draw_route
from cexp.env.utils.route_configuration_parser import parse_routes_file
from srunner.challenge.utils.route_manipulation import interpolate_trajectory

from cexp.env.server_manager import start_test_server, check_test_server

import carla


def draw_routes(world, routes_file, output_folder):

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    routes_descriptions = parse_routes_file('database/' + routes_file)

    for route in routes_descriptions:
        fig = plt.figure()
        plt.xlim(-200, 5500)
        plt.ylim(-200, 5500)

        draw_map(world)

        _, route_interpolated = interpolate_trajectory(world, route['trajectory'])
        draw_route(route_interpolated)

        fig.savefig(os.path.join(output_folder,
                                 'route_' + str(route['id']) + '_.png'),
                    orientation='landscape', bbox_inches='tight', dpi=1200)



def draw_spawn_points():

    # DRaw the points from 0.8

    # Try to match them here on 0.9.X
    pass


if __name__ == '__main__':

    description = ("CARLA AD Challenge evaluation: evaluate your Agent in CARLA scenarios\n")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-t', '--town', default='Town01', help='The town name to be used')

    parser.add_argument('-o', '--output', default='database/output_routes',
                        help='The outputfile json')

    parser.add_argument('-r', '--input-route', default='routes/routes_all.xml',
                        help='The outputfile json')



    arguments = parser.parse_args()

    if not check_test_server(6666):
        start_test_server(6666)
        print (" WAITING FOR DOCKER TO BE STARTED")

    client = carla.Client('localhost', 6666)

    client.set_timeout(30.0)
    world = client.load_world(arguments.town)
    draw_routes(world, arguments.input_route, arguments.output)
