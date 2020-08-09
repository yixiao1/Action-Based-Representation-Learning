import argparse
import numpy as np
import os
import logging
import sys


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from cexp.env.server_manager import start_test_server, check_test_server
from cexp.env.datatools.map_drawer import draw_point, draw_map, draw_text
import carla
from PIL import Image
from tools.converter import Converter

from srunner.challenge.utils.route_manipulation import interpolate_trajectory

""""
This script generates routes based on the compitation of the start and end scenarios.

"""

def estimate_route_distance(route):
    route_length = 0.0  # in meters
    prev_point = route[0][0]
    for current_point, _ in route[1:]:
        dist = current_point.location.distance(prev_point.location)
        route_length += dist
        prev_point = current_point


    return route_length



class CarlaMap(object):

    def __init__(self, city, pixel_density=0.1643, node_density=50):
        dir_path = os.path.dirname(__file__)
        city_file = os.path.join(dir_path, city + '.txt')

        city_map_file = os.path.join(dir_path, city + '.png')
        city_map_file_lanes = os.path.join(dir_path, city + 'Lanes.png')
        city_map_file_center = os.path.join(dir_path, city + 'Central.png')



        self._pixel_density = pixel_density
        # The number of game units per pixel. For now this is fixed.

        self._converter = Converter(city_file, pixel_density, node_density)

        # Load the lanes image
        self.map_image_lanes = Image.open(city_map_file_lanes)
        self.map_image_lanes.load()
        self.map_image_lanes = np.asarray(self.map_image_lanes, dtype="int32")
        # Load the image
        self.map_image = Image.open(city_map_file)
        self.map_image.load()
        self.map_image = np.asarray(self.map_image, dtype="int32")

        # Load the lanes image
        self.map_image_center = Image.open(city_map_file_center)
        self.map_image_center.load()
        self.map_image_center = np.asarray(self.map_image_center, dtype="int32")

    def check_pixel_on_map(self, pixel):

        if pixel[0] < self.map_image_lanes.shape[1] and pixel[0] > 0 and \
                pixel[1] < self.map_image_lanes.shape[0] and pixel[1] > 0:
            return True
        else:
            return False

    def get_map(self, height=None):
        if height is not None:
            img = Image.fromarray(self.map_image.astype(np.uint8))

            aspect_ratio = height / float(self.map_image.shape[0])

            img = img.resize((int(aspect_ratio * self.map_image.shape[1]), height), Image.ANTIALIAS)
            img.load()
            return np.asarray(img, dtype="int32")
        return np.fliplr(self.map_image)


    def convert_to_node(self, input_data):
        """
        Receives a data type (Can Be Pixel or World )
        :param input_data: position in some coordinate
        :return: A node object
        """
        return self._converter.convert_to_node(input_data)

    def convert_to_pixel(self, input_data):
        """
        Receives a data type (Can Be Node or World )
        :param input_data: position in some coordinate
        :return: A node object
        """
        return self._converter.convert_to_pixel(input_data)

    def convert_to_world(self, input_data):
        """
        Receives a data type (Can Be Pixel or Node )
        :param input_data: position in some coordinate
        :return: A node object
        """
        return self._converter.convert_to_world(input_data)








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




def make_routes(filename, positions, spawn_points, town_name):

    routes_vector = []
    for pos_tuple in positions:
        point_a = spawn_points[pos_tuple[1]]
        point_b = spawn_points[pos_tuple[0]]
        if point_a != point_b:
            routes_vector.append([point_a, point_b])
        else:
            print (point_a, point_b)

    write_routes(filename, routes_vector, town_name=town_name)


def view_start_positions(world, positions_to_plot):
    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. The same way as in the client example.
    print('CarlaClient connected')


    count = 0

    fig, ax = plt.subplots(1)

    plt.xlim(-200, 5000)
    plt.ylim(-200, 5000)
    draw_map(world)

    for position in positions_to_plot:

        # Check if position is valid

        # Convert world to pixel coordinates
        draw_point(position.location, (1,0,0), 12, alpha=None)

        draw_text(str(count), position.location, (1,0,0), 5)

        count += 1

    plt.axis('off')
    plt.show()
    fig.savefig('map' + str(count) + '.pdf',
                orientation='landscape', bbox_inches='tight')

def get_positions_further_thresh(filename, world, thresh):

    spawn_points = world.get_map().get_spawn_points()
    routes_vector = []
    count_a = 0
    for point_a in spawn_points:
        count_b = 0
        for point_b in spawn_points:
            # print (point_a, point_b)
            _, route_ab = interpolate_trajectory(world, [point_a, point_b])
            distance = estimate_route_distance(route_ab)
            print ( " Distance ", distance)
            print ()
            if point_a != point_b and distance > thresh:
                routes_vector.append([point_a, point_b])
            count_b += 1
        count_a += 1


    write_routes(filename, routes_vector, world.get_map().name)

def plot_all_spawn_points_carla08():


    pass


if __name__ == '__main__':

    description = ("CARLA AD Challenge evaluation: evaluate your Agent in CARLA scenarios\n")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-t', '--town', default='Town01', help='The town name to be used')

    parser.add_argument('-o', '--output', default='routes_test.xml', help='The outputfile route')

    arguments = parser.parse_args()

    if not check_test_server(6666):
        start_test_server(6666)
        print (" WAITING FOR DOCKER TO BE STARTED")

    client = carla.Client('localhost', 6666)

    client.set_timeout(30.0)
    world = client.load_world(arguments.town)

    spawn_points = world.get_map().get_spawn_points()
    print (spawn_points)
    view_start_positions(world, spawn_points)

    selected_pos = [ [10, 54], [53, 11], [48, 7], [61, 71], [74, 62], [50, 79], [75,49],
                     [80, 53], [80, 50], [60, 80], [83, 61], [94, 72], [43, 74],
                       [13, 66], [89, 64],
                                           [15, 70],  [11, 59],
                       [15, 94], [41, 7],
                       [33, 13], [67, 43],

                       [26, 10], [7, 29], [97, 100], [1, 96] ]

    #selected_pos = [[17, 12], [14, 18], [7, 18], [15, 8], [36, 42], [43, 34], [33, 40], [39,34],
    #                 [35,0], [89,24], [89,67], [30,21], [22,31], [59, 68], [66,69] ,[62,55] ,[57,63],
    #                  [48,52], [51,47], [46,52] , [51,44], [74,76], [75,73], [35,56], [100,24]    ]

    make_routes(arguments.output, selected_pos, spawn_points, world.get_map().name)

