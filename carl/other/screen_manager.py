import numpy as np
import os
import colorsys
import math
from random import randint


import scipy
import cv2
from scipy.misc import imsave
import pygame

clock = pygame.time.Clock()

def draw_vbar_on(img, bar_intensity, x_pos, color=(0, 0, 255)):
    bar_size = int(img.shape[1] / 6 * bar_intensity)
    initial_y_pos = int(img.shape[0] - img.shape[0] / 6)
    # print bar_intensity

    for i in range(bar_size):
        if bar_intensity > 0.0:
            y = initial_y_pos - i
            for j in range(10):
                #img[y, x_pos + j] = color
                img[x_pos + j, y] = color

def get_vec_dist(x_dst, y_dst, x_src, y_src):
    vec = np.array([x_dst, y_dst] - np.array([x_src, y_src]))
    dist = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
    if dist == 0:
        dist = 1
    return vec / dist, dist



def generate_ncolors(num_colors):
    color_pallet = []
    for i in range(0, 360, 360 / num_colors):
        hue = i
        saturation = 90 + float(randint(0, 1000)) / 1000 * 10
        lightness = 50 + float(randint(0, 1000)) / 1000 * 10

        color = colorsys.hsv_to_rgb(float(hue) / 360.0, saturation / 100, lightness / 100)

        color_pallet.append(color)

    # addColor(c);
    return color_pallet


def get_average_over_interval(vector, interval):
    avg_vector = []
    for i in range(0, len(vector), interval):
        initial_train = i
        final_train = i + interval

        avg_point = sum(vector[initial_train:final_train]) / interval
        avg_vector.append(avg_point)

    return avg_vector


def get_average_over_interval_stride(vector, interval, stride):
    avg_vector = []
    for i in range(0, len(vector) - interval, stride):
        initial_train = i
        final_train = i + interval

        avg_point = sum(vector[initial_train:final_train]) / interval
        avg_vector.append(avg_point)

    return avg_vector



# ***** functions to draw lines *****
def draw_pt(img, x, y, color, sz=1):
    row, col = perspective_tform(x, y)
    if row >= 0 and row < img.shape[0] and col >= 0 and col < img.shape[1]:
        img[int(row - sz):int(row + sz), int(col - sz - 65):int(col + sz - 65)] = color


def draw_path(img, path_x, path_y, color):
    for x, y in zip(path_x, path_y):
        draw_pt(img, x, y, color)


# ***** functions to draw predicted path *****

def calc_curvature(v_ego, angle_steers, angle_offset=0):
    deg_to_rad = np.pi / 180.
    slip_fator = 0.0014  # slip factor obtained from real data
    steer_ratio = 15.3
    wheel_base = 2.67

    angle_steers_rad = (angle_steers - angle_offset) * deg_to_rad
    curvature = angle_steers_rad / (steer_ratio * wheel_base * (1. + slip_fator * v_ego ** 2))
    return curvature


def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
    # *** this function return teh lateral offset given the steering angle, speed and the lookahead distance
    curvature = calc_curvature(v_ego, angle_steers, angle_offset)

    # clip is to avoid arcsin NaNs due to too sharp turns
    y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999)) / 2.)
    return y_actual, curvature


def draw_path_on(img, speed_ms, angle_steers, color=(0, 0, 255)):
    path_x = np.arange(0., 50.1, 0.5)
    path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)
    draw_path(img, path_x, path_y, color)






class ScreenManager(object):

    def __init__(self, load_steer=False, save_folder='test_images'):

        pygame.init()
        self.save_folder = save_folder


        self._wheel = cv2.imread('tools/wheel.png') #, cv2.IMREAD_UNCHANGED)
        self._wheel = self._wheel[:,:, ::-1]


    # If we were to load the steering wheel load it

    # take into consideration the resolution when ploting
    # TODO: Resize properly to fit the screen ( MAYBE THIS COULD BE DONE DIRECTLY RESIZING screen and keeping SURFACES)

    def start_screen(self, resolution, aspect_ratio, scale=1, no_display=False):


        if no_display:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        self._resolution = resolution

        self._aspect_ratio = aspect_ratio
        self._scale = scale

        ar = self._wheel.shape[1]/self._wheel.shape[0]
        new = int(self._wheel.shape[0] / 10)

        self._wheel = cv2.resize(self._wheel, (new, int(new*ar)))

        size = (resolution[0] * aspect_ratio[0], resolution[1] * aspect_ratio[1])

        #pygame.display.set_mode((1, 1))

        #self._screen = pygame.surface.Surface(size, 0, 24).convert()
        self._screen = pygame.display.set_mode((size[0] * scale, size[1] * scale))
        #self._screen.set_alpha(None)

        pygame.display.set_caption("Human/Machine - Driving Software")

        self._camera_surfaces = []

        for i in range(aspect_ratio[0] * aspect_ratio[1]):
            camera_surface = pygame.surface.Surface(resolution, 0, 24).convert()

            self._camera_surfaces.append(camera_surface)

    def paint_on_screen(self, size, content, color, position, screen_position):

        myfont = pygame.font.SysFont("monospace", size * self._scale, bold=True)

        position = (position[0] * self._scale, position[1] * self._scale)

        final_position = (position[0] + self._resolution[0] * (self._scale * (screen_position[0])), \
                          position[1] + (self._resolution[1] * (self._scale * (screen_position[1]))))

        content_to_write = myfont.render(content, 1, color)

        self._screen.blit(content_to_write, final_position)

    def set_array(self, array, screen_position, position=(0, 0), scale=None):

        if scale == None:
            scale = self._scale

        if array.shape[0] != self._resolution[1] or array.shape[1] != self._resolution[0]:
            array = scipy.misc.imresize(array, [self._resolution[1], self._resolution[0]])

        # print array.shape, self._resolution

        final_position = (position[0] + self._resolution[0] * (scale * (screen_position[0])), \
                          position[1] + (self._resolution[1] * (scale * (screen_position[1]))))



        self._camera_surfaces[screen_position[0] * screen_position[1]] = \
            pygame.surfarray.make_surface(array.swapaxes(0, 1).astype(np.uint8))
        self._camera_surfaces[screen_position[0] * screen_position[1]].set_colorkey((255, 0, 255))
        #pygame.surfarray.blit_array(self._camera_surfaces[screen_position[0] * screen_position[1]],
        #                        array.swapaxes(0, 1).astype(np.uint32))

        camera_scale = pygame.transform.scale(self._camera_surfaces[screen_position[0] * screen_position[1]],
                                              (int(self._resolution[0] * scale), int(self._resolution[1] * scale)))

        self._screen.blit(camera_scale, final_position)

    def draw_wheel_on(self, steer):

        cols, rows, c = self._wheel.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90 * steer, 1)
        rot_wheel = cv2.warpAffine(self._wheel, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        # scale = 0.5
        position = (self._resolution[0]*1.5 - cols / 2, int(self._resolution[1] / 1.5) - rows / 2)
        # print position

        wheel_surface = pygame.surface.Surface((rot_wheel.shape[1], rot_wheel.shape[0]), 0, 24).convert()


        wheel_surface.set_colorkey((0, 0, 0))
        pygame.surfarray.blit_array(wheel_surface, rot_wheel.swapaxes(0, 1))

        self._screen.blit(wheel_surface, position)

    # This one plot the nice wheel

    def plot_camera(self, sensor_data, screen_position=[0, 0]):

        if sensor_data.shape[2] < 3:
            sensor_data = np.stack((sensor_data,) * 3, axis=2)
            sensor_data = np.squeeze(sensor_data)
        # print sensor_data.shape
        self.set_array(sensor_data, screen_position)

        pygame.display.flip()

    def plot_camera_steer(self, sensor_data, control=None, screen_position=[0, 0], status=None, output_img=None):

        """

        :param sensor_data:
        :param steer:
        :param screen_position:
        :param status: Is the dictionary containing important status from the data
        :return:
        """


        size_x, size_y, size_z = sensor_data.shape

        if sensor_data.shape[2] < 3:
            sensor_data = np.stack((sensor_data,) * 3, axis=2)
            sensor_data = np.squeeze(sensor_data)


        self.set_array(sensor_data, screen_position)

        if control is not None:

            steer, acc, brake = control[0], control[1], control[2]
            #initial_y_pos = size_y - int(size_y / 5)
            draw_vbar_on(sensor_data, acc, int(1.5 * size_x / 8), (0, 128, 0))
            draw_vbar_on(sensor_data, brake, 160, (128, 0, 0))
            initial_y_pos = 10

            self.draw_wheel_on(steer)
            self.paint_on_screen(int(size_x / 8), 'GAS', (0, 128, 0),
                                 (10, initial_y_pos),
                                 screen_position)

            self.paint_on_screen(int(size_x / 8), 'BRAKE', (128, 0, 0),
                                 (150, initial_y_pos),
                                 screen_position)

        if status is not None:
            if status['directions'] == 4:
                text = "GO RIGHT"
            elif status['directions'] == 3:
                text = "GO LEFT"
            else:
                text = "GO STRAIGHT"
            #
            #
            if status['directions'] != 2:
                direction_pos = (int(size_x / 10), int(size_y / 10))

                self.paint_on_screen(int(size_x / 8), text, (0, 255, 0), direction_pos, screen_position)

            self.paint_on_screen(int(size_x / 10), "Direction: %.2f" % status['directions'], (64, 255, 64),
                                 (int(size_x / 4.0), int(size_y / 10)),
                                 screen_position)

            self.paint_on_screen(int(size_x / 10), "Distance: %.2f" % status['distance_intersection'], (64, 255, 64),
                                 (int(size_x / 4.0), int(size_y / 5)),
                                 screen_position)

            self.paint_on_screen(int(size_x / 10), "RoadCurve: %.5f" % status['road_angle'], (64, 255, 64),
                                 (int(size_x / 4.0), int(size_y / 3)),
                                 screen_position)

            self.paint_on_screen(int(size_x / 10), "Scenario: %s" % status['scenario'].split('_')[0], (64, 255, 64),
                                 (int(size_x / 4.0), int(size_y / 2)),
                                 screen_position)


        pygame.display.flip()
        if output_img is not None:
            pygame.image.save(self._screen, output_img)


