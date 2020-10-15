import copy
import logging
import numpy as np
import os
import time
from threading import Thread

from threading import Lock
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.challenge.envs.scene_layout_sensors import SceneLayoutMeasurement, ObjectMeasurements, threaded

class HDMapMeasurement(object):
    def __init__(self, data, frame_number):
        self.data = data
        self.frame_number = frame_number


class HDMapReader(object):
    def __init__(self, vehicle, reading_frequency=1.0):
        self._vehicle = vehicle
        self._reading_frequency = reading_frequency
        self._CARLA_ROOT = os.getenv('CARLA_ROOT', "./")
        self._callback = None
        self._frame_number = 0
        self._run_ps = True
        self.run()

    def __call__(self):
        map_name = os.path.basename(CarlaDataProvider.get_map().name)
        transform = self._vehicle.get_transform()

        return {'map_file': "{}/HDMaps/{}.ply".format(self._CARLA_ROOT, map_name),
                'transform': {'x': transform.location.x,
                              'y': transform.location.y,
                              'z': transform.location.z,
                              'yaw': transform.rotation.yaw,
                              'pitch': transform.rotation.pitch,
                              'roll': transform.rotation.roll}
                }

    @threaded
    def run(self):
        latest_read = time.time()
        while self._run_ps:
            if self._callback is not None:
                capture = time.time()
                if capture - latest_read > (1 / self._reading_frequency):
                    self._callback(HDMapMeasurement(self.__call__(), self._frame_number))
                    self._frame_number += 1
                    latest_read = time.time()
                else:
                    time.sleep(0.001)

    def listen(self, callback):
        # Tell that this function receives what the producer does.
        self._callback = callback

    def stop(self):
        self._run_ps = False

    def destroy(self):
        self._run_ps = False


class CANBusMeasurement(object):
    def __init__(self, data, frame_number):
        self.data = data
        self.frame_number = frame_number


class CANBusSensor(object):
    """
    CAN BUS pseudo sensor that gets to read all the vehicle proprieties including speed.
    This sensor is not placed at the CARLA environment. It is
    only an asynchronous interface to the forward speed.
    """

    def __init__(self, vehicle, reading_frequency):
        # The vehicle where the class reads the speed
        self._vehicle = vehicle
        # How often do you look at your speedometer in hz
        self._reading_frequency = reading_frequency
        self._callback = None
        #  Counts the frames
        self._frame_number = 0
        self._run_ps = True
        self.read_CAN_Bus()

    def _get_forward_speed(self):
        """ Convert the vehicle transform directly to forward speed """

        velocity = self._vehicle.get_velocity()
        transform = self._vehicle.get_transform()
        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def __call__(self):

        """ We convert the vehicle physics information into a convenient dictionary """
        return {
            'speed': self._get_forward_speed()
        }

    @threaded
    def read_CAN_Bus(self):
        latest_speed_read = time.time()
        while self._run_ps:
            if self._callback is not None:
                capture = time.time()
                if capture - latest_speed_read > (1 / self._reading_frequency):
                    self._callback(CANBusMeasurement(self.__call__(), self._frame_number))
                    self._frame_number += 1
                    latest_speed_read = time.time()
                else:
                    time.sleep(0.001)

    def listen(self, callback):
        # Tell that this function receives what the producer does.
        self._callback = callback

    def stop(self):
        self._run_ps = False

    def destroy(self):
        self._run_ps = False

class CallBack(object):
    def __init__(self, tag, sensor, data_provider, writer=None):
        self._tag = tag
        self._data_provider = data_provider
        self._data_provider.register_sensor(tag, sensor)
        self._writer = writer

    def __call__(self, data):
        if isinstance(data, carla.Image):
            self._parse_image_cb(data, self._tag, self._writer)
        elif isinstance(data, carla.LidarMeasurement):
            self._parse_lidar_cb(data, self._tag, self._writer)
        elif isinstance(data, carla.GnssMeasurement):
            self._parse_gnss_cb(data, self._tag, self._writer)
        elif isinstance(data, CANBusMeasurement) or isinstance(data, HDMapMeasurement) \
                or isinstance(data, SceneLayoutMeasurement) or isinstance(data, ObjectMeasurements):
            self._parse_pseudosensor(data, self._tag, self._writer)
        else:
            logging.error('No callback method for this sensor.')

    # Parsing CARLA physical Sensors

    def _parse_image_cb(self, image, tag, writer):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self._data_provider.update_sensor(image, tag, array, image.frame_number, writer)

    def _parse_lidar_cb(self, lidar_data, tag, writer):

        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        #points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        self._data_provider.update_sensor(lidar_data, tag, points, lidar_data.frame_number, writer)

    def _parse_gnss_cb(self, gnss_data, tag, writer=None):

        array = np.array([gnss_data.latitude,
                          gnss_data.longitude,
                          gnss_data.altitude], dtype=np.float32)
        self._data_provider.update_sensor(None, tag, array, gnss_data.frame_number, None)

    # The pseudo sensors already come properly parsed, so we can basically use a single function
    def _parse_pseudosensor(self, package, tag, writer=None):
        if writer is not None:
            writer.write_pseudo(package, tag)

        self._data_provider.update_sensor(None, tag, package.data, package.frame_number, None)
        #self._data_provider.update_sensor(tag, package.data, package.frame_number)


class SensorInterface(object):
    def __init__(self, number_threads_barrier=None):
        self._sensors_objects = {}
        self._data_buffers = {}
        self._timestamps = {}
        self._written = {}
        self._number_sensors = number_threads_barrier
        self._lock = Lock()


    def register_sensor(self, tag, sensor):
        if tag in self._sensors_objects:
            raise ValueError("Duplicated sensor tag [{}]".format(tag))

        self._sensors_objects[tag] = sensor
        self._data_buffers[tag] = None
        self._timestamps[tag] = -1


    def update_sensor(self, raw, tag, data, timestamp, writer):
        if tag not in self._sensors_objects:
            raise ValueError("The sensor with tag [{}] has not been created!".format(tag))
        self._data_buffers[tag] = data
        self._timestamps[tag] = timestamp
        # While all sensors are not ready we cannot way synchronization
        self._synchronize_write(writer, tag, raw)

    def all_sensors_ready(self):
        for key in self._sensors_objects.keys():
            if self._data_buffers[key] is None:
                return False
        # Sensors are ready initialize written as 0
        if not self._written:
            for key in self._sensors_objects.keys():
                self._written[key] = 0
        return True

    def wait_sensors_written(self, writer):
        unsynchronized = True

        while unsynchronized:
            unsynchronized = False
            for tag in self._written.keys():
                if self._written[tag] <=  writer._latest_id:
                    unsynchronized= True

            time.sleep(0.01)

    def _synchronize_write(self, writer, tag, raw):
        """
        Synchronize to check if all sensors have been written.
        """
        if not self.all_sensors_ready():
            return
        self._lock.acquire()
        if writer is not None and self._written[tag] > writer._latest_id:
            self._lock.release()
            return
        if writer is not None:
            writer.write_image(raw, tag)
        self._written[tag] += 1
        self._lock.release()


    def get_data(self):
        data_dict = {}
        for key in self._sensors_objects.keys():
            data_dict[key] = (self._timestamps[key], self._data_buffers[key])
        return data_dict

    def destroy(self):
        self._sensors_objects.clear()
        self._data_buffers.clear()
        self._timestamps.clear()
        self._written.clear()