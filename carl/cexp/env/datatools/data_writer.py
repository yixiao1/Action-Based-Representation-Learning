import os
import glob
import json
import shutil
import subprocess
import numpy as np
import scipy.misc

# TODO write expbatch related data.

class Writer(object):
    """
        Organizing the writing process, note that the sensors are written on a separate thread.
        directly on the sensor interface.
    """

    def __init__(self, dataset_name, env_name, env_number, batch_number, agent_name,
                 other_vehicles=False, road_information=False, walkers=False):
        """
            We have a certain name but also the number of thee environment  ( How many times this env was repeated)
        """

        if "SRL_DATASET_PATH" not in os.environ:
            raise  ValueError("SRL DATASET not defined, set the place where the dataset is going to be saved")

        root_path = os.environ["SRL_DATASET_PATH"]


        self._root_path = root_path
        self._experience_name = env_name
        self._dataset_name = dataset_name
        self._latest_id = 0
        # path for the writter for this specific batch
        self._full_path = os.path.join(root_path, dataset_name, env_name,
                                       str(env_number) + '_' + agent_name, str(batch_number))
        # base path, for writting the metadata for the environment
        self._base_path = os.path.join(root_path, dataset_name, env_name)
        # env full path
        self._env_full_path = os.path.join(root_path, dataset_name, env_name,
                                           str(env_number) + '_' + agent_name)
        # if we save the opponent vehicles , this makes the measurements vec more intesnse.
        self._save_opponents = other_vehicles
        # We can also save the walkers, which also make the measurement vec way bigger
        self._save_walkers = walkers
        # Agent name used for this case
        self._agent_name = agent_name
        if not os.path.exists(self._full_path):
            os.makedirs(self._full_path)


    def _build_measurements(self, world, previous):
        measurements = {"ego_actor": {},
                        "opponents": {},   # Todo add more information on demand, now just ego actor
                        'walkers': {},
                        "lane": {}
                        }
        measurements.update(previous)
        # All the actors present we save their information
        for actor in world.get_actors():
            if 'vehicle' in actor.type_id:
                if actor.attributes['role_name'] == 'scenario':
                    transform = actor.get_transform()
                    velocity = actor.get_velocity()
                    measurements['ego_actor'].update({

                        "position": [transform.location.x, transform.location.y, transform.location.z],
                        "orientation": [transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw],
                        "velocity": [velocity.x, velocity.y, velocity.z]
                     }
                    )
                elif actor.attributes['role_name'] == 'autopilot' and self._save_opponents:
                    transform = actor.get_transform()
                    velocity = actor.get_velocity()
                    measurements['opponents'].update( { actor.id: {

                        "position": [transform.location.x, transform.location.y,
                                     transform.location.z],
                        "orientation": [transform.rotation.roll, transform.rotation.pitch,
                                        transform.rotation.yaw],
                        "velocity": [velocity.x, velocity.y, velocity.z]
                    }})
            elif 'walker.pedestrian' in actor.type_id:
                if actor.attributes['role_name'] == 'walker' and self._save_walkers:
                    transform = actor.get_transform()
                    velocity = actor.get_velocity()
                    measurements['walkers'].update({actor.id: {

                        "position": [transform.location.x, transform.location.y,
                                     transform.location.z],
                        "orientation": [transform.rotation.roll, transform.rotation.pitch,
                                        transform.rotation.yaw],
                        "velocity": [velocity.x, velocity.y, velocity.z]
                    }})

        # Add other actors and lane information
        # general actor info
        # type_id
        # parent
        # semantic_tags
        # is_alive
        # attributes
        # get_world()
        # get_location()
        # get_transform()
        # get_velocity()
        # get_angular_velocity()
        # get_acceleration()

        return measurements

    def _create_scenario_dict(self, scenarios_object_list):

        scenario_info = {}
        for scenario in scenarios_object_list:

            scenario_info.update({'name': scenario.__class__.__name__})

        return scenario_info

    def _write_json_measurements(self, measurements, control, scenario_control, affordances):
        # Build measurements object

        with open(os.path.join(self._full_path, 'measurements_' + str(self._latest_id).zfill(6) + '.json'), 'w') as fo:
            jsonObj = {}
            jsonObj.update(measurements)
            jsonObj.update(affordances)
            jsonObj.update({'steer': np.nan_to_num(control.steer)})
            jsonObj.update({'throttle': np.nan_to_num(control.throttle)})
            jsonObj.update({'brake': np.nan_to_num(control.brake)})
            jsonObj.update({'hand_brake': control.hand_brake})
            jsonObj.update({'reverse': control.reverse})
            jsonObj.update({'steer_noise': scenario_control.steer})
            jsonObj.update({'throttle_noise': scenario_control.throttle})
            jsonObj.update({'brake_noise': scenario_control.brake})

            fo.write(json.dumps(jsonObj, sort_keys=True, indent=4))

    def save_experience(self, world, experience_data, affordances):
        """
         It is also used to step the current data being written
        :param measurements:
        :return:
        """
        # saves the dictionary following the measurements - image - episodes format.  Even though episodes
        # Are completely independent now.
        # We join the building of the measurements with some extra data that was calculated
        self._write_json_measurements(self._build_measurements(world, experience_data['exp_measurements']),
                                      experience_data['ego_controls'],
                                      experience_data['scenario_controls'], affordances
                                     )

        # Before we increment we make sure everyone made their writting
        self._latest_id += 1


    def save_summary(self, statistics):

        with open(os.path.join(self._full_path, 'summary.json'), 'w') as fo:
            jsonObj = {}
            jsonObj.update(statistics)
            fo.write(json.dumps(jsonObj, sort_keys=True, indent=4))


    def save_metadata(self, environment, list_scenarios):

        with open(os.path.join(self._base_path, 'metadata.json'), 'w') as fo:
            jsonObj = {}

            # The full name of the experience ( It can be something different for now we keep the same)
            jsonObj.update({'full_name': environment._environment_name})
            # The sensors dictionary used
            jsonObj.update({'sensors': environment._sensor_desc_vec})

            # The scenarios used and its configuration, a dictionary with the scenarios and their parameters
            # Should also consider the randomly generate parameters from the scenario
            scenario_dict = self._create_scenario_dict(list_scenarios)
            # TODO the full list of scenarios can be already some dictionary on the beggining no need to send instanced ones
            jsonObj.update({'scenarios': scenario_dict})

            # Set of weathers, all the posible
            jsonObj.update({'set_of_weathers': environment._environment_name.split('_')[0]})

            fo.write(json.dumps(jsonObj, sort_keys=True, indent=4))

    def delete(self):
        """
        If the experience was bad, following the scenarios criteria, we may want to delete it.
        :return:
        """
        shutil.rmtree(self._full_path)

    def delete_sensors(self):
        """
        Delete all the PNG files.
        Currently only focus on PNG later we can add more formats
        :return:
        """
        for f in glob.glob(os.path.join(self._full_path,"*.png")):
            os.remove(f)

    def delete_env(self):

        shutil.rmtree(self._env_full_path)

        # TODO check this posible inconsistency
        #if len(os.listdir(self._full_path)) == 0:
        #    shutil.rmtree(self._base_path)

    def make_video(self, sensor_names):
        """
        The idea of this function is to make a low res video for quickly debuging the
        dataset generated. That is good for debuging from remote machines
        :return:
        """
        # We have an arbitrary video path here.
        if not os.path.exists('_videos'):
            os.mkdir('_videos')
        # The folder where the episode is.
        folder_path = self._full_path
        print (" Saving on this full path")

        for sensor_spec in sensor_names:
            if sensor_spec['type'].startswith('sensor.camera'):
                output_name = os.path.join('_videos', self._agent_name + '_' + self._experience_name +
                                                      '_' + sensor_spec['id'] )
                print ( " THis is the output name ", output_name)
                subprocess.call(['ffmpeg', '-f', 'image2', '-i', os.path.join(folder_path,
                                                                  sensor_spec['id'] + '%06d.png'),
                                '-vcodec', 'mpeg4', '-y', output_name + '.mp4'])


    """
        functions called asynchronously by the thread to write the sensors
    """

    def write_image(self, image, tag, resize_images=False):
        if resize_images:
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[65:460, :, :3]
            array = array[:, :, ::-1]
            image = scipy.misc.imresize(array, (88, 200))
            scipy.misc.imsave(os.path.join(self._full_path, tag + '%06d.png' % self._latest_id), image)
        else:
            image.save_to_disk(os.path.join(self._full_path, tag + '%06d.png' % self._latest_id))

    def write_lidar(self, lidar, tag):
        lidar.save_to_disk(os.path.join(self._full_path, tag + '%06d.png' % self._latest_id))

    # in principle these are not needed.
    def write_gnss(self, gnss, tag):
        pass

    def write_pseudo(self, pseudo_data, pseudo_tag):
        pass
