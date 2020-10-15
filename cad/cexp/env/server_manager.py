from enum import Enum
import fcntl
import logging
import os
import psutil
import random
import string
import subprocess
import time
import carla

import socket
from contextlib import closing


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]



class ServerManager(object):
    def __init__(self, opt_dict):
        log_level = logging.INFO
        logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

        self._proc = None
        self._outs = None
        self._errs = None

    def reset(self, host="127.0.0.1", port=2000):
        raise NotImplementedError("This function is to be implemented")

    def wait_until_ready(self, wait=10.0):
        time.sleep(wait)


class ServerManagerBinary(ServerManager):
    def __init__(self, opt_dict):
        super(ServerManagerBinary, self).__init__(opt_dict)

        if 'CARLA_SERVER' in opt_dict:
            self._carla_server_binary = opt_dict['CARLA_SERVER']
        else:
            logging.error('CARLA_SERVER binary not provided!')


    def reset(self, host="127.0.0.1", port=2000):
        self._i = 0
        # first we check if there is need to clean up
        if self._proc is not None:
            logging.info('Stopping previous server [PID=%s]', self._proc.pid)
            self._proc.kill()
            self._outs, self._errs = self._proc.communicate()

        exec_command = "{} -carla-rpc-port={} -benchmark -fps=20 -quality-level=Epic >/dev/null".format(
            self._carla_server_binary, port)
        print(exec_command)
        self._proc = subprocess.Popen(exec_command, shell=True)

    def stop(self):
        parent = psutil.Process(self._proc.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
        self._outs, self._errs = self._proc.communicate()

    def check_input(self):
        while True:
            _ = self._proc.stdout.readline()
            print(self._i)
            self._i += 1


class ServerManagerDocker(ServerManager):

    def __init__(self, opt_dict):
        super(ServerManagerDocker, self).__init__(opt_dict)
        self._docker_name = opt_dict['docker_name']
        self._gpu = opt_dict['gpu']
        self._docker_id = ''

    def reset(self, host="127.0.0.1", port=2000):
        # first we check if there is need to clean up
        if self._proc is not None:
            logging.info('Stopping previous server [PID=%s]', self._proc.pid)
            self.stop()
            self._proc.kill()
            self._outs, self._errs = self._proc.communicate()

        self._docker_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(64))
        # temporary config file
        # TODO quality level to be set here
        my_env = os.environ.copy()
        my_env["NV_GPU"] = str(self._gpu)

        logging.debug("Docker command %s" % ' '.join(['docker', 'run', '--name', self._docker_id,'--rm', '-d', '-p',
                               str(port)+'-'+str(port+2)+':'+str(port)+'-'+str(port+2),
                               '--runtime=nvidia', '-e', 'NVIDIA_VISIBLE_DEVICES='+str(self._gpu), self._docker_name,
                               '/bin/bash', 'CarlaUE4.sh', '-carla-port=' + str(port)]))


        self._proc = subprocess.Popen(['docker', 'run', '--name', self._docker_id,'--rm', '-d', '-p',
                               str(port)+'-'+str(port+2)+':'+str(port)+'-'+str(port+2),
                               '--runtime=nvidia', '-e', 'NVIDIA_VISIBLE_DEVICES='+str(self._gpu), self._docker_name,
                               '/bin/bash', 'CarlaUE4.sh', '-carla-port=' + str(port)], shell=False,
                              stdout=subprocess.PIPE, env=my_env)

        (out, err) = self._proc.communicate()

        logging.debug(" Starting a docker server of id %s at port %d" % (self._docker_id, port))



        time.sleep(170)

    def stop(self):
        logging.debug("Killed a docker of id %s " % self._docker_id)
        exec_command = ['docker', 'kill', '{}'.format(self._docker_id)]
        self._proc = subprocess.Popen(exec_command)




def start_test_server(port=6666, gpu=0, docker_name='carlalatest:latest'):

    params = {'docker_name': docker_name,
              'gpu': gpu
              }

    docker_server = ServerManagerDocker(params)
    docker_server.reset(port=port)



def check_test_server(port):

    # Check if a server is open at some port

    try:
        logging.debug("Trying to Connect to Server")
        client = carla.Client(host='localhost', port=port)
        client.get_server_version()
        logging.debug("Connected to Server port " + str(port))
        del client
        return True
    except:
        return False