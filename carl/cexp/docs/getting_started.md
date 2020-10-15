

####Getting Started

##### Installation

The drive CARLA experience depends on the scenario runner from the CARLA repository

Clone the scenario runner:

    git clone -b development  https://github.com/carla-simulator/scenario_runner.git


Add scenario runner to your PYTHONPATH:
    
    export PYTHONPATH=`pwd`/scenario_runner:$PYTHONPATH


Download the latest version of CARLA, the nightly build.

    http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/Dev/CARLA_Latest.tar.gz


Clone the latest master from the CARLA: 
    
    git clone https://github.com/carla-simulator/carla.git


Make a docker out of it, so you can run no screen without any problem. 

    docker image build -f <path_to_clone_carla_git_master>/Util/Docker/Release.Dockerfile \
      -t carlalatest <path_to_carla_server_root>


Add CARLA binaries to your PYTHONPATH:

    export PYTHONPATH=`pwd`/<path_to_carla_server_root>/PythonAPI/carla/dist/carla-0.9.5-py3.5-linux-x86_64.egg:$PYTHONPATH

Add the CARLA API to your PYTHONPATH:

    export PYTHONPATH=`pwd`/carla/PythonAPI/carla:$PYTHONPATH
    

#### Run some examples

Collect data

    python3 example_npc.py

Train an RL agent

    python3 example_pg.py

    
    
##### Dependencies notes

Use py_trees 0.8.3  not the latest version

 