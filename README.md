# Action-Based Representation Learning for Autonomous Driving

-------------------------------------------------------------
This repository is for running the experiments of paper: Action-Based Representation Learning for Autonomous Driving

Basically, the processes can be defined as four steps:

 * Train an encoder model (Behaviour Cloning (BC), Inverse, Forward, ST-DIM)
 * Train a MLP for affordances outputs. The Pre-trained encoder model need to be used.
 * Validation on affordances prediction.
 * Actual drive using controller tuned with affordances prediction.

-------------------------------------------------------------
### Setting Environments & Getting Datasets

1. Download the dataset with this [link]().

2. Define the path of the dataset folder with SRL_DATASET_PATH:

    export SRL_DATASET_PATH=<Path to where your datasets are>

3. Download this repository

     git clone https://github.com/yixiao1/CoRL2020.git

2. Add packages to your PYTHONPATH:

    export PYTHONPATH=<Path to carla>:<Path to carla .egg file>:<Path to scenario_runner>:<Path to cexp>

    example:

        export PYTHONPATH=/<root dir>/CoRL2020/Carla96ped4/PythonAPI/carla:/<root dir>/CoRL2020/PythonAPI/carla:/<root dir>/CoRL2020/scenario_runner:/<root dir>/CoRL2020/carl

        where `root dir` is the directory you put the downloaded CoRL2020 repository

-------------------------------------------------------------
### Train Encoder

1. You need to define configuration files for training. Refer to files in [configs folder]()

2. Run the main.py file with "train_encoder" process:

   python3 main.py --single-process train_encoder --gpus 0 --encoder-folder ENCODER --encoder-exp BC_im_5Hours_seed1

   where `--single-process` defines the process type, `--gpus` defines the gpu to be used, `--encoder-folder` is the folder to save experiments, `--encoder-exp` is the experiment of encoder training

-------------------------------------------------------------
### Train affordances

1. You need to define configuration files for training. Refer to files in [configs folder]()

2. Run the main.py file with "train" process:

   python3 main.py --single-process train --gpus 0 --encoder-folder ENCODER --encoder-exp BC_im_5Hours_seed1 --encoder-checkpoint 100000 -f EXP -e BC_im_5Hours_seed1_encoder_frozen_1FC_30mins

   where `--single-process` defines the process type, `--gpus` defined the gpu to be used, `--encoder-folder` is the experiment folder of encoder to be used, `--encoder-exp` is the experiment name of encoder to be used, `--encoder-checkpoint` is the specific encoder checkpoint to be used, `-f` is the folder to save experiments of affordances prediction, `-e` is the experiment of affordance training

-------------------------------------------------------------
### Validate on affordances prediction

1. Run the main.py file with "validation" process. You will need to define the path to the json file of validation dataset:

    python3 main.py --single-process validation --gpus 0 --encoder-folder ENCODER --encoder-exp BC_im_5Hours_seed1 --encoder-checkpoint 100000 -f EXP -e BC_im_5Hours_seed1_encoder_frozen_1FC_30mins -vj /<root dir>/CoRL2020/carl/database/CoRL2020/dataset_dynamic_Town01_1Hour_valid.json

    where `-vj` defines the path to your validation json file

-------------------------------------------------------------
### Driving on CARLA benchmark

1. The driving results will be saved to your SRL_DATASET_PATH, you could re-define if you want to save to another path

2. You need to build a docker with your carla version

3. Set up your CARLA drivng PYTHONPATH:

    export PYTHONPATH=<Path to CoRL2020 repository>:<Path to cad>:<Path to carla>::<Path to carla .egg file>:<Path to scenario_runner>

    example:

        export PYTHONPATH=/<root dir>/CoRL2020/:/<root dir>/CoRL2020/cad:/<root dir>/CoRL2020/Carla96ped4/PythonAPI/carla:/<root dir>/CoRL2020/PythonAPI/carla:/<root dir>/CoRL2020/scenario_runner

        where `root dir` is the directory you put the downloaded CoRL2020 repository


4. Define a config.json for using a specific model, and put it inside the logs folder of that model: _logs/(exp folder)/(exp exp)

   check on this [config.json]() example

5. To run the benchmark, go under [driving-benchmarks]() folder, and run:

    python3 benchmark_runner.py -b NoCrash -a /home/yixiao/CoRL2020/drive/AffordancesAgent.py -d carlaped -c /home/yixiao/CoRL2020/_logs/EXP/BC_im_5Hours_seed1_encoder_finetuning_3FC_30mins_s1_100000/config.json --gpu 2

    where `-b` is the benchmark, `-a` is the path to the agent class, `-c` is the [configuration file] () for driving
