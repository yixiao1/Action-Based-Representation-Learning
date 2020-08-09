# Action-Based Representation Learning for Autonomous Driving

-------------------------------------------------------------
This repository is for running the experiments of paper: Action-Based Representation Learning for Autonomous Driving

Basically, the processes can be defined as four steps:

 * Train an encoder model (Behaviour Cloning (BC), Inverse, Forward, ST-DIM)
 * Train a MLP for affordances outputs. The Pre-trained encoder model need to be used.
 * Validation on affordances prediction.
 * Actual drive using controller tuned with affordances prediction.

-------------------------------------------------------------
### Datasets

1. Download the dataset with this [link]().

2. Define the path of the dataset folder with SRL_DATASET_PATH:

    export SRL_DATASET_PATH=<The path to your dataset folder>

-------------------------------------------------------------
### Train Encoder

1. Your need to add some paths to your PYTHONPATH:

	 export PYTHONPATH=<Path to carla>:<Path to scenario_runner>:<Path to carla .egg file>:<Path to cexp>

2. You need to define configuration files for training. Refer to files in [configs folder]()

3. Run main file

   python3 main.py --single-process train_encoder --gpus <the gpu id to be used> -f <the experiment folder> -e <the experiment name>

-------------------------------------------------------------
### Train affordances

1. You need to define the path to the dataset folder with SRL_DATASET_PATH:

    export SRL_DATASET_PATH=<The path to your dataset folder>

2. Your need to add some paths to your PYTHONPATH:

	 export PYTHONPATH=<Path to carla>:<Path to scenario_runner>:<Path to carla .egg file>:<Path to cexp>

3. You need to define configuration files for training. Refer to files in [configs folder]()

3. Run coiltraine.py

   python3 coiltraine.py --single-process train --gpus  <the gpu id to be used> -f <the experiment folder> -e <the experiment name> --encoder-folder <the experiment folder of encoder to be used> --encoder-exp <the experiment name of encoder to be used> --encoder-checkpoint <the checkpoint of encoder to be used>

-------------------------------------------------------------
### Validate on affordances prediction

When you finish affordances training, the validation will be easy to run. You just need to keep the same runing script as before, but simply change the `--single-process train` to `--single-process validation`, and add one more args  `-vj`, which is the path to the validation json file (eg. [this json file](https://github.com/felipecode/cexp/blob/cexp_ICML/database/dataset_ICML_Town01_valid_25mins.json))

   Example:

        python3 coiltraine.py --single-process validation --gpus 0 -f EXP -e ETE_5Hours_1_encoder_finetunning_1Hours_1 --encoder-folder ENCODER --encoder-exp ETE_5Hours_1 --encoder-checkpoint 100000 -vj /datatmp/Experiments/yixiao/carl/database/dataset_ICML_Town01_valid_25mins.json

for one-step-affordances, you don't need to put `--encoder-folder`, `--encoder-exp` and `--encoder-checkpoint`, since we do not train affordances second time

   Example:

        python3 coiltraine.py --single-process validation --gpus 0 -f ENCODER -e one_step_aff_5Hours_1 -vj /datatmp/Experiments/yixiao/carl/database/dataset_ICML_Town01_valid_25mins.json

-------------------------------------------------------------
### Driving on CARLA benchmark

1. The first thing you need to do is to define the sensor_saved path and add some path to PYTHONPATH:

   Examples:

        export SRL_DATASET_PATH = /home/yixiao/Datasets/ICML

        export PYTHONPATH=/home/yixiao/Coiltraine-ICML/:/home/yixiao/Carla96ped4/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg:/home/yixiao/Carla96ped4/PythonAPI/carla/:/home/yixiao/cad/:/home/yixiao/scenario_runner

   (Note that: I modified a bit the code in my CAD folder to debug relative angle error and save affordances results in measurement files, so please set the path to `/home/yixiao/cad/` but not your CAD path.)


2. You need to define a config.json for a specific model, and put it inside the logs folder of that model: _logs/(exp folder)/(exp exp)

   check on this [config.json](https://github.com/yixiao1/Coiltraine-ICML/blob/master/_logs/EXP/ETE_20Hours_1_encoder_finetunning_5Hours_1_100000/config.json) example

3. To run the benchmark, go to /home/yixiao/driving-benchmarks-carla_09_cexp

    (Note: the original code was downloaded from this [branch](https://github.com/carla-simulator/driving-benchmarks/tree/carla_09_cexp), but I modified a bit the code to debug relative angle error and save affordances results in measurement files,
    so please use this folder (but not the online one) to run, then you could get the affordances results in measurement files and also the trajectory of relative angle error.)

   and run:

        python3 benchmark_runner.py -b NoCrashS -a /datatmp/Experiments/yixiao/Coiltraine-ICML/drive/AffordancesAgent.py  --port 8666 -d carlaped -c /datatmp/Experiments/yixiao/Coiltraine-ICML/_logs/EXP/ETE_20Hours_1_encoder_finetunning_5Hours_1_100000/config.json

   where `-b` is the benchmark, `-a` is the path to the agent class, `-c` is the configuration file for driving (like the [config.json](https://github.com/yixiao1/Coiltraine-ICML/blob/master/_logs/EXP/ETE_20Hours_1_encoder_finetunning_5Hours_1_100000/config.json) in step 2))

   Note that you need to define `-c`, otherwise the agent can not access the information for loading driving models
