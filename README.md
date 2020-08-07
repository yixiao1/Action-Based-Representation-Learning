# Action-Based Representation Learning for Autonomous Driving

-------------------------------------------------------------
This repository is for running the experiments of ICML project

Basically, the processes can be defined as four steps:

 * Train an encoder model with specific losses, whose Conv. layers part will be later fine-tunned for prediciting affordances
 * Train FC layers for affordances outputs. Pre-trained encoder models will be fine-tunned
 * Validation on affordances prediction
 * Actual drive using affordances prediction


-------------------------------------------------------------
### Data Collection (Affordances)

YOU CAN SKIP THIS PART IF YOU ALREADY HAD DATASETS

1. Download [this repository](https://github.com/felipecode/cexp/tree/cexp_ICML) to get cexp folder

   Notes: The branch should be "cexp_ICML"

2. Get into the cexp folder: cd ~./cexp

3. The first thing you need to do is define the datasets folder.
This is the folder that will contain your training and validation datasets

   Example:

        export SRL_DATASET_PATH = /home/yixiao/Datasets/ICML


4. You need to add some paths to PYTHONPATH

	 * Path to carla
	 * Path to scenario_runner
	 * Path to carla .egg file
	 * Path to cexp

   Example:

        export PYTHONPATH=/home/yixiao/Carla96b/PythonAPI/carla/:/home/yixiao/scenario_runner:/home/yixiao/Carla96ped4/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg:/datatmp/Experiments/yixiao/carl

5. To get a dataset, you need to define a json file that concludes configurations of those episodes you want to collect, and put it into database folder

   Refer to [This json file](https://github.com/felipecode/cexp/blob/cexp_ICML/database/dataset_dynamic_l0.json)

6. Run multi_gpu_data_collection.py

   Example:

        python3  multi_gpu_data_collection.py  -j  /datatmp/Experiments/yixiao/carl/database/dataset_dynamic_l0.json  -ct  carlaped  -n 1  -ge 0

   where `-j` is the FULL path of json file that you want to collect, `-ct` is the docker name, `-n` defines the number of dockers you want to use, `-ge` is the GPUs you DON'T want to use


-------------------------------------------------------------
### Train Encoder

1. The first thing you need to do is to define the datasets folder and add some path to PYTHONPATH, the same as Section Data Collection (Affordances) (Step 3 & 4)

2. You need to define configuration files for training. Refer to files in [configs folder](https://github.com/yixiao1/Coiltraine-ICML/tree/master/configs/ENCODER)

3. Run coiltraine.py

   Example:

       python3 coiltraine.py --single-process train_encoder --gpus 0 -f ENCODER -e one_step_aff_15mins_2

   where `--single-process` means to do a training for encoder, `--gpus` is the gpu you want to use, `-f` is the exp folder, `-e` is the exp file

-------------------------------------------------------------
### Train affordances

1. The first thing you need to do is to define the datasets folder and add some path to PYTHONPATH, the same as Section Data Collection (Affordances) (Step 3 & 4)

2. You need to define configuration files for training. Refer to files in [configs folder](https://github.com/yixiao1/Coiltraine-ICML/tree/master/configs/EXP)

3. Run coiltraine.py

   Example:

       python3 coiltraine.py --single-process train --gpus 0 -f EXP -e ETE_5Hours_1_encoder_finetunning_1Hours_1 --encoder-folder ENCODER --encoder-exp ETE_5Hours_1 --encoder-checkpoint 100000


   where `--single-process` means to do a training for affordances, `--gpus` is the gpu you want to use, `-f` is the exp folder, `-e` is the exp file,
   `--encoder-folder` is the pre-trained encoder folder, `--encoder-exp` is the pre-trained encoder exp file, and `--encoder-checkpoint` is the pre-trained encoder checkpoint you want to use

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
