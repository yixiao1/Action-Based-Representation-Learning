# Action-Based Representation Learning for Autonomous Driving

-------------------------------------------------------------
This repository is for running the experiments of paper: Action-Based Representation Learning for Autonomous Driving

 <img src="driving_clip.gif" height="350">
  
### Publications
We kindly ask to cite our paper if you find this work useful:
 * Yi Xiao, Felipe Codevilla, Christopher Pal, Antonio M. Lopez, [Action-Based Representation Learning for Autonomous Driving](). ArXiv:

### Video
Please check our online [video](https://drive.google.com/file/d/1kbXF3UtQk70ncDmsI5YQ73QVrENtKU1o/view?usp=sharing)
 
### Related Publications:
Our work is built using the following frameworks:
 * [Coiltraine](https://github.com/felipecode/coiltraine), which can be used to easily train and manage the trainings of imitation learning networks jointly with evaluations on the CARLA simulator. 
 * [Cexp](https://github.com/felipecode/cexp), which is a interface to the CARLA simulator and the scenario runner to produce fully usable environments.

-------------------------------------------------------------
### Experiments Summary

The processes can be defined as four types:

 * Train an encoder model (Behaviour Cloning (BC), Inverse, Forward, ST-DIM)
 * Train a MLP for affordances outputs. The pre-trained encoder model will be used.
 * Validation on affordances prediction.
 * Actual drive using controller tuned with affordances prediction.

-------------------------------------------------------------
### Setting Environments & Getting Datasets

1. Download the dataset with this [link]().

2. Define the path to your dataset folder with SRL_DATASET_PATH:

        export SRL_DATASET_PATH = <Path to where your datasets are>

3. Download the repository

        git clone https://github.com/yixiao1/Action-Based-Representation-Learning.git
        
4. Go to your downloaded repository, and define ACTIONDIR with this directory
        
        cd ~/Action-Based-Representation-Learning

        export ACTIONDIR=$(pwd)

5. Download the CARLA version we used with this [link](https://drive.google.com/file/d/1m4J2yJqL7QcCfaxvMh8erLzdGEyFC5mg/view?usp=sharing), and put it inside your downloaded repository folder

6. Add the following packages to your PYTHONPATH:

    - Path to carla
    - Path to carla .egg fil
    - Path to scenario_runner
    - Path to carl

    Run:

        export PYTHONPATH=$ACTIONDIR/Carla96ped4/PythonAPI/carla:$ACTIONDIR/Carla96ped4/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg:$ACTIONDIR/scenario_runner:$ACTIONDIR/carl

-------------------------------------------------------------
### Training Encoder

1. Define configuration files for training. Refer to files in [configs folder](https://github.com/yixiao1/Action-Based-Representation-Learning/tree/master/configs/ENCODER)

2. Run the main.py file with "train_encoder" process:

        python3 main.py --single-process train_encoder --gpus 0 --encoder-folder ENCODER --encoder-exp BC_im_50Hours_seed1

    where `--single-process` defines the process type, `--gpus` defines the gpu to be used, `--encoder-folder` is the experiment folder name you defined in `config` folder, `--encoder-exp` is the experiment name you defined in `config` folder

-------------------------------------------------------------
### Training MLP for affordances

1. Define configuration files for training. Refer to files in [configs folder](https://github.com/yixiao1/Action-Based-Representation-Learning/tree/master/configs/EXP)

2. Run the main.py file with "train" process:

        python3 main.py --single-process train --gpus 0 --encoder-folder ENCODER --encoder-exp BC_im_50Hours_seed1 --encoder-checkpoint 100000 -f EXP -e BC_im_50Hours_seed1_encoder_frozen_1FC_30mins

   where `--single-process` defines the process type, `--gpus` defined the gpu to be used, `--encoder-folder` is the experiment folder name of the encoder to be used, `--encoder-exp` is the experiment name of encoder to be used, `--encoder-checkpoint` is the specific encoder checkpoint to be used, `-f` is is the experiment folder name you defined in `config` folder for affordances prediction, `-e` is the experiment name you defined in `config` folder for affordances prediction

-------------------------------------------------------------
### Validate on affordances prediction

1. Run the main.py file with "validation" process. You will need to define the path to the json file of validation dataset:

        python3 main.py --single-process validation --gpus 0 --encoder-folder ENCODER --encoder-exp BC_im_50Hours_seed1 --encoder-checkpoint 100000 -f EXP -e BC_im_50Hours_seed1_encoder_frozen_1FC_30mins -vj $ACTIONDIR/carl/database/CoRL2020/dataset_dynamic_Town01_1Hour_valid.json

    where `-vj` defines the path to your validation json file

-------------------------------------------------------------
### Driving on CARLA benchmark

1. The driving results will be saved to your SRL_DATASET_PATH, you could re-define it if you want to save to another path

2. Build a docker with your carla version:

        docker image build -f /.../carla/Util/Docker/Release.Dockerfile -t carlaped $ACTIONDIR/Carla96ped4/

    where `-f` is the path to the [Realease.Dockerfile](https://github.com/carla-simulator/carla/blob/master/Util/Docker/Release.Dockerfile), `-t` defines the name of the docker you want to created, and $ACTIONDIR/Carla96ped4/ is the path to your Carla package

3. Set up PYTHONPATH for CARLA driving:

        export PYTHONPATH=$ACTIONDIR:$ACTIONDIR/cad:$ACTIONDIR/Carla96ped4/PythonAPI/carla:$ACTIONDIR/PythonAPI/carla:$ACTIONDIR/scenario_runner

4. Define a config.json for using a specific model, and put it into the directory of your model in _logs folder: _logs/(experiment folder)/(experiment name)

   check on this [config.json](https://github.com/yixiao1/Action-Based-Representation-Learning/blob/master/_logs/EXP/BC_im_50Hours_seed1_encoder_finetuning_3FC_5Hours_s1_100000/config.json) example

5. To run the benchmark, go to [driving-benchmarks](https://github.com/yixiao1/Action-Based-Representation-Learning/tree/master/driving-benchmarks-carla_09_cexp) folder:

        cd driving-benchmarks-carla_09_cexp/

and run:

        python3 benchmark_runner.py -b NoCrash -a $ACTIONDIR/drive/AffordancesAgent.py -d carlaped -c $ACTIONDIR/_logs/EXP/BC_im_50Hours_seed1_encoder_finetuning_3FC_5Hours_s1_100000/config.json --gpu 2

    where `-b` is the benchmark, `-a` is the path to the agent class, `-c` is the configuration file for driving

6. To drive our affordance-based model with 50 hours Behaviour Cloning (BC) pre-training, you need to download this [_logs](https://drive.google.com/file/d/14N6B6Q_zhCnXZy1sne-HFjaktjNjjTjF/view?usp=sharing) to the same directory of this repository), and run step 5
