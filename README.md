# Action-Based Representation Learning for Autonomous Driving

-------------------------------------------------------------
This repository is for running the experiments of paper: Action-Based Representation Learning for Autonomous Driving

 <img src="driving_clip.gif" height="350">
  
### Publications
We kindly ask to cite our paper if you find this work useful:
 * Yi Xiao, Felipe Codevilla, Christopher Pal, Antonio M. Lopez, [Action-Based Representation Learning for Autonomous Driving](https://arxiv.org/abs/2008.09417).
             
         @inproceedings{Xiao2020ActionBasedRL,
         title={Action-Based Representation Learning for Autonomous Driving},
         author={Yi Xiao and Felipe Codevilla and Christopher Pal and Antonio M. Lopez},
         year={2020}
         }

### Video
Please check our online [video](https://www.youtube.com/watch?v=fFywCMlLbyE)
 
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

1. Download the dataset. The full dataset is not yet ready for publishing, here we provide a [small dataset](https://drive.google.com/file/d/1nGHApXVF8oGRLH9sZ_C1gdqcGkhTvEnb/view?usp=sharing) for simple test. Note that in the following steps, we will use this small dataset for both training and validation, just to illustrate how to run our framework. Once the full dataset is ready, we will provide different datasets for training and validation.

2. Define the path to your dataset folder with SRL_DATASET_PATH:

        export SRL_DATASET_PATH = <Path to where your datasets are>

3. Download the repository

        git clone https://github.com/yixiao1/Action-Based-Representation-Learning.git
        
4. Go to your downloaded repository, and define ACTIONDIR with this directory
        
        cd ~/Action-Based-Representation-Learning

        export ACTIONDIR=$(pwd)

5. Download the CARLA version we used with this [link](https://drive.google.com/file/d/1m4J2yJqL7QcCfaxvMh8erLzdGEyFC5mg/view?usp=sharing), and put it inside your downloaded repository folder

6. To add the following packages to your PYTHONPATH:

    - Path to carla
    - Path to carla .egg file
    - Path to scenario_runner
    - Path to carl

    you need to run:

        export PYTHONPATH=$ACTIONDIR/Carla96ped4/PythonAPI/carla:$ACTIONDIR/Carla96ped4/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg:$ACTIONDIR/scenario_runner:$ACTIONDIR/carl

-------------------------------------------------------------
### Training Encoder

1. Define configuration files for training. Refer to [files](https://github.com/yixiao1/Action-Based-Representation-Learning/tree/master/configs/ENCODER) in configs folder

2. Run the main.py file with "train_encoder" process:

        python3 main.py --single-process train_encoder --gpus 0 --encoder-folder ENCODER --encoder-exp BC_smallDataset_seed1

    where `--single-process` defines the process type, `--gpus` defines the gpu to be used, `--encoder-folder` is the experiment folder you defined in [config folder](https://github.com/yixiao1/Action-Based-Representation-Learning/tree/master/configs), and `--encoder-exp` is the experiment you defined inside the experiment folder.

-------------------------------------------------------------
### Training MLP for affordances

1. Define configuration files for training. Refer to [files](https://github.com/yixiao1/Action-Based-Representation-Learning/tree/master/configs/EXP) in configs folder

2. Run the main.py file with "train" process:

        python3 main.py --single-process train --gpus 0 --encoder-folder ENCODER --encoder-exp BC_smallDataset_seed1 --encoder-checkpoint 1000 -f EXP -e BC_smallDataset_seed1_encoder_frozen_1FC_smallDataset_s1

   where `--single-process` defines the process type, `--gpus` defined the gpu to be used, `--encoder-folder` is the experiment folder name of the encoder to be used, `--encoder-exp` is the experiment name of encoder to be used, `--encoder-checkpoint` is the specific encoder checkpoint to be used, `-f` is the experiment folder you defined in [config folder](https://github.com/yixiao1/Action-Based-Representation-Learning/tree/master/configs) for affordances prediction, `-e` is the experiment name you defined in the experiment folder for affordances prediction.

-------------------------------------------------------------
### Validate on affordances prediction

1. Run the main.py file with "validation" process. You will need to define the path to the json file of validation dataset:

        python3 main.py --single-process validation --gpus 0 --encoder-folder ENCODER --encoder-exp BC_smallDataset_seed1 --encoder-checkpoint 1000 -f EXP -e BC_smallDataset_seed1_encoder_frozen_1FC_smallDataset_s1 -vj $ACTIONDIR/carl/database/CoRL2020/small_dataset.json

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

   check an example on this [config.json](https://github.com/yixiao1/Action-Based-Representation-Learning/blob/master/_logs/EXP/BC_im_50Hours_seed1_encoder_finetuning_3FC_5Hours_s1_100000/config.json) example

5. To run the benchmark, go to [driving-benchmarks](https://github.com/yixiao1/Action-Based-Representation-Learning/tree/master/driving-benchmarks-carla_09_cexp) folder:

        cd driving-benchmarks-carla_09_cexp/

and run:

        python3 benchmark_runner.py -b NoCrash -a $ACTIONDIR/drive/AffordancesAgent.py -d carlaped -c $ACTIONDIR/_logs/EXP/BC_im_50Hours_seed1_encoder_finetuning_3FC_5Hours_s1_100000/config.json --gpu 2

    where `-b` is the benchmark, `-a` is the path to the agent class, `-c` is the configuration file for driving

6. To drive our affordance-based model with 50 hours Behaviour Cloning (BC) pre-training, you need to download this [_logs](https://drive.google.com/file/d/14N6B6Q_zhCnXZy1sne-HFjaktjNjjTjF/view?usp=sharing) to the same directory of this repository), and run step 5
