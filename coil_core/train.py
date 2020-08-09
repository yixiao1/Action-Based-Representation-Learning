import os
import sys
import random
import time
import traceback
import torch
import torch.optim as optim

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel, Loss, adjust_learning_rate_auto, EncoderModel
from input import CoILDataset, Augmenter, select_balancing_strategy
from logger import coil_logger
from coilutils.checkpoint_schedule import is_ready_to_save, get_latest_saved_checkpoint, \
                                    check_loss_validation_stopped
import numpy as np



def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


# The main function maybe we could call it with a default name
def execute(gpu, exp_batch, exp_alias, suppress_output=True, number_of_workers=12, encoder_params = None):
    """
        The main training function. This functions loads the latest checkpoint
        for a given, exp_batch (folder) and exp_alias (experiment configuration).
        With this checkpoint it starts from the beginning or continue some training.
    Args:
        gpu: The GPU number
        exp_batch: the folder with the experiments
        exp_alias: the alias, experiment name
        suppress_output: if the output are going to be saved on a file
        number_of_workers: the number of threads used for data loading

    Returns:
        None

    """
    try:
        # We set the visible cuda devices to select the GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        g_conf.VARIABLE_WEIGHT = {}
        # At this point the log file with the correct naming is created.
        # You merge the yaml file with the global configuration structure.
        merge_with_yaml(os.path.join('configs', exp_batch, exp_alias + '.yaml'), encoder_params)
        set_type_of_process('train')
        # Set the process into loading status.
        coil_logger.add_message('Loading', {'GPU': os.environ["CUDA_VISIBLE_DEVICES"]})

        seed_everything(seed=g_conf.MAGICAL_SEED)

        # Put the output to a separate file if it is the case

        #if suppress_output:
        #    if not os.path.exists('_output_logs'):
        #        os.mkdir('_output_logs')
        #    sys.stdout = open(os.path.join('_output_logs', exp_alias + '_' +
        #                      g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out"), "a",
        #                      buffering=1)
        #    sys.stderr = open(os.path.join('_output_logs',
        #                      exp_alias + '_err_'+g_conf.PROCESS_NAME + '_'
        #                                   + str(os.getpid()) + ".out"),
        #                      "a", buffering=1)

        if coil_logger.check_finish('train'):
            coil_logger.add_message('Finished', {})
            return

        # Preload option
        print( " GOING TO LOAD")
        if g_conf.PRELOAD_MODEL_ALIAS is not None:
            print ( " LOADING A PRELOAD")
            checkpoint = torch.load(os.path.join('_logs', g_conf.PRELOAD_MODEL_BATCH,
                                                  g_conf.PRELOAD_MODEL_ALIAS,
                                                 'checkpoints',
                                                 str(g_conf.PRELOAD_MODEL_CHECKPOINT)+'.pth'))

        else:

            # Get the latest checkpoint to be loaded
            # returns none if there are no checkpoints saved for this model
            checkpoint_file = get_latest_saved_checkpoint()
            if checkpoint_file is not None:
                print('loading previous checkpoint ', checkpoint_file)
                checkpoint = torch.load(os.path.join('_logs', g_conf.EXPERIMENT_BATCH_NAME, g_conf.EXPERIMENT_NAME,
                                        'checkpoints', str(get_latest_saved_checkpoint())))
                iteration = checkpoint['iteration']
                best_loss = checkpoint['best_loss']
                best_loss_iter = checkpoint['best_loss_iter']
            else:
                iteration = 0
                best_loss = 100000000.0
                best_loss_iter = 0


        # Define the dataset. This structure is has the __get_item__ redefined in a way
        # that you can access the positions from the root directory as a in a vector.
        #full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], g_conf.TRAIN_DATASET_NAME)

        # By instantiating the augmenter we get a callable that augment images and transform them
        # into tensors.
        augmenter = Augmenter(g_conf.AUGMENTATION)

        # We can save preload dataset depends on the json file name, then no need to load dataset for each time with the same dataset
        if len(g_conf.EXPERIENCE_FILE) == 1:
            json_file_name = str(g_conf.EXPERIENCE_FILE[0]).split('/')[-1].split('.')[-2]
        else:
            json_file_name = str(g_conf.EXPERIENCE_FILE[0]).split('/')[-1].split('.')[-2] + '_' + str(g_conf.EXPERIENCE_FILE[1]).split('/')[-1].split('.')[-2]
        dataset = CoILDataset(transform=augmenter,
                              preload_name=g_conf.PROCESS_NAME + '_' + json_file_name + '_' + g_conf.DATA_USED,
                              clip_big_values=False)

        #dataset = CoILDataset(transform=augmenter, preload_name=str(g_conf.NUMBER_OF_HOURS)+ 'hours_' + g_conf.TRAIN_DATASET_NAME)
        print ("Loaded Training dataset")

        data_loader = select_balancing_strategy(dataset, iteration, number_of_workers)
        if g_conf.MODEL_TYPE in ['separate-affordances']:
            model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION, g_conf.ENCODER_MODEL_CONFIGURATION)

        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=g_conf.LEARNING_RATE)

        print(model)

        # we use the pre-trained encoder model to extract bottleneck Z and train the E-t-E model

        if g_conf.MODEL_TYPE in ['separate-affordances']:
            encoder_model = EncoderModel(g_conf.ENCODER_MODEL_TYPE, g_conf.ENCODER_MODEL_CONFIGURATION)
            encoder_model.cuda()
            encoder_model.eval()
            # To freeze the pre-trained encoder model
            if g_conf.FREEZE_ENCODER:
                for param_ in encoder_model.parameters():
                    param_.requires_grad = False
            if encoder_params is not None:
                encoder_checkpoint = torch.load(
                    os.path.join('_logs', encoder_params['encoder_folder'], encoder_params['encoder_exp'], 'checkpoints',
                                 str(encoder_params['encoder_checkpoint']) + '.pth'))
                print("Encoder model ", str(encoder_params['encoder_checkpoint']), "loaded from ",
                      os.path.join('_logs', encoder_params['encoder_folder'], encoder_params['encoder_exp'], 'checkpoints'))
                encoder_model.load_state_dict(encoder_checkpoint['state_dict'])
                if g_conf.FREEZE_ENCODER:
                    encoder_model.eval()
                    # To freeze the pre-trained encoder model
                    for param_ in encoder_model.parameters():
                        param_.requires_grad = False
                else:
                    optimizer = optim.Adam(list(model.parameters()) + list(encoder_model.parameters()),
                                           lr=g_conf.LEARNING_RATE)

            for name_encoder, param_encoder in encoder_model.named_parameters():
                if param_encoder.requires_grad:
                    print('  Unfrozen layers', name_encoder)
                else:
                    print('  Frozen layers', name_encoder)


        if checkpoint_file is not None or g_conf.PRELOAD_MODEL_ALIAS is not None:
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            accumulated_time = checkpoint['total_time']
            loss_window = coil_logger.recover_loss_window('train', iteration)
        else:  # We accumulate iteration time and keep the average speed
            accumulated_time = 0
            loss_window = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                print('  Unfrozen layers', name)
            else:
                print('  Frozen layers', name)

        print ("Before the loss")

        # Loss time series window
        for data in data_loader:

            # Basically in this mode of execution, we validate every X Steps, if it goes up 3 times,
            # add a stop on the _logs folder that is going to be read by this process
            if g_conf.FINISH_ON_VALIDATION_STALE is not None and \
                    check_loss_validation_stopped(iteration, g_conf.FINISH_ON_VALIDATION_STALE):
                break
                
            """
                ####################################
                    Main optimization loop
                ####################################
            """

            if iteration % 1000 == 0:
                adjust_learning_rate_auto(optimizer, loss_window)

            model.zero_grad()
            if not g_conf.FREEZE_ENCODER:
                encoder_model.zero_grad()

            if g_conf.LABELS_SUPERVISED:
                inputs_data = torch.cat((data['rgb'],
                                         torch.zeros(g_conf.BATCH_SIZE, 1, 88, 200)), dim=1).cuda()
            else:
                inputs_data = torch.squeeze(data['rgb'].cuda())


            if g_conf.MODEL_TYPE in ['separate-affordances']:
                #TODO: for this two encoder models training, we haven't put speed as input to train yet


                if g_conf.ENCODER_MODEL_TYPE in ['ETE_inverse_model',
                                                 'action_prediction',
                                                 'stdim', 'forward', 'FIMBC',
                                                 'ETEDIM', 'one-step-affordances']:

                    e, inter = encoder_model.forward_encoder(inputs_data,
                                           dataset.extract_inputs(data).cuda(),
                                           # We also add measurements and commands
                                                             torch.squeeze(
                                                                 dataset.extract_commands(
                                                                     data).cuda()))


                elif g_conf.ENCODER_MODEL_TYPE in ['ETE', 'ETE_action_prediction', 'ETE_stdim']:
                    e, inter = encoder_model.forward_encoder(inputs_data,
                                                      dataset.extract_inputs(data).cuda(),
                                                      torch.squeeze(
                                                          dataset.extract_commands(data).cuda()))

                loss_function_params = {
                    'classification_gt': dataset.extract_affordances_targets(data, 'classification').cuda(),
                # harzard stop, red_light....
                    'class_weights': g_conf.AFFORDANCES_CLASS_WEIGHT,
                    'regression_gt': dataset.extract_affordances_targets(data, 'regression').cuda(),
                    'variable_weights': g_conf.AFFORDANCES_VARIABLE_WEIGHT
                }
                loss = model(e, loss_function_params)
                loss.backward()
                optimizer.step()

            else:
                raise RuntimeError(
                    'Not implement yet, this branch is only work for g_conf.MODEL_TYPE in [separate-affordances]')

            """
                ####################################
                    Saving the model if necessary
                ####################################
            """

            if is_ready_to_save(iteration):

                state = {
                    'iteration': iteration,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'total_time': accumulated_time,
                    'optimizer': optimizer.state_dict(),
                    'best_loss_iter': best_loss_iter
                }
                torch.save(state, os.path.join('_logs', g_conf.EXPERIMENT_BATCH_NAME, g_conf.EXPERIMENT_NAME
                                               , 'checkpoints', str(iteration) + '.pth'))

                if not g_conf.FREEZE_ENCODER:
                    encoder_state = {
                        'iteration': iteration,
                        'state_dict': encoder_model.state_dict(),
                        'best_loss': best_loss,
                        'total_time': accumulated_time,
                        'optimizer': optimizer.state_dict(),
                        'best_loss_iter': best_loss_iter
                    }
                    torch.save(encoder_state, os.path.join('_logs', g_conf.EXPERIMENT_BATCH_NAME, g_conf.EXPERIMENT_NAME
                                                   , 'checkpoints', str(iteration) + '_encoder.pth'))

            iteration += 1

            """
                ################################################
                    Adding tensorboard logs.
                    Making calculations for logging purposes.
                    These logs are monitored by the printer module.
                #################################################
            """
            coil_logger.add_scalar('Loss', loss.data, iteration)
            coil_logger.add_image('Image', torch.squeeze(data['rgb']), iteration)


            if loss.data < best_loss:
                best_loss = loss.data.tolist()
                best_loss_iter = iteration


            if iteration % 100 == 0:
                print('Train Iteration: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}'.format(
                    iteration, iteration, g_conf.NUMBER_ITERATIONS,
                    100. * iteration / g_conf.NUMBER_ITERATIONS, loss.data))

        coil_logger.add_message('Finished', {})

    except KeyboardInterrupt:
        coil_logger.add_message('Error', {'Message': 'Killed By User'})

    except RuntimeError as e:

        coil_logger.add_message('Error', {'Message': str(e)})

    except:
        traceback.print_exc()
        coil_logger.add_message('Error', {'Message': 'Something Happened'})
