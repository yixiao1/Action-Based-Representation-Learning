import os
import sys
import time
import random
import numpy as np
import traceback
import torch
import torch.optim as optim

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import Loss, adjust_learning_rate_auto, EncoderModel
from input import CoILDataset, Augmenter, select_balancing_strategy
from logger import coil_logger
from coilutils.checkpoint_schedule import is_ready_to_save, get_latest_saved_checkpoint



def seed_everything(seed=0):
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

# The main function maybe we could call it with a default name
def execute(gpu, exp_batch, exp_alias, suppress_output=True, number_of_workers=12):
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
        merge_with_yaml(os.path.join('configs', exp_batch, exp_alias + '.yaml'))
        set_type_of_process('train_encoder')
        # Set the process into loading status.
        coil_logger.add_message('Loading', {'GPU': os.environ["CUDA_VISIBLE_DEVICES"]})

        seed_everything(seed=g_conf.MAGICAL_SEED)

        # Put the output to a separate file if it is the case

        if suppress_output:
            if not os.path.exists('_output_logs'):
                os.mkdir('_output_logs')
            sys.stdout = open(os.path.join('_output_logs', exp_alias + '_' +
                                           g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out"), "a",
                              buffering=1)
            sys.stderr = open(os.path.join('_output_logs',
                                           exp_alias + '_err_' + g_conf.PROCESS_NAME + '_'
                                           + str(os.getpid()) + ".out"),
                              "a", buffering=1)

        if coil_logger.check_finish('train'):
            coil_logger.add_message('Finished', {})
            return

        # Preload option
        if g_conf.PRELOAD_MODEL_ALIAS is not None:
            checkpoint = torch.load(os.path.join('_logs', g_conf.PRELOAD_MODEL_BATCH,
                                                 g_conf.PRELOAD_MODEL_ALIAS,
                                                 'checkpoints',
                                                 str(g_conf.PRELOAD_MODEL_CHECKPOINT) + '.pth'))

        # Get the latest checkpoint to be loaded
        # returns none if there are no checkpoints saved for this model
        checkpoint_file = get_latest_saved_checkpoint()
        if checkpoint_file is not None:
            checkpoint = torch.load(os.path.join('_logs', exp_batch, exp_alias,
                                                 'checkpoints', str(get_latest_saved_checkpoint())))
            iteration = checkpoint['iteration']
            best_loss = checkpoint['best_loss']
            best_loss_iter = checkpoint['best_loss_iter']
        else:
            iteration = 0
            best_loss = 1000000000.0
            best_loss_iter = 0

        # Define the dataset. This structure is has the __get_item__ redefined in a way
        # that you can access the positions from the root directory as a in a vector.
        # full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], g_conf.TRAIN_DATASET_NAME)

        # By instantiating the augmenter we get a callable that augment images and transform them
        # into tensors.
        augmenter = Augmenter(g_conf.AUGMENTATION)

        if len(g_conf.EXPERIENCE_FILE) == 1:
            json_file_name = str(g_conf.EXPERIENCE_FILE[0]).split('/')[-1].split('.')[-2]
        else:
            json_file_name = str(g_conf.EXPERIENCE_FILE[0]).split('/')[-1].split('.')[-2] + '_' + str(g_conf.EXPERIENCE_FILE[1]).split('/')[-1].split('.')[-2]
        dataset = CoILDataset(transform=augmenter,
                              preload_name=g_conf.PROCESS_NAME + '_' + json_file_name + '_' + g_conf.DATA_USED,
                              clip_big_values=False)

        print ("Loaded dataset")

        data_loader = select_balancing_strategy(dataset, iteration, number_of_workers)

        encoder_model = EncoderModel(g_conf.ENCODER_MODEL_TYPE, g_conf.ENCODER_MODEL_CONFIGURATION)
        encoder_model.cuda()
        encoder_model.train()

        print(encoder_model)

        optimizer = optim.Adam(encoder_model.parameters(), lr=g_conf.LEARNING_RATE)

        if checkpoint_file is not None or g_conf.PRELOAD_MODEL_ALIAS is not None:
            encoder_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            accumulated_time = checkpoint['total_time']
            loss_window = coil_logger.recover_loss_window('train', iteration)
        else:  # We accumulate iteration time and keep the average speed
            accumulated_time = 0
            loss_window = []

        print ("Before the loss")

        if g_conf.ENCODER_MODEL_TYPE in ['ETE', 'ETE_action_prediction']:
            criterion = Loss(g_conf.LOSS_FUNCTION)

        # Loss time series window
        for data in data_loader:
            if iteration % 1000 == 0:
                adjust_learning_rate_auto(optimizer, loss_window)

            capture_time = time.time()
            encoder_model.zero_grad()

            """
                ####################################
                    ENCODER_MODEL_TYPE can be: one-step-affordances, ETE, stdim, action_prediction
                    
                ####################################
              - one-step-affordances: input RGB images, compute affordances loss.
              - ETE: input RGB images and speed, compute action loss (steering, throttle, brake)
              - stdim: input two consecutive RGB images, compute the feature loss
              - action_prediction: input two consecutive RGB images, compute action classification loss
              - forward_infocen: Forward model with action and then perform the infonce.
              - ETE_action_prediction: input two consecutive RGB images, compute action classification loss + control loss
              - ETE_stdim: input two consecutive RGB images, compute features loss + control loss
              
            """

            if g_conf.ENCODER_MODEL_TYPE in ['one-step-affordances']:
                loss_function_params = {
                    'classification_gt': dataset.extract_affordances_targets(data, 'classification').cuda(),
                # harzard stop, red_light....
                    'class_weights': g_conf.AFFORDANCES_CLASS_WEIGHT,
                    'regression_gt': dataset.extract_affordances_targets(data, 'regression').cuda(),
                    'variable_weights': g_conf.AFFORDANCES_VARIABLE_WEIGHT
                }
                # we input RGB images, speed and command to train affordances
                loss = encoder_model(torch.squeeze(data['rgb'].cuda()),
                                     dataset.extract_inputs(data).cuda(),
                                     torch.squeeze(dataset.extract_commands(data).cuda()),
                                     loss_function_params)

                if iteration == 0:
                    state = {
                        'iteration': iteration,
                        'state_dict': encoder_model.state_dict(),
                        'best_loss': best_loss,
                        'total_time': accumulated_time,
                        'optimizer': optimizer.state_dict(),
                        'best_loss_iter': best_loss_iter
                    }
                    torch.save(state, os.path.join('_logs', exp_batch, exp_alias
                                                   , 'checkpoints', 'inital.pth'))

                loss.backward()
                optimizer.step()

            elif g_conf.ENCODER_MODEL_TYPE in ['ETE_inverse_model',
                                               'forward', 'FIMBC']:
                # We sample another batch to avoid the superposition

                inputs_data = [data['rgb'][0].cuda(), data['rgb'][1].cuda()]
                loss, loss_other, loss_ete = encoder_model(inputs_data,
                                           dataset.extract_inputs(data),
                                           # We also add measurements and commands
                                           dataset.extract_commands(data),
                                           dataset.extract_targets(data)[0].cuda()
                                           )
                loss.backward()
                optimizer.step()


            elif g_conf.ENCODER_MODEL_TYPE in ['ETE']:
                branches = encoder_model(torch.squeeze(data['rgb'].cuda()),
                                         dataset.extract_inputs(data).cuda(),
                                         torch.squeeze(dataset.extract_commands(data).cuda()))

                loss_function_params = {
                    'branches': branches,
                    'targets': dataset.extract_targets(data).cuda(),  # steer, throttle, brake
                    'inputs': dataset.extract_inputs(data).cuda(),  # speed
                    'branch_weights': g_conf.BRANCH_LOSS_WEIGHT,
                    'variable_weights': g_conf.VARIABLE_WEIGHT
                }

                loss, _ = criterion(loss_function_params)
                loss.backward()
                optimizer.step()

            elif g_conf.ENCODER_MODEL_TYPE in ['stdim']:
                inputs_data = [data['rgb'][0].cuda(), data['rgb'][1].cuda()]
                loss, _, _ = encoder_model(inputs_data,
                                           dataset.extract_inputs(data),
                                           # We also add measurements and commands
                                           dataset.extract_commands(data)
                                           )
                loss.backward()
                optimizer.step()

            elif g_conf.ENCODER_MODEL_TYPE in ['action_prediction']:
                inputs_data = [data['rgb'][0].cuda(), data['rgb'][1].cuda()]
                loss, _, _ = encoder_model(inputs_data,
                                           dataset.extract_inputs(data),
                                           # We also add measurements and commands
                                           dataset.extract_commands(data),
                                           dataset.extract_targets(data)[0].cuda()
                                           )
                loss.backward()
                optimizer.step()

            elif g_conf.ENCODER_MODEL_TYPE in ['forward_infonce',
                                               'ETEDIM']:
                inputs_data = [data['rgb'][0].cuda(), data['rgb'][1].cuda()]
                targets = dataset.extract_targets(data)[0].cuda()
                loss, loss_other, loss_ete = encoder_model(inputs_data,
                                           dataset.extract_inputs(data),
                                           # We also add measurements and commands
                                           dataset.extract_commands(data), # The continuous one
                                                           targets
                                           )
                loss.backward()
                optimizer.step()


            else:
                raise ValueError("The encoder model type is not know")

            """
                ####################################
                    Saving the model if necessary
                ####################################
            """

            if is_ready_to_save(iteration):
                state = {
                    'iteration': iteration,
                    'state_dict': encoder_model.state_dict(),
                    'best_loss': best_loss,
                    'total_time': accumulated_time,
                    'optimizer': optimizer.state_dict(),
                    'best_loss_iter': best_loss_iter
                }
                torch.save(state, os.path.join('_logs', exp_batch, exp_alias
                                               , 'checkpoints', str(iteration) + '.pth'))

            iteration += 1

            """
                ################################################
                    Adding tensorboard logs.
                    Making calculations for logging purposes.
                    These logs are monitored by the printer module.
                #################################################
            """

            if g_conf.ENCODER_MODEL_TYPE in ['stdim', 'action_prediction', 'ETE_inverse_model',
                                             'ETEDIM', 'forward_infonce', 'forward']:
                coil_logger.add_scalar('Loss', loss.data, iteration)
                coil_logger.add_image('f_t', torch.squeeze(data['rgb'][0]), iteration)
                coil_logger.add_image('f_ti', torch.squeeze(data['rgb'][1]), iteration)

            #elif g_conf.ENCODER_MODEL_TYPE in ['ETE_inverse_model', 'ETEDIM']:
            #    coil_logger.add_scalar('Loss Other', loss_other.data, iteration)
            #    coil_logger.add_scalar('Loss ETE', loss_ete.data, iteration)

            elif g_conf.ENCODER_MODEL_TYPE in ['one-step-affordances', 'ETE']:
                coil_logger.add_scalar('Loss', loss.data, iteration)
                coil_logger.add_image('Image', torch.squeeze(data['rgb']), iteration)

            if loss.data < best_loss:
                best_loss = loss.data.tolist()
                best_loss_iter = iteration

            accumulated_time += time.time() - capture_time
            coil_logger.add_message('Iterating',
                                    {'Iteration': iteration,
                                     'Loss': loss.data.tolist(),
                                     'Images/s': (iteration * g_conf.BATCH_SIZE) / accumulated_time,
                                     'BestLoss': best_loss, 'BestLossIteration': best_loss_iter},
                                    iteration)
            loss_window.append(loss.data.tolist())
            coil_logger.write_on_error_csv('train', loss.data)

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