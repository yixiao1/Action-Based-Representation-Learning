import os
import sys
import torch
import traceback
from torchvision.utils import save_image
from torch.nn import functional as F
import math
import numpy as np
# What do we define as a parameter what not.
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.misc import imresize
from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel, EncoderModel
from input import CoILDataset, Augmenter
from logger import coil_logger
from coilutils.checkpoint_schedule import maximun_checkpoint_reach, get_next_checkpoint, \
    get_next_checkpoint_2, get_latest_evaluated_checkpoint_2



def write_attentions(images, all_layers, iteration, folder_name, layers=None):

    # Plot the attentions that are computed by the network directly here.
    # maybe take the intermediate layers and compute the attentions  ??
    if layers is None:
        layers = [0, 1, 2]


    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    # We save the images directly as a source of comparison
    if not os.path.exists(os.path.join(folder_name, 'images')):
        os.mkdir(os.path.join(folder_name, 'images'))
    for i in range(images.shape[0]):
        save_image(images[i], os.path.join(folder_name, 'images', str(iteration) + '_' + str(i) + '.png' ))


    # We save now the attention maps for the layers
    cmap = plt.get_cmap('inferno')
    for layer in layers:

        if not os.path.exists(os.path.join(folder_name, 'layer' + str(layer))):
            os.mkdir(os.path.join(folder_name, 'layer' + str(layer)))

        y = all_layers[layer]      #shape [120, 64, 22, 50]
        #for i in range(y.shape[0]):

        atts = torch.abs(y).mean(1).data.cpu().numpy()      #shape [120, 22, 50]

        for j in range(atts.shape[0]):
            att = atts[j]                           #shape [22, 50]
            att = att / att.max()                   #shape [22, 50]
            att = scipy.misc.imresize(att, [352, 800])
            scipy.misc.imsave(os.path.join(folder_name, 'layer' + str(layer), str(iteration)+'_'+ str(j) + '.png' ), cmap(att))




def write_regular_output(iteration, output, gt):
    for i in range(len(output)):
        coil_logger.write_on_csv(iteration, [output[i], gt[i]])



# The main function maybe we could call it with a default name
def execute(gpu, exp_batch, exp_alias, json_file_path, suppress_output,
            encoder_params = None, plot_attentions=False):
    try:
        # We set the visible cuda devices
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

        if json_file_path is not None:
            json_file_name = json_file_path.split('/')[-1].split('.')[-2]
        else:
            raise RuntimeError("You need to define the validation json file path")

        # At this point the log file with the correct naming is created.
        merge_with_yaml(os.path.join('configs', exp_batch, exp_alias+'.yaml'), encoder_params)
        if plot_attentions:
            set_type_of_process('validation', json_file_name+'_plotAttention')
        else:
            set_type_of_process('validation', json_file_name)

        if not os.path.exists('_output_logs'):
            os.mkdir('_output_logs')

        if suppress_output:
            sys.stdout = open(os.path.join('_output_logs',
                                           exp_alias + '_' + g_conf.PROCESS_NAME + '_'
                                           + str(os.getpid()) + ".out"),
                              "a", buffering=1)
            sys.stderr = open(os.path.join('_output_logs',
                              exp_alias + '_err_' + g_conf.PROCESS_NAME + '_'
                                           + str(os.getpid()) + ".out"),
                              "a", buffering=1)

        # We create file for saving validation results
        summary_file = os.path.join('_logs', exp_batch, g_conf.EXPERIMENT_NAME, g_conf.PROCESS_NAME + '_csv',
                                    'valid_summary_1camera.csv')
        g_conf.immutable(False)
        g_conf.DATA_USED = 'central'
        g_conf.immutable(True)
        if not os.path.exists(summary_file):
            csv_outfile = open(summary_file, 'w')
            csv_outfile.write("%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n"
                              % ('step', 'accumulated_pedestrian_TP', 'accumulated_pedestrian_FP',
                                 'accumulated_pedestrian_FN', 'accumulated_pedestrian_TN',
                                 'accumulated_vehicle_stop_TP', 'accumulated_vehicle_stop_FP',
                                 'accumulated_vehicle_stop_FN', 'accumulated_vehicle_stop_TN',
                                 'accumulated_red_tl_TP', 'accumulated_red_tl_FP',
                                 'accumulated_red_tl_FN', 'accumulated_red_tl_TN',
                                 'MAE_relative_angle'))
            csv_outfile.close()

        latest = get_latest_evaluated_checkpoint_2(summary_file)

        # Define the dataset. This structure is has the __get_item__ redefined in a way
        # that you can access the HDFILES positions from the root directory as a in a vector.
        #full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], dataset_name)
        augmenter = Augmenter(None)
        # Definition of the dataset to be used. Preload name is just the validation data name
        dataset = CoILDataset(transform=augmenter,
                              preload_name = g_conf.PROCESS_NAME + '_' + g_conf.DATA_USED,
                              process_type='validation', vd_json_file_path = json_file_path)
        print ("Loaded Validation dataset")

        # Creates the sampler, this part is responsible for managing the keys. It divides
        # all keys depending on the measurements and produces a set of keys for each bach.

        # The data loader is the multi threaded module from pytorch that release a number of
        # workers to get all the data.
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=g_conf.BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=g_conf.NUMBER_OF_LOADING_WORKERS,
                                                  pin_memory=True)

        if g_conf.MODEL_TYPE in ['one-step-affordances']:
            # one step training, no need to retrain FC layers, we just get the output of encoder model as prediciton
            model = EncoderModel(g_conf.ENCODER_MODEL_TYPE, g_conf.ENCODER_MODEL_CONFIGURATION)
            model.cuda()
            #print(model)


        elif g_conf.MODEL_TYPE in ['separate-affordances']:
            model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION, g_conf.ENCODER_MODEL_CONFIGURATION)
            model.cuda()
            #print(model)

            encoder_model = EncoderModel(g_conf.ENCODER_MODEL_TYPE, g_conf.ENCODER_MODEL_CONFIGURATION)
            encoder_model.cuda()
            encoder_model.eval()

            # Here we load the pre-trained encoder (not fine-tunned)
            if g_conf.FREEZE_ENCODER:
                if encoder_params is not None:
                    encoder_checkpoint = torch.load(
                    os.path.join('_logs', encoder_params['encoder_folder'],
                                        encoder_params['encoder_exp'],
                                        'checkpoints',
                                        str(encoder_params['encoder_checkpoint']) + '.pth'))
                    print("Encoder model ", str(encoder_params['encoder_checkpoint']), "loaded from ",
                          os.path.join('_logs', encoder_params['encoder_folder'], encoder_params['encoder_exp'],
                                       'checkpoints'))
                    encoder_model.load_state_dict(encoder_checkpoint['state_dict'])
                    encoder_model.eval()
                for param_ in encoder_model.parameters():
                    param_.requires_grad = False

        while not maximun_checkpoint_reach(latest, g_conf.TEST_SCHEDULE):
            latest = get_next_checkpoint_2(g_conf.TEST_SCHEDULE, summary_file)
            if os.path.exists(os.path.join('_logs', exp_batch, g_conf.EXPERIMENT_NAME, 'checkpoints', str(latest) + '.pth')):
                checkpoint = torch.load(os.path.join('_logs', exp_batch, g_conf.EXPERIMENT_NAME, 'checkpoints', str(latest) + '.pth'))
                checkpoint_iteration = checkpoint['iteration']
                model.load_state_dict(checkpoint['state_dict'])
                print("Validation checkpoint ", checkpoint_iteration)
                model.eval()
                for param_ in model.parameters():
                    param_.requires_grad = False

                # Here we load the fine-tunned encoder
                if not g_conf.FREEZE_ENCODER and g_conf.MODEL_TYPE not in ['one-step-affordances']:
                    encoder_checkpoint = torch.load(os.path.join('_logs', exp_batch, g_conf.EXPERIMENT_NAME, 'checkpoints',
                                     str(latest) + '_encoder.pth'))
                    print("FINE TUNNED encoder model ", str(latest) + '_encoder.pth', "loaded from ",
                          os.path.join('_logs', exp_batch, g_conf.EXPERIMENT_NAME,
                                       'checkpoints'))
                    encoder_model.load_state_dict(encoder_checkpoint['state_dict'])
                    encoder_model.eval()
                    for param_ in encoder_model.parameters():
                        param_.requires_grad = False

                accumulated_mae_ra = 0
                accumulated_pedestrian_TP = 0
                accumulated_pedestrian_TN = 0
                accumulated_pedestrian_FN = 0
                accumulated_pedestrian_FP = 0

                accumulated_red_tl_TP = 0
                accumulated_red_tl_TN = 0
                accumulated_red_tl_FP = 0
                accumulated_red_tl_FN = 0

                accumulated_vehicle_stop_TP = 0
                accumulated_vehicle_stop_TN = 0
                accumulated_vehicle_stop_FP = 0
                accumulated_vehicle_stop_FN = 0

                iteration_on_checkpoint = 0

                for data in data_loader:
                    if g_conf.MODEL_TYPE in ['one-step-affordances']:
                        c_output, r_output, layers = model.forward_outputs(torch.squeeze(data['rgb'].cuda()),
                                                                          dataset.extract_inputs(data).cuda(),
                                                                          dataset.extract_commands(
                                                                              data).cuda())

                    elif g_conf.MODEL_TYPE in ['separate-affordances']:
                        if g_conf.ENCODER_MODEL_TYPE in ['action_prediction', 'stdim' ,'ETEDIM',
                                                         'FIMBC', 'one-step-affordances']:
                            e, layers = encoder_model.forward_encoder(torch.squeeze(data['rgb'].cuda()),
                                                                      dataset.extract_inputs(data).cuda(),
                                                                      torch.squeeze(
                                                                      dataset.extract_commands(
                                                                            data).cuda())
                                                              )
                            c_output, r_output = model.forward_test(e)

                        elif g_conf.ENCODER_MODEL_TYPE in ['ETE', 'ETE_inverse_model', 'forward',
                                                           'ETE_stdim']:
                            e, layers = encoder_model.forward_encoder(torch.squeeze(data['rgb'].cuda()),
                                                                   dataset.extract_inputs(data).cuda(),
                                                                   torch.squeeze(
                                                                   dataset.extract_commands(
                                                                      data).cuda())
                                                              )
                            c_output, r_output = model.forward_test(e)

                    if plot_attentions:
                        attentions_path = os.path.join('_logs', exp_batch, g_conf.EXPERIMENT_NAME,
                                                       g_conf.PROCESS_NAME + '_attentions_'+
                                                       str(latest))

                        write_attentions(torch.squeeze(data['rgb']), layers, iteration_on_checkpoint,
                                         attentions_path)

                    # Accurancy = (TP+TN)/(TP+TN+FP+FN)
                    # F1-score = 2*TP / (2*TP + FN + FP)
                    classification_gt = dataset.extract_affordances_targets(data, 'classification')
                    regression_gt = dataset.extract_affordances_targets(data, 'regression')

                    TP = 0
                    FN = 0
                    FP = 0
                    TN = 0
                    for i in range(classification_gt.shape[0]):
                        if classification_gt[i, 0] == (c_output[0][i, 0] < c_output[0][i, 1]).type(torch.FloatTensor) == 1:
                            TP += 1

                        elif classification_gt[i, 0] == 1 and classification_gt[i, 0] != (c_output[0][i, 0] < c_output[0][i, 1]).type(torch.FloatTensor):
                            FN += 1

                        elif classification_gt[i, 0] == 0 and classification_gt[i, 0] != (c_output[0][i, 0] < c_output[0][i, 1]).type(torch.FloatTensor):
                            FP += 1

                        if classification_gt[i, 0] == (c_output[0][i, 0] < c_output[0][i, 1]).type(torch.FloatTensor) == 0:
                            TN += 1

                    accumulated_pedestrian_TP += TP
                    accumulated_pedestrian_TN += TN
                    accumulated_pedestrian_FP += FP
                    accumulated_pedestrian_FN += FN

                    TP = 0
                    FN = 0
                    FP = 0
                    TN = 0
                    for i in range(classification_gt.shape[0]):
                        if classification_gt[i, 1] == (c_output[1][i, 0] < c_output[1][i, 1]).type(torch.FloatTensor) == 1:
                            TP += 1

                        elif classification_gt[i, 1] == 1 and classification_gt[i, 1] != (c_output[1][i, 0] < c_output[1][i, 1]).type(torch.FloatTensor):
                            FN += 1

                        elif classification_gt[i, 1] == 0 and classification_gt[i, 1] != (c_output[1][i, 0] < c_output[1][i, 1]).type(torch.FloatTensor):
                            FP += 1

                        if classification_gt[i, 1] == (c_output[1][i, 0] < c_output[1][i, 1]).type(torch.FloatTensor) == 0:
                            TN += 1

                    accumulated_red_tl_TP += TP
                    accumulated_red_tl_TN += TN
                    accumulated_red_tl_FP += FP
                    accumulated_red_tl_FN += FN

                    TP = 0
                    FN = 0
                    FP = 0
                    TN = 0
                    for i in range(classification_gt.shape[0]):
                        if classification_gt[i, 2] == (c_output[2][i, 0] < c_output[2][i, 1]).type(
                                torch.FloatTensor) == 1:
                            TP += 1

                        elif classification_gt[i, 2] == 1 and classification_gt[i, 2] !=\
                                (c_output[2][i, 0] < c_output[2][i, 1]).type(torch.FloatTensor):
                            FN += 1

                        elif classification_gt[i, 2] == 0 and classification_gt[i, 2] !=\
                                (c_output[2][i, 0] < c_output[2][i, 1]).type(torch.FloatTensor):
                            FP += 1

                        if classification_gt[i, 2] == (c_output[2][i, 0] <
                                                       c_output[2][i, 1]).type(torch.FloatTensor) == 0:
                            TN += 1

                    accumulated_vehicle_stop_TP += TP
                    accumulated_vehicle_stop_TN += TN
                    accumulated_vehicle_stop_FP += FP
                    accumulated_vehicle_stop_FN += FN

                    # if the data was normalized during training, we need to transform it to its unit

                    write_regular_output(checkpoint_iteration, torch.squeeze(r_output[0]),
                                         regression_gt[:, 0])
                    mae_ra = torch.abs(regression_gt[:, 0] -
                                       torch.squeeze(r_output[0]).type(torch.FloatTensor)).\
                                            numpy()
                    accumulated_mae_ra += np.sum(mae_ra)

                    if iteration_on_checkpoint % 100 == 0:
                        print("Validation iteration: %d [%d/%d)] on Checkpoint %d " % (iteration_on_checkpoint,
                                                                                       iteration_on_checkpoint,
                                                                                       len(data_loader),
                                                                                       checkpoint_iteration))

                    iteration_on_checkpoint += 1


                # Here also need a better analysis. TODO divide into curve and other things
                MAE_relative_angle = accumulated_mae_ra / (len(dataset))

                csv_outfile = open(summary_file, 'a')
                csv_outfile.write("%s, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f" %
                                                        (checkpoint_iteration,
                                                          accumulated_pedestrian_TP,
                                                          accumulated_pedestrian_FP,
                                                          accumulated_pedestrian_FN,
                                                          accumulated_pedestrian_TN,
                                                          accumulated_vehicle_stop_TP,
                                                          accumulated_vehicle_stop_FP,
                                                          accumulated_vehicle_stop_FN,
                                                          accumulated_vehicle_stop_TN,
                                                          accumulated_red_tl_TP,
                                                          accumulated_red_tl_FP,
                                                          accumulated_red_tl_FN,
                                                          accumulated_red_tl_TN,
                                                          MAE_relative_angle))


                csv_outfile.write("\n")
                csv_outfile.close()

            else:
                print('The checkpoint you want to validate is not yet ready ', str(latest))



        coil_logger.add_message('Finished', {})
        print('VALIDATION FINISHED !!')
        print('  Validation results saved in ==> ', summary_file)

    except KeyboardInterrupt:
        coil_logger.add_message('Error', {'Message': 'Killed By User'})
        # We erase the output that was unfinished due to some process stop.
        if latest is not None:
            coil_logger.erase_csv(latest)

    except RuntimeError as e:
        if latest is not None:
            coil_logger.erase_csv(latest)
        coil_logger.add_message('Error', {'Message': str(e)})

    except:
        traceback.print_exc()
        coil_logger.add_message('Error', {'Message': 'Something Happened'})
        # We erase the output that was unfinished due to some process stop.
        if latest is not None:
            coil_logger.erase_csv(latest)
