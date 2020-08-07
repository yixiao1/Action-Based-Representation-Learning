from . import loss_functional as LF
import torch
from configs import g_conf
from torch.nn import functional as F


def CE(params):
    CE = F.cross_entropy(params['outputs'], params['consecutive_gt'].long())

    return CE

def CE_L1_L1(params):
    if g_conf.MODEL_TYPE in ['separate-supervised-NoSpeed']:
        L1_steering_gas_brake = branched_loss_without_speed(LF.l1_loss_without_speed, params)
        affordances_loss = CrossEntropy_L1(params)

    elif g_conf.MODEL_TYPE in ['separate-supervised']:
        L1_steering_gas_brake, _ = branched_loss(LF.l1_loss, params)
        affordances_loss = CrossEntropy_L1(params)

    return affordances_loss + L1_steering_gas_brake

def branched_loss_without_speed(loss_function, params):

    """
    Args
        loss_function: The loss functional that is actually computing the loss
        params: all the parameters, including
                branches: The tensor containing all the branches branches output from the network
                targets: The ground truth targets that the network should produce
                controls: the controls used for each point
                branches weights: the weigths that each branch will have on the loss function
                speed_gt: the ground truth speed for these data points
                variable_weights: The weights for each of the variables used

                For other losses it could contain more parameters

    Returns
        The computed loss function, but also a dictionary with plotable variables for tensorboard
    """

    controls_mask = LF.compute_branches_masks(params['controls'],
                                              params['branches'][0].shape[1])

    # Update the dictionary to add also the controls mask.
    params.update({'controls_mask': controls_mask})

    # calculate loss for each branch with specific activation
    loss_branches_vec = loss_function(params)

    # Apply the variable weights
    # This is applied to all branches except the last one, that is the speed branch...
    # TODO This is hardcoded to have 4 branches not using speed.
    for i in range(4):
        loss_branches_vec[i] = loss_branches_vec[i][:, 0] * params['variable_weights']['Steer'] \
                               + loss_branches_vec[i][:, 1] * params['variable_weights']['Gas'] \
                               + loss_branches_vec[i][:, 2] * params['variable_weights']['Brake']

    loss = loss_branches_vec[0] + loss_branches_vec[1] + loss_branches_vec[2] + \
                    loss_branches_vec[3]

    return torch.sum(loss) / (params['branches'][0].shape[0])

# TODO: This is hardcoded
def CrossEntropy_L1(params):
    relative_angle_output = params['affordances_output'][:, 0]
    hazard_stop_output = params['affordances_output'][:, 0:2]
    relative_angle_gt = params['affordances_gt'][:, 0]
    hazard_stop_gt = params['affordances_gt'][:, 1]

    CE = F.cross_entropy(hazard_stop_output, hazard_stop_gt.long(), weight = torch.FloatTensor(params['class_weights']['hazard_stop']).cuda())
    L1 = F.l1_loss(relative_angle_output, relative_angle_gt)

    # TODO: hardcoded......
    return L1 * g_conf.AFFORDANCES_LOSS_WEIGHT[0] + CE * g_conf.AFFORDANCES_LOSS_WEIGHT[1]


def CrossEntropy_MAE(params):
    MAE_loss = 0
    hazard_stop_output = params['outputs'][:,0:2]
    red_light_output = params['outputs'][:, 2:4]
    vehicle_stop_output = params['outputs'][:, 4:6]
    hazard_stop_gt = params['targets'][:, 0]
    red_light_gt = params['targets'][:, 1]
    vehicle_stop_gt = params['targets'][:, 2]
    regression_output = params['outputs'][:, 6:9]
    regression_gt = params['targets'][:, 3:6]
    #regression_output = params['outputs'][:, 6:7]
    #regression_gt = params['targets'][:, 3]

    CE_1 = F.cross_entropy(hazard_stop_output, hazard_stop_gt.long(), weight = torch.FloatTensor(params['class_weights']['hazard_stop']).cuda())
    CE_2 = F.cross_entropy(red_light_output, red_light_gt.long(), weight = torch.FloatTensor(params['class_weights']['red_traffic_light']).cuda())
    CE_3 = F.cross_entropy(vehicle_stop_output, vehicle_stop_gt.long(), weight=torch.FloatTensor(params['class_weights']['vehicle_stop']).cuda())
    CE_loss = (CE_1 + CE_2 + CE_3) / 3.0

    i = 0
    for k in params['variable_weights']:
        MAE_loss += F.l1_loss(regression_output[:, i], regression_gt[:, i]) * params['variable_weights'][k]
        #MAE_loss += F.l1_loss(regression_output, regression_gt)* params['variable_weights'][k]
        i += 1

    return (CE_loss + MAE_loss) / 2.0

def l1_KLD(params):
    ETE_loss, plotable_params = branched_loss(LF.l1_loss, params)
    KLD = -0.5 * torch.sum(1 + params['logvar'] - params['mu'].pow(2) - params['logvar'].exp())
    return ETE_loss+KLD


def l1_MSE_KLD(params):
    ete_loss, plotable_params = l1(params)
    vae_loss = MSE_KLD(params)
    return ete_loss+vae_loss, plotable_params


def l1_BCE_KLD(params):
    ete_loss, plotable_params = l1(params)
    vae_loss = BCE_KLD(params)
    return ete_loss+vae_loss, plotable_params

def l2_MSE_KLD(params):
    ete_loss, plotable_params = l2(params)
    vae_loss = MSE_KLD(params)
    return ete_loss+vae_loss, plotable_params

def l2_BCE_KLD(params):
    ete_loss, plotable_params = l2(params)
    vae_loss = BCE_KLD(params)
    return ete_loss+vae_loss, plotable_params


def KLD(params):
    mu = params['outputs']['mu']
    logvar = params['outputs']['logvar']
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

def BCE_KLD_L1(params):
    recon_x = params['outputs']['predictions']
    mu = params['outputs']['mu']
    logvar = params['outputs']['logvar']
    if g_conf.LABELS_SUPERVISED:
        x = params['labels']
    else:
        x = params['inputs_img']

    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    l1_speed = torch.sum(torch.abs(params['outputs']['speed_output'] - params['speed_gt']))/g_conf.BATCH_SIZE

    return BCE + g_conf.DISENTANGLE_BETA * KLD + l1_speed

def BCE_KLD_L1_CE(params):
    recon_x = params['outputs']['predictions']
    mu = params['outputs']['mu']
    logvar = params['outputs']['logvar']
    if g_conf.LABELS_SUPERVISED:
        x = params['labels']
    else:
        x = params['inputs_img']

    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    affordances_params = {'outputs': params['outputs']['affordances_output'],
                          'targets': params['affordances_gt'],  # harzard stop, red_light....
                          'class_weights': g_conf.AFFORDANCES_CLASS_WEIGHT,
                          'variable_weights': g_conf.AFFORDANCES_VARIABLE_WEIGHT}
    affordances_loss = CrossEntropy_MAE(affordances_params)

    return BCE + g_conf.DISENTANGLE_BETA * KLD + affordances_loss


def BCE_KLD(params):
    recon_x = params['outputs']['predictions']
    mu = params['outputs']['mu']
    logvar = params['outputs']['logvar']
    if g_conf.LABELS_SUPERVISED:
        x = params['labels']
    else:
        x = params['inputs_img']

    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + g_conf.DISENTANGLE_BETA * KLD


def BCE(params):
    recon_x = params['outputs']['predictions']
    if g_conf.LABELS_SUPERVISED:
        x = params['labels']
    else:
        x = params['inputs_img']

    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    return BCE


def MSE_KLD(params):
    recon_x = params['outputs']['predictions']
    mu = params['outputs']['mu']
    logvar = params['outputs']['logvar']

    if g_conf.LABELS_SUPERVISED:
        x = params['labels']
    else:
        x = params['inputs_img']

    MSE = F.mse_loss(recon_x, x, reduction='sum')

    # VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + g_conf.DISENTANGLE_BETA * KLD

def l1(params):
    if g_conf.MODEL_TYPE in ['coil-icra-NoSpeed']:
        return command_input_loss(LF.l1_loss_without_speed, params)

    else:
        return command_input_loss(LF.l1_loss, params)


def l2(params):
    return command_input_loss(LF.l2_loss, params)




def branched_loss(loss_function, params):

    """
    Args
        loss_function: The loss functional that is actually computing the loss
        params: all the parameters, including
                branches: The tensor containing all the branches branches output from the network
                targets: The ground truth targets that the network should produce
                controls: the controls used for each point
                branches weights: the weigths that each branch will have on the loss function
                speed_gt: the ground truth speed for these data points
                variable_weights: The weights for each of the variables used

                For other losses it could contain more parameters

    Returns
        The computed loss function, but also a dictionary with plotable variables for tensorboard
    """

    controls_mask = LF.compute_branches_masks(params['controls'],
                                              params['branches'][0].shape[1])

    # Update the dictionary to add also the controls mask.
    params.update({'controls_mask': controls_mask})

    # calculate loss for each branch with specific activation
    loss_branches_vec, plotable_params = loss_function(params)

    # Apply the variable weights
    # This is applied to all branches except the last one, that is the speed branch...
    # TODO This is hardcoded to  have 4 branches not using speed.

    for i in range(4):
        loss_branches_vec[i] = loss_branches_vec[i][:, 0] * params['variable_weights']['Steer'] \
                               + loss_branches_vec[i][:, 1] * params['variable_weights']['Gas'] \
                               + loss_branches_vec[i][:, 2] * params['variable_weights']['Brake']

    loss = loss_branches_vec[0] + loss_branches_vec[1] + loss_branches_vec[2] + \
                    loss_branches_vec[3]

    speed_loss = loss_branches_vec[4]

    return torch.sum(loss) / (params['branches'][0].shape[0])\
                + torch.sum(speed_loss) / (params['branches'][0].shape[0]), plotable_params


def command_input_loss(loss_function, params):

    """
    Args
        loss_function: The loss functional that is actually computing the loss
        params: all the parameters, including
                branches: The tensor containing all the branches branches output from the network
                targets: The ground truth targets that the network should produce
                controls: the controls used for each point
                branches weights: the weigths that each branch will have on the loss function
                speed_gt: the ground truth speed for these data points
                variable_weights: The weights for each of the variables used
                For other losses it could contain more parameters
    Returns
        The computed loss function, but also a dictionary with plotable variables for tensorboard
    """
    # Update the dictionary to add also the controls mask.
    # TODO branches name is not updated.
    params.update({'controls_mask': torch.ones(params['branches'][0].size()).cuda()})
    # calculate loss for each branch with specific activation
    loss_branches_vec, plotable_params = loss_function(params)

    # Apply the variable weights
    # This is applied to all branches except the last one, that is the speed branch...
    # TODO This is hardcoded to  have 4 branches not using speed.

    loss_branches_vec[0] = loss_branches_vec[0][:, 0] * params['variable_weights']['Steer'] \
                               + loss_branches_vec[0][:, 1] * params['variable_weights']['Gas'] \
                               + loss_branches_vec[0][:, 2] * params['variable_weights']['Brake']

    loss_function = loss_branches_vec[0]
    speed_loss = loss_branches_vec[-1]/(params['branches'][0].shape[0])

    return torch.sum(loss_function) / (params['branches'][0].shape[0])\
                + torch.sum(speed_loss) / (params['branches'][0].shape[0]),\
           plotable_params



def Loss(loss_name):
    """ Factory function

        Note: It is defined with the first letter as uppercase even though is a function to contrast
        the actual use of this function that is making classes
    """
    # TODO: this could be extended to some more arbitrary definition

    if g_conf.MODEL_TYPE in ['VAE']:

        if loss_name == 'BCE+KLD':
            return BCE_KLD

        elif loss_name == 'BCE':
            return BCE

        elif loss_name == 'MSE+KLD':
            return MSE_KLD

        else:
            raise ValueError(" Not found Loss name")

    elif g_conf.MODEL_TYPE in ['VAE-speed', 'VAE-affordances']:
        if loss_name == 'BCE+KLD+L1':
            return BCE_KLD_L1

        elif loss_name == 'BCE+KLD+L1+CE':
            return BCE_KLD_L1_CE

    elif g_conf.MODEL_TYPE in ['coil-icra-VAE']:

        if loss_name == 'L1':
            if g_conf.VAE_LOSS_FUNCTION is None:
                return l1

            elif g_conf.VAE_LOSS_FUNCTION == 'MSE+KLD':
                return l1_MSE_KLD

            elif g_conf.VAE_LOSS_FUNCTION == 'BCE+KLD':
                return l1_BCE_KLD

        elif loss_name == 'L2':
            if g_conf.VAE_LOSS_FUNCTION is None:
                return l2

            elif g_conf.VAE_LOSS_FUNCTION == 'MSE+KLD':
                return l2_MSE_KLD

            elif g_conf.VAE_LOSS_FUNCTION == 'BCE+KLD':
                return l2_BCE_KLD

        else:
            raise ValueError(" Not found Loss name")

    elif g_conf.MODEL_TYPE in ['coil-icra-VAE-affordances', 'coil-icra-affordances', 'coil-icra-Supervised-affordances', 'Affordances', 'affordances-affordances']:
        if loss_name == 'CrossEntropy+MAE':
            return CrossEntropy_MAE

    elif g_conf.MODEL_TYPE in ['coil-icra-KLD']:
        if loss_name == 'KLD+L1':
            return l1_KLD

    # TODO: this is hardcoded
    elif g_conf.MODEL_TYPE in ['separate-supervised-NoSpeed', 'separate-supervised']:
        if loss_name == 'CE+L1+L1':
            return CE_L1_L1

    elif g_conf.MODEL_TYPE in ['coil-icra-NoSpeed']:
        if loss_name == 'L1':

            return l1

    elif g_conf.MODEL_TYPE in ['detect_consecutive']:
        if loss_name == 'CE':
            return CE

    else:

        if loss_name == 'L1':

            return l1

        elif loss_name == 'L2':

            return l2

        else:
            raise ValueError(" Not found Loss name")


