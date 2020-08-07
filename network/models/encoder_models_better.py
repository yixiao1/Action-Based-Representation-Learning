from logger import coil_logger
import torch.nn as nn
import torch
import importlib
from torch.nn import functional as F

from configs import g_conf
from coilutils.general import command_number_to_index


from logger import coil_logger
import numpy as np
import torch.nn as nn
import torch
import importlib
from torch.nn import functional as F

from configs import g_conf
from coilutils.general import command_number_to_index

from .building_blocks import Conv, Conv_Encode, ConvTrans_Decode
from .building_blocks import Branching
from .building_blocks import Mu_Logvar
from .building_blocks import FC, FC_Bottleneck
from .building_blocks import Join
from .building_blocks import Resnet34_Encode, Resnet34_Decode
from .building_blocks import utils


class VAE(nn.Module):

    def __init__(self, params):
        super(VAE, self).__init__()
        self.params = params

        number_input_channels = 0

        for sensor_name, sizes in g_conf.SENSORS.items():
            # TODO: NEEED TO BE BETTER CODED
            if 'labels' in sensor_name:
                number_input_channels += 1 * g_conf.NUMBER_FRAMES_FUSION
            else:
                number_input_channels += sizes[0] * g_conf.NUMBER_FRAMES_FUSION

        sensor_input_shape = next(iter(g_conf.SENSORS.values()))
        sensor_input_shape = [number_input_channels, sensor_input_shape[1],
                              sensor_input_shape[2]]


        if 'conv' in params['encode']['perception']:
            self.encode_conv = Conv_Encode(params={'channels': [number_input_channels] +
                                                       params['encode']['perception']['conv']['channels'],
                                                   'kernels': params['encode']['perception']['conv']['kernels'],
                                                   'strides': params['encode']['perception']['conv']['strides']})

        elif 'res' in params['encode']['perception']:
            if params['encode']['perception']['res']['name'] == 'resnet34':
                self.encode_conv = Resnet34_Encode(params['encode']['perception']['res']['name'], pretrained = False, progress=False,
                                                   num_classes = params['encode']['perception']['res']['num_classes'])


        # building two seperate fc parts for mu and logvar
        branches_for_mu_logvar = []
        for i in range(2):
            branches_for_mu_logvar.append(FC_Bottleneck(params={'neurons': [self.encode_conv.get_conv_output(sensor_input_shape)] +
                                                                             params['bottleneck']['neurons']['h_dim']}))

        self.bottleneck_h = Mu_Logvar(branches_for_mu_logvar)

        self.bottleneck_z = FC_Bottleneck(params={'neurons': params['bottleneck']['neurons']['z_dim'] + [self.encode_conv.get_conv_output(sensor_input_shape)]})

        if 'conv_trans' in params['decode']['perception']:
            self.decode_convTrans = ConvTrans_Decode(params={'encode_output_size': self.encode_conv.get_conv_output(sensor_input_shape),
                                                             'channels': params['decode']['perception']['conv_trans']['channels'] + [number_input_channels],
                                                             'kernels': params['decode']['perception']['conv_trans']['kernels'],
                                                             'strides': params['decode']['perception']['conv_trans']['strides']})

        elif 'res' in params['decode']['perception']:
            print('Not yet implement for resnet34 decoder in VAE')


    def encode(self, x):
        # the size need to be rewrote for generalization later
        h, h_shape = self.encode_conv(x)
        [mu, logvar] = self.bottleneck_h(h)
        return mu, logvar, h_shape

    def bottleneck(self, mu,logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z, h_shape):
        z_unflatten = self.bottleneck_z(z)
        recon_x = self.decode_convTrans(z_unflatten, h_shape)
        return recon_x

    def forward(self, x):
        mu, logvar, h_shape= self.encode(x)
        z = self.bottleneck(mu, logvar)
        recon_x = self.decode(z, h_shape)
        return recon_x, mu, logvar, z

    def forward_encoder(self,x):
        mu, logvar, h_shape = self.encode(x)
        z = self.bottleneck(mu, logvar)

        # i think we should retrun mu instead of z for getting bottleneck to train control network
        return mu


class Affordances_Separate(nn.Module):
    def __init__(self, params):
        super(Affordances_Separate, self).__init__()
        self.params = params

        if 'res' in params['encode']['perception']:  # pre defined residual networks
            resnet_module = importlib.import_module('network.models.building_blocks.resnet')
            resnet_module = getattr(resnet_module, params['encode']['perception']['res']['name'])
            self.encode_conv = resnet_module(pretrained=g_conf.PRE_TRAINED,
                                             num_classes=params['encode']['perception']['res']['num_classes'])

            number_output_neurons = params['encode']['perception']['res']['num_classes']

        self.command = FC(params={'neurons': [4] + params['encode']['command']['fc']['neurons'],
                                       'dropouts': params['encode']['command']['fc']['dropouts'],
                                       'end_layer': False})

        self.speed = FC(params={'neurons': [1] + params['encode']['speed']['fc']['neurons'],
                                       'dropouts': params['encode']['speed']['fc']['dropouts'],
                                       'end_layer': False})

        self.join = Join(
            params={'after_process':
                        FC(params={'neurons':
                                       [params['encode']['speed']['fc']['neurons'][-1] + params['encode']['command']['fc']['neurons'][-1]+
                                        number_output_neurons] +
                                       params['join']['fc']['neurons'],
                                   'dropouts': params['join']['fc']['dropouts'],
                                   'end_layer': False}),
                    'mode': 'cat'
                    } )

        # TODO: THIS is the hard coding for two fully conncected layers which output the affordances
        # Create the fc vector separatedely
        affordances_classification_fc_vector = []
        affordances_regression_fc_vector = []
        for i in range(params['affordances']['number_of_classification']):
            if not 'c_fc' in params['affordances']:
                affordances_classification_fc_vector.append(
                    FC(params={'neurons': [params['join']['fc']['neurons'][-1]] + [2],
                               'dropouts': [0.0],
                               'end_layer': True}))

            else:
                affordances_classification_fc_vector.append(
                    FC(params={'neurons': [params['join']['fc']['neurons'][-1]] + params['affordances']['c_fc']['neurons'] +[2],
                               'dropouts': params['affordances']['c_fc']['dropouts'] ,
                               'end_layer': True}))

        for j in range(params['affordances']['number_of_regression']):
            if not 'r_fc' in params['affordances']:
                affordances_regression_fc_vector.append(
                    FC(params={'neurons': [params['join']['fc']['neurons'][-1]] + [1],
                               'dropouts': [0.0],
                               'end_layer': True}))
            else:
                affordances_regression_fc_vector.append(
                    FC(params={'neurons': [params['join']['fc']['neurons'][-1]] + params['affordances']['r_fc']['neurons'] + [1],
                               'dropouts': params['affordances']['r_fc']['dropouts'] ,
                               'end_layer': True}))

        self.affordances_classification = Branching(affordances_classification_fc_vector)  # Here we set branching automatically
        self.affordances_regression = Branching(affordances_regression_fc_vector)  # Here we set branching automatically


    def forward(self, x, m, c, params):
        z, _ = self.encode_conv(x)
        s = self.speed(m)
        c = self.command(c)
        z = self.join(z, s, c)
        c_output = self.affordances_classification(z)
        r_output = self.affordances_regression(z)

        # here we compute the loss for classification and regression
        c_loss = 0.0
        r_loss = 0.0

        for i in range(len(c_output)):
            c_loss += F.cross_entropy(c_output[i], params['classification_gt'][:,i].long(), weight=torch.FloatTensor(params['class_weights'][i]).cuda())

        for j in range(len(r_output)):
            r_loss += F.l1_loss(torch.squeeze(r_output[j]), params['regression_gt'][:, j]) *params['variable_weights'][j]

        return c_loss + r_loss

    def forward_encoder(self, x, m, c):
        z, _ = self.encode_conv(x)
        s = self.speed(m)
        c = self.command(c)
        z = self.join.forward_three(z, s, c)
        # The encoder now it is a joined observation that is sufficient for
        # the case.
        return z

    def forward_outputs(self, x,  s, c,):
        z, inter = self.encode_conv(x)
        s = self.speed(s)
        c = self.command(c)
        z = self.join.forward(z, s, c)
        c_output = self.affordances_classification(z)
        r_output = self.affordances_regression(z)

        return c_output, r_output, inter

class Encoder(nn.Module):

    def __init__(self, params):
        self.params = params

        number_first_layer_channels = 0

        for _, sizes in g_conf.SENSORS.items():
            number_first_layer_channels += sizes[0] * g_conf.NUMBER_FRAMES_FUSION

        # Get one item from the dict
        sensor_input_shape = next(iter(g_conf.SENSORS.values()))
        sensor_input_shape = [number_first_layer_channels, sensor_input_shape[1],
                              sensor_input_shape[2]]

        # For this case we check if the perception layer is of the type "conv"
        if 'conv' in params['perception']:
            perception_convs = Conv(params={'channels': [number_first_layer_channels] +
                                                        params['perception']['conv']['channels'],
                                            'kernels': params['perception']['conv']['kernels'],
                                            'strides': params['perception']['conv']['strides'],
                                            'dropouts': params['perception']['conv']['dropouts'],
                                            'end_layer': True})

            perception_fc = FC(params={'neurons': [perception_convs.get_conv_output(sensor_input_shape)]
                                                  + params['perception']['fc']['neurons'],
                                       'dropouts': params['perception']['fc']['dropouts'],
                                       'end_layer': False})

            self.perception = nn.Sequential(*[perception_convs, perception_fc])

            number_output_neurons = params['perception']['fc']['neurons'][-1]

        elif 'res' in params['perception']:  # pre defined residual networks
            resnet_module = importlib.import_module('network.models.building_blocks.resnet')
            resnet_module = getattr(resnet_module, params['perception']['res']['name'])
            self.perception = resnet_module(pretrained=g_conf.PRE_TRAINED,
                                            num_classes=params['perception']['res']['num_classes'])

            number_output_neurons = params['perception']['res']['num_classes']

        else:

            raise ValueError("invalid convolution layer type")

        self.measurements = FC(params={'neurons': [1] +
                                                  params['measurements']['fc']['neurons'],
                                       'dropouts': params['measurements']['fc']['dropouts'],
                                       'end_layer': False})

        self.command = FC(params={'neurons': [4] +
                                             params['command']['fc']['neurons'],
                                  'dropouts': params['command']['fc']['dropouts'],
                                  'end_layer': False})

        self.join = Join(
            params={'after_process':
                        FC(params={'neurons':
                                       [params['measurements']['fc']['neurons'][-1] +
                                        params['command']['fc']['neurons'][-1] +
                                        number_output_neurons] +
                                       params['join']['fc']['neurons'],
                                   'dropouts': params['join']['fc']['dropouts'],
                                   'end_layer': False}),
                    'mode': 'cat'
                    }
        )

    def forward(self, x, a, c):
        """ ###### APPLY THE PERCEPTION MODULE """
        x, inter = self.perception(x)
        ## Not a variable, just to store intermediate layers for future vizualization
        # self.intermediate_layers = inter

        """ ###### APPLY THE MEASUREMENT MODULE """
        m = self.measurements(a)
        """ ###### APPLY THE Command MODULE  that can also be output of another network"""
        c = self.command(c)
        """ Join measurements and perception"""
        z = self.join(x, m, c)

        return z, x, inter

class BehaviorClonning(nn.Module):

    def __init__(self, params):
        # TODO: Improve the model autonaming function

        super(BehaviorClonning, self).__init__()

        self.speed_branch = FC(params={'neurons': [params['join']['fc']['neurons'][-1]] +
                                                   params['speed_branch']['fc']['neurons'] + [1],
                                       'dropouts': params['speed_branch']['fc']['dropouts'] + [0.0],
                                       'end_layer': True})

        self.action = FC(params={'neurons': [params['join']['fc']['neurons'][-1]] +
                                                         params['action']['fc']['neurons'] +
                                                         [len(g_conf.TARGETS)],
                                               'dropouts': params['action']['fc']['dropouts'] + [0.0],
                                               'end_layer': True})

        if 'conv' in params['perception']:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.1)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.1)


    def forward(self, x, a, c):
        """ ###### APPLY THE ENCODER MODULE """
        features, x, inter = self.encoder(x, a, c)

        branch_outputs = self.action(features)

        speed_branch_output = self.speed_branch(x)

        # We concatenate speed with the rest.

        if g_conf.MODEL_TYPE in ['coil-icra-affordances']:
            return [branch_outputs] + [speed_branch_output], x

        else:
            return [branch_outputs] + [speed_branch_output]



    def forward_encoder(self, x, s, c):
        """ ###### APPLY THE PERCEPTION MODULE """
        x, inter = self.perception(x)
        m = self.measurements(s)
        c = self.command(c)
        j = self.join(x, m, c)

        return j, inter


    def forward_action(self, x,  s, c):
        """
        DO a forward operation and return a single branch.

        Args:
            x: the image input
            a: speed measurement
            c: the branch number to be returned

        Returns:
            the forward operation on the selected branch

        """
        # Convert to integer just in case .
        # TODO: take four branches, this is hardcoded
        j, inter = self.forward_encoder(x, s, c)
        return self.action(j)



    def get_perception_layers(self, x):
        return self.perception.get_layers_features(x)

    def extract_branch(self, output_vec, branch_number):

        branch_number = command_number_to_index(branch_number)

        if len(branch_number) > 1:
            branch_number = torch.squeeze(branch_number.type(torch.cuda.LongTensor))
        else:
            branch_number = branch_number.type(torch.cuda.LongTensor)

        branch_number = torch.stack([branch_number,
                                     torch.cuda.LongTensor(range(0, len(branch_number)))])

        return output_vec[branch_number[0], branch_number[1], :]



class InverseModel(nn.Module):

    def __init__(self, params):
        super(InverseModel, self).__init__()
        self.params = params

        number_input_channels = 0

        self.join = Join(
            params={'after_process': FC(params={'neurons':[params['join']['fc']['neurons'][-1] +
                                                            params['join']['fc']['neurons'][-1]] +
                                                          params['join']['fc']['neurons'],
                                               'dropouts': params['join']['fc']['dropouts'],
                                               'end_layer': False}),
                    'mode': 'cat'})
        number_action_class = 0
        for i in g_conf.ACTION_CLASS_RANGE:
            number_action_class += len(g_conf.ACTION_CLASS_RANGE[i]) + 1

        self.decode = FC(params={'neurons': [params['join']['fc']['neurons'][-1]] +
                                            params['decode']['fc']['neurons'] +
                                            [number_action_class],
                                 'dropouts': params['decode']['fc']['dropouts'] + [0.0],
                                 'end_layer': True})



    def forward(self, f_t, f_ti, a_c):
        """
        Args:
            features are an input
            a_c: the action (steer, throttle, brake) classification ground truth at time t_prev.
            The time difference should be enough so that the action has effect.
            I think incorporating the action will improve the predictability of what the
            model does.
        Returns:
        """
        # Perform the action prediction
        j = self.join.forward(f_t, f_ti)  # Maybe the multiplication works here
        out = self.decode(j)

        x = len(g_conf.ACTION_CLASS_RANGE['steer']) + 1
        y = len(g_conf.ACTION_CLASS_RANGE['throttle']) + 1
        z = len(g_conf.ACTION_CLASS_RANGE['brake']) + 1
        loss1 = F.cross_entropy(out[:, 0:x], a_c[:, 0].long(), weight=torch.FloatTensor(g_conf.ACTION_VARIABLE_WEIGHT['steer']).cuda())
        loss2 = F.cross_entropy(out[:, x:x+y], a_c[:, 1].long(), weight=torch.FloatTensor(g_conf.ACTION_VARIABLE_WEIGHT['throttle']).cuda())
        loss3 = F.cross_entropy(out[:, x+y:x+y+z], a_c[:, 2].long(), weight=torch.FloatTensor(g_conf.ACTION_VARIABLE_WEIGHT['brake']).cuda())
        loss = loss1+loss2+loss3

        return loss, f_t, f_ti


class ETE_inverse_model(nn.Module):
    def __init__(self, params):
        super(ETE_inverse_model, self).__init__()
        self.params = params

        self.encoder = Encoder(params)

        self.join = Join(
            params={'after_process': FC(params={'neurons':[params['join']['fc']['neurons'][-1] +
                                                            params['join']['fc']['neurons'][-1]] +
                                                          params['join']['fc']['neurons'],
                                               'dropouts': params['join']['fc']['dropouts'],
                                               'end_layer': False}),
                    'mode': 'cat'})
        number_action_class = 0
        for i in g_conf.ACTION_CLASS_RANGE:
            number_action_class += len(g_conf.ACTION_CLASS_RANGE[i]) + 1

        self.speed_branch = FC(params={'neurons': [params['join']['fc']['neurons'][-1]] +
                                                   params['speed_branch']['fc']['neurons'] + [1],
                                       'dropouts': params['speed_branch']['fc']['dropouts'] + [0.0],
                                       'end_layer': True})

        self.ete = FC(params={'neurons': [params['join']['fc']['neurons'][-1]] +
                                            params['decode']['fc']['neurons'] +
                                         [len(g_conf.TARGETS)],
                                 'dropouts': params['decode']['fc']['dropouts'] + [0.0],
                                 'end_layer': True})
        self.inverse = FC(params={'neurons': [params['join']['fc']['neurons'][-1]] +
                                            params['decode']['fc']['neurons'] +
                                            [number_action_class],
                                 'dropouts': params['decode']['fc']['dropouts'] + [0.0],
                                 'end_layer': True})

        # Create the fc vector separatedely


        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x,  m,  c,  a, a_c):
        """
        Args:
            x: the input data at time t_prev and t, with the size of [[mini_batch, 3, 88, 200], [mini_batch, 3, 88, 200]]
            s: the speed input
            The time difference should be enough so that the action has effect.
            I think incorporating the action will improve the predictability of what the
            model does.
        Returns:
        """

        # lateral cameral at time t after action becomes central camera time t+1
        # the forward part of model
        # To input image to model
        x_t, inter = self.perception(x[0])  # frame before the action
        x_ti, inter = self.perception(x[1]) # frame after the action

        """ ###### APPLY THE MEASUREMENT MODULE """
        m_t = self.measurements(m[0])
        m_ti = self.measurements(m[1])
        """ ###### APPLY THE Command MODULE  that can also be output of another network"""
        c_t = self.command(c[0])
        c_ti = self.command(c[1])
        """ Join measurements and perception"""
        f_t = self.join_obs(x_t, m_t, c_t)
        f_ti = self.join_obs(x_ti, m_ti, c_ti)

        # Perform the action prediction

        j_f = self.join(f_t, f_ti)  # Maybe the multiplication works here
        inverse_out = self.inverse(j_f)


        x_pos = len(g_conf.ACTION_CLASS_RANGE['steer']) + 1
        y_pos = len(g_conf.ACTION_CLASS_RANGE['throttle']) + 1
        z_pos = len(g_conf.ACTION_CLASS_RANGE['brake']) + 1
        loss1 = F.cross_entropy(inverse_out[:, 0:x_pos], a_c[:, 0].long(),
                                weight=torch.FloatTensor(g_conf.ACTION_VARIABLE_WEIGHT['steer']).cuda())
        loss2 = F.cross_entropy(inverse_out[:, x_pos:x_pos+y_pos], a_c[:, 1].long(),
                                weight=torch.FloatTensor(g_conf.ACTION_VARIABLE_WEIGHT['throttle']).cuda())
        loss3 = F.cross_entropy(inverse_out[:, x_pos+y_pos:x_pos+y_pos+z_pos], a_c[:, 2].long(),
                                weight=torch.FloatTensor(g_conf.ACTION_VARIABLE_WEIGHT['brake']).cuda())
        inverse_loss = loss1 + loss2 + loss3

        # We compute control loss for current frame t+i
        #c_t = self.command(c[0])
        #x_t, inter = self.perception(x[0])  # frame before the action
        #m_t = self.measurements(m[0])
        #f_t = self.join_obs(x_t, m_t, c_t)
        a_pred = self.ete(f_t)

        loss_action =  torch.abs(a_pred - a)
        loss_action = loss_action[:, 0] * g_conf.VARIABLE_WEIGHT['Steer'] \
                       + loss_action[:, 1] * g_conf.VARIABLE_WEIGHT['Gas'] \
                       + loss_action[:, 2] * g_conf.VARIABLE_WEIGHT['Brake']

        loss_speed = torch.abs(self.speed_branch(x_t) - m[0])
        N = x_t.size(0)
        loss_ete = torch.sum(loss_action) * g_conf.BRANCH_LOSS_WEIGHT[0]/N \
                + torch.sum(loss_speed) * g_conf.BRANCH_LOSS_WEIGHT[1]/N


        loss = inverse_loss * g_conf.LOSSES_WEIGHTS['inverse'] + \
               loss_ete * g_conf.LOSSES_WEIGHTS['control_t']
        print ( " A LOSS ")
        return loss, inverse_loss, loss_ete

    def get_perception_layers(self, x):
        return self.perception.get_layers_features(x)

    def extract_branch(self, output_vec, branch_number):

        branch_number = command_number_to_index(branch_number)

        if len(branch_number) > 1:
            branch_number = torch.squeeze(branch_number.type(torch.cuda.LongTensor))
        else:
            branch_number = branch_number.type(torch.cuda.LongTensor)

        branch_number = torch.stack([branch_number,
                                     torch.cuda.LongTensor(range(0, len(branch_number)))])

        return output_vec[branch_number[0], branch_number[1], :]

    def forward_encoder(self, x, m, c):
        """
        Args:
            x: the input data at time t and ti, with the size of [[mini_batch, 3, 88, 200], [mini_batch, 3, 88, 200]]
            a: the action(steering, accelerate, gas) at time t.
        Returns:

        """
        # the forward part of model
        # To input image to model
        x, inter = self.perception(x)  # encoding with the resnet.

        """ ###### APPLY THE MEASUREMENT MODULE """
        m = self.measurements(m)
        """ ###### APPLY THE Command MODULE  that can also be output of another network"""
        c = self.command(c)
        """ Join measurements and perception"""
        f_global = self.join_obs(x, m, c)

        return f_global, inter

class STDIM(nn.Module):

    def __init__(self, params):
        super(STDIM, self).__init__()
        self.params = params


    def forward(self, f_t_global,
                inter, inter_prev,
                m_t, m_t_prev,
                c_t, c_t_prev):
        """
        Args:
            all already encoded

        Returns:
        """

        # x[0] is the previous frame, frame t -1
        # x[1] is current frame t
        # NOTE all the samples are positive. Less than N frames distance.


        # local features from previous t-1 and current t
        x_t_prev_local = inter_prev[4]  # get the feature maps of the last resnet layer
        x_t_local = inter[4]

        sy = x_t_prev_local.size(2)
        sx = x_t_prev_local.size(3)

        # Loss 1: Global at time t-1, the local features from previous patches at time t
        N = f_t_global.size(0)
        loss1 = 0.
        # iterate over the local features that have 512 channels
        for y in range(sy):
            for x in range(sx):

                # take the global features of time t -> shape [120, 512]
                predictions = f_t_global
                # the local features of time t-1 at [y,x] -> shape [120, 512]
                positive = self.join_obs(x_t_prev_local[:, :, y, x],
                                         m_t_prev, c_t_prev)
                logits = torch.matmul(predictions, positive.t())
                step_loss = F.cross_entropy(logits, torch.arange(N).cuda())
                loss1 += step_loss
        loss1 = loss1 / (sx * sy)

        loss2 = 0.
        for y in range(sy):
            for x in range(sx):

                # take the local features of time t
                predictions = self.join_obs(x_t_local[:, :, y, x],
                                            m_t, c_t)
                # the local features of time t-1
                positive = self.join_obs(x_t_prev_local[:, :, y, x],
                                         m_t_prev, c_t_prev)

                logits = torch.matmul(predictions, positive.t())
                step_loss = F.cross_entropy(logits, torch.arange(N).cuda())
                loss2 += step_loss
        loss2 = loss2 / (sx * sy)
        loss = loss1 + loss2

        return loss, x_t_prev_local, x_t_local


class ETEDIM(nn.Module):

    # The normal forward model that uses the ground truth and
    def __init__(self, params):
        super(ETEDIM, self).__init__()
        self.params = params

        number_input_channels = 0

        for sensor_name, sizes in g_conf.SENSORS.items():
            # TODO: NEEED TO BE BETTER CODED
            if 'labels' in sensor_name:
                number_input_channels += 1 * g_conf.NUMBER_FRAMES_FUSION
            else:
                number_input_channels += sizes[0] * g_conf.NUMBER_FRAMES_FUSION

        sensor_input_shape = next(iter(g_conf.SENSORS.values()))
        sensor_input_shape = [number_input_channels, sensor_input_shape[1],
                              sensor_input_shape[2]]

        if 'conv' in params['encode']['perception']:
            self.encode_conv = Conv_Encode(params={'channels': [number_input_channels] +
                                                       params['encode']['perception']['conv']['channels'],
                                                   'kernels': params['encode']['perception']['conv']['kernels'],
                                                   'strides': params['encode']['perception']['conv']['strides']})

            perception_fc = FC(params={'neurons': [Conv_Encode.get_conv_output(sensor_input_shape)]
                                                  + params['encode']['perception']['fc']['neurons'],
                                       'dropouts': params['encode']['perception']['fc']['dropouts'],
                                       'end_layer': False})

            self.perception = nn.Sequential(*[Conv_Encode, perception_fc])

            number_output_neurons = params['encode']['perception']['fc']['neurons'][-1]


        elif 'res' in params['encode']['perception']:  # pre defined residual networks
            resnet_module = importlib.import_module('network.models.building_blocks.resnet')
            resnet_module = getattr(resnet_module, params['encode']['perception']['res']['name'])
            self.encode_conv = resnet_module(pretrained=g_conf.PRE_TRAINED,
                                             num_classes=params['encode']['perception']['res']['num_classes'])

            number_output_neurons = params['encode']['perception']['res']['num_classes']

        self.measurements = FC(params={'neurons': [1] +
                                                   params['measurements']['fc']['neurons'],
                                       'dropouts': params['measurements']['fc']['dropouts'],
                                       'end_layer': False})

        self.command = FC(params={'neurons': [4] +
                                                   params['command']['fc']['neurons'],
                                       'dropouts': params['command']['fc']['dropouts'],
                                       'end_layer': False})
        self.join_obs = Join(
            params={'after_process': FC(params={'neurons': [number_output_neurons
                                                        + params['command']['fc']['neurons'][-1]
                                                        + params['measurements']['fc']['neurons'][-1]
                                                            ]
                                                       + params['join']['fc']['neurons'],
                                            'dropouts': params['join']['fc']['dropouts'],
                                            'end_layer': False}),
                    'mode': 'cat'})



        self.actionfeat = FC(params={'neurons': params['join']['fc']['neurons'] +
                                            params['action']['fc']['neurons'],
                                 'dropouts': params['action']['fc']['dropouts'],
                                 'end_layer': False})

        self.action = FC(params={'neurons': [params['action']['fc']['neurons'][-1]] +
                                            [len(g_conf.TARGETS)],
                                 'dropouts':  [0.0],
                                 'end_layer': True})
        self.join = Join(
            params={'after_process': FC(params={'neurons':[params['action']['fc']['neurons'][-1] +
                                                            params['join']['fc']['neurons'][-1]] +
                                                          params['join']['fc']['neurons'],
                                               'dropouts': params['join']['fc']['dropouts'],
                                               'end_layer': False}),
                    'mode': 'cat'})

        self.speed_branch = FC(params={'neurons': [params['join']['fc']['neurons'][-1]] +
                                                   params['speed_branch']['fc']['neurons'] + [1],
                                       'dropouts': params['speed_branch']['fc']['dropouts'] + [0.0],
                                       'end_layer': True})


    def forward(self, x, m, c, a):
        """
        Args:
            x: the input data at time t_prev and t, with the size of [[mini_batch, 3, 88, 200], [mini_batch, 3, 88, 200]]
            m: the measurements that are useful for this step ( speed)
            c: the command that is useful for this step
            a_c: the action (steer, throttle, brake) classification ground truth at time t_prev.
            The time difference should be enough so that the action has effect.
            I think incorporating the action will improve the predictability of what the
            model does.
        Returns:
        """
        """
        Args:
            x: the input data at time t and ti, with the size of [[mini_batch, 3, 88, 200], [mini_batch, 3, 88, 200]]
            The time difference should be enough so that the action has effect.
            I think incorporating the action will improve the predictability of what the
            model does.
        Returns:
        """

        # x[0] is the previous frame, frame t -1
        # x[1] is current frame t
        # NOTE all the samples are positive. Less than N frames distance.

        # global features from previous t-1 and current t
        x_t, inter_prev = self.encode_conv(x[0])  # frame before the action
        x_ti, inter = self.encode_conv(x[1])  # frame after the action

        """ ###### APPLY THE MEASUREMENT MODULE """
        m_t = self.measurements(m[0])
        m_ti = self.measurements(m[1])
        """ ###### APPLY THE Command MODULE  that can also be output of another network"""
        c_t = self.command(c[0])
        c_ti = self.command(c[1])
        """ Join measurements and perception"""
        f_t_global = self.join_obs(x_ti, m_ti, c_ti)
        # The N is the batch size used on all losses
        N = f_t_global.size(0)
        """ Loss 1: The prediction based on the labelled loss (ACTION)"""

        obs_t_prev = self.join_obs(x_t, m_t, c_t)

        # Get the last layers features
        a_feat_prev = self.actionfeat(obs_t_prev)

        a_pred = self.action(a_feat_prev)

        loss_action =  torch.abs(a_pred - a)
        loss_action = loss_action[:, 0] * g_conf.VARIABLE_WEIGHT['Steer'] \
                       + loss_action[:, 1] * g_conf.VARIABLE_WEIGHT['Gas'] \
                       + loss_action[:, 2] * g_conf.VARIABLE_WEIGHT['Brake']

        loss_speed = torch.abs(self.speed_branch(x_t) - m[0])

        loss1 = torch.sum(loss_action) * g_conf.BRANCH_LOSS_WEIGHT[0]/N \
                + torch.sum(loss_speed) * g_conf.BRANCH_LOSS_WEIGHT[1]/N

        """ INFONCE based LOSSES  LOCAL AND GLOBAL"""
        # local features from previous t-1 and current t
        x_t_prev_local = inter_prev[4]  # get the feature maps of the last resnet layer
        x_t_local = inter[4]
        sy = x_t_prev_local.size(2)
        sx = x_t_prev_local.size(3)

        # Loss 2: Global at time t-1, the local features from previous patches at time t

        loss2 = 0.
        # iterate over the local features that have 512 channels
        for y in range(sy):
            for x in range(sx):
                # take the global features of time t -> shape [120, 512]
                predictions = f_t_global
                # the local features of time t-1 at [y,x] -> shape [120, 512]
                obs_t_prev_local = self.join_obs(x_t_prev_local[:, :, y, x],
                                                 m_t, c_t)

                f_ti_pred = self.join.forward(obs_t_prev_local, a_feat_prev)  # Maybe the multiplication works here
                positive = f_ti_pred

                logits = torch.matmul(predictions, positive.t())
                step_loss = F.cross_entropy(logits, torch.arange(N).cuda())
                loss1 += step_loss
        loss2 = loss2 / (sx * sy)

        # loss 3
        loss3 = 0.
        for y in range(sy):
            for x in range(sx):

                # take the local features of time t
                predictions = self.join_obs(x_t_local[:, :, y, x],
                                            m_ti, c_ti)
                # the local features of time t-1
                obs_t_prev = self.join_obs(x_t_prev_local[:, :, y, x],
                                           m_t, c_t)
                # Perform the action space expansion
                f_ti_pred = self.join.forward(obs_t_prev, a_feat_prev)  # Maybe the multiplication works here
                positive = f_ti_pred

                logits = torch.matmul(predictions, positive.t())
                step_loss = F.cross_entropy(logits, torch.arange(N).cuda())
                loss3 += step_loss
        loss3 = loss3 / (sx * sy)
        loss = loss1 * g_conf.LOSSES_WEIGHTS['control_t'] + \
               loss2 * g_conf.LOSSES_WEIGHTS['stdim']/2.0 + \
               loss3 * g_conf.LOSSES_WEIGHTS['stdim']/2.0

        return loss, loss1, loss2 * g_conf.LOSSES_WEIGHTS['stdim']/2.0 + \
                            loss3 * g_conf.LOSSES_WEIGHTS['stdim']/2.0

    def forward_encoder(self, x, m, c):
        """
        Args:
            x: the input data at time t
            a: the action(steering, accelerate, gas) at time t.
        Returns:
        """
        # the forward part of model
        # To input image to model        x_t, inter = self.encode_conv(x[0])  # frame before the action
        x_t, inter = self.encode_conv(x) # frame after the action
        """ ###### APPLY THE MEASUREMENT MODULE """
        m_t = self.measurements(m)
        """ ###### APPLY THE Command MODULE  that can also be output of another network"""
        c_t = self.command(c)
        """ Join measurements and perception"""
        f_t = self.join_obs(x_t, m_t, c_t)

        return f_t, inter




class ALL(nn.Module):

    # The normal forward model that uses the ground truth and
    def __init__(self, params):
        super(ALL, self).__init__()
        self.params = params

        self.encoder = Encoder(params)

        self.stdim = STDIM(params)

        self.im = InverseModel(params)

        self.join = Join(
            params={'after_process': FC(params={'neurons': [params['action']['fc']['neurons'][-1] +
                                                            params['join']['fc']['neurons'][-1]] +
                                                           params['join']['fc']['neurons'],
                                                'dropouts': params['join']['fc']['dropouts'],
                                                'end_layer': False}),
                    'mode': 'cat'})

    def forward(self, x, m, c, a):
        """
        Args:
            x: the input data at time t_prev and t, with the size of [[mini_batch, 3, 88, 200], [mini_batch, 3, 88, 200]]
            m: the measurements that are useful for this step ( speed)
            c: the command that is useful for this step
            a_c: the action (steer, throttle, brake) classification ground truth at time t_prev.
            The time difference should be enough so that the action has effect.
            I think incorporating the action will improve the predictability of what the
            model does.
        Returns:
        """

        f_t, x_t, m_t, c_t, inter = self.encoder(x[0], m[0], c[0])
        f_ti, x_ti, m_ti, c_ti, inter_2 = self.encoder(x[1], m[1], c[1])

        """ STDIM !!!"""

        loss_stdim = self.stdim(f_t,inter, inter_2, m_t, m_ti, c_t, c_ti )

        """ ACTION """
        loss_

        """ ETE """

        return loss

    def forward_encoder(self, x, m, c):
        """
        Args:
            x: the input data at time t
            a: the action(steering, accelerate, gas) at time t.
        Returns:
        """

        f_t, x_t, inter = self.encoder(x[0], m[0], c[0])
        return f_t, inter