from logger import coil_logger
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


class Separate_Affordances(nn.Module):
    def __init__(self, params, ENCODER_params = None):
        super(Separate_Affordances, self).__init__()
        self.params = params


        # TODO: THIS is the hard coding for two fully conncected layers which output the affordances
        # Create the fc vector separatedely
        affordances_classification_fc_vector = []
        affordances_regression_fc_vector = []

        for i in range(params['affordances']['number_of_classification']):
            if 'c_res' in params['affordances']:
                if g_conf.ENCODER_MODEL_TYPE in ['ETE']:
                    resnet_module = importlib.import_module('network.models.building_blocks.resnet_mlp')
                    resnet_module = getattr(resnet_module,
                                            params['affordances']['c_res']['name'])
                    res_mlp = resnet_module(
                        inplanes=ENCODER_params['perception']['res']['num_classes'],
                        num_classes=params['affordances']['c_res'][
                                                         'num_classes'])

                elif g_conf.ENCODER_MODEL_TYPE in ['action_prediction', 'stdim']:
                    resnet_module = importlib.import_module('network.models.building_blocks.resnet_mlp')
                    resnet_module = getattr(resnet_module,
                                            params['affordances']['c_res']['name'])
                    res_mlp = resnet_module(
                        inplanes=ENCODER_params['encode']['perception']['res']['num_classes'],
                        num_classes=params['affordances']['c_res'][
                            'num_classes'])

                if 'c_fc' in params['affordances']:
                    if g_conf.ENCODER_MODEL_TYPE in ['ETE', 'action_prediction']:
                        aff_classification_fc = FC(params={'neurons': [params['affordances']['c_res']['num_classes']] + params['affordances']['c_fc']['neurons'] + [2],
                                       'dropouts': params['affordances']['c_fc']['dropouts'],
                                       'end_layer': True})

                    affordances_classification = nn.Sequential(*[res_mlp, aff_classification_fc ])

                    affordances_classification_fc_vector.append(affordances_classification)

                else:
                    affordances_classification_fc_vector.append(res_mlp)



            elif  'c_fc' not in params['affordances']:
                if g_conf.ENCODER_MODEL_TYPE in ['VAE']:
                    affordances_classification_fc_vector.append(
                        FC(params={'neurons': ENCODER_params['bottleneck']['neurons']['z_dim'] + [2],
                                   'dropouts': [0.0],
                                   'end_layer': True}))

                elif g_conf.ENCODER_MODEL_TYPE in ['ETE', 'ETE_inverse_model', 'forward']:
                    affordances_classification_fc_vector.append(
                        FC(params={'neurons': [ENCODER_params['perception']['res']['num_classes']] + [2],
                                   'dropouts': [0.0],
                                   'end_layer': True}))

                elif g_conf.ENCODER_MODEL_TYPE in ['action_prediction', 'stdim', 'one-step-affordances']:
                    affordances_classification_fc_vector.append(
                        FC(params={'neurons': [ENCODER_params['encode']['perception']['res']['num_classes']] + [2],
                                   'dropouts': [0.0],
                                   'end_layer': True}))
            else:
                if g_conf.ENCODER_MODEL_TYPE in ['ETE', 'ETE_inverse_model', 'forward']:
                    affordances_classification_fc_vector.append(
                        FC(params={'neurons': [ENCODER_params['perception']['res']['num_classes']] +params['affordances']['c_fc']['neurons']+ [2],
                                   'dropouts': params['affordances']['c_fc']['dropouts'],
                                   'end_layer': True}))

                elif g_conf.ENCODER_MODEL_TYPE in ['action_prediction', 'stdim', 'one-step-affordances']:
                    affordances_classification_fc_vector.append(
                        FC(params={'neurons': [ENCODER_params['encode']['perception']['res']['num_classes']] +params['affordances']['c_fc']['neurons']+ [2],
                                   'dropouts': params['affordances']['c_fc']['dropouts'],
                                   'end_layer': True}))

        for j in range(params['affordances']['number_of_regression']):
            if 'r_res' in params['affordances']:
                if g_conf.ENCODER_MODEL_TYPE in ['ETE']:
                    resnet_module = importlib.import_module('network.models.building_blocks.resnet_mlp')
                    resnet_module = getattr(resnet_module,
                                            params['affordances']['r_res']['name'])
                    res_mlp = resnet_module(
                        inplanes=ENCODER_params['perception']['res']['num_classes'],
                        num_classes=params['affordances']['r_res'][
                                                         'num_classes'])
                elif g_conf.ENCODER_MODEL_TYPE in ['action_prediction', 'stdim']:
                    resnet_module = importlib.import_module('network.models.building_blocks.resnet_mlp')
                    resnet_module = getattr(resnet_module,
                                            params['affordances']['r_res']['name'])
                    res_mlp = resnet_module(
                        inplanes=ENCODER_params['encode']['perception']['res']['num_classes'],
                        num_classes=params['affordances']['r_res'][
                            'num_classes'])

                if 'r_fc' in params['affordances']:
                    if g_conf.ENCODER_MODEL_TYPE in ['ETE', 'action_prediction']:
                        aff_regression_fc = FC(params={'neurons': [params['affordances']['r_res']['num_classes']] + params['affordances']['r_fc']['neurons'] + [1],
                                       'dropouts': params['affordances']['r_fc']['dropouts'],
                                       'end_layer': True})

                    affordances_regression = nn.Sequential(*[res_mlp, aff_regression_fc])

                    affordances_regression_fc_vector.append(affordances_regression)
                else:
                    affordances_regression_fc_vector.append(res_mlp)

            elif 'r_fc' not in params['affordances']:
                if g_conf.ENCODER_MODEL_TYPE in ['VAE']:
                    affordances_regression_fc_vector.append(
                        FC(params={'neurons': ENCODER_params['bottleneck']['neurons']['z_dim'] + [1],
                                   'dropouts': [0.0],
                                   'end_layer': True}))

                elif g_conf.ENCODER_MODEL_TYPE in ['ETE', 'ETE_inverse_model', 'forward']:
                    affordances_regression_fc_vector.append(
                        FC(params={'neurons': [ENCODER_params['perception']['res']['num_classes']] +
                                              [1],
                                   'dropouts': [0.0],
                                   'end_layer': True}))

                elif g_conf.ENCODER_MODEL_TYPE in ['action_prediction', 'stdim', 'one-step-affordances']:
                    affordances_regression_fc_vector.append(
                        FC(params={'neurons': [ENCODER_params['encode']['perception']['res']['num_classes']]
                                              + [1],
                                   'dropouts': [0.0],
                                   'end_layer': True}))
            else:
                if g_conf.ENCODER_MODEL_TYPE in ['ETE', 'ETE_inverse_model', 'forward']:
                    affordances_regression_fc_vector.append(
                        FC(params={'neurons': [ENCODER_params['perception']['res']['num_classes']] +
                                              params['affordances']['r_fc']['neurons'] + [1],
                                   'dropouts': params['affordances']['r_fc']['dropouts'],
                                   'end_layer': True}))

                elif g_conf.ENCODER_MODEL_TYPE in ['action_prediction', 'stdim', 'one-step-affordances']:
                    affordances_regression_fc_vector.append(
                        FC(params={'neurons': [ENCODER_params['encode']['perception']['res']['num_classes']]
                                              + params['affordances']['r_fc']['neurons'] + [1],
                                   'dropouts': params['affordances']['r_fc']['dropouts'],
                                   'end_layer': True}))

        self.affordances_classification = Branching(affordances_classification_fc_vector)  # Here we set branching automatically
        self.affordances_regression = Branching(affordances_regression_fc_vector)  # Here we set branching automatically


    def forward(self, z, params):
        c_output = self.affordances_classification(z)
        r_output = self.affordances_regression(z)

        # here we compute the loss for classification and regression
        c_loss = 0.0
        r_loss = 0.0

        for i in range(len(c_output)):
            c_loss += F.cross_entropy(c_output[i], params['classification_gt'][:, i].long(),
                                      weight=torch.FloatTensor(params['class_weights'][i]).cuda())

        for j in range(len(r_output)):
            r_loss += F.l1_loss(torch.squeeze(r_output[j]), params['regression_gt'][:, j]) *\
                      params['variable_weights'][j]

        return c_loss + r_loss

    def forward_test(self, z):
        c_output = self.affordances_classification(z)
        r_output = self.affordances_regression(z)

        return c_output, r_output



