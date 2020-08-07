
class ForwardInfonce(nn.Module):

    # The normal forward model that uses the ground truth and
    def __init__(self, params):
        super(ForwardInfonce, self).__init__()
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



        self.action = FC(params={'neurons': [3] +
                                            params['join']['fc']['neurons'] +
                                            params['action']['fc']['neurons'],
                                 'dropouts': params['action']['fc']['dropouts'] + [0.0],
                                 'end_layer': False})
        self.join = Join(
            params={'after_process': FC(params={'neurons':[params['action']['fc']['neurons'][-1] +
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

        # local features from previous t-1 and current t
        x_t_prev_local = inter_prev[4]  # get the feature maps of the last resnet layer
        x_t_local = inter[4]

        sy = x_t_prev_local.size(2)
        sx = x_t_prev_local.size(3)
        a_t_prev = self.action(a)
        # Loss 1: Global at time t-1, the local features from previous patches at time t
        N = f_t_global.size(0)
        loss1 = 0.
        # iterate over the local features that have 512 channels
        for y in range(sy):
            for x in range(sx):
                # take the global features of time t -> shape [120, 512]
                predictions = f_t_global
                # the local features of time t-1 at [y,x] -> shape [120, 512]
                obs_t_prev = self.join_obs(x_t_prev_local[:, :, y, x],
                                           m_t, c_t)
                # Perform the action space expansion
                f_ti_pred = self.join.forward(obs_t_prev, a_t_prev)  # Maybe the multiplication works here
                positive = f_ti_pred

                logits = torch.matmul(predictions, positive.t())
                step_loss = F.cross_entropy(logits, torch.arange(N).cuda())
                loss1 += step_loss
        loss1 = loss1 / (sx * sy)

        loss2 = 0.
        for y in range(sy):
            for x in range(sx):

                # take the local features of time t
                predictions = self.join_obs(x_t_local[:, :, y, x],
                                            m_ti, c_ti)
                # the local features of time t-1
                obs_t_prev = self.join_obs(x_t_prev_local[:, :, y, x],
                                           m_t, c_t)

                f_ti_pred = self.join.forward(obs_t_prev, a_t_prev)  # Maybe the multiplication works here
                positive = f_ti_pred

                logits = torch.matmul(predictions, positive.t())
                step_loss = F.cross_entropy(logits, torch.arange(N).cuda())
                loss2 += step_loss
        loss2 = loss2 / (sx * sy)
        loss = loss1 + loss2

        return loss, loss1, loss2

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

        return f_t



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