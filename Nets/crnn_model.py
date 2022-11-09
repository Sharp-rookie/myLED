import time
import torch
import torch.nn as nn
from torch.autograd import Variable

from .activations import getActivation
from .beta_vae import Beta_Vae
from .zoneoutlayer import ZoneoutLayer
from .rnn_mlp_wrapper import RNN_MLP_wrapper
from .mlp_autoencoders import MLPEncoder, MLPDecoder
from .conv_1D_autoencoder import getEncoderModel, getDecoderModel
from .dummy import viewEliminateChannels, viewAddChannels


class crnn_model(nn.Module):

    def __init__(self, model):
        super(crnn_model, self).__init__()
        self.parent = model
        self.buildNetwork()

        # Define latent state scaler
        self.has_latent_scaler = self.parent.has_rnn and self.parent.has_autoencoder and (not self.parent.RNN_convolutional) and self.parent.load_trained_AE
        if self.has_latent_scaler:
            self.defineLatentStateParams()

        # Changing the data type of the network
        self.double()
        for modules in self.module_list:
            modules.double()
            for layer in modules:
                layer.double()

        # Changing the device of the network
        if self.parent.gpu:
            self.cuda()
            for modules in self.module_list:
                for model in modules:
                    model.cuda()
    

    def buildNetwork(self):
        
        ##################################################################
        # Part1: ENCODER
        ##################################################################
        if self.parent.has_autoencoder:
            if not self.parent.AE_convolutional:
                encoder = MLPEncoder(
                    channels=self.parent.channels,
                    input_dim=self.parent.input_dim,
                    Dx=self.parent.Dx,
                    Dy=self.parent.Dy,
                    output_dim=self.parent.latent_state_dim,
                    activation=self.parent.params["activation_str_general"],
                    activation_output=self.parent.params["activation_str_general"],
                    layers_size=self.parent.layers_encoder,
                    dropout_keep_prob=self.parent.dropout_keep_prob,
                )
                self.ENCODER = encoder.layers
            elif self.parent.AE_convolutional:
                if self.parent.channels == 1:
                    encoder = getEncoderModel(self.parent, self.parent.mode)
                elif self.parent.channels == 2:
                    assert False, "2D Convolutional AE not implemented"
                    encoder = conv_mlp_model_2D.getEncoderModel(self.parent)
                else:
                    raise ValueError("Not implemented.")
                self.ENCODER = encoder.layers
        
        elif self.parent.has_dummy_autoencoder:
            self.ENCODER = nn.ModuleList()
            self.ENCODER.append(viewEliminateChannels(self.parent.params))
        
        else:
            self.ENCODER = nn.ModuleList()

        ##################################################################
        # Part2: BETA_VAE
        ##################################################################
        if self.parent.has_autoencoder and self.parent.AE_convolutional:
            # assert False, "Convolutional AE not implemented"
            sizes = encoder.printDimensions()
            latent_state_dim = sizes[-1]
            iter_ = 1
            for dim in latent_state_dim:
                iter_ *= dim
            latent_state_dim = iter_
            self.parent.RNN_state_dim = latent_state_dim
            self.parent.latent_state_dim = latent_state_dim
        else:
            latent_state_dim = self.parent.latent_state_dim
        self.BETA_VAE = nn.ModuleList()
        if self.parent.has_autoencoder and self.parent.beta_vae:
            temp = Beta_Vae(embedding_size=latent_state_dim)
            self.BETA_VAE.append(temp)

        ##################################################################
        # Part3: RNN
        ##################################################################
        self.RNN = nn.ModuleList()
        if self.parent.has_rnn:
            if self.parent.RNN_cell_type == "mlp":
                """ dummy wrapper to implement an MLP ignoring the hidden state """
                input_size = self.parent.RNN_state_dim
                self.RNN.append(RNN_MLP_wrapper(
                        input_size=input_size,
                        output_size=input_size,
                        hidden_sizes=self.parent.layers_rnn,
                        act=self.parent.RNN_activation_str,
                        act_output=self.parent.RNN_activation_str_output,
                    ))

            elif self.parent.RNN_cell_type == 'lstm':
                """ normal recurrent neural network """
                if not self.parent.RNN_convolutional:
                    # Parsing the layers of the RNN
                    # The input to the RNN is an embedding (effective dynamics) vector, or latent vector

                    if not self.parent.RNN_convolutional:
                        # Determining cell type
                        if self.parent.RNN_cell_type == "lstm":
                            self.RNN_cell = nn.LSTMCell
                        elif self.parent.RNN_cell_type == "gru":
                            self.RNN_cell = nn.GRUCell
                        else:
                            assert False, "{:} not implemented".format(self.parent.RNN_cell_type)

                    input_size = self.parent.RNN_state_dim
                    for ln in range(len(self.parent.layers_rnn)):
                        self.RNN.append(ZoneoutLayer(
                            self.RNN_cell(input_size=input_size, hidden_size=self.parent.layers_rnn[ln]), 
                            self.parent.zoneout_keep_prob
                            )
                        )
                else:
                    assert False, "Convolutional RNN not implemented"
                    input_size = self.parent.RNN_state_dim
                    # Convolutional RNN Cell
                    for ln in range(len(self.parent.layers_rnn)):
                        if self.parent.channels == 1:
                            self.RNN.append(
                                conv_rnn_cell_1D.ConvRNNCell(
                                    input_size,
                                    self.parent.layers_rnn[ln],
                                    self.parent.RNN_kernel_size,
                                    activation=self.parent.RNN_activation_str,
                                    cell_type=self.parent.RNN_cell_type,
                                    torch_dtype=self.parent.torch_dtype))
                        elif self.parent.channels == 2:
                            self.RNN.append(
                                conv_rnn_cell_2D.ConvRNNCell(
                                    input_size,
                                    self.parent.layers_rnn[ln],
                                    self.parent.RNN_kernel_size,
                                    activation=self.parent.RNN_activation_str,
                                    cell_type=self.parent.RNN_cell_type,
                                    torch_dtype=self.parent.torch_dtype))
                        input_size = self.parent.layers_rnn[ln]
        
        # Initial RNN hidden states (in case of trainable)
        if self.parent.RNN_trainable_init_hidden_state: # TODO: 初始化权重的取值还能训练优化？
            self.augmentRNNwithTrainableInitialHiddenState()

        ##################################################################
        # Part4: RNN OUTPUT
        ##################################################################
        self.RNN_OUTPUT = nn.ModuleList()
        if self.parent.has_rnn:

            if self.parent.RNN_cell_type == "mlp":

                self.RNN_OUTPUT.extend([nn.Identity()])

            elif self.parent.RNN_cell_type == 'lstm':
                if not self.parent.RNN_convolutional:
                    self.RNN_OUTPUT.extend([
                        nn.Linear(
                            self.parent.layers_rnn[-1],
                            self.parent.RNN_state_dim,
                            bias=True,
                        )
                    ])
                    # Here RNN is in the latent space (general activation string)
                    self.RNN_OUTPUT.extend([getActivation(self.parent.RNN_activation_str_output)])
                else:
                    assert False, "Convolutional RNN not implemented"
                    if self.parent.channels == 2:
                        self.RNN_OUTPUT.extend([
                            nn.Conv2d(self.parent.RNN_layers_size,
                                      self.parent.input_dim,
                                      1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
                        ])
                    elif self.parent.channels == 1:
                        self.RNN_OUTPUT.extend([
                            nn.Conv1d(self.parent.RNN_layers_size,
                                      self.parent.input_dim,
                                      1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
                        ])
                    # Here RNN is at the output (output activation string)
                    self.RNN_OUTPUT.extend([getActivation(self.parent.RNN_activation_str_output)])
            else:
                assert False, "{:} not implemented!".format(self.parent.RNN_cell_type)

        ##################################################################
        # Part5: DECODER
        ##################################################################
        if self.parent.has_autoencoder:
            # Building the layers of the decoder (additional input is the latent noise)
            if not self.parent.AE_convolutional:
                decoder = MLPDecoder(
                    channels=self.parent.channels,
                    input_dim=self.parent.latent_state_dim,
                    Dx=self.parent.Dx,
                    Dy=self.parent.Dy,
                    output_dim=self.parent.output_dim,
                    activation=self.parent.params["activation_str_general"],
                    activation_output=self.parent.params["activation_str_output"],
                    layers_size=self.parent.layers_decoder,
                    dropout_keep_prob=self.parent.dropout_keep_prob,
                )
                self.DECODER = decoder.layers

            elif self.parent.AE_convolutional:
                if self.parent.channels == 1:
                    decoder = getDecoderModel(self.parent, encoder, self.parent.mode)
                elif self.parent.channels == 2:
                    assert False, "2D Convolutional AE not implemented"
                    decoder = conv_mlp_model_2D.getDecoderModel(self.parent, encoder)
                else:
                    raise ValueError("Not implemented.")
                self.DECODER = decoder.layers
        elif self.parent.has_dummy_autoencoder:
            self.DECODER = nn.ModuleList()
            self.DECODER.append(viewAddChannels(self.parent.params))
        else:
            self.DECODER = nn.ModuleList()


        self.module_list = [self.ENCODER, self.BETA_VAE, self.DECODER, self.RNN, self.RNN_OUTPUT]

        if self.parent.has_autoencoder and self.parent.AE_convolutional:
            # assert False, "Convolutional AE not implemented"
            latent_state_dim = sizes[-1]
            iter_ = 1
            for dim in latent_state_dim:
                iter_ *= dim
            latent_state_dim = iter_
            self.parent.latent_state_dim = latent_state_dim
    

    def forward(
        self,
        inputs, # The input to the RNN
        init_hidden_state, # The initial hidden state
        is_train=False, # Whether it is training or evaluation
        is_iterative_forecasting=False, # Whether to feed the predicted output back in the input (iteratively)
        iterative_forecasting_prob=0,
        iterative_forecasting_gradient=1,
        horizon=None, # The iterative forecasting horizon
        input_is_latent=False, # Whether the input is latent state/original state
        iterative_propagation_is_latent=False, # To iteratively propagate the latent state or the output
        detach_output=False,
    ):

        if is_iterative_forecasting:
            return self.forecast(
                inputs,
                init_hidden_state,
                horizon,
                is_train=is_train,
                iterative_forecasting_prob=iterative_forecasting_prob,
                iterative_forecasting_gradient=iterative_forecasting_gradient,
                input_is_latent=input_is_latent,
                iterative_propagation_is_latent=iterative_propagation_is_latent,
            )
        else:
            assert (input_is_latent == iterative_propagation_is_latent)
            return self.forward_(
                inputs,
                init_hidden_state,
                is_train=is_train,
                is_latent=input_is_latent,
                detach_output=detach_output,
            )


    def forecast(
        self,
        inputs,
        init_hidden_state,
        horizon=None,
        is_train=False,
        iterative_forecasting_prob=0,
        iterative_forecasting_gradient=1,
        iterative_propagation_is_latent=False,
        input_is_latent=False,
    ):
    
        # Set Network State: train or eval
        if is_train:
            self.train()
        else:
            self.eval()
        if self.parent.has_rnn and self.parent.train_RNN_only:
            self.setBatchNormalizationLayersToEvalMode()

        # Guarantee the propagated object: latent/original state
        assert input_is_latent == iterative_propagation_is_latent, "input_is_latent and not iterative_propagation_is_latent, Not implemented!"

        with torch.set_grad_enabled(is_train):

            # inputs is either the inputs of the encoder or the latent state when input_is_latent=True
            outputs = []
            inputs_decoded = []

            latent_states = []
            latent_states_pred = []
            RNN_internal_states = []
            RNN_outputs = []

            inputsize = inputs.size()
            K, T, D = inputsize[0], inputsize[1], inputsize[2]

            if (horizon is not None):
                if (not (K == 1)) or (not (T == 1)):
                    raise ValueError("Forward iterative called with K!=1 or T!=1 and a horizon. This is not allowed! K={:}, T={:}, D={:}".format(K, T, D))
                else:
                    # Horizon is not None and T=1, so forecast called in the testing phase
                    pass
            else:
                horizon = T

            if iterative_forecasting_prob == 0:
                assert T == horizon, "If iterative forecasting, with iterative_forecasting_prob={:}>0, the provided time-steps T cannot be {:}, but have to be horizon={:}.".format(iterative_forecasting_prob, T, horizon)

            # When T>1, only inputs[:,0,:] is taken into account. The network is propagating its own predictions.
            input_t = inputs[:, 0].view(K, 1, *inputs.size()[2:])
            assert (T > 0)
            assert (horizon > 0)
            time_latent_prop = 0.0
            
            for t in range(horizon):
                # BE CAREFULL: input may be the latent input!
                output, next_hidden_state, latent_state, latent_state_pred, RNN_output, input_decoded, time_latent_prop_t, _, _ = self.forward_(
                    input_t,
                    init_hidden_state,
                    is_train=is_train,
                    is_latent=input_is_latent)
                time_latent_prop += time_latent_prop_t

                # TODO: iterative_forecasting_prob决定前向推理时，依概率决定，迭代的是输入的真实数据还是自身推理的中间预测值
                # Settting the next input if t < horizon - 1
                if t < horizon - 1:
                    if iterative_forecasting_prob > 0.0:
                        # Iterative forecasting:
                        # with probability iterative_forecasting_prob propagate the state
                        # with probability (1-iterative_forecasting_prob) propagate the data
                        temp = torch.rand(1).data[0].item()
                    else:
                        temp = 0.0

                    if temp < (1 - iterative_forecasting_prob):
                        # with probability (1-iterative_forecasting_prob) propagate the data
                        input_t = inputs[:,t + 1].view(K, 1, *inputs.size()[2:])
                        input_is_latent = False
                    else:
                        # with probability iterative_forecasting_prob propagate the state
                        if iterative_propagation_is_latent:
                            # Changing the propagation to latent
                            input_is_latent = True

                            if iterative_forecasting_gradient:
                                # Forecasting the prediction as a tensor in graph
                                input_t = latent_state_pred
                            else:
                                # Deatching, and propagating the prediction as data
                                input_t = latent_state_pred.detach()
                        
                        else:
                            if iterative_forecasting_gradient:
                                # Forecasting the prediction as a tensor in graph
                                if 'only_inhibitor' in self.parent.mode:
                                    input_t = output[:,:,1].unsqueeze(-2)
                                elif 'only_activator' in self.parent.mode:
                                    input_t = output[:,:,0].unsqueeze(-2)
                                else:
                                    input_t = output
                            else:
                                # Deatching, and propagating the prediction as data
                                if 'only_inhibitor' in self.parent.mode:
                                    input_t = Variable(output[:,:,1].unsqueeze(-2).data)
                                elif 'only_activator' in self.parent.mode:
                                    input_t = Variable(output[:,:,0].unsqueeze(-2).data)
                                else:
                                    input_t = Variable(output.data)

                outputs.append(output[:, 0])
                inputs_decoded.append(input_decoded[:, 0])

                latent_states.append(latent_state[:, 0])
                latent_states_pred.append(latent_state_pred[:, 0])
                RNN_internal_states.append(next_hidden_state)
                RNN_outputs.append(RNN_output[:, 0])

                init_hidden_state = next_hidden_state

            outputs = torch.stack(outputs)
            outputs = outputs.transpose(1, 0)
            inputs_decoded = torch.stack(inputs_decoded)
            inputs_decoded = inputs_decoded.transpose(1, 0)

            latent_states = torch.stack(latent_states)
            latent_states_pred = torch.stack(latent_states_pred)
            RNN_outputs = torch.stack(RNN_outputs)

            latent_states = latent_states.transpose(1, 0)
            latent_states_pred = latent_states_pred.transpose(1, 0)
            RNN_outputs = RNN_outputs.transpose(1, 0)

        # Two additional dummy outputs for the case of Beta-VAE
        return outputs, next_hidden_state, latent_states, latent_states_pred, RNN_outputs, inputs_decoded, time_latent_prop, None, None


    def forward_(
        self,
        inputs,
        init_hidden_state,
        is_train=True,
        is_latent=False,
        detach_output=False,
    ):

        # TRANSPOSE FROM BATCH FIRST TO LAYER FIRST
        if self.parent.has_rnn:
            init_hidden_state = self.transposeHiddenState(init_hidden_state)

        # Set Network State: train or eval
        if is_train:
            self.train()
        else:
            self.eval()
        if self.parent.has_rnn and self.parent.train_RNN_only:
            self.setBatchNormalizationLayersToEvalMode()

        time_latent_prop = 0.0 # Time spent in propagation of the latent state

        with torch.set_grad_enabled(is_train):

            inputsize = inputs.size()
            K, T, D = inputsize[0], inputsize[1], inputsize[2]

            # Swapping the inputs to RNN [K,T,LD]->[T, K, LD] (time first) LD=latent dimension
            inputs = inputs.transpose(1, 0)

            if (K != self.parent.batch_size and is_train == True and (not self.parent.device_count > 1)):
                raise ValueError("Batch size {:d} does not match {:d} and model not in multiple GPUs.".format(K, self.parent.batch_size))

            ########################
            # ENCODER
            ########################
            if (self.parent.has_autoencoder or self.parent.has_dummy_autoencoder) and not is_latent:
                if D != self.parent.input_dim:
                    if 'only_inhibitor' in self.parent.mode:
                        inputs = inputs[:,:,1].unsqueeze(-2)
                    elif 'only_activator' in self.parent.mode:
                        inputs = inputs[:,:,0].unsqueeze(-2)
                    else:
                        raise ValueError("Input dimension D={:d} does not match self.parent.input_dim={:d}.".format(D, self.parent.input_dim))
                # Forward the encoder only in the original space
                encoder_output = self.forwardEncoder(inputs)
            else:
                encoder_output = inputs

            if detach_output:
                del inputs
                encoder_output = encoder_output.detach()

            ########################
            # BETA VAE
            ########################
            if self.parent.beta_vae:
                latent_state_shape = encoder_output.size()
                n_dims = len(list(encoder_output.size()))
                encoder_output_flat = encoder_output.flatten(start_dim=2, end_dim=n_dims - 1)
                z, beta_vae_mu, beta_vae_logvar = self.BETA_VAE[0](encoder_output_flat)

                if (not is_train) or self.parent.has_rnn:
                    # In case of testing, or RNN training, use the mean value
                    decoder_input = torch.reshape(beta_vae_mu, latent_state_shape)
                else:
                    decoder_input = torch.reshape(z, latent_state_shape)

            else:
                decoder_input = encoder_output
                beta_vae_mu = None
                beta_vae_logvar = None

            latent_states = encoder_output

            ########################
            # DECODER
            ########################
            if not detach_output:
                if self.parent.has_autoencoder or self.parent.has_dummy_autoencoder:
                    inputs_decoded = self.forwardDecoder(decoder_input)
                else:
                    inputs_decoded = decoder_input
            else:
                inputs_decoded = []

            ########################
            # RNN
            ########################
            if self.parent.has_rnn:
                time_start = time.time()

                if not self.parent.beta_vae:
                    rnn_input = encoder_output
                else:
                    rnn_input = beta_vae_mu

                if self.has_latent_scaler:
                    rnn_input = self.scaleLatentState(rnn_input)

                # Latent states are the autoencoded states BEFORE being past through the RNN
                RNN_outputs, next_hidden_state = self.forwardRNN(rnn_input, init_hidden_state, is_train)

                # Output of the RNN passed through MLP
                latent_states_pred = self.forwardRNNOutput(RNN_outputs)

                if self.has_latent_scaler:
                    latent_states_pred = self.descaleLatentState(latent_states_pred)

                time_latent_prop += (time.time() - time_start)

                # The predicted latent states are the autoencoded states AFTER being past through the RNN, before beeing decoded
                decoder_input_pred = latent_states_pred

                # TRANSPOSING BATCH_SIZE WITH TIME
                latent_states_pred = latent_states_pred.transpose(1, 0).contiguous()
                RNN_outputs = RNN_outputs.transpose(1, 0).contiguous()

                # TRANSPOSE BACK FROM LAYER FIRST TO BATCH FIRST
                next_hidden_state = self.transposeHiddenState(next_hidden_state)

                # Output of the RNN (after the MLP) has dimension: [T, K, latend_dim]

                if not detach_output:
                    if self.parent.has_autoencoder or self.parent.has_dummy_autoencoder:
                        outputs = self.forwardDecoder(decoder_input_pred)
                        outputs = outputs.transpose(1, 0).contiguous()
                    else:
                        outputs = latent_states_pred
                else:
                    outputs = []

            else:
                outputs = []
                RNN_outputs = []
                latent_states_pred = []
                next_hidden_state = []

            latent_states = latent_states.transpose(1, 0).contiguous()
            if not detach_output:
                inputs_decoded = inputs_decoded.transpose(1, 0).contiguous()
        
        return outputs, next_hidden_state, latent_states, latent_states_pred, RNN_outputs, inputs_decoded, time_latent_prop, beta_vae_mu, beta_vae_logvar
    

    def forwardEncoder(self, inputs):
        
        # PROPAGATING THROUGH THE ENCODER TO GET THE LATENT STATE
        outputs = inputs
        shape_ = outputs.size()
        T, K = shape_[0], shape_[1]

        outputs = torch.reshape(outputs, (T * K, *shape_[2:]))
        
        for l in range(len(self.ENCODER)):
            outputs = self.ENCODER[l](outputs)
        shape_ = outputs.size()
        
        outputs = torch.reshape(outputs, (T, K, *shape_[1:]))
        
        return outputs
    

    def forwardDecoder(self, inputs):
        
        # Dimension of inputs: [T, K, latend_dim + noise_dim]
        outputs = inputs

        shape_ = outputs.size()
        T, K = shape_[0], shape_[1]
        outputs = torch.reshape(outputs, (T * K, *shape_[2:]))
        
        for l in range(len(self.DECODER)):
            outputs = self.DECODER[l](outputs)
        shape_ = outputs.size()
        outputs = torch.reshape(outputs, (T, K, *shape_[1:]))

        return outputs
    

    def forwardRNNOutput(self, inputs):
        
        outputs = []
        for input_t in inputs:
            output = input_t
            for l in range(len(self.RNN_OUTPUT)):
                output = self.RNN_OUTPUT[l](output)
            outputs.append(output)
        outputs = torch.stack(outputs)
        
        return outputs
    

    def forwardRNN(self, inputs, init_hidden_state, is_train):
        
        # The inputs are the latent_states
        T = inputs.size()[0]
        RNN_outputs = []
        for t in range(T):
            input_t = inputs[t]
            next_hidden_state = []
            for ln in range(len(self.RNN)):
                hidden_state = init_hidden_state[ln]

                if self.parent.params["RNN_cell_type"] == "lstm" and not self.parent.RNN_convolutional:
                    hidden_state = self.transform2Tuple(init_hidden_state[ln])

                RNN_output, next_hidden_state_layer = self.RNN[ln].forward(input_t, hidden_state, is_train=is_train)

                if self.parent.params["RNN_cell_type"] == "lstm" and not self.parent.RNN_convolutional:
                    hx, cx = next_hidden_state_layer
                    next_hidden_state_layer = torch.stack([hx, cx])

                next_hidden_state.append(next_hidden_state_layer)
                input_t = RNN_output

            init_hidden_state = next_hidden_state
            RNN_outputs.append(RNN_output)

        RNN_outputs = torch.stack(RNN_outputs)
        next_hidden_state = torch.stack(next_hidden_state)

        return RNN_outputs, next_hidden_state
    

    def transform2Tuple(self, hidden_state):
        
        hx, cx = hidden_state
        hidden_state = tuple([hx, cx])
        return hidden_state


    def transposeHiddenState(self, hidden_state):

        # Transpose hidden state from batch_first to Layer first
        # (gru)  [K, L, H]    -> [L, K, H]
        # (lstm) [K, 2, L, H] -> [L, 2, K, H]
        if self.parent.params["RNN_cell_type"] == "gru":
            hidden_state = hidden_state.transpose(0, 1)  # gru
        elif "lstm" in self.parent.params["RNN_cell_type"]:
            hidden_state = hidden_state.transpose(0, 2)  # lstm
        elif self.parent.params["RNN_cell_type"] == "mlp":
            pass
        else:
            raise ValueError("RNN_cell_type {:} not recognized".format(self.parent.params["RNN_cell_type"]))
        return hidden_state
    

    def scaleLatentState(self, latent_state):

        K, T, D = latent_state.size()

        if self.parent.params["latent_space_scaler"] == "MinMaxZeroOne":
            latent_state_min = self.repeatTensor(self.latent_state_min, K, T)
            latent_state_max = self.repeatTensor(self.latent_state_max, K, T)
            if self.parent.gpu:
                latent_state_min = latent_state_min.cuda()
                latent_state_max = latent_state_max.cuda()
            assert (latent_state.size() == latent_state_min.size())
            assert (latent_state.size() == latent_state_max.size())
            return (latent_state - latent_state_min) / (latent_state_max - latent_state_min)
        
        elif self.parent.params["latent_space_scaler"] == "Standard":
            latent_state_mean = self.repeatTensor(self.latent_state_mean, K, T)
            latent_state_std = self.repeatTensor(self.latent_state_std, K, T)
            if self.parent.gpu:
                latent_state_mean = latent_state_mean.cuda()
                latent_state_std = latent_state_std.cuda()
            assert (latent_state.size() == latent_state_mean.size())
            assert (latent_state.size() == latent_state_std.size())
            return (latent_state - latent_state_mean) / latent_state_std
        
        else:
            raise ValueError("Invalid latent_space_scaler {:}".format(self.parent.params["latent_space_scaler"]))
    

    def descaleLatentState(self, latent_state):

        K, T, D = latent_state.size()
        if self.parent.params["latent_space_scaler"] == "MinMaxZeroOne":
            assert self.parent.RNN_activation_str_output == "tanhplus", "Latent space scaler is {:}, while activation at the output of the RNN is {:}.".format(self.parent.params["latent_space_scaler"], self.parent.RNN_activation_str_output)
            latent_state_min = self.repeatTensor(self.latent_state_min, K, T)
            latent_state_max = self.repeatTensor(self.latent_state_max, K, T)
            if self.parent.gpu:
                latent_state_min = latent_state_min.cuda()
                latent_state_max = latent_state_max.cuda()

            assert (latent_state.size() == latent_state_min.size())
            assert (latent_state.size() == latent_state_max.size())
            return latent_state * (latent_state_max - latent_state_min) + latent_state_min
        elif self.parent.params["latent_space_scaler"] == "Standard":
            assert self.parent.RNN_activation_str_output == "identity", "Latent space scaler is {:}, while activation at the output of the RNN is {:}.".format(self.parent.params["latent_space_scaler"],self.parent.RNN_activation_str_output)
            latent_state_mean = self.repeatTensor(self.latent_state_mean, K, T)
            latent_state_std = self.repeatTensor(self.latent_state_std, K, T)
            if self.parent.gpu:
                latent_state_mean = latent_state_mean.cuda()
                latent_state_std = latent_state_std.cuda()
            assert (latent_state.size() == latent_state_mean.size())
            assert (latent_state.size() == latent_state_std.size())
            return latent_state * latent_state_std + latent_state_mean
        else:
            raise ValueError("Invalid latent_space_scaler {:}".format(self.parent.params["latent_space_scaler"]))
    

    def repeatTensor(self, temp, K, T):
        
        tensor = temp.unsqueeze(0)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.repeat(K, T, 1)
        return tensor

    
    def getParams(self):

        params = list()
        named_params = list()
        for layers in self.module_list:
            for layer in layers:
                params += layer.parameters()
                named_params += layer.named_parameters()
        return params, named_params
    

    def getAutoencoderParams(self):

        params = list()
        named_params = list()
        for layers in [self.ENCODER, self.DECODER]:
            for layer in layers:
                params += layer.parameters()
                named_params += layer.named_parameters()
        return params, named_params

    
    def getRNNParams(self):
        
        params = list()
        named_params = list()
        for layers in [self.RNN, self.RNN_OUTPUT]:
            for layer in layers:
                params += layer.parameters()
                named_params += layer.named_parameters()
        return params, named_params


    def defineLatentStateParams(self):
        import ipdb;ipdb.set_trace()
        if self.parent.params["latent_space_scaler"] == "MinMaxZeroOne":
            assert self.parent.RNN_activation_str_output == "tanhplus", "Latent space scaler is {:}, while activation at the output of the RNN is {:}.".format(self.parent.params["latent_space_scaler"], self.parent.RNN_activation_str_output)
        elif self.parent.params["latent_space_scaler"] == "Standard":
            assert self.parent.RNN_activation_str_output == "identity", "Latent space scaler is {:}, while activation at the output of the RNN is {:}.".format(self.parent.params["latent_space_scaler"], self.parent.RNN_activation_str_output)
        else:
            raise ValueError("Invalid latent_space_scaler {:}".format(self.parent.params["latent_space_scaler"]))
        self.latent_state_min = torch.nn.Parameter(torch.zeros(self.parent.params["latent_state_dim"]))
        self.latent_state_max = torch.nn.Parameter(torch.zeros(self.parent.params["latent_state_dim"]))
        self.latent_state_mean = torch.nn.Parameter(torch.zeros(self.parent.params["latent_state_dim"]))
        self.latent_state_std = torch.nn.Parameter(torch.zeros(self.parent.params["latent_state_dim"]))
        self.latent_state_min.requires_grad = False
        self.latent_state_max.requires_grad = False
        self.latent_state_mean.requires_grad = False
        self.latent_state_std.requires_grad = False
    

    def initializeWeights(self):

        for modules in self.module_list:
            for module in modules:
                for name, param in module.named_parameters(): # Initializing RNN, GRU, LSTM cells
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif self._ifAnyIn(["Wxi.weight", "Wxf.weight", "Wxc.weight", "Wxo.weight", "Bci", "Bcf", "Bco", "Bcc",], name):
                        torch.nn.init.xavier_uniform_(param.data)
                    elif self._ifAnyIn(["Wco", "Wcf", "Wci", "Whi.weight", "Whf.weight", "Whc.weight", "Who.weight"], name):
                        torch.nn.init.orthogonal_(param.data)
                    elif self._ifAnyIn(["Whi.bias", "Wxi.bias", "Wxf.bias", "Whf.bias", "Wxc.bias", "Whc.bias", "Wxo.bias", "Who.bias"], name):
                        param.data.fill_(0.001) # param.data.fill_(0)
                    elif 'weight' in name:
                        ndim = len(list(param.data.size()))
                        if ndim > 1:
                            torch.nn.init.xavier_uniform_(param.data)
                        else:
                            print("[crnn_model] Module {:}, Params {:} default initialization.".format(module, name))
                    elif 'bias' in name:
                        param.data.fill_(0.001)
                    elif 'initial_hidden_state' in name:
                        param.data.fill_(0.00001)
                    else:
                        raise ValueError("[crnn_model] NAME {:} NOT FOUND!".format(name))
    

    def setLatentStateBounds(self, min_=None, max_=None, mean_=None, std_=None, slack=0.05):
        
        latent_state_min = self.parent.torch_dtype(min_)
        latent_state_max = self.parent.torch_dtype(max_)

        latent_state_mean = self.parent.torch_dtype(mean_)
        latent_state_std = self.parent.torch_dtype(std_)

        range_ = latent_state_max - latent_state_min
        latent_state_min = latent_state_min - slack * range_
        latent_state_max = latent_state_max + slack * range_
        del self.latent_state_min
        del self.latent_state_max
        del self.latent_state_mean
        del self.latent_state_std

        self.latent_state_min = torch.nn.Parameter(latent_state_min)
        self.latent_state_max = torch.nn.Parameter(latent_state_max)
        self.latent_state_mean = torch.nn.Parameter(latent_state_mean)
        self.latent_state_std = torch.nn.Parameter(latent_state_std)
        self.latent_state_min.requires_grad = False
        self.latent_state_max.requires_grad = False
        self.latent_state_mean.requires_grad = False
        self.latent_state_std.requires_grad = False
    

    def augmentRNNwithTrainableInitialHiddenState(self):
        for ln in range(len(self.RNN)):
            hidden_channels = self.RNN[ln].hidden_channels
            self.RNN[ln].initial_hidden_state = self._RNNInitHidStateLayer(hidden_channels)
    

    def eval(self):
        for modules in [self.ENCODER, self.BETA_VAE, self.DECODER, self.RNN, self.RNN_OUTPUT,]:
            for layer in modules:
                layer.eval()


    def train(self):
        for modules in [self.ENCODER, self.BETA_VAE, self.DECODER, self.RNN, self.RNN_OUTPUT,]:
            for layer in modules:
                layer.train()
    

    def setBatchNormalizationLayersToEvalMode(self):

        for layer in self.ENCODER:
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()
            elif isinstance(layer, nn.modules.Dropout):
                layer.eval()

        for layer in self.DECODER:
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()
            elif isinstance(layer, nn.modules.Dropout):
                layer.eval()
    

    def _ifAnyIn(self, list_, name):
        for element in list_:
            if element in name:
                return True
        return False
    

    def _RNNInitHidStateLayer(self, hidden_units):
        
        if not self.parent.RNN_convolutional:
            
            if self.parent.params["RNN_cell_type"] == "lstm":
                hx = torch.nn.Parameter(data=torch.zeros(hidden_units), requires_grad=True)
                cx = torch.nn.Parameter(data=torch.zeros(hidden_units), requires_grad=True)
                hidden_state = torch.stack([hx, cx])
                return hidden_state
            
            elif self.parent.params["RNN_cell_type"] == "gru":
                return torch.nn.Parameter(data=torch.zeros(hidden_units), requires_grad=True)
            
            elif self.RNN_cell_type == "mlp":
                return torch.zeros(1)
            
            else:
                raise ValueError("Unknown cell type {}.".format(self.parent.params["RNN_cell_type"]))
        else:
            assert False, "Convolutional RNN not implemented"
            if self.parent.params["RNN_cell_type"] == "lstm":
                hx = torch.nn.Parameter(data=self.getZeroState(hidden_units),requires_grad=True)
                cx = torch.nn.Parameter(data=self.getZeroState(hidden_units),requires_grad=True)
                hidden_state = torch.stack([hx, cx])
                return hidden_state
            elif self.parent.params["RNN_cell_type"] == "gru":
                return torch.nn.Parameter(data=self.getZeroState(hidden_units),requires_grad=True)
            else:
                raise ValueError("Unknown cell type {}.".format(self.parent.params["RNN_cell_type"]))