import json
import time
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import warnings;warnings.simplefilter('ignore')

import Utils
import Systems
import plotwork
import testwork
import Nets.crnn_model as crnn_model
import Parser.argparser as argparser
from Multiscale import utils_multiscale


class crnn():

    def __init__(self, params):

        super(crnn, self).__init__()

        ##################################################################
        # SYSTEM SET
        ##################################################################
        self.mode = params['mode']
        self.start_time = time.time()
        self.params = params.copy()
        self.system_name = params["system_name"] # System
        self.save_format = params["save_format"] # Save format: 'pickle, ...'

        # Check device and Set tensor datatype
        self.gpu = torch.cuda.is_available()
        self.gpu_id = params['gpu_id']
        if self.gpu:
            self.torch_dtype = torch.cuda.DoubleTensor
            torch.backends.cudnn.benchmark = True # conv speed-up
            self.device_count = torch.cuda.device_count()
            torch.cuda.set_device(self.gpu_id)
            torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        else:
            self.torch_dtype = torch.DoubleTensor            

        # Random seed
        self.random_seed = params["random_seed"]
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if self.gpu: torch.cuda.manual_seed(self.random_seed)

        ##################################################################
        # PATH SET
        ##################################################################
        # Data path
        self.data_path_train = params['data_path'] + '/train'
        self.data_path_val = params['data_path'] + '/val'
        self.data_path_test = params['data_path'] + '/test'
        self.data_path_gen = params['data_path']

        # Result path
        self.saving_path = params['saving_path']
        self.model_dir = params['model_dir']
        self.fig_dir = params['fig_dir']
        self.results_dir = params['results_dir']
        self.logfile_dir = params["logfile_dir"]
        self.write_to_log = True # Whether to write log
        self.display_output = True # Whether to display in the output


        ##################################################################
        # RNN SET
        ##################################################################
        self.RNN_activation_str = params['RNN_activation_str'] # tanh, ...
        self.RNN_activation_str_output = params['RNN_activation_str_output'] # tanh, ...
        self.RNN_cell_type = params['RNN_cell_type'] # mlp, lstm, gru
        self.zoneout_keep_prob = params["zoneout_keep_prob"] # Zoneout probability for RNN regularizing
        self.sequence_length = params['sequence_length']
        self.prediction_horizon = params["prediction_horizon"] # The prediction horizon
        self.prediction_length = params['prediction_length'] if params['prediction_length'] else self.sequence_length # The prediction length of BPTT
        self.n_warmup_train = params["n_warmup_train"] # The number of warming-up steps during training
        self.noise_level = params['noise_level'] # The noise level in the training data

        # Convolutional RNN set
        self.RNN_convolutional = params['RNN_convolutional'] # Whether use convolutional RNN
        self.RNN_kernel_size = params['RNN_kernel_size']
        self.RNN_layers_size = params['RNN_layers_size']
        self.RNN_trainable_init_hidden_state = params['RNN_trainable_init_hidden_state'] # If the initial state of the RNN is trainable
        self.RNN_statefull = params['RNN_statefull'] # If the RNN is statefull, iteratively propagating the hidden state during training

        # TODO
        self.iterative_propagation_during_training_is_latent = params["iterative_propagation_during_training_is_latent"]
        self.iterative_loss_schedule_and_gradient = params["iterative_loss_schedule_and_gradient"]
        self.iterative_loss_validation = params["iterative_loss_validation"]
        if self.iterative_loss_schedule_and_gradient not in [
                "none",
                "exponential_with_gradient",
                "linear_with_gradient",
                "inverse_sigmoidal_with_gradient",
                "exponential_without_gradient",
                "linear_without_gradient",
                "inverse_sigmoidal_without_gradient",
        ]:
            raise ValueError("Iterative loss schedule {:} not recognized.".format(self.iterative_loss_schedule_and_gradient))
        else:
            if "without_gradient" in self.iterative_loss_schedule_and_gradient or "none" in self.iterative_loss_schedule_and_gradient:
                self.iterative_loss_gradient = False
            elif "with_gradient" in self.iterative_loss_schedule_and_gradient:
                self.iterative_loss_gradient = True
            else:
                raise ValueError("self.iterative_loss_schedule_and_gradient={:} not recognized.".format(self.iterative_loss_schedule_and_gradient))
        
        # The number of IC to test on
        self.num_test_ICS = params["num_test_ICS"]

        ##################################################################
        # AE SET
        ##################################################################

        if 'only_inhibitor' in params['mode'] and params['input_dim']!= 1:
            assert False, "input_dim must be 1 in only_inhibitor mode!"
        elif 'only_activator' in params['mode'] and params['input_dim']!= 1:
            assert False, "input_dim must be 1 in only_activator mode!"
        
        self.input_dim = params['input_dim']
        self.output_dim = params['output_dim']
        self.channels = params['channels']
        self.Dz, self.Dy, self.Dx = Utils.getChannels(self.channels, params)
        self.dropout_keep_prob = params["dropout_keep_prob"] # Dropout probabilities for AE regularizing

        # Convolutional AE Set
        self.AE_convolutional = params["AE_convolutional"] 
        self.AE_batch_norm = params["AE_batch_norm"]
        self.AE_conv_transpose = params["AE_conv_transpose"]
        self.AE_pool_type = params["AE_pool_type"]

        ##################################################################
        # BETA-VAE SET
        ##################################################################
        self.beta_vae = params["beta_vae"]
        self.beta_vae_weight_max = params["beta_vae_weight_max"]

        ##################################################################
        # AE and RNN Layers Set
        ##################################################################
        # Check RNN exist
        self.layers_rnn = [self.params['RNN_layers_size']] * self.params['RNN_layers_num']
        self.has_rnn = len(self.layers_rnn) > 0

        # Adding the AE and RNN latent dimension
        if bool(params["latent_state_dim"]): # micro, both RNN and AE
            if not self.AE_convolutional:
                # encoder layers
                self.layers_encoder = [self.params['AE_layers_size']] * self.params['AE_layers_num']
                self.layers_decoder = self.layers_encoder[::-1]
                self.latent_state_dim = params["latent_state_dim"]
                # rnn layers
                self.RNN_state_dim = self.latent_state_dim
                # decoder layers
                self.decoder_input_dim = self.RNN_state_dim
            
            self.iterative_propagation_during_training_is_latent = params["iterative_propagation_during_training_is_latent"]
            self.has_dummy_autoencoder = False
        
        else: # macro, only RNN and no AE
            self.layers_encoder = []

            if self.RNN_convolutional:
                assert False, "RNN_convolutional not implmented!"
                self.RNN_state_dim = params['input_dim']
                self.has_dummy_autoencoder = False
                assert self.RNN_kernel_size > 0
            else:
                if self.channels == 1:
                    total_dim = params['input_dim'] * self.Dx
                elif self.channels == 2:
                    total_dim = params['input_dim'] * self.Dx * self.Dy
                else:
                    raise ValueError("Not implemented.")
                self.RNN_state_dim = total_dim
                self.has_dummy_autoencoder = True

            self.iterative_propagation_during_training_is_latent = False

        ##################################################################
        # TRAINING PARAMETERS
        ##################################################################
        self.scaler = params["scaler"] # 'MinMaxZeroOne', ...
        self.batch_size = params['batch_size']
        self.optimizer_str = params["optimizer_str"] # 'adam', ...
        self.training_loss = params["training_loss"] # 'mse', ...
        self.max_epochs = params['max_epochs']
        self.max_rounds = params['max_rounds']
        self.learning_rate = params['learning_rate']
        self.lr_reduction_factor = params['lr_reduction_factor']
        self.weight_decay = params['weight_decay']
        self.retrain = params['retrain'] # whether to retrain or not
        self.overfitting_patience = params['overfitting_patience']
        
        # Training object
        self.train_AE_only = params["train_AE_only"]
        self.train_RNN_only = params["train_RNN_only"]
        self.load_trained_AE = params["load_trained_AE"]

        # Loss function type
        self.output_forecasting_loss = params["output_forecasting_loss"] # AE+Vae+RNN
        self.latent_forecasting_loss = params["latent_forecasting_loss"] # RNN
        self.reconstruction_loss = params["reconstruction_loss"] # AE
        self.c1_latent_smoothness_loss = params["c1_latent_smoothness_loss"] # RNN smoothing
        self.c1_latent_smoothness_loss_factor = params["c1_latent_smoothness_loss_factor"]
        self.has_autoencoder = bool(params["latent_state_dim"])
        if self.c1_latent_smoothness_loss:
            assert self.sequence_length > 1, "c1_latent_smoothness_loss cannot be used with sequence_length={:}<1.".format(self.sequence_length)
        if (self.latent_forecasting_loss == True or self.reconstruction_loss == True) and (self.has_autoencoder == False):
            raise ValueError("latent_forecasting_loss and reconstruction_loss are not meaningfull without latent state (Autoencoder mode).")

        ##################################################################
        # BUILD MODEL
        ##################################################################
        self.model = crnn_model.crnn_model(self)
        self.latent_state_dim = params["latent_state_dim"]

        self.model_name = self.params['model_name']
        self.saving_model_path = Utils.getModelDir(self) + "/model.pth"
        Utils.makeDirectories(self) # Make directories

        self.model_parameters, self.model_named_params = self.model.getParams()
        self.model.initializeWeights() # Initialize model parameters

        self.data_info_dict = Systems.getSystemDataInfo(self)
    

    def loadAutoencoderModel(self, in_cpu=False):
        model_name_autoencoder = self.params['AE_name']
        AE_path = self.saving_path + self.model_dir + model_name_autoencoder + "/model.pth"
        
        try:
            if not in_cpu and self.gpu:
                self.model.load_state_dict(torch.load(AE_path),strict=False)
            else:
                self.model.load_state_dict(torch.load(AE_path, map_location=torch.device('cpu')), strict=False)
        except Exception as inst:
            print("[Error] MODEL {:s} NOT FOUND. Are you testing ? Did you already train the autoencoder ? If you run on a cluster, is the GPU detected ? Did you use the srun command ?".format(AE_path))
            raise ValueError(inst)
        
        print("LOADING autoencoder model successfully!")


    def train(self):
        
        # Data
        data_loader_train, dataset_train = Utils.getDataLoader(self.data_path_train, self.data_info_dict, self.batch_size, shuffle=True, gpu=self.gpu, mode=self.mode)
        data_loader_val, dataset_val = Utils.getDataLoader(self.data_path_val, self.data_info_dict, self.batch_size, shuffle=False, gpu=self.gpu, mode=self.mode)

        # Before starting training, scale the latent space
        if self.model.has_latent_scaler:
            self.loadAutoencoderLatentStateLimits()

        # Optimizer
        self.declareOptimizer(self.learning_rate)

        # Load model file
        if self.retrain == 1:
            assert False, "retrain not implemented!"
            print("[crnn] RESTORING pytorch model")
            self.load()
        elif self.train_RNN_only == 1:
            self.loadAutoencoderModel()
            # Saving the initial state
            torch.save(self.model.state_dict(), self.saving_model_path)
        elif self.load_trained_AE == 1:
            self.loadAutoencoderModel()
            # Saving the initial state
            torch.save(self.model.state_dict(), self.saving_model_path)

        self.loss_total_train_vec = []
        self.loss_total_val_vec = []

        self.losses_train_vec = []
        self.losses_val_vec = []

        self.losses_time_train_vec = []
        self.losses_time_val_vec = []

        self.ifp_train_vec = []
        self.ifp_val_vec = []

        self.learning_rate_vec = []

        self.beta_vae_weight_vec = []

        # tqdm
        bar_format = '{desc}{n_fmt:>2s}/{total_fmt:<3s}|{bar}|{postfix}'
        pbar = tqdm(range(self.max_epochs), desc='Epoch', bar_format=bar_format)

        # Termination criterion:
        # If the training procedure completed the maximum number of epochs
        self.epochs_iter = 0
        self.epochs_iter_global = self.epochs_iter
        self.rounds_iter = 0
        while self.epochs_iter < self.max_epochs and self.rounds_iter < self.max_rounds:
            
            ##################################################################
            # TrainRound
            ##################################################################
            
            # learning rate decay
            if self.rounds_iter == 0:
                if not self.retrain:
                    self.learning_rate_round = self.learning_rate
                    self.previous_round_converged = False # Whether last epoch converged (Learning rate decrease criterion)
                else:
                    assert False, "retrain not implemented!"
                    if self.retrain_model_data_found:
                        self.learning_rate_round = self.learning_rate_round
                        self.previous_round_converged = False
            elif self.previous_round_converged == True:
                self.previous_round_converged = False
                self.learning_rate_round = self.learning_rate_round * self.lr_reduction_factor

            # Load best model in last iter and build a optimizer for it
            if self.rounds_iter > 0:
                # Restore the model
                self.model.load_state_dict(torch.load(self.saving_model_path))
                # Optimizer has to be re-declared
                del self.optimizer
                self.declareOptimizer(self.learning_rate_round)

            # TODO: 这里训练是干什么？不是在下面训练吗？先关掉
            # # Train Epoch
            # losses_train, ifp_train, time_train, beta_vae_weight = self.trainEpoch(data_loader_train, is_train=False, dataset=dataset_train)
            # if self.iterative_loss_validation: assert (ifp_train == 1.0)
            # # Validate Epoch
            # losses_val, ifp_val, time_val, beta_vae_weight = self.trainEpoch(data_loader_val, is_train=False, dataset=dataset_val)
            # if self.iterative_loss_validation: assert (ifp_val == 1.0)
            # self.printLosses("TRAIN", losses_train)
            # self.printLosses("VAL  ", losses_val)

            # self.min_val_total_loss = losses_val[0]
            # self.loss_total_train = losses_train[0]
            # RNN_loss_round_val_vec = []
            # RNN_loss_round_val_vec.append(losses_val[0])
            # self.loss_total_train_vec.append(losses_train[0])
            # self.loss_total_val_vec.append(losses_val[0])
            # self.losses_train_vec.append(losses_train)
            # self.losses_time_train_vec.append(time_train)
            # self.losses_val_vec.append(losses_val)
            # self.losses_time_val_vec.append(time_val)

            # Record the best cal loss in last iter
            losses_val, ifp_val, time_val, beta_vae_weight = self.trainEpoch(data_loader_val, is_train=False, dataset=dataset_val)
            self.min_val_total_loss = losses_val[0]
            
            RNN_loss_round_val_vec = []
            for epochs_iter in range(self.epochs_iter, self.max_epochs + 1):

                pbar.set_description(f'\33[36m🌌 Round {self.rounds_iter}/{self.max_rounds}')

                epochs_in_round = epochs_iter - self.epochs_iter
                self.epochs_iter_global = epochs_iter

                losses_train, ifp_train, time_train, beta_vae_weight = self.trainEpoch(data_loader_train, is_train=True, dataset=dataset_train)
                losses_val, ifp_val, time_val, beta_vae_weight = self.trainEpoch(data_loader_val, is_train=False, dataset=dataset_val)
                RNN_loss_round_val_vec.append(losses_val[0])
                self.loss_total_train_vec.append(losses_train[0])
                self.loss_total_val_vec.append(losses_val[0])

                self.losses_train_vec.append(losses_train)
                self.losses_time_train_vec.append(time_train)
                self.losses_val_vec.append(losses_val)
                self.losses_time_val_vec.append(time_val)

                self.ifp_val_vec.append(ifp_val)
                self.ifp_train_vec.append(ifp_train)
                self.beta_vae_weight_vec.append(beta_vae_weight)

                self.learning_rate_vec.append(self.learning_rate_round)

                # Save best model
                if losses_val[0] < self.min_val_total_loss:
                    self.min_val_total_loss = losses_val[0]
                    self.loss_total_train = losses_train[0]
                    torch.save(self.model.state_dict(), self.saving_model_path)

                if epochs_in_round > self.overfitting_patience:
                    # Learning rate decrease criterion:
                    # If the validation loss does not improve for some epochs (self.overfitting_patience)
                    # the round is terminated, the learning rate decreased and training
                    # proceeds in the next round.
                    if all(self.min_val_total_loss < np.array(RNN_loss_round_val_vec[-self.overfitting_patience:])):
                        self.previous_round_converged = True
                        break
                
                self.printLosses("TRAIN", losses_train)
                self.printLosses("VAL  ", losses_val)
                pbar.set_postfix_str(f'[Val loss]  avg:{losses_val[0]:.5f} | E2E:{losses_val[1]:.5f} | RNN:{losses_val[2]:.5f} | AE:{losses_val[3]:.5f} | VAE:{losses_val[4]:.5f} | smooth:{losses_val[5]:.5f}')
                pbar.update()

            self.rounds_iter += 1
            self.epochs_iter = epochs_iter

            pbar.reset()

        # Save model
        if self.epochs_iter == self.max_epochs:
            print("Max epoch reached")
        elif self.rounds_iter == self.max_rounds:
            print("Max round reached")
        self.saveModel()

        if self.params["plotting"]:
            Utils.plotTrainingLosses(self, self.loss_total_train_vec, self.loss_total_val_vec, self.min_val_total_loss)
            Utils.plotAllLosses(self, self.losses_train_vec, self.losses_time_train_vec, self.losses_val_vec, self.losses_time_val_vec, self.min_val_total_loss)
            Utils.plotScheduleLoss(self, self.ifp_train_vec, self.ifp_val_vec)
            Utils.plotScheduleLearningRate(self, self.learning_rate_vec)
            
            if self.beta_vae:
                Utils.plotScheduleKLLoss(self, self.beta_vae_weight_vec)

    
    def trainEpoch(self, data_loader, is_train=False, dataset=None):
        epoch_losses_vec = []

        for batch_of_sequences in data_loader:
            losses, iterative_forecasting_prob, beta_vae_weight = self.trainOnBatch(batch_of_sequences, is_train=is_train, dataset=dataset)
            epoch_losses_vec.append(losses)

        epoch_losses = np.mean(np.array(epoch_losses_vec), axis=0)
        time_ = time.time() - self.start_time

        return epoch_losses, iterative_forecasting_prob, time_, beta_vae_weight

    def trainOnBatch(self, batch_of_sequences, is_train=False, dataset=None):
        
        batch_size = len(batch_of_sequences)
        initial_hidden_states = self.getInitialRNNHiddenState(batch_size)

        if self.data_info_dict["structured"]:
            T = dataset.seq_paths[0]["num_timesteps"]
        else:
            T = np.shape(batch_of_sequences)[1]

        ##################################################################
        # Warm up for each batch
        ##################################################################
        
        predict_on = self.n_warmup_train
        if self.n_warmup_train > 0:
            
            # Setting the optimizer to zero grad
            self.optimizer.zero_grad()

            # Data
            input_batch =  Utils.getDataBatch(self, batch_of_sequences, predict_on - self.n_warmup_train, predict_on, dataset=dataset)
            target_batch =  Utils.getDataBatch(self, batch_of_sequences, predict_on - self.n_warmup_train + 1, predict_on + 1, dataset=dataset)

            # Transform to tensor
            input_batch =  Utils.transform2Tensor(self, input_batch)
            target_batch =  Utils.transform2Tensor(self, target_batch)
            initial_hidden_states =  Utils.transform2Tensor(self, initial_hidden_states)

            # Adding noise to the input data for regularization
            if self.noise_level > 0.0:
                input_batch += self.noise_level * torch.randn_like(input_batch)

            # Warm up
            output_batch, last_hidden_state, latent_states, latent_states_pred, RNN_outputs, input_batch_decoded, time_latent_prop, beta_vae_mu, beta_vae_logvar = self.model.forward(
                input_batch,
                initial_hidden_states,
                is_train=False,
                is_iterative_forecasting=False,
                iterative_forecasting_prob=0,
                iterative_forecasting_gradient=0,
                iterative_propagation_is_latent=False,
                horizon=None,
                input_is_latent=False,
            )
            initial_hidden_states = self.detachHiddenState(last_hidden_state)

        ##################################################################
        # Trainging
        ##################################################################

        num_propagations = int((T - 1 - self.n_warmup_train) / self.sequence_length)
        assert num_propagations >= 1, "Number of propagations int((T - 1 - self.n_warmup_train) / self.sequence_length) = {:} has to be larger than or equal to one. T={:}, self.n_warmup_train={:}, self.sequence_length={:}".format(num_propagations, T, self.n_warmup_train, self.sequence_length)

        losses_vec = []
        predict_on += self.sequence_length
        for p in range(num_propagations):
            
            # Setting the optimizer to zero grad
            self.optimizer.zero_grad()
            
            # Data
            input_batch =  Utils.getDataBatch(self, batch_of_sequences, predict_on - self.sequence_length, predict_on, dataset=dataset)
            target_batch =  Utils.getDataBatch(self, batch_of_sequences, predict_on -self.sequence_length + 1, predict_on + 1, dataset=dataset)

            # Transform to tensor
            input_batch =  Utils.transform2Tensor(self, input_batch)
            target_batch =  Utils.transform2Tensor(self, target_batch)
            initial_hidden_states =  Utils.transform2Tensor(self, initial_hidden_states)

            # Adding noise to the input data for regularization
            if self.noise_level > 0.0:
                input_batch += self.noise_level * torch.randn_like(input_batch)

            # TODO
            if not is_train and self.iterative_loss_validation:
                # set iterative forecasting to True in case of validation
                iterative_forecasting_prob = 1.0
                # Latent iterative propagation is relevant only when: is_iterative_forecasting = True
                iterative_propagation_is_latent = self.iterative_propagation_during_training_is_latent
                is_iterative_forecasting = True
                # ----------------------------------------
                iterative_forecasting_gradient = False
            elif self.iterative_loss_schedule_and_gradient in ["none"]: # only AE, no RNN
                iterative_forecasting_prob = 0.0
                # Latent iterative propagation is relevant only when: is_iterative_forecasting = True
                iterative_propagation_is_latent = False
                is_iterative_forecasting = False
                # ----------------------------------------
                iterative_forecasting_gradient = 0
            elif any(x in self.iterative_loss_schedule_and_gradient for x in ["linear", "inverse_sigmoidal", "exponential"]):
                assert (self.iterative_loss_validation == 1)
                iterative_forecasting_prob = self.getIterativeForecastingProb(self.epochs_iter_global, self.iterative_loss_schedule_and_gradient)
                # Latent iterative propagation is relevant only when: is_iterative_forecasting = True
                iterative_propagation_is_latent = self.iterative_propagation_during_training_is_latent
                is_iterative_forecasting = True
                # ----------------------------------------
                iterative_forecasting_gradient = self.iterative_loss_gradient
            else:
                raise ValueError("self.iterative_loss_schedule_and_gradient={:} not recognized.".format(self.iterative_loss_schedule_and_gradient))
            self.iterative_forecasting_prob = iterative_forecasting_prob

            if self.has_rnn and self.has_autoencoder and self.latent_forecasting_loss and not self.output_forecasting_loss:
                detach_output = True
                del target_batch
            else:
                detach_output = False

            # Forward
            output_batch, last_hidden_state, latent_states, latent_states_pred, RNN_outputs, input_batch_decoded, time_latent_prop, beta_vae_mu, beta_vae_logvar = self.model.forward(
                input_batch,
                initial_hidden_states,
                is_train=is_train,
                is_iterative_forecasting=is_iterative_forecasting,
                iterative_forecasting_prob=iterative_forecasting_prob,
                iterative_forecasting_gradient=iterative_forecasting_gradient,
                iterative_propagation_is_latent=iterative_propagation_is_latent,
                horizon=None,
                input_is_latent=False,
                detach_output=detach_output,
            )

            if detach_output: 
                del input_batch

            # Loss: RNN (with AE, Vae)
            if self.output_forecasting_loss:
                output_batch = output_batch[:, -self.params["prediction_length"]:]
                target_batch = target_batch[:, -self.params["prediction_length"]:]
                loss_fwd = self.getLoss(output_batch, target_batch)
            else:
                loss_fwd = self.torch_dtype([0.0])[0]
            
            if not detach_output:
                if self.has_rnn:
                    assert output_batch.size() == target_batch.size(), "ERROR: Output of network ({:}) does not match with target ({:}).".format(output_batch.size(), target_batch.size())
                else:
                    assert input_batch.size() == input_batch_decoded.size(), "ERROR: Output of DECODER network ({:}) does not match INPUT ({:}).".format(input_batch_decoded.size(), input_batch.size())

            # Loss: Beta-Vae
            if self.beta_vae and not self.has_rnn: # TODO: 为什么有RNN就不能单独算VAE的KL Loss呢？
                loss_kl = self.getKLLoss(beta_vae_mu, beta_vae_logvar)
            else:
                loss_kl = self.torch_dtype([0.0])[0]

            # Loss: RNN
            if self.latent_forecasting_loss:
                outputs = latent_states_pred[:, :-1, :]
                targets = latent_states[:, 1:, :]
                assert outputs.size() == targets.size(), "ERROR: Latent output of network ({:}) does not match with target ({:}).".format(outputs.size(), targets.size())
                outputs = outputs[:, -self.params["prediction_length"]:]
                targets = targets[:, -self.params["prediction_length"]:]
                loss_dyn_fwd = self.getLoss(outputs, targets, is_latent=True,)
            else:
                loss_dyn_fwd = self.torch_dtype([0.0])[0]

            # Loss: AE
            if self.reconstruction_loss:
                loss_auto_fwd = self.getLoss(input_batch_decoded, input_batch)
            else:
                loss_auto_fwd = self.torch_dtype([0.0])[0]

            # Loss: RNN smoothing predict loss(MSE)
            if self.c1_latent_smoothness_loss and not self.has_rnn:
                loss_auto_fwd_c1 = self.getC1Loss(latent_states)
            else:
                loss_auto_fwd_c1 = self.torch_dtype([0.0])[0]

            # Add all loss and backward
            loss_batch = 0.0
            num_losses = 0
            if self.output_forecasting_loss:
                loss_batch += loss_fwd
                num_losses += 1
            if self.latent_forecasting_loss:
                loss_batch += loss_dyn_fwd
                num_losses += 1
            if self.reconstruction_loss:
                loss_batch += loss_auto_fwd
                num_losses += 1
            if self.c1_latent_smoothness_loss and not self.has_rnn:
                loss_auto_fwd_c1 *= self.c1_latent_smoothness_loss_factor
                loss_batch += loss_auto_fwd_c1
                num_losses += 1
            if self.beta_vae and not self.has_rnn:
                beta_vae_weight = self.beta_vae_weight_max * self.getKLLossSchedule(self.epochs_iter_global)
                loss_batch += beta_vae_weight * loss_kl
                num_losses += 1
            else:
                beta_vae_weight = 0.0

            if is_train:
                loss_batch.backward()
                self.optimizer.step()

            # Record all loss
            loss_batch /= num_losses
            loss_batch = loss_batch.cpu().detach().numpy()
            loss_fwd = loss_fwd.cpu().detach().numpy()
            loss_dyn_fwd = loss_dyn_fwd.cpu().detach().numpy()
            loss_auto_fwd = loss_auto_fwd.cpu().detach().numpy()
            loss_kl = loss_kl.cpu().detach().numpy()
            loss_auto_fwd_c1 = loss_auto_fwd_c1.cpu().detach().numpy()
            losses_batch = [loss_batch, loss_fwd, loss_dyn_fwd, loss_auto_fwd, loss_kl, loss_auto_fwd_c1]
            losses_vec.append(losses_batch)

            if self.RNN_statefull:
                # Propagating the hidden state
                last_hidden_state = self.detachHiddenState(last_hidden_state)
                initial_hidden_states = last_hidden_state
            else:
                initial_hidden_states = self.getInitialRNNHiddenState(batch_size)

            predict_on = predict_on + self.sequence_length

        losses = np.mean(np.array(losses_vec), axis=0)

        return losses, iterative_forecasting_prob, beta_vae_weight


    def test(self):

        self.load()
        
        # MODEL LOADED IN EVALUATION MODE
        with torch.no_grad():
            test_on = []
            
            if self.has_rnn:
                self.n_warmup = self.params["n_warmup"]
                assert self.n_warmup > 0
                testing_modes = self.getRNNTestingModes()
            else:
                testing_modes = self.getAutoencoderTestingModes()
            
            test_on = []
            if self.params["test_on_test"]: test_on.append("test")
            if self.params["test_on_val"]: test_on.append("val")
            if self.params["test_on_train"]: test_on.append("train")

            for set_ in test_on:
                print("\n\n[Data Set]: {:}".format(set_))
                testwork.testModesOnSet(self, set_=set_, testing_modes=testing_modes, gpu=self.gpu, mode=self.mode)
    

    def plot(self):
        
        if self.has_rnn:
            testing_modes = self.getRNNTestingModes()
        elif self.has_autoencoder:
            testing_modes = self.getAutoencoderTestingModes()
        if self.write_to_log:
            for testing_mode in testing_modes:
                plotwork.writeLogfiles(self, testing_mode=testing_mode)
        else:
            print("write_to_log = False, no write")
        
        if self.params["plotting"]:
            for testing_mode in testing_modes:
                plotwork.plot(self, testing_mode=testing_mode)
        else:
            print("plotting = False, no plot")
        
    
    def forward(self, input_sequence, init_hidden_state, input_is_latent=False, iterative_propagation_is_latent=False):

        input_sequence = Utils.transform2Tensor(self, input_sequence)
        init_hidden_state = Utils.transform2Tensor(self, init_hidden_state)
        outputs, next_hidden_state, latent_states, latent_states_pred, _, _, time_latent_prop, _, _ = self.model.forward(
            input_sequence,
            init_hidden_state,
            is_train=False,
            is_iterative_forecasting=False,
            iterative_forecasting_prob=0,
            iterative_forecasting_gradient=0,
            horizon=None,
            input_is_latent=input_is_latent,
            iterative_propagation_is_latent=iterative_propagation_is_latent,
        )
        outputs = outputs.detach().cpu().numpy()
        latent_states_pred = latent_states_pred.detach().cpu().numpy()
        latent_states = latent_states.detach().cpu().numpy()
        
        return outputs, next_hidden_state, latent_states, latent_states_pred, time_latent_prop
    

    def forecast(self, input_sequence, hidden_state, horizon):

        input_sequence = Utils.transform2Tensor(self, input_sequence)
        hidden_state = Utils.transform2Tensor(self, hidden_state)
        outputs, next_hidden_state, latent_states, latent_states_pred, _, _, time_latent_prop, _, _ = self.model.forward(
            input_sequence,
            hidden_state,
            is_train=False,
            is_iterative_forecasting=True,
            iterative_forecasting_prob=1.0,
            horizon=horizon,
            iterative_propagation_is_latent=True,
            input_is_latent=True,
        )
        outputs = outputs.detach().cpu().numpy()
        latent_states_pred = latent_states_pred.detach().cpu().numpy()
        latent_states = latent_states.detach().cpu().numpy()
        
        return outputs, next_hidden_state, latent_states, latent_states_pred, time_latent_prop
    

    def getTestingModes(self):
        modes = self.getAutoencoderTestingModes() + self.getRNNTestingModes()
        return modes
    

    def getRNNTestingModes(self):
        
        modes = []
        if self.params["iterative_state_forecasting"]:
            modes.append("iterative_state_forecasting")
        if self.params["iterative_latent_forecasting"]:
            modes.append("iterative_latent_forecasting")
        if self.params["teacher_forcing_forecasting"]:
            modes.append("teacher_forcing_forecasting")
        
        return modes


    def getAutoencoderTestingModes(self):
        return ["autoencoder_testing"]
    

    def encodeDecode(self, input_sequence):

        input_sequence = Utils.transform2Tensor(self, input_sequence)
        initial_hidden_states = self.getInitialRNNHiddenState(len(input_sequence))

        _, _, latent_states, _, _, input_decoded, _, _, _ = self.model.forward(input_sequence, initial_hidden_states, is_train=False)
        input_decoded = input_decoded.detach().cpu().numpy()
        latent_states = latent_states.detach().cpu().numpy()
        
        return input_decoded, latent_states
    

    def predictSequence(self, input_sequence, testing_mode=None, dt=1, prediction_horizon=None):
        
        if prediction_horizon is None:
            prediction_horizon = self.prediction_horizon

        # PREDICTION LENGTH
        N = np.shape(input_sequence)[0]
        if N - self.n_warmup != prediction_horizon:
            raise ValueError("Error! N ({:}) - self.n_warmup ({:}) != prediction_horizon ({:})".format(N, self.n_warmup, prediction_horizon))

        # PREPARING THE HIDDEN STATES
        initial_hidden_states = self.getInitialRNNHiddenState(1)

        if self.has_rnn:
            assert self.n_warmup >= 1, "Warm up steps cannot be < 1 in RNNs. Increase the iterative prediction length."
        elif self.has_predictor:
            assert self.n_warmup == 1, "Warm up steps cannot be != 1 in Predictor."

        warmup_data_input = input_sequence[:self.n_warmup - 1]
        warmup_data_input = warmup_data_input[np.newaxis]
        warmup_data_target = input_sequence[1:self.n_warmup]
        warmup_data_target = warmup_data_target[np.newaxis]

        if testing_mode in self.getRNNTestingModes():
            target = input_sequence[self.n_warmup:self.n_warmup + prediction_horizon]
        else:
            raise ValueError("Testing mode {:} not recognized.".format(testing_mode))

        warmup_data_input = Utils.transform2Tensor(self, warmup_data_input)
        initial_hidden_states = Utils.transform2Tensor(self, initial_hidden_states)

        if self.n_warmup > 1:
            warmup_data_output, last_hidden_state, warmup_latent_states, latent_states_pred, _, _, _, _, _ = self.model.forward(warmup_data_input, initial_hidden_states, is_train=False)
        else:
            last_hidden_state = initial_hidden_states

        prediction = []

        if ("iterative_latent" in testing_mode):
            iterative_propagation_is_latent = 1
            # GETTING THE LAST LATENT STATE (K, T, LD)
            # In iterative latent forecasting, the input is the latent state
            input_latent = latent_states_pred[:, -1, :]
            input_latent.unsqueeze_(0)
            input_t = input_latent
        elif ("iterative_state" in testing_mode):
            iterative_propagation_is_latent = 0
            # LATTENT PROPAGATION
            input_t = input_sequence[self.n_warmup - 1]
            input_t = input_t[np.newaxis, np.newaxis, :]
        elif "teacher_forcing" in testing_mode:
            iterative_propagation_is_latent = 0
            input_t = input_sequence[self.n_warmup - 1:-1]
            input_t = input_t.cpu().detach().numpy()
            input_t = input_t[np.newaxis]
        else:
            raise ValueError("Do not know how to initialize the state for {:}.".format(testing_mode))

        input_t = Utils.transform2Tensor(self, input_t)
        last_hidden_state = Utils.transform2Tensor(self, last_hidden_state)

        if "teacher_forcing" in testing_mode:
            prediction, last_hidden_state, latent_states, latent_states_pred, RNN_outputs, input_decoded, time_latent_prop, _, _ = self.model.forward(
                input_t,
                last_hidden_state,
                is_iterative_forecasting=False,
                horizon=prediction_horizon,
                is_train=False,
                iterative_propagation_is_latent=iterative_propagation_is_latent,
                input_is_latent=False)
        elif "iterative_latent" in testing_mode:
            # LATENT/ORIGINAL DYNAMICS PROPAGATION
            prediction, last_hidden_state, latent_states, latent_states_pred, RNN_outputs, input_decoded, time_latent_prop, _, _ = self.model.forward(
                input_t,
                last_hidden_state,
                is_iterative_forecasting=True,
                iterative_forecasting_prob=1.0,
                horizon=prediction_horizon,
                is_train=False,
                iterative_propagation_is_latent=iterative_propagation_is_latent,
                input_is_latent=iterative_propagation_is_latent,
            )
        elif "iterative_state" in testing_mode:
            # LATENT/ORIGINAL DYNAMICS PROPAGATION
            prediction, last_hidden_state, latent_states, latent_states_pred, RNN_outputs, input_decoded, time_latent_prop, _, _ = self.model.forward(
                input_t,
                last_hidden_state,
                is_iterative_forecasting=True,
                iterative_forecasting_prob=1.0,
                horizon=prediction_horizon,
                is_train=False,
                iterative_propagation_is_latent=iterative_propagation_is_latent,
                input_is_latent=iterative_propagation_is_latent,
            )
        else:
            raise ValueError("Testing mode {:} not recognized.".format(testing_mode))
        
        # Correcting the time-measurement in case of evolution of the original system (in this case, we do not need to internally propagate the latent space of the RNN)
        time_total = time_latent_prop

        time_total_per_iter = time_total / prediction_horizon

        prediction = prediction[0]
        if self.has_rnn: RNN_outputs = RNN_outputs[0]
        latent_states = latent_states[0]

        target = target.cpu().detach().numpy()
        prediction = prediction.cpu().detach().numpy()
        latent_states = latent_states.cpu().detach().numpy()
        if self.has_rnn: RNN_outputs = RNN_outputs.cpu().detach().numpy()

        target = np.array(target)
        prediction = np.array(prediction)
        latent_states = np.array(latent_states)
        if self.has_rnn: RNN_outputs = np.array(RNN_outputs)

        # print("Shapes of prediction/target/latent_states:")
        # print("{:}".format(np.shape(prediction)))
        # print("{:}".format(np.shape(target)))
        # print("{:}".format(np.shape(latent_states)))

        if self.n_warmup > 1:
            warmup_data_target = warmup_data_target.cpu().detach().numpy()
            warmup_data_output = warmup_data_output.cpu().detach().numpy()
            warmup_latent_states = warmup_latent_states.cpu().detach().numpy()

            target_augment = np.concatenate((warmup_data_target[0], target),axis=0)
            prediction_augment = np.concatenate((warmup_data_output[0], prediction), axis=0)
            latent_states_augmented = np.concatenate((warmup_latent_states[0], latent_states), axis=0)
        else:
            target_augment = target
            prediction_augment = prediction
            latent_states_augmented = latent_states

        return prediction, target, prediction_augment, target_augment, latent_states, latent_states_augmented, time_total_per_iter
    

    def load(self, in_cpu=False):
        try:
            if not in_cpu and self.gpu:
                self.model.load_state_dict(torch.load(self.saving_model_path))
            else:
                self.model.load_state_dict(torch.load(self.saving_model_path, map_location=torch.device('cpu')))

        except Exception as inst:
            print("[Error] MODEL {:s} NOT FOUND. Are you testing ? Did you already train the model?".format(self.saving_model_path))
            raise ValueError(inst)

        data_path = Utils.getModelDir(self) + "/data"

        try:
            data = Utils.loadData(data_path, "pickle")
            self.loss_total_train_vec = data["loss_total_train_vec"]
            self.loss_total_val_vec = data["loss_total_val_vec"]
            self.min_val_total_loss = data["min_val_total_loss"]
            self.losses_time_train_vec = data["losses_time_train_vec"]
            self.losses_time_val_vec = data["losses_time_val_vec"]
            self.losses_val_vec = data["losses_val_vec"]
            self.losses_train_vec = data["losses_train_vec"]
            self.losses_labels = data["losses_labels"]
            self.ifp_train_vec = data["ifp_train_vec"]
            self.ifp_val_vec = data["ifp_val_vec"]
            self.learning_rate_vec = data["learning_rate_vec"]
            self.learning_rate_round = data["learning_rate_round"]
            self.beta_vae_weight_vec = data["beta_vae_weight_vec"]
            del data

            self.retrain_model_data_found = True

        except Exception as inst:
            print("[Error (soft)] Model {:s} found. The data from training (result, losses, etc.), however, is missing.".format(self.saving_model_path))
            self.retrain_model_data_found = False
    

    def getLoss(self, output, target, is_latent=False,):

        if "mse" in self.training_loss:
            if self.beta_vae and not self.has_rnn:
                # Sum reduction
                loss = output - target
                loss = loss.pow(2.0)
                # Mean over all dimensions
                loss = loss.sum()
            else:
                # Mean squared loss
                loss = output - target
                loss = loss.pow(2.0)
                # Mean over all dimensions
                loss = loss.mean(2)
                # Mean over all batches
                loss = loss.mean(0)
                # Mean over all time-steps
                loss = loss.mean()
        elif "crossentropy" in self.training_loss:
            # Cross-entropy loss
            cross_entropy = torch.nn.BCELoss(reduction='none')
            loss = cross_entropy(output, target)
            loss = loss.mean()

        return loss
    

    def getKLLoss(self, mu, logvar):

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return kl
    

    def getC1Loss(self, latent_states):
        
        c1_loss = torch.pow(latent_states[:, 1:] - latent_states[:, :-1], 2.0)
        c1_loss = torch.mean(c1_loss)
        
        return c1_loss
    

    def getInitialRNNHiddenState(self, batch_size):

        if self.has_rnn and self.RNN_trainable_init_hidden_state:
            hidden_state = self.model.getRnnHiddenState(batch_size)
        else:
            hidden_state = self.getZeroRnnHiddenState(batch_size)
        
        if (self.torch_dtype == torch.DoubleTensor) or (self.torch_dtype == torch.cuda.DoubleTensor):
            if torch.is_tensor(hidden_state):
                hidden_state = hidden_state.double()
        
        return hidden_state
    

    def getZeroRnnHiddenState(self, batch_size):
        
        if self.has_rnn:
            hidden_state = []
            for ln in self.layers_rnn:
                hidden_state.append(self.getZeroRnnHiddenStateLayer(batch_size, ln))
            hidden_state = torch.stack(hidden_state)
            hidden_state = self.model.transposeHiddenState(hidden_state)
        else:
            hidden_state = []
        
        return hidden_state
    

    def getZeroRnnHiddenStateLayer(self, batch_size, hidden_units):
        
        if self.RNN_cell_type == "mlp": return torch.zeros(1)
        
        if self.RNN_convolutional:
            assert False, "RNN_convolutional not implemented!"
            hx = Variable(self.getZeroState(batch_size, hidden_units))
            if "lstm" in self.params["RNN_cell_type"]:
                cx = Variable(self.getZeroState(batch_size, hidden_units))
                hidden_state = torch.stack([hx, cx])
                return hidden_state
            elif self.params["RNN_cell_type"] == "gru":
                return hx
            else:
                raise ValueError("Unknown cell type {}.".format(
                    self.params["RNN_cell_type"]))
        else:
            hx = Variable(torch.zeros(batch_size, hidden_units))
            
            if "lstm" in self.params["RNN_cell_type"]:
                cx = Variable(torch.zeros(batch_size, hidden_units))
                hidden_state = torch.stack([hx, cx])
                return hidden_state
            elif self.params["RNN_cell_type"] == "gru":
                return hx
            else:
                raise ValueError("Unknown cell type {}.".format(self.params["RNN_cell_type"]))
    

    def detachHiddenState(self, h_state):

        if self.has_rnn:
            return h_state.detach()
        else:
            return h_state
    

    def declareOptimizer(self, lr):
        
        # Choose trainable network parameters
        if self.train_AE_only:
            self.params_trainable, self.params_named_trainable = self.model.getAutoencoderParams()
            if self.has_rnn:
                rnn_params, rnn_named_params = self.model.getRNNParams()
                for name, param in rnn_named_params:
                    param.requires_grad = False
        elif self.train_RNN_only:
            self.params_trainable, self.params_named_trainable = self.model.getRNNParams()
            AE_params, AE_named_params = self.model.getAutoencoderParams()
            for name, param in AE_named_params:
                param.requires_grad = False
        else:
            self.params_trainable = self.model_parameters
            self.params_named_trainable = self.model_named_params

        # Weight decay only when training the autoencoder
        if self.has_rnn and not self.train_AE_only:
            weight_decay = 0.0
            if self.weight_decay > 0:
                print("[crnn] No weight decay in RNN training.")
        else:
            weight_decay = self.weight_decay

        # Define optimizer
        if self.optimizer_str == "adam":
            self.optimizer = torch.optim.Adam(self.params_trainable, lr=lr, weight_decay=weight_decay)
        elif self.optimizer_str == "sgd":
            self.optimizer = torch.optim.SGD(self.params_trainable, lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif self.optimizer_str == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.params_trainable, lr=lr, weight_decay=weight_decay)
        elif self.optimizer_str == "adabelief":
            from adabelief_pytorch import AdaBelief
            self.optimizer = AdaBelief(self.params_trainable, lr=lr, eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=False)
        else:
            raise ValueError("Optimizer {:} not recognized.".format(self.optimizer_str))
    

    def loadAutoencoderLatentStateLimits(self):
        
        model_name_autoencoder = self.params['AE_name']
        AE_results_testing_path = self.saving_path + self.results_dir + model_name_autoencoder + "/results_autoencoder_testing_val"
        try:
            data = Utils.loadData(AE_results_testing_path, "pickle")
        except Exception as inst:
            print("[Error] AE testing results file:\n{:}\nNOT FOUND. Result file from AE testing needed to load the bounds of the latent state.".format(AE_results_testing_path))
            raise ValueError(inst)

        if "latent_state_info" in data.keys():
            latent_state_info = data["latent_state_info"]
        else:
            print("latent bounds not found in AE testing file. Computing them...")
            # Loading the bounds of the latent state
            latent_states_all = data["latent_states_all"]
            latent_state_info = self.computeLatentStateInfo(latent_states_all)
        self.model.setLatentStateBounds(min_=latent_state_info["min"],
                                        max_=latent_state_info["max"],
                                        mean_=latent_state_info["mean"],
                                        std_=latent_state_info["std"])
        del data
    

    def computeLatentStateInfo(self, latent_states_all):
        #########################################################
        # In case of plain CNN (no MLP between encoder-decoder):
        # shape either  (n_ics, T, latent_state, 1, 1)
        # shape or      (n_ics, T, 1, 1, latent_state)
        #########################################################
        # In case of CNN-MLP (encoder-MLP-latent_space-decoder):
        # Shape (n_ics, T, latent_state)
        assert len(np.shape(latent_states_all)) == 3, "np.shape(latent_states_all)={:}".format(np.shape(latent_states_all))
        latent_states_all = np.reshape(latent_states_all, (-1, self.params["latent_state_dim"]))
        min_ = np.min(latent_states_all, axis=0)
        max_ = np.max(latent_states_all, axis=0)
        mean_ = np.mean(latent_states_all, axis=0)
        std_ = np.std(latent_states_all, axis=0)
        latent_state_info = {}
        latent_state_info["min"] = min_
        latent_state_info["max"] = max_
        latent_state_info["mean"] = mean_
        latent_state_info["std"] = std_
        
        return latent_state_info

    
    def printLosses(self, label, losses):
        
        self.losses_labels = ["Total", "FWD", "DYN-FWD", "AUTO-REC", "KL", "C1"] # loss vector include these item
        loss_name = {
            "Total": "Avg", 
            "FWD": "AE+RNN", 
            "DYN-FWD": "RNN", 
            "AUTO-REC": "AE", 
            "KL": "VAE", 
            "C1": "RNN-smooth"
        }
        idx = np.nonzero(losses)[0] # choose non-zero item and print
        to_print = "[{:s}] Loss: ".format(label)
        for i in range(len(idx)):
            to_print += "{:}={:.6} |".format(loss_name[self.losses_labels[idx[i]]], losses[idx[i]])
        # print(to_print)
    

    def saveModel(self):

        self.total_training_time = time.time() - self.start_time
        if hasattr(self, 'loss_total_train_vec'):
            if len(self.loss_total_train_vec) != 0:
                self.training_time = self.total_training_time / len(self.loss_total_train_vec)
            else:
                self.training_time = self.total_training_time
        else:
            self.training_time = self.total_training_time
        print("Training time per epoch is {:}".format(Utils.secondsToTimeStr(self.training_time)))
        print("Total training time is {:}".format(Utils.secondsToTimeStr(self.total_training_time)))
        self.memory = Utils.getMemory()
        print("Script used {:} MB".format(self.memory))
        
        data = {
            "params": self.params,
            "model_name": self.model_name,
            "losses_labels": self.losses_labels,
            "memory": self.memory,
            "total_training_time": self.total_training_time,
            "training_time": self.training_time,
            "loss_total_train_vec": self.loss_total_train_vec,
            "loss_total_val_vec": self.loss_total_val_vec,
            "min_val_total_loss": self.min_val_total_loss,
            "loss_total_train": self.loss_total_train,
            "losses_train_vec": self.losses_train_vec,
            "losses_time_train_vec": self.losses_time_train_vec,
            "losses_val_vec": self.losses_val_vec,
            "losses_time_val_vec": self.losses_time_val_vec,
            "ifp_val_vec": self.ifp_val_vec,
            "ifp_train_vec": self.ifp_train_vec,
            "learning_rate_vec": self.learning_rate_vec,
            "learning_rate": self.learning_rate,
            "learning_rate_round": self.learning_rate_round,
            "beta_vae_weight_vec": self.beta_vae_weight_vec,
        }
        
        fields_to_write = [
            "memory",
            "total_training_time",
            "min_val_total_loss",
        ]
        
        if self.write_to_log == 1:
            logfile_train = Utils.getLogFileDir(self) + "/train.txt"
            Utils.writeToLogFile(self, logfile_train, data, fields_to_write)
        
        data_path = Utils.getModelDir(self) + "/data"
        Utils.saveData(data, data_path, "pickle")


if __name__ == '__main__':
    
    parser = argparser.defineParser().parse_args().__dict__

    # Load shell \params for debug
    # with open('4_args.txt', 'w') as f:
    #     json.dump(parser, f, indent=2)
    # exit(0)
    # with open('4_args.txt', 'r') as f:
    #     parser = json.load(f)

    M = crnn(params=parser)
    print(M.model)

    if parser['mode'] in ['all', 'only_inhibitor', 'only_activator']:
        M.train()
        # M.test()
        # M.plot()

    elif 'multiscale' in parser['mode']:
        multiscale_testing = utils_multiscale.multiscaleTestingClass(M, parser)
        # multiscale_testing.test()
        multiscale_testing.plot()