#!/bin/bash

# System
mode=all
system_name=FHN
CUDA_DEVICES=0
cudnn_benchmark=1
random_seed=114
random_seed_in_name=1
random_seed_in_AE_name=$random_seed

# Training
train_RNN_only=0
load_trained_AE=0
retrain=0
max_epochs=50
max_rounds=4
overfitting_patience=10
batch_size=16
learning_rate=0.001
weight_decay=0.0
dropout_keep_prob=0.99
noise_level=0.0
optimizer_str=adabelief
iterative_loss_validation=0
iterative_loss_schedule_and_gradient=none
reconstruction_loss=1
output_forecasting_loss=1
latent_forecasting_loss=1

# Testing
num_test_ICS=100

# Log
plotting=1
make_videos=0
write_to_log=0
plot_testing_ics_examples=1

# Input
Dx=101
channels=1
input_dim=2
truncate_data_batches=128

# Model
model_name=AE-LSTM
activation_str_general=relu
activation_str_output=tanh
scaler=MinMaxZeroOne
latent_space_scaler=Standard
latent_state_dim=4
sequence_length=10 # number of time step in data_input
prediction_length=10 # less than sequence_length then invalid
prediction_horizon=8000
iterative_propagation_during_training_is_latent=1

# AE
AE_name=AE
AE_convolutional=0
AE_batch_norm=0
AE_conv_transpose=0
AE_pool_type="avg"
AE_conv_architecture=conv_latent_1

# Vae
VAE_name=VAE
beta_vae=0
beta_vae_weight_max=1.0

# RNN
RNN_name=LSTM
RNN_cell_type="lstm"
RNN_layers_num=1
RNN_layers_size=16
RNN_activation_str_output="identity"
n_warmup_train=60
n_warmup=60
c1_latent_smoothness_loss=0
c1_latent_smoothness_loss_factor=0.1

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python crnn.py \
--mode $mode \
--system_name $system_name \
--cudnn_benchmark $cudnn_benchmark \
--write_to_log $write_to_log \
--input_dim $input_dim \
--channels $channels \
--Dx $Dx \
--optimizer_str $optimizer_str \
--beta_vae $beta_vae \
--beta_vae_weight_max $beta_vae_weight_max \
--c1_latent_smoothness_loss $c1_latent_smoothness_loss \
--c1_latent_smoothness_loss_factor $c1_latent_smoothness_loss_factor \
--iterative_loss_validation $iterative_loss_validation \
--iterative_loss_schedule_and_gradient $iterative_loss_schedule_and_gradient \
--output_forecasting_loss $output_forecasting_loss \
--latent_forecasting_loss $latent_forecasting_loss \
--reconstruction_loss $reconstruction_loss \
--activation_str_general $activation_str_general \
--activation_str_output $activation_str_output \
--AE_convolutional $AE_convolutional \
--AE_batch_norm $AE_batch_norm \
--AE_conv_transpose $AE_conv_transpose \
--AE_pool_type $AE_pool_type \
--AE_conv_architecture $AE_conv_architecture \
--train_RNN_only $train_RNN_only \
--load_trained_AE $load_trained_AE \
--RNN_cell_type $RNN_cell_type \
--RNN_layers_num $RNN_layers_num \
--RNN_layers_size $RNN_layers_size \
--RNN_activation_str_output $RNN_activation_str_output \
--latent_state_dim $latent_state_dim  \
--sequence_length $sequence_length \
--prediction_length $prediction_length \
--scaler $scaler \
--learning_rate $learning_rate \
--weight_decay $weight_decay \
--dropout_keep_prob $dropout_keep_prob \
--noise_level $noise_level \
--batch_size $batch_size \
--overfitting_patience $overfitting_patience \
--max_epochs $max_epochs \
--max_rounds $max_rounds \
--num_test_ICS $num_test_ICS \
--prediction_horizon $prediction_horizon \
--display_output 1 \
--random_seed $random_seed \
--random_seed_in_name $random_seed_in_name \
--random_seed_in_AE_name $random_seed_in_AE_name \
--teacher_forcing_forecasting 1 \
--iterative_state_forecasting 1 \
--iterative_latent_forecasting 1 \
--iterative_propagation_during_training_is_latent $iterative_propagation_during_training_is_latent \
--make_videos $make_videos \
--retrain $retrain \
--compute_spectrum 0 \
--plot_state_distributions 0 \
--plot_system 0 \
--plot_latent_dynamics 1 \
--truncate_data_batches $truncate_data_batches \
--plot_testing_ics_examples $plot_testing_ics_examples \
--plotting $plotting \
--test_on_test 1 \
--n_warmup_train $n_warmup_train \
--n_warmup $n_warmup \
--latent_space_scaler $latent_space_scaler \
--AE_name $AE_name \
--model_name $model_name