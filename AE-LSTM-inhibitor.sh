#!/bin/bash

# Mode
mode="only_inhibitor"
CUDA_DEVICES=0
gpu_id=$CUDA_DEVICES
input_dim=1
num_test_ICS=1
max_epochs=500
max_rounds=30
model_name=AE-LSTM-only_inhibitor-epoch$max_epochs


# ---------------------------------------------- Train ----------------------------------------------


# System
system_name=FHN
cudnn_benchmark=1
random_seed=114
random_seed_in_name=1
random_seed_in_AE_name=$random_seed

# Training
train_RNN_only=0
load_trained_AE=0
retrain=0
overfitting_patience=10
batch_size=32
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

# Log
plot_gif=1
plotting=1
make_videos=0
write_to_log=0
plot_testing_ics_examples=1

# Input
Dx=101
channels=1
output_dim=2
truncate_data_batches=128

# Model
scaler="MinMaxZeroOne"
latent_state_dim=4
sequence_length=40 # number of time step in data_input
prediction_length=40 # less than sequence_length then invalid
prediction_horizon=8000
iterative_propagation_during_training_is_latent=1

# AE
AE_name=AE
AE_layers_size=100
AE_layers_num=3
activation_str_general="celu"
activation_str_output="tanhplus"
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
RNN_layers_size=32
RNN_activation_str="tanh"
RNN_activation_str_output="tanhplus"
latent_space_scaler="Standard"
n_warmup_train=40
n_warmup=40
c1_latent_smoothness_loss=0
c1_latent_smoothness_loss_factor=0.1

# CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python crnn.py \
# --mode $mode \
# --system_name $system_name \
# --cudnn_benchmark $cudnn_benchmark \
# --write_to_log $write_to_log \
# --input_dim $input_dim \
# --output_dim $output_dim \
# --channels $channels \
# --Dx $Dx \
# --optimizer_str $optimizer_str \
# --beta_vae $beta_vae \
# --beta_vae_weight_max $beta_vae_weight_max \
# --c1_latent_smoothness_loss $c1_latent_smoothness_loss \
# --c1_latent_smoothness_loss_factor $c1_latent_smoothness_loss_factor \
# --iterative_loss_validation $iterative_loss_validation \
# --iterative_loss_schedule_and_gradient $iterative_loss_schedule_and_gradient \
# --output_forecasting_loss $output_forecasting_loss \
# --latent_forecasting_loss $latent_forecasting_loss \
# --reconstruction_loss $reconstruction_loss \
# --activation_str_general $activation_str_general \
# --activation_str_output $activation_str_output \
# --AE_convolutional $AE_convolutional \
# --AE_batch_norm $AE_batch_norm \
# --AE_conv_transpose $AE_conv_transpose \
# --AE_pool_type $AE_pool_type \
# --AE_conv_architecture $AE_conv_architecture \
# --train_RNN_only $train_RNN_only \
# --load_trained_AE $load_trained_AE \
# --RNN_cell_type $RNN_cell_type \
# --RNN_layers_num $RNN_layers_num \
# --RNN_layers_size $RNN_layers_size \
# --RNN_activation_str $RNN_activation_str \
# --RNN_activation_str_output $RNN_activation_str_output \
# --latent_state_dim $latent_state_dim  \
# --sequence_length $sequence_length \
# --prediction_length $prediction_length \
# --scaler $scaler \
# --learning_rate $learning_rate \
# --weight_decay $weight_decay \
# --dropout_keep_prob $dropout_keep_prob \
# --noise_level $noise_level \
# --batch_size $batch_size \
# --overfitting_patience $overfitting_patience \
# --max_epochs $max_epochs \
# --max_rounds $max_rounds \
# --num_test_ICS $num_test_ICS \
# --prediction_horizon $prediction_horizon \
# --display_output 1 \
# --random_seed $random_seed \
# --random_seed_in_name $random_seed_in_name \
# --random_seed_in_AE_name $random_seed_in_AE_name \
# --teacher_forcing_forecasting 1 \
# --iterative_state_forecasting 1 \
# --iterative_latent_forecasting 1 \
# --iterative_propagation_during_training_is_latent $iterative_propagation_during_training_is_latent \
# --make_videos $make_videos \
# --retrain $retrain \
# --compute_spectrum 0 \
# --plot_state_distributions 0 \
# --plot_system 0 \
# --plot_latent_dynamics 1 \
# --truncate_data_batches $truncate_data_batches \
# --plot_testing_ics_examples $plot_testing_ics_examples \
# --plotting $plotting \
# --test_on_test 1 \
# --n_warmup_train $n_warmup_train \
# --n_warmup $n_warmup \
# --latent_space_scaler $latent_space_scaler \
# --AE_name $AE_name \
# --AE_layers_size $AE_layers_size \
# --AE_layers_num $AE_layers_num \
# --model_name $model_name



# ---------------------------------------------- Test ----------------------------------------------

mode="only_inhibitor_multiscale"

# Testing
multiscale_testing=1
plot_multiscale_results_comparison=1

# Log
plotting=1
make_videos=0
write_to_log=0
plot_testing_ics_examples=0

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python crnn.py \
--mode $mode \
--system_name $system_name \
--cudnn_benchmark $cudnn_benchmark \
--write_to_log $write_to_log \
--input_dim $input_dim \
--output_dim $output_dim \
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
--RNN_activation_str $RNN_activation_str \
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
--AE_layers_size $AE_layers_size \
--AE_layers_num $AE_layers_num \
--model_name $model_name \
--multiscale_testing 1 \
--plot_gif $plot_gif \
--plot_multiscale_results_comparison 1 \
--multiscale_micro_steps_list 10 \
--multiscale_macro_steps_list 0 \
--multiscale_macro_steps_list 1 \
--multiscale_macro_steps_list 3 \
--multiscale_macro_steps_list 5 \
--multiscale_macro_steps_list 7 \
--multiscale_macro_steps_list 9 \
--multiscale_macro_steps_list 11 \
--multiscale_macro_steps_list 13 \
--multiscale_macro_steps_list 15 \
--multiscale_macro_steps_list 50 \
--multiscale_macro_steps_list 100 \
--multiscale_macro_steps_list 250 \
--multiscale_macro_steps_list 500 \
--multiscale_macro_steps_list 1000 \
--multiscale_macro_steps_list 8000 \