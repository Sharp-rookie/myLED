#--------------------------------2S2F--------------------------------
model=ours
system=2S2F
enc_net=MLP
e1_layer_n=2
sample_type=static
channel_num=1
data_dim=4
obs_dim=$data_dim
trace_num=200
total_t=5.1
dt=0.01
lr=0.001
batch_size=128
id_epoch=100
learn_epoch=100
# seed_num=10
seed_num=1
tau_unit=0.1
tau_1=0.
tau_N=3.0
tau_s=0.1
embedding_dim=64
slow_dim=2
koopman_dim=$slow_dim
device=cpu
cpu_num=1
data_dir=Data/$system/data/
id_log_dir=logs/$system-$sample_type/TimeSelection/
learn_log_dir=logs/$system-$sample_type/LearnDynamics/
result_dir=Results/$system-$sample_type/
gpu=1


# #--------------------------------1S1F--------------------------------
# model=ours
# system=1S1F
# enc_net=MLP
# e1_layer_n=2
# sample_type=sliding_window
# channel_num=1
# data_dim=2
# obs_dim=$data_dim
# trace_num=100
# total_t=10.1
# dt=0.01
# lr=0.001
# batch_size=128
# id_epoch=30
# learn_epoch=30
# # seed_num=10
# seed_num=1
# tau_unit=0.01
# tau_1=0.
# tau_N=0.1
# tau_s=0.01
# embedding_dim=64
# slow_dim=1
# koopman_dim=4
# device=cpu
# cpu_num=1
# data_dir=Data/$system/data/
# id_log_dir=logs/$system-$sample_type/TimeSelection/
# learn_log_dir=logs/$system-$sample_type/LearnDynamics/
# result_dir=Results/$system-$sample_type/
# gpu=1


# #--------------------------------1S2F--------------------------------
# model=ours
# system=1S2F
# enc_net=MLP
# e1_layer_n=2
# sample_type=static
# channel_num=1
# data_dim=3
# obs_dim=$data_dim
# trace_num=100
# total_t=15.1
# dt=0.01
# lr=0.001
# batch_size=128
# id_epoch=500
# learn_epoch=30
# # seed_num=10
# seed_num=1
# tau_unit=0.3
# tau_1=0.
# tau_N=7.0
# tau_s=3.0
# embedding_dim=64
# slow_dim=1
# koopman_dim=4
# device=cpu
# cpu_num=1
# data_dir=Data/$system/data/
# id_log_dir=logs/$system-$sample_type/TimeSelection/
# learn_log_dir=logs/$system-$sample_type/LearnDynamics/
# result_dir=Results/$system-$sample_type/
# gpu=1


# #--------------------------------FHN--------------------------------
# model=ours
# enc_net=GRU2
# e1_layer_n=3
# sample_type=sliding_window
# channel_num=2
# data_dim=5
# system=FHN_$data_dim
# obs_dim=$data_dim
# trace_num=6
# total_t=800.1
# dt=0.01
# lr=0.001
# batch_size=128
# id_epoch=30
# learn_epoch=50
# # seed_num=10
# seed_num=1
# tau_unit=20.
# tau_1=0.
# tau_N=200.0
# tau_s=5.0
# embedding_dim=64
# slow_dim=1
# koopman_dim=4
# device=cpu
# cpu_num=1
# data_dir=Data/$system/data/
# id_log_dir=logs/$system-$sample_type/TimeSelection/
# learn_log_dir=logs/$system-$sample_type/LearnDynamics/
# result_dir=Results/$system-$sample_type/
# gpu=1

# #--------------------------------HalfMoon--------------------------------
# model=ours
# system=HalfMoon
# enc_net=GRU2
# e1_layer_n=3
# sample_type=sliding_window
# channel_num=1
# data_dim=4
# obs_dim=2
# trace_num=10
# # total_t=12560.
# # dt=0.1
# total_t=31400.
# dt=1.
# lr=0.001
# batch_size=128
# id_epoch=30
# learn_epoch=30
# # seed_num=10
# seed_num=1
# tau_unit=2.
# tau_1=0.
# tau_N=100.
# tau_s=20.
# embedding_dim=64
# slow_dim=2
# koopman_dim=$slow_dim
# device=cpu
# cpu_num=1
# data_dir=Data/$system/data/
# id_log_dir=logs/$system-$sample_type/TimeSelection/
# learn_log_dir=logs/$system-$sample_type/LearnDynamics/
# result_dir=Results/$system-$sample_type/
# gpu=1

# #--------------------------------SC--------------------------------
# model=ours
# system=SC
# enc_net=GRU2
# e1_layer_n=3
# sample_type=static
# channel_num=1
# data_dim=2
# obs_dim=$data_dim
# trace_num=100
# total_t=20.
# dt=0.1
# lr=0.001
# batch_size=128
# id_epoch=50
# learn_epoch=50
# # seed_num=10
# seed_num=1
# tau_unit=0.1
# tau_1=0.
# tau_N=6.28
# tau_s=10.
# embedding_dim=64
# slow_dim=1
# koopman_dim=$slow_dim
# device=cpu
# cpu_num=1
# data_dir=Data/$system/data/
# id_log_dir=logs/$system-$sample_type/TimeSelection/
# learn_log_dir=logs/$system-$sample_type/LearnDynamics/
# result_dir=Results/$system-$sample_type/
# gpu=1


CUDA_VISIBLE_DEVICES=$gpu python run.py \
--model $model \
--system $system \
--enc_net $enc_net \
--e1_layer_n $e1_layer_n \
--channel_num $channel_num \
--obs_dim $obs_dim \
--data_dim $data_dim \
--trace_num $trace_num \
--total_t $total_t \
--dt $dt \
--lr $lr \
--batch_size $batch_size \
--id_epoch $id_epoch \
--learn_epoch $learn_epoch \
--seed_num $seed_num \
--tau_unit $tau_unit \
--tau_1 $tau_1 \
--tau_N $tau_N \
--tau_s $tau_s \
--embedding_dim $embedding_dim \
--slow_dim $slow_dim \
--koopman_dim $koopman_dim \
--device $device \
--cpu_num $cpu_num \
--data_dir $data_dir \
--id_log_dir $id_log_dir \
--learn_log_dir $learn_log_dir \
--result_dir $result_dir \
--sample_type $sample_type \
--parallel \
# --plot_mi \
# --plot_corr \