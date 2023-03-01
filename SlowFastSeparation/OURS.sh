# #--------------------------------2S2F--------------------------------
# model=ours
# system=2S2F
# obs_dim=4
# data_dim=4
# trace_num=200
# total_t=5.1
# dt=0.01
# lr=0.001
# batch_size=128
# id_epoch=100
# learn_epoch=100
# # seed_num=10
# seed_num=1
# tau_unit=0.1
# tau_1=0.
# # tau_N=3.0
# tau_N=1.0
# tau_s=0.8
# embedding_dim=64
# slow_dim=2
# koopman_dim=4
# device=cpu
# cpu_num=1
# data_dir=Data/$system/data/
# id_log_dir=logs/$system/TimeSelection/
# learn_log_dir=logs/$system/LearnDynamics/
# result_dir=Results/$system/
# gpu=1


# #--------------------------------1S2F--------------------------------
# model=ours
# system=1S2F
# obs_dim=3
# data_dim=3
# trace_num=100
# total_t=15.1
# dt=0.01
# lr=0.001
# batch_size=128
# id_epoch=30
# learn_epoch=50
# # seed_num=10
# seed_num=1
# tau_unit=0.3
# tau_1=0.
# # tau_N=7.0
# tau_N=4.5
# tau_s=3.0
# embedding_dim=64
# slow_dim=1
# koopman_dim=4
# device=cpu
# cpu_num=1
# data_dir=Data/$system/data/
# id_log_dir=logs/$system/TimeSelection/
# learn_log_dir=logs/$system/LearnDynamics/
# result_dir=Results/$system/
# gpu=1


# #--------------------------------FHN--------------------------------
# model=ours
# system=FHN
# obs_dim=2
# data_dim=3
# trace_num=100
# total_t=20.1
# dt=0.01
# lr=0.001
# batch_size=128
# id_epoch=100
# learn_epoch=50
# # seed_num=10
# seed_num=2
# tau_unit=0.4
# tau_1=0.
# tau_N=12.0
# tau_s=5.2
# embedding_dim=64
# slow_dim=1
# koopman_dim=4
# device=cpu
# cpu_num=1
# data_dir=Data/$system/data/
# id_log_dir=logs/$system/TimeSelection/
# learn_log_dir=logs/$system/LearnDynamics/
# result_dir=Results/$system/
# gpu=1

#--------------------------------1S1F--------------------------------
model=ours
system=1S1F
obs_dim=2
data_dim=4
trace_num=10
# total_t=12560.
# dt=0.1
total_t=31400.
dt=1.
lr=0.0001
batch_size=1024
id_epoch=50
learn_epoch=50
# seed_num=10
seed_num=1
tau_unit=1.
tau_1=0.
# tau_N=40.
tau_N=20.
tau_s=11.
embedding_dim=64
slow_dim=1
koopman_dim=1
device=cpu
cpu_num=1
data_dir=Data/$system/data/
id_log_dir=logs/$system/TimeSelection/
learn_log_dir=logs/$system/LearnDynamics/
result_dir=Results/$system/
gpu=4


CUDA_VISIBLE_DEVICES=$gpu python run.py \
--model $model \
--system $system \
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
--parallel \
# --plot_mi \
# --plot_corr \