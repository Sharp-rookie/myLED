#--------------------------------FHN-------------------------------- 
model=ours
enc_net=MLP
e1_layer_n=2
sample_type=static
channel_num=1
data_dim=2
u_max=20
u_min=-20
v_max=20
v_min=-20
grid=50
system=FHN_2d_grid$grid
obs_dim=$data_dim
trace_num=1000
total_t=36.
dt=0.001
lr=0.001
batch_size=128
id_epoch=500
learn_epoch=100
seed_num=1
tau_unit=0.001
tau_1=0.
tau_N=0.05
tau_s=0.03
embedding_dim=64
slow_dim=1
koopman_dim=4
device=cpu
cpu_num=1
data_dir=Data/$system/data/
id_log_dir=logs/$system-$sample_type/TimeSelection/
learn_log_dir=logs/$system-$sample_type/LearnDynamics/
result_dir=Results/$system-$sample_type/
gpu=1


CUDA_VISIBLE_DEVICES=$gpu python run.py \
--model $model \
--system $system \
--enc_net $enc_net \
--e1_layer_n $e1_layer_n \
--channel_num $channel_num \
--obs_dim $obs_dim \
--data_dim $data_dim \
--u_max $u_max \
--u_min $u_min \
--v_max $v_max \
--v_min $v_min \
--grid $grid \
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