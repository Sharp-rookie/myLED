#--------------------------------PNAS17-------------------------------- 
model=ours
enc_net=MLP
e1_layer_n=2
# enc_net=Conv1d
# e1_layer_n=1
delta=0.2
du=0.0
init_type=circle
sample_type=sliding_window
channel_num=2
xdim=1
clone=5
noise=1
data_dim=$((${xdim}*${clone}))
system=PNAS17_xdim${xdim}_clone${clone}_noise${noise}
obs_dim=$data_dim
trace_num=10000
total_t=0.01
dt=0.001
lr=0.001
batch_size=128
id_epoch=200
learn_epoch=50
# seed_num=10
seed_num=1
tau_unit=0.001
tau_1=0.
tau_N=0.0
tau_s=0.01
embedding_dim=128
slow_dim=1
koopman_dim=1
device=cpu
cpu_num=1
data_dir=Data/${system}_delta${delta}_du${du}_t${total_t}-${init_type}/data/
id_log_dir=logs/${system}_delta${delta}_du${du}-${sample_type}-${init_type}/TimeSelection/
learn_log_dir=logs/${system}_delta${delta}_du${du}-${sample_type}-${init_type}/LearnDynamics/
result_dir=Results/${system}_delta${delta}_du${du}-${sample_type}-${init_type}/
gpu=0


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
--delta $delta \
--du $du \
--init_type $init_type \
--xdim $xdim \
--clone $clone \
--noise $noise \
--parallel \
# --plot_mi \
# --plot_corr \