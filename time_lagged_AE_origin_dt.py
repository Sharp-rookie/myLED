# -*- coding: utf-8 -*-
import os
import time
import torch
from torch import nn
import traceback
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, JoinableQueue
from pytorch_lightning import seed_everything
import warnings;warnings.simplefilter('ignore')

from Data.gillespie import generate_origin
from Data.data_process import time_discretization
from utils import set_cpu_num
from utils.pnas_dataset import PNASDataset
from utils.intrinsic_dimension import eval_id_embedding


class TIME_LAGGED_AE(nn.Module):
    
    def __init__(self, in_channels, input_1d_width, output_1d_width):
        super(TIME_LAGGED_AE, self).__init__()
        
        self.in_channels = in_channels
        
        # TODO: 把embedding形状从（batchsize，64，1）改成（batchsize，64）；另外维度64用传参来指定

        # MLP_encoder_layer, (batchsize,1,3)-->(batchsize,64,1)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels*input_1d_width, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, 64, bias=True),
            nn.Tanh(),
            nn.Unflatten(-1, (64, 1))
        )
        
        # Conv_time-lagged_decoder_layer,(batchsize,64,1)-->(batchsize,1,3)
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, 3, bias=True),
            nn.Tanh(),
            nn.Unflatten(-1, (self.in_channels, int(output_1d_width/self.in_channels)))
        )
        
        def weights_normal_init(m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                torch.nn.init.zeros_(m.bias)
        self.apply(weights_normal_init)

    def forward(self,x):
        
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out, latent


def generate_original_data(trace_num, total_t):

    seed_everything(trace_num+729)
    os.makedirs('Data/origin', exist_ok=True)

    # generate original data by gillespie algorithm
    subprocess = []
    for seed in range(1, trace_num+1):
        if not os.path.exists(f'Data/origin/{seed}/origin.npz'):
            IC = [np.random.randint(0,200), np.random.randint(0,100), np.random.randint(0,5000)]
            subprocess.append(Process(target=generate_origin, args=(total_t, seed, IC), daemon=True))
            subprocess[-1].start()
            print(f'\rStart process[seed={seed}] for origin data' + ' '*30)
        else:
            pass
    while any([subp.exitcode == None for subp in subprocess]):
        pass
    
    # time discretization by time-forward NearestNeighbor interpolate
    subprocess = []
    for seed in range(1, trace_num+1):
        if not os.path.exists(f'Data/origin/{seed}/data.npz'):
            subprocess.append(Process(target=time_discretization, args=(seed, total_t, True), daemon=True))
            subprocess[-1].start()
            print(f'\rStart process[seed={seed}] for time-discrete data' + ' '*30)
    while any([subp.exitcode == None for subp in subprocess]):
        pass

    print(f'save origin data form seed 1 to {trace_num} at Data/origin/')


def generate_dataset(trace_num, tau, sample_num=None, is_print=False):

    if os.path.exists(f"Data/data/tau_{tau}/train.npz") and os.path.exists(f"Data/data/tau_{tau}/val.npz") and os.path.exists(f"Data/data/tau_{tau}/test.npz"):
        return

    # load original data
    if is_print: print('loading original trace data:')
    data = []
    from tqdm import tqdm
    for trace_id in tqdm(range(1, trace_num+1)):
        tmp = np.load(f"Data/origin/{trace_id}/data.npz")
        X = np.array(tmp['X'])[:, np.newaxis]
        Y = np.array(tmp['Y'])[:, np.newaxis]
        Z = np.array(tmp['Z'])[:, np.newaxis]

        trace = np.concatenate((X, Y, Z), axis=1)
        data.append(trace[np.newaxis])
    data = np.concatenate(data, axis=0)

    # subsampling
    dt = tmp['dt']
    subsampling = int(tau/dt) if tau!=0. else 1
    data = data[:, ::subsampling]
    if is_print: print(f'tau[{tau}]', 'data shape', data.shape, '# (trace_num, time_length, feature_num)')

    # save statistic information
    data_dir = f"Data/data/tau_{tau}"
    os.makedirs(data_dir, exist_ok=True)
    np.savetxt(data_dir + "/data_mean.txt", np.mean(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_std.txt", np.std(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_max.txt", np.max(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_min.txt", np.min(data, axis=(0,1)))
    np.savetxt(data_dir + "/tau.txt", [tau]) # Save the timestep

    # single-sample time steps for train
    sequence_length = 2 if tau != 0. else 1

    #######################j
    # Create [train,val,test] dataset
    #######################
    trace_list = {'train':range(500), 'val':range(500,550), 'test':range(550,600)}
    for item in ['train','val','test']:
        
        if os.path.exists(data_dir+f'/{item}.npz'): continue
        
        # select trace num
        N_TRACE = len(trace_list[item])
        data_item = data[trace_list[item]]

        # select sliding window index from 2 trace
        idxs_timestep = []
        idxs_ic = []
        for ic in range(N_TRACE):
            seq_data = data_item[ic]
            idxs = np.arange(0, np.shape(seq_data)[0]-sequence_length, 1)
            for idx_ in idxs:
                idxs_ic.append(ic)
                idxs_timestep.append(idx_)

        # generator item dataset
        sequences = []
        for bn in range(len(idxs_timestep)):
            idx_ic = idxs_ic[bn]
            idx_timestep = idxs_timestep[bn]
            tmp = data_item[idx_ic, idx_timestep:idx_timestep+sequence_length]
            sequences.append(tmp)

        sequences = np.array(sequences) 
        if is_print: print(f'tau[{tau}]', f"original {item} dataset", np.shape(sequences))

        # keep sequences_length equal to sample_num
        if sample_num is not None:
            repeat_num = int(np.floor(N_TRACE*sample_num/len(sequences)))
            idx = np.random.choice(range(len(sequences)), N_TRACE*sample_num-len(sequences)*repeat_num, replace=False)
            idx = np.sort(idx)
            tmp1 = sequences[idx]
            tmp2 = None
            for i in range(repeat_num):
                if i == 0:
                    tmp2 = sequences
                else:
                    tmp2 = np.concatenate((tmp2, sequences), axis=0)
            sequences = tmp1 if tmp2 is None else np.concatenate((tmp1, tmp2), axis=0)
        if is_print: print(f'tau[{tau}]', f"processed {item} dataset", np.shape(sequences))

        # save item dataset
        np.savez(data_dir+f'/{item}.npz', data=sequences)

        # plot
        plt.figure(figsize=(16,10))
        for i in range(3):
            ax = plt.subplot(3,1,i+1)
            ax.set_title(['X','Y','Z'][i])
            plt.plot(sequences[:, 0, i])
        plt.title(f'{item.capitalize()} Data' + f' | sample_num[{len(sequences) if sample_num is None else sample_num}]')
        plt.subplots_adjust(left=0.05, bottom=0.05,  right=0.95,  top=0.95,  hspace=0.35)
        plt.savefig(data_dir+f'/{item}.jpg', dpi=300)


def train_time_lagged_ae(tau, is_print=False):
    
    # prepare
    device = torch.device('cpu')
    data_filepath = 'Data/data/tau_' + str(tau)
    log_dir = 'logs/time-lagged/tau_' + str(tau)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)
    
    # init model
    model = TIME_LAGGED_AE(in_channels=1, input_1d_width=3, output_1d_width=3)
    model.to(device)
    
    # training params
    lr = 0.01
    batch_size = 256
    max_epoch = 50
    weight_decay = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.MSELoss()
    
    # dataset
    train_dataset = PNASDataset(data_filepath, 'train', 'MinMaxZeroOne',)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = PNASDataset(data_filepath, 'val', 'MinMaxZeroOne')
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    # training pipeline
    losses = []
    loss_curve = []
    for epoch in range(1, max_epoch+1):
        
        # train
        model.train()
        for input, target in train_loader:
            output, _ = model.forward(input.to(device))
            
            loss = loss_func(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.detach().item())
            
        loss_curve.append(np.mean(losses))
        
        # validate
        with torch.no_grad():
            targets = []
            outputs = []
            
            model.eval()
            for input, target in val_loader:
                output, _ = model.forward(input.to(device))
                outputs.append(output.cpu().detach())
                targets.append(target.cpu().detach())
                
            targets = torch.concat(targets, axis=0)
            outputs = torch.concat(outputs, axis=0)
            mse = loss_func(outputs, targets)
            if is_print: print(f'\rTau[{tau}] | epoch[{epoch}/{max_epoch}] val-MSE={mse:.5f}', end='')
        
        # save each epoch model
        model.train()
        torch.save(model.state_dict(), log_dir+f"/checkpoints/epoch-{epoch}.ckpt")
        
    # plot loss curve
    plt.figure()
    plt.plot(loss_curve)
    plt.xlabel('epoch')
    plt.title('Train MSELoss Curve')
    plt.savefig(log_dir+'/loss_curve.jpg', dpi=300)
    np.save(log_dir+'/loss_curve.npy', loss_curve)
    

def testing_and_save_embeddings_of_time_lagged_ae(tau, checkpoint_filepath=None, is_print=False):
    
    # prepare
    device = torch.device('cpu')
    data_filepath = 'Data/data/tau_' + str(tau)
    log_dir = 'logs/time-lagged/tau_' + str(tau)
    os.makedirs(log_dir+'/test', exist_ok=True)
    
    # testing params
    batch_size = 256
    max_epoch = 50
    loss_func = nn.MSELoss()
    
    # init model
    model = TIME_LAGGED_AE(in_channels=1, input_1d_width=3, output_1d_width=3)
    
    # dataset
    train_dataset = PNASDataset(data_filepath, 'train', 'MinMaxZeroOne',)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = PNASDataset(data_filepath, 'test', 'MinMaxZeroOne',)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # testing pipeline
    fp = open(log_dir+'/test/log.txt', 'a')
    for ep in range(max_epoch):
        
        # load weight file
        epoch = ep
        if checkpoint_filepath is not None:
            epoch = ep + 1
            ckpt_path = checkpoint_filepath + f"/checkpoints/" + f'epoch-{epoch}.ckpt'
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt)
        model = model.to(device)
        model.eval()
        
        all_embeddings = []
        test_outputs = np.array([])
        test_targets = np.array([])
        train_outputs = np.array([])
        train_targets = np.array([])
        var_log_dir = log_dir + f'/test/epoch-{epoch}'
        os.makedirs(var_log_dir, exist_ok=True)
        
        # testing
        with torch.no_grad():
            
            # train-dataset
            for batch_idx, (input, target) in enumerate(train_loader):
                output, _ = model.forward(input.to(device))
                train_outputs = output.cpu() if not len(train_outputs) else torch.concat((train_outputs, output.cpu()), axis=0)
                train_targets = target.cpu() if not len(train_targets) else torch.concat((train_targets, target.cpu()), axis=0)
                
                if batch_idx >= len(test_loader): break

            # test-dataset
            for input, target in test_loader:
                output, embedding = model.forward(input.to(device))
                # save the embedding vectors
                # TODO: 这里的代码好奇怪？
                for idx in range(input.shape[0]):
                    embedding_tmp = embedding[idx].view(1, -1)[0]
                    embedding_tmp = embedding_tmp.cpu().detach().numpy()
                    all_embeddings.append(embedding_tmp)
                
                test_outputs = output.cpu() if not len(test_outputs) else torch.concat((test_outputs, output.cpu()), axis=0)
                test_targets = target.cpu() if not len(test_targets) else torch.concat((test_targets, target.cpu()), axis=0)
            # test mse
            # TODO: 用沿轴操作，一行搞定x、y、z三个各自的test_mse计算
            mse_x = loss_func(test_outputs[:,0,0], test_targets[:,0,0])
            mse_y = loss_func(test_outputs[:,0,1], test_targets[:,0,1])
            mse_z = loss_func(test_outputs[:,0,2], test_targets[:,0,2])
        
        # plot
        test_plot, train_plot = [[], [], []], [[], [], []]
        for i in range(len(test_outputs)):
            for j in range(len(test_plot)):
                test_plot[j].append([test_outputs[i,0,j], test_targets[i,0,j]])
                train_plot[j].append([train_outputs[i,0,j], train_targets[i,0,j]])
        plt.figure(figsize=(16,9))
        for i, item in enumerate(['test', 'train']):
            for j in range(len(test_plot)):
                ax = plt.subplot(2,3,j+1+3*i)
                ax.set_title(item+'_'+['X','Y','Z'][j])
                plt.plot(np.array(test_plot[j])[:,1], label='true')
                plt.plot(np.array(test_plot[j])[:,0], label='predict')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
        plt.savefig(var_log_dir+"/result.jpg", dpi=300)
        plt.close()

        # save embedding
        np.save(var_log_dir+'/embedding.npy', all_embeddings)

        # calculae ID
        LB_id = cal_id_embedding(tau, epoch, 'MLE')
        # MiND_id = cal_id_embedding(tau, epoch, 'MiND_ML')
        # MADA_id = cal_id_embedding(tau, epoch, 'MADA')
        # PCA_id = cal_id_embedding(tau, epoch, 'PCA')

        # logging
        # fp.write(f"{tau},{random_seed},{mse_x},{mse_y},{mse_z},{epoch},{LB_id},{MiND_id},{MADA_id},{PCA_id}\n")
        fp.write(f"{tau},0,{mse_x},{mse_y},{mse_z},{epoch},{LB_id},0,0,0\n")
        fp.flush()

        if is_print: print(f'\rTau[{tau}] | Test epoch[{epoch}/{max_epoch}]               ', end='')
        
        if checkpoint_filepath is None: break
        
    fp.close()


def cal_id_embedding(tau, epoch, method='MLE', is_print=False):

    var_log_dir = f'logs/time-lagged/tau_{tau}/test/epoch-{epoch}'
    eval_id_embedding(var_log_dir, method=method, is_print=is_print, max_point=100)
    dims = np.load(var_log_dir+f'/id_{method}.npy')

    return np.mean(dims)


def pipeline(trace_num, tau, random_seed, is_print, queue: JoinableQueue):

    time.sleep(1)
    set_cpu_num(1)

    try:
        # generate dataset
        generate_dataset(trace_num=trace_num, tau=tau, sample_num=50, is_print=is_print)
        
        # random seed
        seed_everything(random_seed)

        # untrained net for ID
        testing_and_save_embeddings_of_time_lagged_ae(tau, None, is_print)

        # training
        # if not os.path.exists(f'logs/time-lagged/tau_{tau}/loss_curve.jpg'):
        train_time_lagged_ae(tau, is_print)
        
        # testing and calculating ID
        testing_and_save_embeddings_of_time_lagged_ae(tau, f"logs/time-lagged/tau_{tau}", is_print)
        
        queue.put_nowait([f'Part--{tau}--{random_seed}'])
    
    except:
        if random_seed is None:
            queue.put_nowait([f'Data Generate Error--{tau}', traceback.format_exc()])
        else:
            queue.put_nowait([f'Error--{tau}--{random_seed}', traceback.format_exc()])


if __name__ == '__main__':

    # generate original data
    trace_num = 500+50+50
    total_t = 6
    generate_original_data(trace_num=trace_num, total_t=total_t)

    # start pipeline-subprocess of different tau
    random_seed = 1
    tau_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 3.0]
    queue = JoinableQueue()
    subprocesses = []
    for tau in tau_list:
        tau = round(tau, 5)
        is_print = True if len(subprocesses)==0 else False
        subprocesses.append(Process(target=pipeline, args=(trace_num, tau, random_seed, is_print, queue, ), daemon=True))
        subprocesses[-1].start()
        print(f'Start process[tau={tau}]')

    # join main-process
    os.makedirs('logs/time-lagged/', exist_ok=True)
    log_fp = open(f'logs/time-lagged/log_tau{tau_list[0]}to{tau_list[-1]}.txt', 'w')

    while any([subp.exitcode == None for subp in subprocesses]):
        # listen
        if not queue.empty():
            pkt = queue.get_nowait()
            if 'Part' in pkt[0]:
                tau = float(pkt[0].split("--")[1])
                random_seed = float(pkt[0].split("--")[2])
                log_fp.write(f'Processing[tau={tau}] finish seed {int(random_seed)}\n')
                log_fp.flush()
            elif 'Data' in pkt[0]:
                tau = float(pkt[0].split("--")[1])
                log_fp.write(f'Processing[tau={tau}] error in data-generating\n')
                log_fp.write(str(pkt[1]))
                log_fp.flush()
            elif 'Error' in pkt[0]:
                tau = float(pkt[0].split("--")[1])
                random_seed = float(pkt[0].split("--")[2])
                log_fp.write(f'Processing[tau={tau}] error in seed {int(random_seed)}\n')
                log_fp.write(str(pkt[1]))
                log_fp.flush()
    
    log_fp.close()
    print()