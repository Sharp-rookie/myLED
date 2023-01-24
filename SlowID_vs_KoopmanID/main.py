# -*- coding: utf-8 -*-
import os
import time
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Process
from pytorch_lightning import seed_everything
import warnings;warnings.simplefilter('ignore')

import models
from Data.dataset import JCP12Dataset
from Data.generator import generate_dataset, generate_original_data
from util import set_cpu_num
from util.plot import plot_epoch_test_log
from util.intrinsic_dimension import eval_id_embedding


def train_time_lagged(tau, is_print=False, observation_dim=4, koopman_dim=2, random_seed=729):
    
    # prepare
    device = torch.device('cpu')
    data_filepath = f'Data/data/k_{koopman_dim}/tau_' + str(tau)
    log_dir = f'logs/time-lagged/k_{koopman_dim}/tau_' + str(tau) + f'/seed_{random_seed}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)
    
    # init model
    model = models.TIME_LAGGED_AE(in_channels=1, input_1d_width=observation_dim, embed_dim=64)
    model.apply(models.weights_normal_init)
    model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min.txt").astype(np.float32)).unsqueeze(0)
    model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max.txt").astype(np.float32)).unsqueeze(0)
    model.to(device)
    
    # training params
    lr = 0.005
    batch_size = 128
    max_epoch = 100
    weight_decay = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.MSELoss()

    # dataset
    train_dataset = JCP12Dataset(data_filepath, 'train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataset = JCP12Dataset(data_filepath, 'val')
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # training pipeline
    losses = []
    loss_curve = []
    for epoch in range(1, max_epoch+1):
        
        # train
        model.train()
        for input, target in train_loader:
            input = model.scale(input) # (batchsize,1,1,4)
            target = model.scale(target)
            
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
                input = model.scale(input)
                target = model.scale(target)
            
                output, _ = model.forward(input.to(device))
                outputs.append(output.cpu())
                targets.append(target.cpu())
                
            targets = torch.concat(targets, axis=0)
            outputs = torch.concat(outputs, axis=0)
            mse = loss_func(outputs, targets)
            if is_print: print(f'\rTau[{tau}] | epoch[{epoch}/{max_epoch}] val-MSE={mse:.6f}', end='')
        
        # save each epoch model
        model.train()
        torch.save(
            {'model': model.state_dict(),
             'encoder': model.encoder.state_dict(),}, 
            log_dir+f"/checkpoints/epoch-{epoch}.ckpt")
        
    # plot loss curve
    plt.figure()
    plt.plot(loss_curve)
    plt.xlabel('epoch')
    plt.title('Train MSELoss Curve')
    plt.savefig(log_dir+'/loss_curve.jpg', dpi=300)
    np.save(log_dir+'/loss_curve.npy', loss_curve)
    
    if is_print: print()
    

def test_and_save_embeddings_of_time_lagged(tau, checkpoint_filepath=None, is_print=False, observation_dim=4, koopman_dim=2, random_seed=729):
    
    # prepare
    device = torch.device('cpu')
    data_filepath = f'Data/data/k_{koopman_dim}/tau_' + str(tau)
    log_dir = f'logs/time-lagged/k_{koopman_dim}/tau_' + str(tau) + f'/seed_{random_seed}'
    os.makedirs(log_dir+'/test', exist_ok=True)
    
    # testing params
    batch_size = 128
    max_epoch = 100
    loss_func = nn.MSELoss()
    
    # init model
    model = models.TIME_LAGGED_AE(in_channels=1, input_1d_width=observation_dim, embed_dim=64)
    if checkpoint_filepath is None: # not trained
        model.apply(models.weights_normal_init)
        model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min.txt").astype(np.float32)).unsqueeze(0)
        model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max.txt").astype(np.float32)).unsqueeze(0)

    # dataset
    train_dataset = JCP12Dataset(data_filepath, 'train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_dataset = JCP12Dataset(data_filepath, 'test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # testing pipeline
    fp = open(f'logs/time-lagged/k_{koopman_dim}/tau_{tau}/test_log.txt', 'a')
    for ep in range(max_epoch):
        
        # load weight file
        epoch = ep
        if checkpoint_filepath is not None:
            epoch = ep + 1
            ckpt_path = checkpoint_filepath + f"/checkpoints/" + f'epoch-{epoch}.ckpt'
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['model'])
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
                input = model.scale(input) # (batchsize,1,1,4)
                target = model.scale(target)
                
                output, _ = model.forward(input.to(device))
                
                train_outputs = output.cpu() if not len(train_outputs) else torch.concat((train_outputs, output.cpu()), axis=0)
                train_targets = target.cpu() if not len(train_targets) else torch.concat((train_targets, target.cpu()), axis=0)
                                
                if batch_idx >= len(test_loader): break

            # test-dataset
            for input, target in test_loader:
                input = model.scale(input) # (batchsize,1,1,4)
                target = model.scale(target)
                
                output, embedding = model.forward(input.to(device))
                # save the embedding vectors
                # TODO: 这里的代码好奇怪？
                for idx in range(input.shape[0]):
                    embedding_tmp = embedding[idx].view(1, -1)[0]
                    embedding_tmp = embedding_tmp.cpu().numpy()
                    all_embeddings.append(embedding_tmp)

                test_outputs = output.cpu() if not len(test_outputs) else torch.concat((test_outputs, output.cpu()), axis=0)
                test_targets = target.cpu() if not len(test_targets) else torch.concat((test_targets, target.cpu()), axis=0)
                                
            # test mse
            mse_c1 = loss_func(test_outputs[:,0,0,0], test_targets[:,0,0,0])
            mse_c2 = loss_func(test_outputs[:,0,0,1], test_targets[:,0,0,1])
            mse_c3 = loss_func(test_outputs[:,0,0,2], test_targets[:,0,0,2])
            mse_c4 = loss_func(test_outputs[:,0,0,3], test_targets[:,0,0,3])
        
        # plot
        test_plot, train_plot = [[], [], [], []], [[], [], [], []]
        for i in range(len(test_outputs)):
            for j in range(4):
                test_plot[j].append([test_outputs[i,0,0,j], test_targets[i,0,0,j]])
                train_plot[j].append([train_outputs[i,0,0,j], train_targets[i,0,0,j]])
        plt.figure(figsize=(16,9))
        for i, item in enumerate(['test', 'train']):
            plot_data = test_plot if i == 0 else train_plot
            for j in range(4):
                ax = plt.subplot(2,4,j+1+4*i)
                ax.set_title(item+'_'+['c1','c2','c3','c4'][j])
                plt.plot(np.array(plot_data[j])[:100,1], label='true')
                plt.plot(np.array(plot_data[j])[:100,0], label='predict')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
        plt.savefig(var_log_dir+"/result.jpg", dpi=300)
        plt.close()

        # save embedding
        np.save(var_log_dir+'/embedding.npy', all_embeddings)
        
        # calculae ID
        def cal_id_embedding(tau, epoch, method='MLE', is_print=False):
            var_log_dir = f'logs/time-lagged/k_{koopman_dim}/tau_{tau}/seed_{random_seed}/test/epoch-{epoch}'
            eval_id_embedding(var_log_dir, method=method, is_print=is_print, max_point=100)
            dims = np.load(var_log_dir+f'/id_{method}.npy')
            return np.mean(dims)
        LB_id = cal_id_embedding(tau, epoch, 'MLE')
        MiND_id = cal_id_embedding(tau, epoch, 'MiND_ML')
        MADA_id = cal_id_embedding(tau, epoch, 'MADA')
        PCA_id = cal_id_embedding(tau, epoch, 'PCA')

        # logging
        # fp.write(f"{tau},0,{mse_c1},{mse_c2},{mse_c3},{mse_c4},{epoch},{LB_id},{MiND_id},{MADA_id},{PCA_id},{koopman_dim}\n")
        fp.write(f"{tau},{random_seed},{mse_c1},{mse_c2},{mse_c3},{mse_c4},{epoch},{LB_id},{MiND_id},{MADA_id},{PCA_id}\n")
        fp.flush()

        mse = loss_func(test_outputs, test_targets)
        if is_print: print(f'\rTau[{tau}] | Test epoch[{epoch}/{max_epoch}] | MSE={mse:.6f} | MLE={LB_id:.1f}, MinD={MiND_id:.1f}, MADA={MADA_id:.1f}, PCA={PCA_id:.1f}   ', end='')
        # if is_print: print(f'\rTau[{tau}] | Test epoch[{epoch}/{max_epoch}] | MSE: {loss_func(test_outputs, test_targets):.6f} | MLE={LB_id:.1f}   ', end='')
        
        if checkpoint_filepath is None: break
        
    fp.close()
    if is_print: print()
        
    
def worker_1(tau, koopman_dim=2, trace_num=256+32+32, random_seed=729, cpu_num=1, is_print=False, observation_dim=4):
    
    time.sleep(0.1)
    seed_everything(random_seed)
    set_cpu_num(cpu_num)
    
    sample_num = None

    # generate dataset
    generate_dataset(trace_num, koopman_dim, tau, sample_num, is_print=is_print)
    # train
    train_time_lagged(tau, is_print, observation_dim, koopman_dim, random_seed)
    # test and calculating ID
    # test_and_save_embeddings_of_time_lagged(tau, None, is_print)
    test_and_save_embeddings_of_time_lagged(tau, f"logs/time-lagged/k_{koopman_dim}/tau_{tau}/seed_{random_seed}", is_print, observation_dim, koopman_dim, random_seed)
    # plot id of each epoch
    plot_epoch_test_log(tau, koopman_dim, max_epoch=100+1)

    
def data_generator_pipeline(trace_num=256+32+32, time_step=100, observation_dim=4, koopman_dim=2):
    
    seed_everything(729)
    
    # generate original data
    generate_original_data(trace_num=trace_num, time_step=time_step, observation_dim=observation_dim, koopman_dim=koopman_dim)
    
    
def id_esitimate_pipeline(cpu_num=1, trace_num=256+32+32, observation_dim=4, koopman_dim=2):
    
    tau_list = [0,5,10,15,20,25]
    workers = []
    
    # id esitimate sub-process
    for random_seed in [1,2,3,4,5,6,7,8,9,10]:
        for tau in tau_list:
            is_print = True if len(workers)==0 else False
            workers.append(Process(target=worker_1, args=(tau, koopman_dim, trace_num, random_seed, cpu_num, is_print, observation_dim), daemon=True))
            workers[-1].start()
    while any([sub.exitcode==None for sub in workers]):
        pass
    
    print('ID Esitimate Over!')


if __name__ == '__main__':
    
    trace_num = 1000
    time_step = 100
    observation_dim = 16

    for koopman_dim in [2,4,6,8]:
        observation_dim = koopman_dim
        data_generator_pipeline(trace_num=trace_num, time_step=time_step, observation_dim=observation_dim, koopman_dim=koopman_dim)
        # id_esitimate_pipeline(trace_num=trace_num, observation_dim=observation_dim, koopman_dim=koopman_dim)