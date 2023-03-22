# -*- coding: utf-8 -*-
import os
import torch
from torch import nn
import numpy as np
import scienceplots
import matplotlib.pyplot as plt;plt.style.use(['science']);plt.rcParams.update({'font.size':16})
import warnings;warnings.simplefilter('ignore')

import models
from Data.dataset import Dataset
from util.intrinsic_dimension import eval_id_embedding


def train_time_lagged(
        system,
        embedding_dim,
        channel_num,
        obs_dim,
        tau, 
        max_epoch, 
        is_print=False, 
        random_seed=729, 
        data_dir='Data/2S2F/data/', 
        log_dir='logs/2S2F/TimeSelection/', 
        device=torch.device('cpu'),
        data_dim=4,
        lr=0.001,
        batch_size=128,
        enc_net='MLP',
        e1_layer_n=3,
        sliding_window=True
        ):
    
    # prepare
    data_filepath = data_dir + 'tau_' + str(tau)
    log_dir = log_dir + 'tau_' + str(tau) + f'/seed{random_seed}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)
    
    # init model
    model = models.TimeLaggedAE(in_channels=channel_num, feature_dim=obs_dim, embed_dim=embedding_dim, data_dim=data_dim, enc_net=enc_net, e1_layer_n=e1_layer_n)
    tmp = '' if sliding_window else '_static'
    model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min" + tmp + ".txt").reshape(channel_num,obs_dim).astype(np.float32))
    model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max" + tmp + ".txt").reshape(channel_num,obs_dim).astype(np.float32))
    model.to(device)
    
    # training params
    weight_decay = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    # dataset
    train_dataset = Dataset(data_filepath, 'train', sliding_window=sliding_window)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataset = Dataset(data_filepath, 'val', sliding_window=sliding_window)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=len(val_dataset), shuffle=False, drop_last=False)
    
    # training pipeline
    losses = []
    loss_curve = []
    for epoch in range(1, max_epoch+1):
        
        # train
        model.train()
        for x_t0, x_t1 in train_loader:
            x_t0 = model.scale(x_t0.to(device))[..., :obs_dim] # (batchsize,1,channel_num,feature_dim)
            x_t1 = model.scale(x_t1.to(device))[..., :obs_dim]
            
            prior, embed1 = model.forward(x_t0, direct='prior')
            reverse, embed2 = model.forward(x_t1, direct='reverse')
            
            prior_loss = mse_loss(prior, x_t1)
            reverse_loss = mse_loss(reverse, x_t0)
            symmetry_loss = l1_loss(embed1, embed2)
            # loss = prior_loss + reverse_loss + symmetry_loss
            loss = prior_loss + reverse_loss + 0.1*symmetry_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append([prior_loss.detach().item(), reverse_loss.detach().item(), symmetry_loss.detach().item()])
            
        loss_curve.append(np.mean(losses, axis=0))
        
        # validate
        with torch.no_grad():
            
            model.eval()
            for x_t0, x_t1 in val_loader:
                x_t0 = model.scale(x_t0.to(device))[..., :obs_dim]
                x_t1 = model.scale(x_t1.to(device))[..., :obs_dim]
            
                prior, embed1 = model.forward(x_t0, direct='prior')
                reverse, embed2 = model.forward(x_t1, direct='reverse')
                
            prior_loss = mse_loss(prior, x_t1)
            reverse_loss = mse_loss(reverse, x_t0)
            symmetry_loss = l1_loss(embed1, embed2)
            if is_print: print(f'\rTau[{tau}] | epoch[{epoch}/{max_epoch}] val-MSE: prior={prior_loss:.5f}, reverse={reverse_loss:.5f}, symmetry={symmetry_loss:.5f}', end='')
        
        # save each epoch model
        model.train()
        torch.save({'model': model.state_dict(), 'encoder': model.encoder.state_dict(),}, log_dir+f"/checkpoints/epoch-{epoch}.ckpt")
        
    # plot loss curve
    plt.figure()
    plt.plot(np.array(loss_curve)[:,0], label='prior')
    plt.plot(np.array(loss_curve)[:,1], label='reverse')
    plt.plot(np.array(loss_curve)[:,2], label='symmetry')
    plt.legend()
    plt.xlabel('epoch')
    plt.title('Training Loss Curve')
    plt.savefig(log_dir+'/training_loss.pdf', dpi=300)
    np.save(log_dir+'/training_loss.npy', loss_curve)
    if is_print: print()
        

def test_and_save_embeddings_of_time_lagged(
        system,
        embedding_dim,
        channel_num,
        obs_dim,
        tau, 
        max_epoch, 
        checkpoint_filepath=None, 
        is_print=False, 
        random_seed=729, 
        data_dir='Data/2S2F/data/', 
        log_dir='logs/2S2F/TimeSelection/', 
        device=torch.device('cpu'),
        data_dim=4,
        batch_size=128,
        enc_net='MLP',
        e1_layer_n=3,
        sliding_window=True
        ):
    
    # prepare
    data_filepath = data_dir + 'tau_' + str(tau)
    
    # testing params
    loss_func = nn.MSELoss()
    
    # init model
    model = models.TimeLaggedAE(in_channels=channel_num, feature_dim=obs_dim, embed_dim=embedding_dim, data_dim=data_dim, enc_net=enc_net, e1_layer_n=e1_layer_n)
    if checkpoint_filepath is None: # not trained
        tmp = '' if sliding_window else '_static'
        model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min" + tmp + ".txt").reshape(channel_num,obs_dim).astype(np.float32))
        model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max" + tmp + ".txt").reshape(channel_num,obs_dim).astype(np.float32))

    # dataset
    test_dataset = Dataset(data_filepath, 'test', sliding_window=sliding_window)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # testing pipeline
    fp = open(log_dir + 'tau_' + str(tau) + '/test_log.txt', 'a')
    interval = 1
    for ep in range(1, max_epoch, interval):
        
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
        var_log_dir = log_dir + 'tau_' + str(tau) + f'/seed{random_seed}/test/epoch-{epoch}'
        os.makedirs(var_log_dir, exist_ok=True)
        
        # testing
        with torch.no_grad():
            for input, target in test_loader:
                input = model.scale(input.to(device))[..., :obs_dim] # (batchsize,1,1,4)
                target = model.scale(target.to(device))[..., :obs_dim]
                
                output, embeddings = model.forward(input)
                
                # save the embedding vectors
                for embedding in embeddings:
                    all_embeddings.append(embedding.cpu().numpy())

                test_outputs = output.cpu() if not len(test_outputs) else torch.concat((test_outputs, output.cpu()), axis=0)
                test_targets = target.cpu() if not len(test_targets) else torch.concat((test_targets, target.cpu()), axis=0)
                                
            # test mse
            mse_ = []
            for i in range(test_outputs.shape[-1]):
                mse_.append(loss_func(test_outputs[:,0,0,i], test_targets[:,0,0,i]))
        
        # plot
        plt.figure(figsize=(16,5))
        for j in range(test_outputs.shape[-1]):
            data = []
            for i in range(len(test_outputs)):
                data.append([test_outputs[i,0,0,j], test_targets[i,0,0,j]])
            ax = plt.subplot(1,test_outputs.shape[-1],j+1)
            ax.set_title('test_'+f'c{j+1}')
            ax.plot(np.array(data)[:,1], label='true')
            ax.plot(np.array(data)[:,0], label='predict')
            ax.legend()
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
        plt.savefig(var_log_dir+"/result.pdf", dpi=300)
        plt.close()

        if 'FHN' in system:
            plt.figure(figsize=(16,5))
            for j in range(test_outputs.shape[-1]):
                data = []
                for i in range(len(test_outputs)):
                    data.append([test_outputs[i,0,1,j], test_targets[i,0,1,j]])
                ax = plt.subplot(1,test_outputs.shape[-1],j+1)
                ax.set_title('test_'+f'c{j+1}')
                ax.plot(np.array(data)[:,1], label='true')
                ax.plot(np.array(data)[:,0], label='predict')
                ax.legend()
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
            plt.savefig(var_log_dir+"/result_inhibitor.pdf", dpi=300)
            plt.close()

        # save embedding
        np.save(var_log_dir+'/embedding.npy', all_embeddings)
        
        # calculae ID
        def cal_id_embedding(method='MLE', is_print=False):
            # eval_id_embedding(var_log_dir, method=method, is_print=is_print, max_point=100)
            eval_id_embedding(var_log_dir, method=method, is_print=False, max_point=100)
            dims = np.load(var_log_dir+f'/id_{method}.npy')
            return np.mean(dims)
        MLE_id = cal_id_embedding('MLE', is_print)

        # logging
        if system == '2S2F':
            fp.write(f"{tau},{random_seed},{mse_[0]},{mse_[1]},{mse_[2]},{mse_[3]},{epoch},{MLE_id}\n")
        elif system == '1S1F':
            fp.write(f"{tau},{random_seed},{mse_[0]},{mse_[1]},{epoch},{MLE_id}\n")
        elif system == '1S2F':
            fp.write(f"{tau},{random_seed},{mse_[0]},{mse_[1]},{mse_[2]},{epoch},{MLE_id}\n")
        elif 'FHN' in system:
            fp.write(f"{tau},{random_seed},{mse_[0]},{mse_[1]},{epoch},{MLE_id}\n")
        elif system == 'HalfMoon':
            fp.write(f"{tau},{random_seed},{mse_[0]},{mse_[1]},{epoch},{MLE_id}\n")
        elif system == 'SC':
            fp.write(f"{tau},{random_seed},{mse_[0]},{mse_[1]},{epoch},{MLE_id}\n")
        fp.flush()

        if is_print: print(f'\rTau[{tau}] | Test epoch[{epoch}/{max_epoch}] | MSE: {loss_func(test_outputs, test_targets):.6f} | MLE={MLE_id:.1f}', end='')
        # if is_print: print(f'\rTau[{tau}] | Test epoch[{epoch}/{max_epoch}] | MSE: {loss_func(test_outputs, test_targets):.6f}', end='')
        
        if checkpoint_filepath is None: break
        
    fp.close()
