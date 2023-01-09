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
from Data.dataset import PNASDataset
from Data.generator import generate_dataset, generate_original_data
from util import set_cpu_num
from util.plot import plot_epoch_test_log, plot_slow_ae_loss
from util.intrinsic_dimension import eval_id_embedding


def train_time_lagged(tau, is_print=False):
    
    # prepare
    device = torch.device('cpu')
    data_filepath = 'Data/data/tau_' + str(tau)
    log_dir = 'logs/time-lagged/tau_' + str(tau)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)
    
    # init model
    model = models.TIME_LAGGED_AE(in_channels=1, input_1d_width=3, embed_dim=64)
    model.apply(models.weights_normal_init)
    model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min.txt").astype(np.float32)).unsqueeze(0)
    model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max.txt").astype(np.float32)).unsqueeze(0)
    model.to(device)
    
    # training params
    lr = 0.001
    batch_size = 128
    max_epoch = 50
    weight_decay = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.MSELoss()

    # dataset
    train_dataset = PNASDataset(data_filepath, 'train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataset = PNASDataset(data_filepath, 'val')
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    # training pipeline
    losses = []
    loss_curve = []
    for epoch in range(1, max_epoch+1):
        
        # train
        model.train()
        for input, target in train_loader:
            input = model.scale(input) # (batchsize,1,1,3)
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
            if is_print: print(f'\rTau[{tau}] | epoch[{epoch}/{max_epoch}] val-MSE={mse:.5f}', end='')
        
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
    

def test_and_save_embeddings_of_time_lagged(tau, checkpoint_filepath=None, is_print=False):
    
    # prepare
    device = torch.device('cpu')
    data_filepath = 'Data/data/tau_' + str(tau)
    log_dir = 'logs/time-lagged/tau_' + str(tau)
    os.makedirs(log_dir+'/test', exist_ok=True)
    
    # testing params
    batch_size = 128
    max_epoch = 50
    loss_func = nn.MSELoss()
    
    # init model
    model = models.TIME_LAGGED_AE(in_channels=1, input_1d_width=3, embed_dim=64)
    if checkpoint_filepath is None: # not trained
        model.apply(models.weights_normal_init)
        model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min.txt").astype(np.float32)).unsqueeze(0)
        model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max.txt").astype(np.float32)).unsqueeze(0)

    # dataset
    train_dataset = PNASDataset(data_filepath, 'train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataset = PNASDataset(data_filepath, 'test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # testing pipeline
    fp = open(log_dir+'/test/log.txt', 'a')
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
                input = model.scale(input) # (batchsize,1,1,3)
                target = model.scale(target)
                
                output, _ = model.forward(input.to(device))
                
                train_outputs = output.cpu() if not len(train_outputs) else torch.concat((train_outputs, output.cpu()), axis=0)
                train_targets = target.cpu() if not len(train_targets) else torch.concat((train_targets, target.cpu()), axis=0)
                                
                if batch_idx >= len(test_loader): break

            # test-dataset
            for input, target in test_loader:
                input = model.scale(input) # (batchsize,1,1,3)
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
            mse_x = loss_func(test_outputs[:,0,0,0], test_targets[:,0,0,0])
            mse_y = loss_func(test_outputs[:,0,0,1], test_targets[:,0,0,1])
            mse_z = loss_func(test_outputs[:,0,0,2], test_targets[:,0,0,2])
        
        # plot
        test_plot, train_plot = [[], [], []], [[], [], []]
        for i in range(len(test_outputs)):
            for j in range(len(test_plot)):
                test_plot[j].append([test_outputs[i,0,0,j], test_targets[i,0,0,j]])
                train_plot[j].append([train_outputs[i,0,0,j], train_targets[i,0,0,j]])
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
        def cal_id_embedding(tau, epoch, method='MLE', is_print=False):
            var_log_dir = f'logs/time-lagged/tau_{tau}/test/epoch-{epoch}'
            eval_id_embedding(var_log_dir, method=method, is_print=is_print, max_point=100)
            dims = np.load(var_log_dir+f'/id_{method}.npy')
            return np.mean(dims)
        LB_id = cal_id_embedding(tau, epoch, 'MLE')
        MiND_id = cal_id_embedding(tau, epoch, 'MiND_ML')
        MADA_id = cal_id_embedding(tau, epoch, 'MADA')
        PCA_id = cal_id_embedding(tau, epoch, 'PCA')

        # logging
        fp.write(f"{tau},0,{mse_x},{mse_y},{mse_z},{epoch},{LB_id},{MiND_id},{MADA_id},{PCA_id}\n")
        fp.flush()

        if is_print: print(f'\rTau[{tau}] | Test epoch[{epoch}/{max_epoch}] | MLE={LB_id:.1f}, MinD={MiND_id:.1f}, MADA={MADA_id:.1f}, PCA={PCA_id:.1f}   ', end='')
        
        if checkpoint_filepath is None: break
        
    fp.close()
    if is_print: print()
    
    
def train_slow_extract_and_evolve(tau, pretrain_epoch, slow_id, delta_t, is_print=False):
        
    # prepare
    device = torch.device('cpu')
    data_filepath = 'Data/data/tau_' + str(delta_t)
    log_dir = f'logs/slow_extract_and_evolve/tau_{tau}/pretrain_epoch{pretrain_epoch}/delta_t{delta_t}/id{slow_id}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)

    # init model
    model = models.EVOLVER(in_channels=1, input_1d_width=3, embed_dim=64, slow_dim=slow_id, delta_t=delta_t)
    model.apply(models.weights_normal_init)
    model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min.txt").astype(np.float32)).unsqueeze(0)
    model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max.txt").astype(np.float32)).unsqueeze(0)
    
    # load pretrained time-lagged AE
    ckpt_path = f'logs/time-lagged/tau_{tau}/checkpoints/epoch-{pretrain_epoch}.ckpt'
    ckpt = torch.load(ckpt_path)
    model.encoder_1.load_state_dict(ckpt['encoder'])
    model = model.to(device)
    
    # training params
    lr = 0.001
    batch_size = 128
    max_epoch = 50
    weight_decay = 0.001
    L1_loss = nn.L1Loss()
    MSE_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        [{'params': model.encoder_2.parameters()},
         {'params': model.decoder.parameters()}, 
         {'params': model.K_opt.parameters()},
         {'params': model.lstm.parameters()}],
        lr=lr, weight_decay=weight_decay) # not involve encoder_1 (freezen)
    # TODO: 需要对比把fast和slow的evolve用不同optimizer分别训练，与，现在这样端到端训练的效果
    
    # dataset
    train_dataset = PNASDataset(data_filepath, 'train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataset = PNASDataset(data_filepath, 'val')
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    # training pipeline
    losses = []
    train_loss = []
    val_loss = []
    best_loss = 1.
    for epoch in range(1, max_epoch+1):
        
        # train
        model.train()
        for input, target in train_loader:
            input = model.scale(input) # (batchsize,1,1,3)
            target = model.scale(target)
            
            # slow extract
            slow_var, embed = model.extract(input.to(device))
            slow_info = model.recover(slow_var)
            _, embed_from_info = model.extract(slow_info)
            
            # slow evolve
            slow_var_next = model.koopman_evolve(slow_var, T=1)
            slow_info_next = model.recover(slow_var_next)
            
            # fast evolve
            # TODO: 需要对比测试加不加detach()的区别，如果不加看看fast会怎么影响slow提取
            fast_info = input - slow_info.detach() # detach() to avoid the gradiant of fast evolver impact the slow evolver
            fast_info_next = model.lstm_evolve(fast_info, T=1)
            
            # total evolve
            total_info_next = slow_info_next + fast_info_next
            
            adiabatic_loss = L1_loss(embed, embed_from_info)
            slow_reconstruct_loss = MSE_loss(slow_info, input)
            evolve_loss = MSE_loss(total_info_next, target)
            all_loss = 0.5*slow_reconstruct_loss + 0.5*evolve_loss + 0.05*adiabatic_loss
            
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            
            losses.append(all_loss.detach().item())
            
        train_loss.append(np.mean(losses))
        
        # validate
        with torch.no_grad():
            inputs = []
            slow_vars = []
            targets = []
            slow_infos = []
            slow_infos_next = []
            fast_infos = []
            fast_infos_next = []
            total_infos_next = []
            
            model.eval()
            for input, target in val_loader:
                input = model.scale(input) # (batchsize,1,1,3)
                target = model.scale(target)
                
                # slow extract
                slow_var, embed = model.extract(input.to(device))
                slow_info = model.recover(slow_var)
                _, embed_from_info = model.extract(slow_info)
                
                # slow evolve
                slow_var_next = model.koopman_evolve(slow_var, T=1)
                slow_info_next = model.recover(slow_var_next)
                
                # fast evolve
                fast_info = input - slow_info
                fast_info_next = model.lstm_evolve(fast_info, T=1)
                
                # total evolve
                total_info_next = slow_info_next + fast_info_next

                inputs.append(input.cpu())
                slow_vars.append(slow_var.cpu())
                targets.append(target.cpu())
                slow_infos.append(slow_info.cpu())
                slow_infos_next.append(slow_info_next.cpu())
                fast_infos.append(fast_info.cpu())
                fast_infos_next.append(fast_info_next.cpu())
                total_infos_next.append(total_info_next.cpu())
            
            inputs = torch.concat(inputs, axis=0)
            slow_vars = torch.concat(slow_vars, axis=0)
            targets = torch.concat(targets, axis=0)
            slow_infos = torch.concat(slow_infos, axis=0)
            slow_infos_next = torch.concat(slow_infos_next, axis=0)
            fast_infos = torch.concat(fast_infos, axis=0)
            fast_infos_next = torch.concat(fast_infos_next, axis=0)
            total_infos_next = torch.concat(total_infos_next, axis=0)
            
            adiabatic_loss = L1_loss(embed, embed_from_info)
            slow_reconstruct_loss = MSE_loss(slow_info, input)
            evolve_loss = MSE_loss(total_info_next, target)
            all_loss = 0.5*slow_reconstruct_loss + 0.5*evolve_loss + 0.05*adiabatic_loss
            if is_print: print(f'\rTau[{tau}] | epoch[{epoch}/{max_epoch}] | val: adiab_loss={adiabatic_loss:.5f}, recons_loss={slow_reconstruct_loss:.5f}, evol_loss={evolve_loss:.5f}', end='')
            
            val_loss.append(all_loss.detach().item())
            
            os.makedirs(log_dir+f"/val/epoch-{epoch}/", exist_ok=True)

            # TODO: 把类似的plot写进for循环，压缩行数
            # plot slow variable
            plt.figure(figsize=(12,5+2*(slow_id-1)))
            plt.title('Val Reconstruction Curve')
            for id_var in range(slow_id):
                for index, item in enumerate(['X', 'Y', 'Z']):
                    plt.subplot(slow_id, 3, index+1+3*(id_var))
                    plt.scatter(inputs[:,0,0,index], slow_vars[:, id_var], s=5)
                    plt.xlabel(item)
                    plt.ylabel(f'U{id_var+1}')
            plt.subplots_adjust(wspace=0.35, hspace=0.35)
            plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_variable.jpg", dpi=300)
            plt.close()
            
            # plot slow infomation reconstruction curve
            plt.figure(figsize=(16,5))
            for j, item in enumerate(['X','Y','Z']):
                ax = plt.subplot(1,3,j+1)
                ax.set_title(item)
                plt.plot(inputs[:,0,0,j], label='all_info')
                plt.plot(slow_infos[:,0,0,j], label='slow_info')
            plt.subplots_adjust(wspace=0.2)
            plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_info.jpg", dpi=300)
            plt.close()
            
            # plot fast infomation curve (== origin_data - slow_info_recons)
            plt.figure(figsize=(16,5))
            for j, item in enumerate(['X','Y','Z']):
                ax = plt.subplot(1,3,j+1)
                ax.set_title(item)
                plt.plot(inputs[:,0,0,j], label='all_info')
                plt.plot(fast_infos[:,0,0,j], label='fast_info')
            plt.subplots_adjust(wspace=0.2)
            plt.savefig(log_dir+f"/val/epoch-{epoch}/fast_info.jpg", dpi=300)
            plt.close()
            
            # plot slow infomation one-step prediction curve
            plt.figure(figsize=(16,5))
            for j, item in enumerate(['X','Y','Z']):
                ax = plt.subplot(1,3,j+1)
                ax.set_title(item)
                plt.plot(targets[:,0,0,j], label='all_true')
                plt.plot(slow_infos_next[:,0,0,j], label='slow_predict')
            plt.subplots_adjust(wspace=0.2)
            plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_predict.jpg", dpi=300)
            plt.close()
            
            # plot fast infomation one-step prediction curve
            plt.figure(figsize=(16,5))
            for j, item in enumerate(['X','Y','Z']):
                ax = plt.subplot(1,3,j+1)
                ax.set_title(item)
                plt.plot(targets[:,0,0,j], label='all_true')
                plt.plot(fast_infos_next[:,0,0,j], label='fast_predict')
            plt.subplots_adjust(wspace=0.2)
            plt.savefig(log_dir+f"/val/epoch-{epoch}/fast_predict.jpg", dpi=300)
            plt.close()
            
            # plot total infomation one-step prediction curve
            plt.figure(figsize=(16,5))
            for j, item in enumerate(['X','Y','Z']):
                ax = plt.subplot(1,3,j+1)
                ax.set_title(item)
                plt.plot(targets[:,0,0,j], label='all_true')
                plt.plot(total_infos_next[:,0,0,j], label='all_predict')
            plt.subplots_adjust(wspace=0.2)
            plt.savefig(log_dir+f"/val/epoch-{epoch}/all_predict.jpg", dpi=300)
            plt.close()
        
            # record best model
            if all_loss < best_loss:
                best_loss = all_loss
                best_model = model.state_dict()

    # save model
    torch.save(best_model, log_dir+f"/checkpoints/epoch-{epoch}.ckpt")
    if is_print: print(f'\nsave best model at {log_dir}/checkpoints/epoch-{epoch}.ckpt (val best_loss={best_loss})')
    
    # TODO: 分别画出每种loss曲线
    # plot loss curve
    plt.figure()
    plt.plot(train_loss)
    plt.xlabel('epoch')
    plt.title('Train MSELoss Curve')
    plt.savefig(log_dir+'/train_loss_curve.jpg', dpi=300)
    np.save(log_dir+'/val_loss_curve.npy', val_loss)


def test_evolve(tau, pretrain_epoch, ckpt_epoch, slow_id, delta_t, T_max, is_print=False):
        
    # prepare
    device = torch.device('cpu')
    data_filepath = 'Data/data/tau_' + str(delta_t)
    log_dir = f'logs/slow_extract_and_evolve/tau_{tau}/pretrain_epoch{pretrain_epoch}/delta_t{delta_t}/id{slow_id}'

    # load model
    batch_size = 128
    model = models.EVOLVER(in_channels=1, input_1d_width=3, embed_dim=64, slow_dim=slow_id, delta_t=delta_t)
    ckpt_path = log_dir+f'/checkpoints/epoch-{ckpt_epoch}.ckpt'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)
    
    mse_slow, mse_fast, mse_total = [], [], []
    l1_fn = nn.L1Loss()
    
    for T in range(1, T_max):
        # dataset
        test_dataset = PNASDataset(data_filepath, 'test', T=T, T_max=T_max)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        
        # testing pipeline        
        with torch.no_grad():
            targets = []
            slow_infos_next = []
            fast_infos_next = []
            total_infos_next = []
            
            model.eval()
            for input, target in test_loader:
                input = model.scale(input)
                target = model.scale(target)
            
                # slow extract
                slow_var, _ = model.extract(input.to(device))
                slow_info = model.recover(slow_var)
                
                # slow evolve
                slow_var_next = model.koopman_evolve(slow_var, T=1)
                slow_info_next = model.recover(slow_var_next)
                
                # fast evolve
                fast_info = input - slow_info
                fast_info_next = model.lstm_evolve(fast_info, T=1)
                
                # total evolve
                total_info_next = slow_info_next + fast_info_next

                targets.append(target.cpu())
                slow_infos_next.append(slow_info_next.cpu())
                fast_infos_next.append(fast_info_next.cpu())
                total_infos_next.append(total_info_next.cpu())
            
            targets = torch.concat(targets, axis=0)
            slow_infos_next = torch.concat(slow_infos_next, axis=0)
            fast_infos_next = torch.concat(fast_infos_next, axis=0)
            total_infos_next = torch.concat(total_infos_next, axis=0)
            
            mse_slow.append(l1_fn(slow_infos_next, targets))
            mse_fast.append(l1_fn(fast_infos_next, targets))
            mse_total.append(l1_fn(total_infos_next, targets))
            if is_print: print(f'\rTesting Koopman evolvement | tau[{tau}] | pretrain[{pretrain_epoch}] | delta_t[{delta_t}] | T[{T}/{T_max}] | total_mse={mse_total[-1]:.5f}     ', end='')
            
            os.makedirs(log_dir+f"/test/", exist_ok=True)

            # plot slow infomation prediction curve
            plt.figure(figsize=(16,5))
            for j, item in enumerate(['X','Y','Z']):
                ax = plt.subplot(1,3,j+1)
                ax.set_title(item)
                plt.plot(targets[:,0,0,j], label='true')
                plt.plot(slow_infos_next[:,0,0,j], label='predict')
            plt.subplots_adjust(wspace=0.2)
            plt.savefig(log_dir+f"/test/pred_deltaT_{T*delta_t:.3f}_slow.jpg", dpi=300)
            plt.close()
            
            # plot fast infomation prediction curve
            plt.figure(figsize=(16,5))
            for j, item in enumerate(['X','Y','Z']):
                ax = plt.subplot(1,3,j+1)
                ax.set_title(item)
                plt.plot(targets[:,0,0,j], label='true')
                plt.plot(fast_infos_next[:,0,0,j], label='predict')
            plt.subplots_adjust(wspace=0.2)
            plt.savefig(log_dir+f"/test/pred_deltaT_{T*delta_t:.3f}_fast.jpg", dpi=300)
            plt.close()
            
            # plot total infomation prediction curve
            plt.figure(figsize=(16,5))
            for j, item in enumerate(['X','Y','Z']):
                ax = plt.subplot(1,3,j+1)
                ax.set_title(item)
                plt.plot(targets[:,0,0,j], label='true')
                plt.plot(total_infos_next[:,0,0,j], label='predict')
            plt.subplots_adjust(wspace=0.2)
            plt.savefig(log_dir+f"/test/pred_deltaT_{T*delta_t:.3f}_total.jpg", dpi=300)
            plt.close()
    
    # plot mse per T
    plt.figure()
    for item, mse in zip(['slow','fast','total'], [mse_slow, mse_fast, mse_total]):
        plt.plot(np.arange(1,len(mse)+1)*delta_t, mse, label=item)
    plt.xlabel(f'delta_t/s')
    plt.title(f'Evolve L1-dist curve | delta_t[{delta_t}]')
    plt.legend()
    plt.savefig(log_dir+f"/test/all_mse.jpg", dpi=300)
        
    
def worker_1(tau, random_seed=729, cpu_num=1, is_print=False):
    
    time.sleep(0.1)
    seed_everything(random_seed)
    set_cpu_num(cpu_num)
    
    sample_num = 50
    trace_num = 256+32+32

    # generate dataset
    generate_dataset(trace_num, tau, sample_num, is_print=is_print)
    # train
    train_time_lagged(tau, is_print)
    # test and calculating ID
    test_and_save_embeddings_of_time_lagged(tau, None, is_print)
    test_and_save_embeddings_of_time_lagged(tau, f"logs/time-lagged/tau_{tau}", is_print)
    # plot id of each epoch
    plot_epoch_test_log(tau, max_epoch=50+1)


def worker_2(tau, pretrain_epoch, slow_id, delta_t, random_seed=729, cpu_num=1, is_print=False, id_list=[1,2,3,4]):
    
    time.sleep(0.1)
    seed_everything(random_seed)
    set_cpu_num(cpu_num)

    sequence_length = int(tau/delta_t)
    sample_num = 50

    # train
    train_slow_extract_and_evolve(tau, pretrain_epoch, slow_id, delta_t, is_print=is_print)
    # plot mse curve of each id
    try: plot_slow_ae_loss(tau, pretrain_epoch, delta_t, id_list) 
    except: pass
    # test evolve
    test_evolve(tau, pretrain_epoch, sample_num, slow_id, delta_t, sequence_length, is_print)
    
    
def data_generator_pipeline():
    
    # generate original data
    trace_num = 256+32+32
    total_t = 9
    generate_original_data(trace_num=trace_num, total_t=total_t)
    
    
def id_esitimate_pipeline(cpu_num=1):
    
    tau_list = [0.0, 1.5, 3.0, 4.5]
    workers = []
    
    # id esitimate sub-process
    for tau in tau_list:
        is_print = True if len(workers)==0 else False
        workers.append(Process(target=worker_1, args=(tau, 729, cpu_num, is_print), daemon=True))
        workers[-1].start()
    while any([sub.exitcode==None for sub in workers]):
        pass
    
    print('ID Esitimate Over!')


def slow_evolve_pipeline(delta_t=0.01, cpu_num=1):
    
    tau_list = [1.5, 3.0, 4.5]
    id_list = [2, 4, 6]
    workers = []
    
    # generate dataset sub-process
    sample_num = 50
    trace_num = 256+32+32
    generate_dataset(trace_num, delta_t, sample_num, is_print=True, sequence_length=None)
    for tau in tau_list:
        sequence_length = int(tau/delta_t)
        is_print = True if len(workers)==0 else False
        workers.append(Process(target=generate_dataset, args=(trace_num, delta_t, sample_num, is_print, sequence_length), daemon=True))
        workers[-1].start()
    while any([sub.exitcode==None for sub in workers]):
        pass
    workers = []
    
    # slow evolve sub-process
    for tau in tau_list:
        for pretrain_epoch in [5, 30]:
            for slow_id in id_list:
                is_print = True if len(workers)==0 else False
                workers.append(Process(target=worker_2, args=(tau, pretrain_epoch, slow_id, delta_t, 729, cpu_num, is_print, id_list), daemon=True))
                workers[-1].start()
    while any([sub.exitcode==None for sub in workers]):
        pass
    
    print('Slow-Infomation Evolve Over!')
    

if __name__ == '__main__':
    
    data_generator_pipeline()
    id_esitimate_pipeline()
    slow_evolve_pipeline()