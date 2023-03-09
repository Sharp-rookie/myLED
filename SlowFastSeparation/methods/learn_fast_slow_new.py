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
    
    
def train_slow_extract_and_evolve(
        system,
        embedding_dim,
        channel_num,
        obs_dim,
        tau_s, 
        slow_dim, 
        koopman_dim, 
        delta_t, 
        n, 
        ckpt_path,
        is_print=False, 
        random_seed=729, 
        learn_max_epoch=100, 
        data_dir='Data/2S2F/data/', 
        log_dir='logs/2S2F/LearnDynamics/', 
        device='cpu',
        data_dim=4,
        lr=0.01,
        batch_size=128,
        enc_net='MLP'
        ):
        
    # prepare
    data_filepath = data_dir + 'tau_' + str(delta_t)
    log_dir = log_dir + f'seed{random_seed}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)

    # init model
    assert koopman_dim>=slow_dim, f"Value Error, koopman_dim is smaller than slow_dim({koopman_dim}<{slow_dim})"
    model = models.DynamicsEvolver(in_channels=channel_num, feature_dim=obs_dim, embed_dim=embedding_dim, slow_dim=slow_dim, redundant_dim=koopman_dim-slow_dim, tau_s=tau_s, device=device, data_dim=data_dim, enc_net=enc_net)
    model.apply(models.weights_normal_init)
    model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min.txt").astype(np.float32)).unsqueeze(0)
    model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max.txt").astype(np.float32)).unsqueeze(0)
    
    # load pretrained time-lagged AE
    ckpt = torch.load(ckpt_path)
    model.encoder_1.load_state_dict(ckpt['encoder'])
    model = model.to(device)
    
    # training params
    weight_decay = 0.001
    L1_loss = nn.L1Loss()
    MSE_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        [{'params': model.encoder_2.parameters()},
        #  {'params': model.encoder_2_1.parameters()},
        #  {'params': model.encoder_2_2.parameters()},
         {'params': model.encoder_3.parameters()},
        #  {'params': model.decoder_1_1.parameters()},
        #  {'params': model.decoder_1_2.parameters()}, 
         {'params': model.decoder_1.parameters()}, 
         {'params': model.decoder_2.parameters()}, 
        #  {'params': model.decoder_2_1.parameters()},
        #  {'params': model.decoder_2_2.parameters()},
         {'params': model.K_opt.parameters()},
         {'params': model.lstm.parameters()}],
        lr=lr, weight_decay=weight_decay) # not involve encoder_1 (freezen)
    
    # dataset
    train_dataset = Dataset(data_filepath, 'train', length=n)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataset = Dataset(data_filepath, 'val', length=n)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # training pipeline
    train_loss = []
    lambda_curve = [[] for _ in range(slow_dim)]
    for epoch in range(1, learn_max_epoch+1):
        
        losses = [[],[],[]]
        
        # train
        model.train()
        [lambda_curve[i].append(model.K_opt.Lambda[i].detach().cpu()) for i in range(slow_dim) ]
        for input, _, internl_units in train_loader:
            
            input = model.scale(input.to(device))[..., :obs_dim] # (batchsize,1,1,4)
            
            # obs —— embedding —— slow representation —— embedding(reconstruction) —— obs(reconstruction)
            slow_var, embed = model.obs2slow(input)
            # slow_var, embed = model.obs2slow_new2(input)
            recons_obs, recons_embed = model.slow2obs(slow_var)
            # recons_obs, recons_embed = model.slow2obs_new2(slow_var)
            _, adiabatic_embed = model.obs2slow(recons_obs)
            
            # calculate loss value
            adiabatic_loss = L1_loss(embed, adiabatic_embed)
            embed_reconstruct_loss = L1_loss(recons_embed, embed)
            obs_reconstruct_loss = MSE_loss(recons_obs, input)

            # slow_var1, slow_var2, embed = model.obs2slow_new(input)
            # recons_obs, recons_embed1, recons_embed2 = model.slow2obs_new(slow_var1, slow_var2)
            # _, _, adiabatic_embed = model.obs2slow_new(recons_obs)

            # adiabatic_loss = L1_loss(embed, adiabatic_embed)
            # # embed_reconstruct_loss = L1_loss(recons_embed1, embed) + L1_loss(recons_embed1+recons_embed2, embed)
            # # embed_reconstruct_loss = L1_loss(recons_embed1, embed) + L1_loss(recons_embed2, embed-recons_embed1.detach()) + L1_loss(recons_embed1+recons_embed2, embed)
            # # embed_reconstruct_loss = L1_loss(recons_embed1, embed-recons_embed2.detach()) + L1_loss(recons_embed2, embed-recons_embed1.detach()) + L1_loss(recons_embed1+recons_embed2, embed)
            # embed_reconstruct_loss = L1_loss(recons_embed1, embed) + L1_loss(recons_embed2, embed-recons_embed1.detach()) + L1_loss(recons_embed1+recons_embed2, embed)
            # obs_reconstruct_loss = MSE_loss(recons_obs, input)
            
            ###########
            # optimize
            ###########
            # all_loss = (slow_reconstruct_loss + 0.1*adiabatic_loss) + (koopman_evol_loss + slow_evol_loss + obs_evol_loss) / n
            all_loss = 0.1*adiabatic_loss + embed_reconstruct_loss + obs_reconstruct_loss
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            
            # record loss
            losses[0].append(adiabatic_loss.detach().item())
            losses[1].append(embed_reconstruct_loss.detach().item())
            losses[2].append(obs_reconstruct_loss.detach().item())
        
        train_loss.append([np.mean(losses[0]), np.mean(losses[1]), np.mean(losses[2])])
        
        # validate 
        with torch.no_grad():
            inputs = []
            slow_vars = []
            targets = []
            recons_obses = []
            embeds = []
            recons_embeds = []
            adiabatic_embeds = []
            hidden_slow = []
            
            model.eval()
            for input, target, _ in val_loader:

                if system == 'FHN' or system == '1S1F':
                    hidden_slow.append(input[..., obs_dim:].cpu())
                
                input = model.scale(input.to(device))[..., :obs_dim] # (batchsize,1,channel_num,feature_dim)
                target = model.scale(target.to(device))[..., :obs_dim]

                # obs —— embedding —— slow representation —— embedding(reconstruction) —— obs(reconstruction)
                slow_var, embed = model.obs2slow(input)
                recons_obs, recons_embed = model.slow2obs(slow_var)
                # recons_obs, recons_embed = model.slow2obs_new2(slow_var)
                _, adiabatic_embed = model.obs2slow(recons_obs)
                # slow_var1, slow_var2, embed = model.obs2slow_new(input)
                # recons_obs, recons_embed1, recons_embed2 = model.slow2obs_new(slow_var1, slow_var2)
                # _, _, adiabatic_embed = model.obs2slow_new(recons_obs)
                # slow_var = torch.concat([slow_var1,slow_var2], dim=-1)
                # recons_embed = recons_embed1 + recons_embed2

                # record results
                inputs.append(input.cpu())
                slow_vars.append(slow_var.cpu())
                targets.append(target.cpu())
                recons_obses.append(recons_obs.cpu())
                embeds.append(embed.cpu())
                recons_embeds.append(recons_embed.cpu())
                adiabatic_embeds.append(adiabatic_embed.cpu())
            
            # trans to tensor
            inputs = torch.concat(inputs, axis=0)
            slow_vars = torch.concat(slow_vars, axis=0)
            targets = torch.concat(targets, axis=0)
            recons_obses = torch.concat(recons_obses, axis=0)
            embeds = torch.concat(embeds, axis=0)
            recons_embeds = torch.concat(recons_embeds, axis=0)
            adiabatic_embeds = torch.concat(adiabatic_embeds, axis=0)
            if system == 'FHN' or system == '1S1F':
                hidden_slow = torch.concat(hidden_slow, axis=0)

            # slow_var1 = slow_vars[:, :1]
            # slow_var2 = slow_vars[:, 1:2]
            # cos_sim = torch.nn.functional.cosine_similarity(slow_var1, slow_var2, dim=0)[0]
            
            # cal loss
            adiabatic_loss = L1_loss(embeds, adiabatic_embeds)
            embed_reconstruct_loss = L1_loss(recons_embeds, embeds)
            obs_reconstruct_loss = MSE_loss(recons_obses, inputs)
            if is_print: print(f'\rTau[{tau_s}] | epoch[{epoch}/{learn_max_epoch}] | val: adiabatic={adiabatic_loss:.5f}, emebd_recons={embed_reconstruct_loss:.5f}, obs_recons={obs_reconstruct_loss:.5f}', end='')
            # if is_print: print(f'\rTau[{tau_s}] | epoch[{epoch}/{learn_max_epoch}] | val: adiabatic={adiabatic_loss:.5f}, emebd_recons={embed_reconstruct_loss:.5f}, obs_recons={obs_reconstruct_loss:.5f}, cosine_similarity={cos_sim:.5f}', end='')
            
            
            # plot per 5 epoch
            if epoch % 1 == 0:
                os.makedirs(log_dir+f"/val/epoch-{epoch}/", exist_ok=True)
                
                sample = 10
                # plot slow variable vs input
                plt.figure(figsize=(16,5+2*(slow_dim-1)))
                plt.title('Val Reconstruction Curve')
                for id_var in range(slow_dim):
                    for index, item in enumerate([f'c{k}' for k in range(inputs.shape[-1])]):
                        plt.subplot(slow_dim, inputs.shape[-1], index+1+inputs.shape[-1]*(id_var))
                        plt.scatter(inputs[::sample,0,0,index], slow_vars[::sample, id_var], s=5)
                        plt.xlabel(item)
                        plt.ylabel(f'U{id_var+1}')
                plt.subplots_adjust(wspace=0.35, hspace=0.35)
                plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_vs_input.pdf", dpi=300)
                plt.close()

                if system == 'FHN' or system == '1S1F':
                    # plot slow variable vs hidden variable
                    plt.figure(figsize=(8*(hidden_slow.shape[-1]),5+2*(slow_dim-1)))
                    plt.title('Extracted VS Hidden Slow')
                    for id_var in range(slow_dim):
                        for index, item in enumerate([f'c{k}' for k in range(hidden_slow.shape[-1])]):
                            plt.subplot(slow_dim, hidden_slow.shape[-1], index+1+hidden_slow.shape[-1]*(id_var))
                            plt.scatter(hidden_slow[::sample,0,0,index], slow_vars[::sample, id_var], s=5)
                            plt.xlabel(item)
                            plt.ylabel(f'U{id_var+1}')
                    plt.subplots_adjust(wspace=0.35, hspace=0.35)
                    plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_vs_hidden.pdf", dpi=300)
                    plt.close()
                
                # plot slow variable
                plt.figure(figsize=(12,5+2*(slow_dim-1)))
                plt.title('Slow variable Curve')
                for id_var in range(slow_dim):
                    ax = plt.subplot(slow_dim, 1, 1+id_var)
                    for idx in range(slow_dim):
                        ax.plot(inputs[:,0,0,idx], label=f'c{idx+1}')
                    ax.plot(slow_vars[:, id_var], label=f'U{id_var+1}')
                    plt.xlabel(item)
                    ax.legend()
                plt.subplots_adjust(wspace=0.35, hspace=0.35)
                plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_variable.pdf", dpi=300)
                plt.close()

                # plot observation and reconstruction
                plt.figure(figsize=(16,5))
                for j, item in enumerate([f'c{k}' for k in range(targets.shape[-1])]):
                    ax = plt.subplot(1,targets.shape[-1],j+1)
                    ax.plot(inputs[:,0,0,j], label='all_obs')
                    ax.plot(recons_obses[:,0,0,j], label='slow_obs')
                    ax.set_title(item)
                    ax.legend()
                plt.subplots_adjust(wspace=0.2)
                plt.savefig(log_dir+f"/val/epoch-{epoch}/recons_obs.pdf", dpi=300)
                plt.close()
        
                # save model
                torch.save(model.state_dict(), log_dir+f"/checkpoints/epoch-{epoch}.ckpt")
    
    # plot loss curve
    train_loss = np.array(train_loss)
    plt.figure()
    for i, item in enumerate(['adiabatic','embed_reconstruct_loss','obs_reconstruct_loss']):
        plt.plot(train_loss[:, i], label=item)
    plt.xlabel('epoch')
    plt.legend()
    plt.title('Training Loss Curve')
    plt.savefig(log_dir+'/train_loss_curve.pdf', dpi=300)

    # plot Koopman Lambda curve
    plt.figure(figsize=(6,6))
    marker = ['o', '^', '+', 's', '*', 'x']
    for i in range(slow_dim):
        plt.plot(lambda_curve[i], marker=marker[i%len(marker)], markersize=6, label=rf'$\lambda_{i}$')
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel(r'$\Lambda$', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(log_dir+'/K_lambda_curve.pdf', dpi=300)
    np.savez(log_dir+'/K_lambda_curve.npz',lambda_curve=lambda_curve)


def test_evolve(
        system,
        embedding_dim,
        channel_num,
        obs_dim,
        tau_s, 
        ckpt_epoch, 
        slow_dim,
        koopman_dim, 
        delta_t, 
        n, 
        is_print=False, 
        random_seed=729, 
        data_dir='Data/2S2F/data/', 
        log_dir='logs/2S2F/LearnDynamics/', 
        device='cpu',
        data_dim=4,
        batch_size=128,
        enc_net='MLP'
        ):
        
    # prepare
    data_filepath = data_dir + 'tau_' + str(delta_t)
    log_dir = log_dir + f'seed{random_seed}'

    # load model
    model = models.DynamicsEvolver(in_channels=channel_num, feature_dim=obs_dim, embed_dim=embedding_dim, slow_dim=slow_dim, redundant_dim=koopman_dim-slow_dim, tau_s=tau_s, device=device, data_dim=data_dim, enc_net=enc_net)
    ckpt_path = log_dir+f'/checkpoints/epoch-{ckpt_epoch}.ckpt'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)

    if is_print and delta_t==0.1:
        print('Koopman V:')
        print(model.K_opt.V.detach().cpu().numpy())
        print('Koopman Lambda:')
        print(model.K_opt.Lambda.detach().cpu().numpy())
    
    # dataset
    test_dataset = Dataset(data_filepath, 'test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # testing pipeline        
    with torch.no_grad():
        inputs = []
        targets = []
        slow_vars = []
        recons_obses = []
        recons_obses_next = []
        fast_obses_next = []
        total_obses_next = []
        slow_vars_next = []
        slow_vars_truth = []
        hidden_slow = []
        
        model.eval()
        for input, target in test_loader:

            if system == 'FHN' or system == '1S1F':
                hidden_slow.append(input[..., obs_dim:].cpu())
            
            input = model.scale(input.to(device))[..., :obs_dim]
            target = model.scale(target.to(device))[..., :obs_dim]
        
            # obs ——> slow ——> koopman
            slow_var, _ = model.obs2slow(input)
            koopman_var = model.slow2koopman(slow_var)
            slow_obs = model.slow2obs(slow_var)
            
            # koopman evolve
            t = torch.tensor([delta_t], device=device)
            koopman_var_next = model.koopman_evolve(koopman_var, tau=t)
            slow_var_next = model.koopman2slow(koopman_var_next)
            slow_obs_next = model.slow2obs(slow_var_next)
            
            # fast obs evolve
            fast_obs = input - slow_obs
            fast_obs_next, _ = model.lstm_evolve(fast_obs, T=n)
            
            # total obs evolve
            total_obs_next = slow_obs_next + fast_obs_next

            inputs.append(input)
            targets.append(target)
            slow_vars.append(slow_var)
            recons_obses.append(slow_obs)
            recons_obses_next.append(slow_obs_next)
            fast_obses_next.append(fast_obs_next)
            total_obses_next.append(total_obs_next)   
            slow_vars_next.append(slow_var_next)   
            slow_vars_truth.append(model.obs2slow(target)[0])
        
        inputs = model.descale(torch.concat(inputs, axis=0)).cpu()
        recons_obses = model.descale(torch.concat(recons_obses, axis=0)).cpu()
        recons_obses_next = model.descale(torch.concat(recons_obses_next, axis=0)).cpu()
        fast_obses_next = model.descale(torch.concat(fast_obses_next, axis=0)).cpu()
        slow_vars = torch.concat(slow_vars, axis=0).cpu()
        slow_vars_next = torch.concat(slow_vars_next, axis=0).cpu()
        slow_vars_truth = torch.concat(slow_vars_truth, axis=0).cpu()
        hidden_slow = torch.concat(hidden_slow, axis=0)
        
        targets = torch.concat(targets, axis=0)
        total_obses_next = torch.concat(total_obses_next, axis=0)
    
    # metrics
    pred = total_obses_next.detach().cpu().numpy()
    true = targets.detach().cpu().numpy()
    MAPE = np.mean(np.abs((pred - true) / true))
    
    targets = model.descale(targets)
    total_obses_next = model.descale(total_obses_next)
    pred = total_obses_next.detach().cpu().numpy()
    true = targets.detach().cpu().numpy()
    MSE = np.mean((pred - true) ** 2)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(pred - true))
                
    os.makedirs(log_dir+f"/test/{delta_t}/", exist_ok=True)

    # plot slow extract from original data
    sample = 1
    plt.figure(figsize=(16,5))
    for j, item in enumerate([rf'$c_{k}$' for k in range(targets.shape[-1])]):
        ax = plt.subplot(1,targets.shape[-1],j+1)
        t = torch.range(0,len(inputs)-1) * 0.01
        ax.plot(t[::sample], inputs[::sample,0,0,j], label=r'$X$')
        ax.plot(t[::sample], recons_obses[::sample,0,0,j], marker="^", markersize=4, label=r'$X_s$')
        ax.legend()
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('t / s', fontsize=20)
        plt.ylabel(item, fontsize=20)
    plt.subplots_adjust(wspace=0.35)
    plt.savefig(log_dir+f"/test/{delta_t}/slow_extract.pdf", dpi=300)
    plt.close()

    # plot slow variable vs input
    sample = 4
    plt.figure(figsize=(16,5+2*(slow_dim-1)))
    for id_var in range(slow_dim):
        for index, item in enumerate([rf'$c_{k}$' for k in range(inputs.shape[-1])]):
            plt.subplot(slow_dim, inputs.shape[-1], index+1+inputs.shape[-1]*(id_var))
            plt.scatter(inputs[::sample,0,0,index], slow_vars[::sample, id_var], s=2)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.xlabel(item, fontsize=20)
            plt.ylabel(rf'$u_{id_var+1}$', fontsize=20)
    plt.subplots_adjust(wspace=0.55, hspace=0.35)
    plt.savefig(log_dir+f"/test/{delta_t}/slow_vs_input.pdf", dpi=150)
    plt.close()

    if system == 'FHN' or system == '1S1F':
        # plot slow variable vs hidden variable
        plt.figure(figsize=(8*(hidden_slow.shape[-1]),5+2*(slow_dim-1)))
        plt.title('Extracted VS Hidden Slow')
        for id_var in range(slow_dim):
            for index, item in enumerate([f'c{k}' for k in range(hidden_slow.shape[-1])]):
                plt.subplot(slow_dim, hidden_slow.shape[-1], index+1+hidden_slow.shape[-1]*(id_var))
                plt.scatter(hidden_slow[::sample,0,0,index], slow_vars[::sample, id_var], s=5)
                plt.xlabel(item)
                plt.ylabel(f'U{id_var+1}')
        plt.subplots_adjust(wspace=0.35, hspace=0.35)
        plt.savefig(log_dir+f"/test/{delta_t}/slow_vs_hidden.pdf", dpi=300)
        plt.close()

    for i, figname in enumerate(['slow_pred', 'fast_pred', 'total_pred']):
        plt.figure(figsize=(16,5))
        for j, item in enumerate([rf'$c_{k}$' for k in range(targets.shape[-1])]):
            ax = plt.subplot(1,targets.shape[-1],j+1)
            
            ax.plot(true[:,0,0,j], label='true')

            # plot slow observation prediction curve
            if i == 0:
                ax.plot(recons_obses_next[:,0,0,j], label='predict')
            # plot fast observation prediction curve
            elif i == 1:
                ax.plot(fast_obses_next[:,0,0,j], label='predict')
            # plot total observation prediction curve
            elif i == 2:
                ax.plot(pred[:,0,0,j], label='predict')
            
            ax.set_title(item)
            ax.legend()
        plt.subplots_adjust(wspace=0.2)
        plt.savefig(log_dir+f"/test/{delta_t}/{figname}.pdf", dpi=300)
        plt.close()

    if system == '2S2F':
        c1_evolve_mae = torch.mean(torch.abs(recons_obses_next[:,0,0,0] - true[:,0,0,0]))
        c2_evolve_mae = torch.mean(torch.abs(recons_obses_next[:,0,0,1] - true[:,0,0,1]))
        return MSE, RMSE, MAE, MAPE, c1_evolve_mae.item(), c2_evolve_mae.item()
    elif system == '1S2F' or system == 'FHN' or system == '1S1F':
        return MSE, RMSE, MAE, MAPE
