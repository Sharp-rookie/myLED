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
        enc_net='MLP',
        e1_layer_n=3,
        sliding_window=True
        ):
        
    # prepare
    data_filepath = data_dir + 'tau_' + str(delta_t)
    log_dir = log_dir + f'seed{random_seed}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)

    # init model
    assert koopman_dim>=slow_dim, f"Value Error, koopman_dim is smaller than slow_dim({koopman_dim}<{slow_dim})"
    model = models.DynamicsEvolver(in_channels=channel_num, feature_dim=obs_dim, embed_dim=embedding_dim, slow_dim=slow_dim, redundant_dim=koopman_dim-slow_dim, tau_s=tau_s, device=device, data_dim=data_dim, enc_net=enc_net, e1_layer_n=e1_layer_n)
    tmp = '' if sliding_window else '_static'
    model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min" + tmp + ".txt").reshape(channel_num,obs_dim).astype(np.float32))
    model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max" + tmp + ".txt").reshape(channel_num,obs_dim).astype(np.float32))
    
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
         {'params': model.encoder_3.parameters()},
         {'params': model.decoder_1.parameters()}, 
         {'params': model.decoder_2.parameters()},
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
        
        losses = [[],[],[],[],[],[]]
        
        # train
        model.train()
        [lambda_curve[i].append(model.K_opt.Lambda[i].detach().cpu()) for i in range(slow_dim) ]
        for input, _, internl_units in train_loader:
            
            input = model.scale(input.to(device))[..., :obs_dim] # (batchsize,1,channel,obs_dim)
            
            # obs —— embedding —— slow representation —— embedding(reconstruction) —— obs(reconstruction)
            slow_var, embed = model.obs2slow(input)
            recons_obs, recons_embed = model.slow2obs(slow_var)
            _, adiabatic_embed = model.obs2slow(recons_obs)
            
            # calculate loss value
            adiabatic_loss = L1_loss(embed, adiabatic_embed)
            embed_reconstruct_loss = L1_loss(recons_embed, embed)
            obs_reconstruct_loss = MSE_loss(recons_obs, input)

            # ################
            # # n-step evolve
            # ################
            # koopman_var = model.slow2koopman(slow_var)
            # fast_obs = input - recons_obs
            # obs_evol_loss, slow_evol_loss, koopman_evol_loss = 0, 0, 0
            # for i in range(1, len(internl_units)):
                
            #     unit = model.scale(internl_units[i].to(device))[..., :obs_dim] # t+i
                
            #     #######################
            #     # slow component evolve
            #     #######################
            #     # obs ——> slow ——> koopman
            #     unit_slow_var, _ = model.obs2slow(unit)
            #     unit_koopman_var = model.slow2koopman(unit_slow_var)

            #     # slow evolve
            #     t = torch.tensor([delta_t * i], device=device) # delta_t
            #     unit_koopman_var_next = model.koopman_evolve(koopman_var, tau=t) # t ——> t + i*delta_t
            #     unit_slow_var_next = model.koopman2slow(unit_koopman_var_next)

            #     # koopman ——> slow ——> obs
            #     unit_slow_obs_next, _ = model.slow2obs(unit_slow_var_next)
                
            #     #######################
            #     # fast component evolve
            #     #######################
            #     # fast obs evolve
            #     unit_fast_obs_next, _ = model.lstm_evolve(fast_obs, T=i) # t ——> t + i*delta_t
                
            #     ################
            #     # calculate loss
            #     ################
            #     # total obs evolve
            #     unit_obs_next = unit_slow_obs_next + unit_fast_obs_next
                
            #     # evolve loss
            #     koopman_evol_loss += MSE_loss(unit_koopman_var_next, unit_koopman_var)
            #     slow_evol_loss += MSE_loss(unit_slow_var_next, unit_slow_var)
            #     obs_evol_loss += MSE_loss(unit_obs_next, unit)
            
            ###########
            # optimize
            ###########
            all_loss = 0.1*adiabatic_loss + embed_reconstruct_loss + obs_reconstruct_loss
            # all_loss = (0.1*adiabatic_loss + embed_reconstruct_loss + obs_reconstruct_loss) + (koopman_evol_loss + slow_evol_loss + obs_evol_loss) / n
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            
            # record loss
            losses[0].append(adiabatic_loss.detach().item())
            losses[1].append(embed_reconstruct_loss.detach().item())
            losses[2].append(obs_reconstruct_loss.detach().item())
            # losses[3].append(koopman_evol_loss.detach().item())
            # losses[4].append(slow_evol_loss.detach().item())
            # losses[5].append(obs_evol_loss.detach().item())
        
        train_loss.append([np.mean(losses[0]), np.mean(losses[1]), np.mean(losses[2])])
        # train_loss.append([np.mean(losses[0]), np.mean(losses[1]), np.mean(losses[2]), np.mean(losses[3]), np.mean(losses[4]), np.mean(losses[5])])
        
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
            # slow_obses_next = []
            # total_obses_next = []
            
            model.eval()
            for input, target, _ in val_loader:

                if system in ['HalfMoon', 'ToggleSwitch']:
                    hidden_slow.append(input[..., obs_dim:].cpu())
                
                input = model.scale(input.to(device))[..., :obs_dim] # (batchsize,1,channel_num,feature_dim)
                target = model.scale(target.to(device))[..., :obs_dim]

                # obs —— embedding —— slow representation —— embedding(reconstruction) —— obs(reconstruction)
                slow_var, embed = model.obs2slow(input)
                recons_obs, recons_embed = model.slow2obs(slow_var)
                _, adiabatic_embed = model.obs2slow(recons_obs)

                # # koopman evolve
                # koopman_var = model.slow2koopman(slow_var)
                # t = torch.tensor([tau_s-delta_t], device=device)
                # koopman_var_next = model.koopman_evolve(koopman_var, tau=t)
                # slow_var_next = model.koopman2slow(koopman_var_next)
                # slow_obs_next, _ = model.slow2obs(slow_var_next)
                
                # # fast obs evolve
                # fast_obs = input - recons_obs
                # fast_obs_next, _ = model.lstm_evolve(fast_obs, T=n)
                
                # # total obs evolve
                # total_obs_next = slow_obs_next + fast_obs_next

                # record results
                inputs.append(input.cpu())
                slow_vars.append(slow_var.cpu())
                targets.append(target.cpu())
                recons_obses.append(recons_obs.cpu())
                embeds.append(embed.cpu())
                recons_embeds.append(recons_embed.cpu())
                adiabatic_embeds.append(adiabatic_embed.cpu())
                # slow_obses_next.append(slow_obs_next.cpu())
                # total_obses_next.append(total_obs_next.cpu())
            
            # trans to tensor
            inputs = torch.concat(inputs, axis=0)
            slow_vars = torch.concat(slow_vars, axis=0)
            targets = torch.concat(targets, axis=0)
            recons_obses = torch.concat(recons_obses, axis=0)
            embeds = torch.concat(embeds, axis=0)
            recons_embeds = torch.concat(recons_embeds, axis=0)
            adiabatic_embeds = torch.concat(adiabatic_embeds, axis=0)
            # slow_obses_next = torch.concat(slow_obses_next, axis=0)
            # total_obses_next = torch.concat(total_obses_next, axis=0)
            if system in ['HalfMoon', 'ToggleSwitch']:
                hidden_slow = torch.concat(hidden_slow, axis=0)

            # slow_var1 = slow_vars[:, :1]
            # slow_var2 = slow_vars[:, 1:2]
            # cos_sim = torch.nn.functional.cosine_similarity(slow_var1, slow_var2, dim=0)[0]
            
            # cal loss
            adiabatic_loss = L1_loss(embeds, adiabatic_embeds)
            embed_reconstruct_loss = L1_loss(recons_embeds, embeds)
            obs_reconstruct_loss = MSE_loss(recons_obses, inputs)
            # obs_evol_loss = MSE_loss(total_obses_next, targets)
            if is_print: print(f'\rTau[{tau_s}] | epoch[{epoch}/{learn_max_epoch}] | val: adiabatic={adiabatic_loss:.5f}, emebd_recons={embed_reconstruct_loss:.5f}, obs_recons={obs_reconstruct_loss:.5f}', end='')
            # if is_print: print(f'\rTau[{tau_s}] | epoch[{epoch}/{learn_max_epoch}] | val: adiabatic={adiabatic_loss:.5f}, emebd_recons={embed_reconstruct_loss:.5f}, obs_recons={obs_reconstruct_loss:.5f}, cosine_similarity={cos_sim:.5f}', end='')
            # if is_print: print(f'\rTau[{tau_s}] | epoch[{epoch}/{learn_max_epoch}] | val: adiabatic={adiabatic_loss:.5f}, emebd_recons={embed_reconstruct_loss:.5f}, obs_recons={obs_reconstruct_loss:.5f}, obs_evol={obs_evol_loss:.5f}, cosine_similarity={cos_sim:.5f}', end='')
            
            
            # plot per 5 epoch
            if epoch % 5 == 0:
                os.makedirs(log_dir+f"/val/epoch-{epoch}/", exist_ok=True)
                sample = 1
                input_data = model.descale(inputs)

                # plot slow variable vs input
                if 'FHN' in system:
                    dim = inputs.shape[-1]
                    num = 5
                    interval = dim//num

                    plt.figure(figsize=(16,5+2*(slow_dim-1)))
                    plt.title('Val Reconstruction Curve')
                    for id_var in range(slow_dim):
                        for index, item in enumerate([f'c{k}' for k in range(0, dim, interval)]):
                            plt.subplot(slow_dim, num, index+1+dim*(id_var))
                            plt.scatter(input_data[::sample,0,0,index*interval], slow_vars[::sample, id_var], s=5)
                            plt.xlabel(item)
                            plt.ylabel(f'U{id_var+1}')
                    plt.subplots_adjust(wspace=0.35, hspace=0.35)
                    plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_vs_input_activator.png", dpi=300)
                    plt.close()

                    plt.figure(figsize=(16,5+2*(slow_dim-1)))
                    plt.title('Val Reconstruction Curve')
                    for id_var in range(slow_dim):
                        for index, item in enumerate([f'c{k}' for k in range(0, dim, interval)]):
                            plt.subplot(slow_dim, num, index+1+dim*(id_var))
                            plt.scatter(input_data[::sample,0,1,index*interval], slow_vars[::sample, id_var], s=5)
                            plt.xlabel(item)
                            plt.ylabel(f'U{id_var+1}')
                    plt.subplots_adjust(wspace=0.35, hspace=0.35)
                    plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_vs_input_inhibitor.png", dpi=300)
                    plt.close()
                elif 'PNAS17' in system:
                    plt.figure(figsize=(16,5+2*(slow_dim-1)))
                    plt.title('Val Reconstruction Curve')
                    for id_var in range(slow_dim):
                        ax1 = plt.subplot(slow_dim, 2, 1+2*(id_var))
                        ax1.scatter(input_data[::sample,0,0,0], slow_vars[::sample, id_var], s=5)
                        ax1.set_xlabel('u')
                        ax1.set_ylabel(f'U{id_var+1}')
                        ax2 = plt.subplot(slow_dim, 2, 2+2*(id_var))
                        ax2.scatter(input_data[::sample,0,1,0], slow_vars[::sample, id_var], s=5)
                        ax2.set_xlabel('v')
                        ax2.set_ylabel(f'U{id_var+1}')
                    plt.subplots_adjust(wspace=0.35, hspace=0.35)
                    plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_vs_input.png", dpi=300)
                    plt.close()
                else:
                    plt.figure(figsize=(16,5+2*(slow_dim-1)))
                    plt.title('Val Reconstruction Curve')
                    for id_var in range(slow_dim):
                        for index, item in enumerate([f'c{k}' for k in range(inputs.shape[-1])]):
                            plt.subplot(slow_dim, inputs.shape[-1], index+1+inputs.shape[-1]*(id_var))
                            plt.scatter(input_data[::sample,0,0,index], slow_vars[::sample, id_var], s=5)
                            plt.xlabel(item)
                            plt.ylabel(f'U{id_var+1}')
                    plt.subplots_adjust(wspace=0.35, hspace=0.35)
                    plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_vs_input.png", dpi=300)
                    plt.close()

                if system in ['HalfMoon', 'ToggleSwitch']:
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
                    plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_vs_hidden.png", dpi=300)
                    plt.close()
                
                if system == '1S1F':
                    # plot slow variable vs constant variable
                    plt.figure(figsize=(9,9))
                    plt.title('Extracted VS Constant')
                    plt.scatter(input_data[::sample,0,0,0]+2*input_data[::sample,0,0,0], slow_vars[::sample, 0], s=5)
                    plt.xlabel('X+2Y')
                    plt.ylabel('U')
                    plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_vs_constant.png", dpi=300)
                    plt.close()
                
                # plot slow variable
                if 'PNAS17' in system:
                    plt.figure(figsize=(16,5+2*(slow_dim-1)))
                    plt.title('Slow variable Curve')
                    for id_var in range(slow_dim):
                        ax1 = plt.subplot(slow_dim, 2, 1+2*id_var)
                        ax1.plot(inputs[:,0,0,0], label=f'u')
                        ax1.plot(slow_vars[:, id_var], label=f'U{id_var+1}')
                        ax1.set_xlabel('t / sample interval')
                        ax1.legend()
                        ax2 = plt.subplot(slow_dim, 2, 2+2*id_var)
                        ax2.plot(inputs[:,0,1,0], label=f'v')
                        ax2.plot(slow_vars[:, id_var], label=f'U{id_var+1}')
                        ax2.set_xlabel('t / sample interval')
                        ax2.legend()
                    plt.subplots_adjust(wspace=0.35, hspace=0.35)
                    plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_variable.png", dpi=300)
                    plt.close()
                else:
                    plt.figure(figsize=(12,5+2*(slow_dim-1)))
                    plt.title('Slow variable Curve')
                    for id_var in range(slow_dim):
                        ax = plt.subplot(slow_dim, 1, 1+id_var)
                        for idx in range(slow_dim):
                            ax.plot(inputs[:,0,0,idx], label=f'c{idx+1}')
                        ax.plot(slow_vars[:, id_var], label=f'U{id_var+1}')
                        ax.set_xlabel('t / sample interval')
                        ax.legend()
                    plt.subplots_adjust(wspace=0.35, hspace=0.35)
                    plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_variable.png", dpi=300)
                    plt.close()

                # plot observation and reconstruction
                if 'FHN' in system:
                    dim = targets.shape[-1]
                    num = 5
                    interval = dim//num

                    plt.figure(figsize=(16,5))
                    for j, item in enumerate([f'c{k}' for k in range(0, dim, interval)]):
                        ax = plt.subplot(1,num,j+1)
                        ax.plot(inputs[:,0,0,j*interval], label='activator')
                        ax.plot(recons_obses[:,0,0,j*interval], label='slow_obs')
                        ax.set_title(item)
                        ax.legend()
                    plt.subplots_adjust(wspace=0.2)
                    plt.savefig(log_dir+f"/val/epoch-{epoch}/recons_obs_activator.png", dpi=300)
                    plt.close()

                    plt.figure(figsize=(16,5))
                    for j, item in enumerate([f'c{k}' for k in range(0, dim, interval)]):
                        ax = plt.subplot(1,num,j+1)
                        ax.plot(inputs[:,0,1,j*interval], label='inhibitor')
                        ax.plot(recons_obses[:,0,1,j*interval], label='slow_obs')
                        ax.set_title(item)
                        ax.legend()
                    plt.subplots_adjust(wspace=0.2)
                    plt.savefig(log_dir+f"/val/epoch-{epoch}/recons_obs_inhibitor.png", dpi=300)
                    plt.close()
                elif 'PNAS17' in system:
                    plt.figure(figsize=(16,5))
                    ax = plt.subplot(121)
                    ax.plot(inputs[:,0,0,0], label='all_obs')
                    ax.plot(recons_obses[:,0,0,0], label='slow_obs')
                    ax.set_title('u')
                    ax.legend()
                    ax = plt.subplot(122)
                    ax.plot(inputs[:,0,1,0], label='all_obs')
                    ax.plot(recons_obses[:,0,1,0], label='slow_obs')
                    ax.set_title('v')
                    ax.legend()
                    plt.subplots_adjust(wspace=0.2)
                    plt.savefig(log_dir+f"/val/epoch-{epoch}/recons_obs.png", dpi=300)
                    plt.close()
                    
                    # phase
                    plt.figure(figsize=(6,6))
                    plt.plot(recons_obses[:,0,1,0], recons_obses[:,0,0,0], label='identified slow variable')
                    plt.plot(inputs[:,0,1,0], inputs[:,0,0,0], label='trajectory')
                    # u1 = recons_obses.cpu().detach().numpy()[:,0,0,0]
                    # v1 = u1 - u1**3/3
                    # plt.plot(v1, u1, label='slow manifold')
                    plt.xlabel('v')
                    plt.ylabel('u')
                    plt.legend()
                    plt.savefig(log_dir+f"/val/epoch-{epoch}/phase.png", dpi=300)
                    plt.close()
                else:
                    plt.figure(figsize=(16,5))
                    for j, item in enumerate([f'c{k}' for k in range(targets.shape[-1])]):
                        ax = plt.subplot(1,targets.shape[-1],j+1)
                        ax.plot(inputs[:,0,0,j], label='all_obs')
                        ax.plot(recons_obses[:,0,0,j], label='slow_obs')
                        ax.set_title(item)
                        ax.legend()
                    plt.subplots_adjust(wspace=0.2)
                    plt.savefig(log_dir+f"/val/epoch-{epoch}/recons_obs.png", dpi=300)
                    plt.close()

                if system == '2S2F':

                    # TODO: 有问题！！！

                    np.savez(log_dir+f"/val/epoch-{epoch}/slow_vs_input.npz", slow_vars=slow_vars, inputs=inputs, recons=recons_obses)

                    # num = int(5.1/0.01)
                    
                    # c1, c2 = np.meshgrid(np.linspace(-3, 3, 60), np.linspace(-3, 3, 60))
                    # omega = 3
                    # c3 = np.sin(omega*c1)*np.sin(omega*c2)
                    # c4 = 1/((1+np.exp(-omega*c1))*(1+np.exp(-omega*c2)))
                    
                    # fig = plt.figure(figsize=(16,6))
                    # descale = model.descale(recons_obses)
                    # for i, (c, trace) in enumerate(zip([c3,c4], [descale[:num,0,0,2:3],descale[:num,0,0,3:4]])):
                    #     ax = plt.subplot(1,2,i+1,projection='3d')

                    #     # plot the slow manifold and c3,c4 trajectory
                    #     ax.scatter(c1, c2, c, marker='.', color='k', label=rf'Points on slow-manifold surface')
                    #     ax.plot(descale[:num,0,0,:1], descale[:num,0,0,1:2], trace, linewidth=2, color="r", label=rf'Slow trajectory')
                    #     print(descale[:num,0,0,:1].item());exit(0)
                    #     ax.set_xlabel(r"$c_1$", labelpad=10, fontsize=18)
                    #     ax.set_ylabel(r"$c_2$", labelpad=10, fontsize=18)
                    #     ax.set_zlim(0, 2)
                    #     ax.text2D(0.85, 0.65, rf"$c_{2+i+1}$", fontsize=18, transform=ax.transAxes)
                    #     # ax.zaxis.set_rotate_label(False)  # disable automatic rotation
                    #     ax.invert_xaxis()
                    #     ax.invert_yaxis()
                    #     ax.grid(False)
                    #     ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                    #     ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                    #     ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                    #     # ax.view_init(elev=25., azim=-60.) # view direction: elve=vertical angle ,azim=horizontal angle
                    #     ax.view_init(elev=0., azim=-90.) # view direction: elve=vertical angle ,azim=horizontal angle
                    #     plt.tick_params(labelsize=16)
                    #     ax.xaxis.set_major_locator(plt.MultipleLocator(1))
                    #     ax.yaxis.set_major_locator(plt.MultipleLocator(1))
                    #     ax.zaxis.set_major_locator(plt.MultipleLocator(1))
                    #     if i == 1:
                    #         plt.legend()
                    #     plt.subplots_adjust(bottom=0., top=1.)
                    
                    plt.savefig(log_dir+f"/val/epoch-{epoch}/recons_manifold.png", dpi=300)
                    plt.close()

                # # plot prediction vs true
                # plt.figure(figsize=(16,5))
                # plt.title('Prediction')
                # for j, item in enumerate([f'c{k}' for k in range(targets.shape[-1])]):
                #     ax = plt.subplot(1,targets.shape[-1],j+1)
                #     ax.plot(targets[:,0,0,j], label='true')
                #     ax.plot(total_obses_next[:,0,0,j], label='pred')
                #     ax.set_title(item)
                #     ax.legend()
                # plt.subplots_adjust(wspace=0.2)
                # plt.savefig(log_dir+f"/val/epoch-{epoch}/prediction.png", dpi=300)
                # plt.close()
        
                # save model
                torch.save(model.state_dict(), log_dir+f"/checkpoints/epoch-{epoch}.ckpt")
    
    # plot loss curve
    train_loss = np.array(train_loss)
    plt.figure()
    for i, item in enumerate(['adiabatic','embed_reconstruct_loss','obs_reconstruct_loss']):
    # for i, item in enumerate(['adiabatic','embed_reconstruct_loss','obs_reconstruct_loss', 'koopman_evol_loss', 'slow_evol_loss', 'obs_evol_loss']):
        plt.plot(train_loss[:, i], label=item)
    plt.xlabel('epoch')
    plt.legend()
    plt.title('Training Loss Curve')
    plt.savefig(log_dir+'/train_loss_curve.png', dpi=300)

    # plot Koopman Lambda curve
    plt.figure(figsize=(10,10))
    marker = ['o', '^', '+', 's', '*', 'x']
    for i in range(slow_dim):
        plt.plot(lambda_curve[i], marker=marker[i%len(marker)], markersize=6, label=rf'$\lambda_{i}$')
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel(r'$\Lambda$', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(log_dir+'/K_lambda_curve.png', dpi=300)
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
        enc_net='MLP',
        e1_layer_n=3
        ):
        
    # prepare
    data_filepath = data_dir + 'tau_' + str(delta_t)
    log_dir = log_dir + f'seed{random_seed}'

    # load model
    model = models.DynamicsEvolver(in_channels=channel_num, feature_dim=obs_dim, embed_dim=embedding_dim, slow_dim=slow_dim, redundant_dim=koopman_dim-slow_dim, tau_s=tau_s, device=device, data_dim=data_dim, enc_net=enc_net, e1_layer_n=e1_layer_n)
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

            if system in ['HalfMoon', 'ToggleSwitch']:
                hidden_slow.append(input[..., obs_dim:].cpu())
            
            input = model.scale(input.to(device))[..., :obs_dim]
            target = model.scale(target.to(device))[..., :obs_dim]
        
            # obs —— embedding —— slow representation —— koopman
            slow_var, _ = model.obs2slow(input)
            recons_obs, _ = model.slow2obs(slow_var)
            koopman_var = model.slow2koopman(slow_var)
            
            # koopman evolve
            t = torch.tensor([delta_t], device=device)
            koopman_var_next = model.koopman_evolve(koopman_var, tau=t)
            slow_var_next = model.koopman2slow(koopman_var_next)
            slow_obs_next = model.slow2obs(slow_var_next)
            
            # fast obs evolve
            fast_obs = input - recons_obs
            fast_obs_next, _ = model.lstm_evolve(fast_obs, T=n)
            
            # total obs evolve
            total_obs_next = slow_obs_next + fast_obs_next

            inputs.append(input)
            targets.append(target)
            slow_vars.append(slow_var)
            recons_obses.append(recons_obs)
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
    plt.savefig(log_dir+f"/test/{delta_t}/slow_extract.png", dpi=300)
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
    plt.savefig(log_dir+f"/test/{delta_t}/slow_vs_input.png", dpi=150)
    plt.close()

    if system in ['HalfMoon', 'ToggleSwitch']:
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
        plt.savefig(log_dir+f"/test/{delta_t}/slow_vs_hidden.png", dpi=300)
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
        plt.savefig(log_dir+f"/test/{delta_t}/{figname}.png", dpi=300)
        plt.close()

    if system == '2S2F':
        c1_evolve_mae = torch.mean(torch.abs(recons_obses_next[:,0,0,0] - true[:,0,0,0]))
        c2_evolve_mae = torch.mean(torch.abs(recons_obses_next[:,0,0,1] - true[:,0,0,1]))
        return MSE, RMSE, MAE, MAPE, c1_evolve_mae.item(), c2_evolve_mae.item()
    elif system in ['1S1F', '1S2F', 'ToggleSwitch', 'SignalingCascade', 'HalfMoon'] or 'FHN' in system:
        return MSE, RMSE, MAE, MAPE
