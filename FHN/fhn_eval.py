import os
import glob
import yaml
import torch
import shutil
import numpy as np
from tqdm import tqdm
from munch import munchify
from pytorch_lightning import seed_everything
import warnings
warnings.simplefilter('ignore')

from fhn_model import FHN_VisDynamicsModel
from fhn_dataset import FHNDataset, scaler


def set_cpu_num(cpu_num):
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)


def fhn_gather_latent_from_trained_high_dim_model(config_filepath, checkpoint_filepath):
    
    checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[0]
    cfg = load_config(filepath=config_filepath)
    # pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)
    set_cpu_num(cfg.cpu_num)

    log_dir = '_'.join([cfg.log_dir,
                        cfg.dataset,
                        cfg.model_name,
                        str(cfg.seed)])

    model = FHN_VisDynamicsModel(
        lr=cfg.lr,
        seed=cfg.seed,
        if_cuda=cfg.if_cuda,
        if_test=False,
        gamma=cfg.gamma,
        log_dir=log_dir,
        train_batch=cfg.train_batch,
        val_batch=cfg.val_batch,
        test_batch=cfg.test_batch,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        model_name=cfg.model_name,
        data_filepath=cfg.data_filepath,
        dataset=cfg.dataset,
        lr_schedule=cfg.lr_schedule
    )

    ckpt = torch.load(checkpoint_filepath)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to('cuda')
    model.eval()
    model.freeze()

    # prepare train and val dataset
    kwargs = {'num_workers': cfg.num_workers, 'pin_memory': True} if cfg.if_cuda else {}
    data_info_dict = {
                'truncate_data_batches': 2048, 
                'scaler': scaler(
                        scaler_type='MinMaxZeroOne',
                        data_min=np.loadtxt(cfg.data_filepath+"/data_min.txt"),
                        data_max=np.loadtxt(cfg.data_filepath+"/data_max.txt"),
                        channels=1,
                        common_scaling_per_input_dim=0,
                        common_scaling_per_channels=1,  # Common scaling for all channels
                    ), 
                }
    # data_info_dict['truncate_data_batches'] = 8192
    # train_dataset = FHNDataset('Data/Data/train',
    #                             data_cache_size=3,
    #                             data_info_dict=data_info_dict)
    data_info_dict['truncate_data_batches'] = 4096
    val_dataset = FHNDataset(cfg.data_filepath+'/val',
                              data_cache_size=3,
                              data_info_dict=data_info_dict)
    # prepare train and val loader
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                            batch_size=cfg.train_batch,
    #                                            shuffle=False,
    #                                            **kwargs)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=cfg.val_batch,
                                             shuffle=False,
                                             **kwargs)

    # # run train forward pass to save the latent vector for training the refine network later
    # all_latents = []
    # var_log_dir = os.path.join(log_dir, 'variables')
    # for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
    #     output, latent = model.model(data.cuda())
    #     # save the latent vectors
    #     for idx in range(data.shape[0]):
    #         latent_tmp = latent[idx].view(1, -1)[0]
    #         latent_tmp = latent_tmp.cpu().detach().numpy()
    #         all_latents.append(latent_tmp)

    # mkdir(var_log_dir+'_train')
    # np.save(os.path.join(var_log_dir+'_train', 'latent.npy'), all_latents)

    # run val forward pass to save the latent vector for validating the refine network later
    all_latents = []
    var_log_dir = os.path.join(log_dir, 'variables')
    plot_act_true = []
    plot_act_pred = []
    for batch_idx, (data, target) in enumerate(tqdm(val_loader)):
        output, latent = model.model(data.cuda())
        # save the latent vectors
        for idx in range(data.shape[0]):
            latent_tmp = latent[idx].view(1, -1)[0]
            latent_tmp = latent_tmp.cpu().detach().numpy()
            all_latents.append(latent_tmp)
        
        for i in range(target.shape[0]):
            if i%2 == 0:
                plot_act_true.append(target[i, 0, :101].unsqueeze(0))
                plot_act_true.append(target[i, 0, 101:].unsqueeze(0))
                plot_act_pred.append(output[i, 0, :101].unsqueeze(0))
                plot_act_pred.append(output[i, 0, 101:].unsqueeze(0))
    
    import matplotlib.pyplot as plt
    os.system('export MPLBACKEND=Agg')
    plot_act_true = torch.cat(plot_act_true, dim=0)[::10]
    plot_act_pred = torch.cat(plot_act_pred, dim=0)[::10]
    dimension = 55
    length = plot_act_true.shape[0]
    plt.figure()
    plt.plot(range(length), plot_act_true[:length, dimension].cpu(), label='True')
    plt.plot(range(length), plot_act_pred[:length, dimension].cpu(), label='Predict')
    plt.legend()
    plt.savefig(f"act_tau{[0.005,0.01,0.025,0.05,0.075,0.1,0.3,0.5][config_id-1]}_dimension{dimension}_seed{cfg.seed}.jpg", dpi=300)

    mkdir(var_log_dir+'_val')
    print(f'latent.npy save at: {var_log_dir}_val/latent.npy')
    np.save(os.path.join(var_log_dir+'_val', 'latent.npy'), all_latents)


if __name__ == '__main__':

    for config_id in range(8):
        checkpoint_filepath = f"logs/logs_tau{[0.005,0.01,0.025,0.05,0.075,0.1,0.3,0.5][config_id]}_fhn_fhn-ae_1/lightning_logs/checkpoints"
        fhn_gather_latent_from_trained_high_dim_model(f"config/config{config_id+1}.yaml", checkpoint_filepath)