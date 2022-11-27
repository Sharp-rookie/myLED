import os
import yaml
import torch
from munch import munchify
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings
warnings.simplefilter('ignore')

from fhn_model import FHN_VisDynamicsModel


def set_cpu_num(cpu_num):
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

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


def fhn_main(config_filepath):
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

    # define callback for selecting checkpoints during training
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir+"/lightning_logs/checkpoints/",
        filename="{epoch}_{val_loss}",
        verbose=True,
        monitor='val_loss',
        mode='min',
        save_top_k=-1
    )

    # define trainer
    trainer = Trainer(gpus=cfg.num_gpus,
                      max_epochs=cfg.epochs,
                      deterministic=True,
                      strategy='ddp_find_unused_parameters_false',
                      amp_backend='native',
                      default_root_dir=log_dir,
                      val_check_interval=1.0,
                      callbacks=checkpoint_callback
    )

    trainer.fit(model)

    print("Best model path:", checkpoint_callback.best_model_path)


if __name__ == '__main__':

    random_seed = 6
    config_filepath = f"config/config{random_seed}.yaml"
    fhn_main(config_filepath)