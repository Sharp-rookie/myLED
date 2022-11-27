import os
import yaml
import numpy as np
from munch import munchify


def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)


def calculate_intrinsic_dimension_statistics():
    
    filename = 'intrinsic_dimension.npy'
    dims_all = []
    
    # Val
    for config_filepath in os.listdir("config/"):
        cfg = load_config(filepath=os.path.join("config/", config_filepath))
        cfg = munchify(cfg)
        log_dir = '_'.join([cfg.log_dir, cfg.dataset, cfg.model_name, str(cfg.seed)])
        vars_filepath = os.path.join(log_dir, 'variables')
        dims = np.load(os.path.join(vars_filepath, filename))
        dims_all.append(dims)
        dim_mean = np.mean(dims_all)
        dim_std = np.std(dims_all)
        print(f'Seed[{cfg.seed}] Mean (±std):', '%.2f (±%.2f)' % (dim_mean, dim_std))
        print(f'Seed[{cfg.seed}] Confidence interval:', '(%.1f, %.1f)' % (dim_mean-1.96*dim_std, dim_mean+1.96*dim_std))


if __name__ == '__main__':

    calculate_intrinsic_dimension_statistics()