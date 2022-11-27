import numpy as np
import os
import yaml
import pprint
from munch import munchify
from intrinsic_dimension_estimation import ID_Estimator


def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def remove_duplicates(X):
    return np.unique(X, axis=0)


def eval_id_latent(vars_filepath, if_refine, if_all_methods):
    if if_refine:
        latent = np.load(os.path.join(vars_filepath, 'refine_latent.npy'))
    else:
        latent = np.load(os.path.join(vars_filepath, 'latent.npy'))
    print(f'Number of samples: {latent.shape[0]}; Latent dimension: {latent.shape[1]}')
    latent = remove_duplicates(latent)
    print(f'Number of samples (duplicates removed): {latent.shape[0]}')
    
    estimator = ID_Estimator()
    k_list = (latent.shape[0] * np.linspace(0.008, 0.016, 5)).astype('int')
    print(f'List of numbers of nearest neighbors: {k_list}')
    if if_all_methods:
        dims = estimator.fit_all_methods(latent, k_list)
        np.save(os.path.join(vars_filepath, 'intrinsic_dimension_all_methods.npy'), dims)
    else:
        dims = estimator.fit(latent, k_list)
        np.save(os.path.join(vars_filepath, 'intrinsic_dimension.npy'), dims)


if __name__ == '__main__':
    
    for config_filepath in os.listdir("config/"):
        cfg = load_config(filepath=os.path.join("config/", config_filepath))
        cfg = munchify(cfg)
    
        log_dir = '_'.join([cfg.log_dir, cfg.dataset, cfg.model_name, str(cfg.seed)])
        vars_filepath = os.path.join(log_dir, 'variables_val')
        dims = eval_id_latent(vars_filepath, if_refine=False, if_all_methods=False)