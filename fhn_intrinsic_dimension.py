import numpy as np
import os
# import yaml
import pprint
# from munch import munchify
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
    # print(f'Number of samples: {latent.shape[0]}; Latent dimension: {latent.shape[1]}')
    latent = remove_duplicates(latent)
    print(f'Samples (unique): {latent.shape[0]}')
    
    estimator = ID_Estimator()
    # k_list = (latent.shape[0] * np.linspace(0.005, 0.025, 10)).astype('int')
    k_list = np.array(range(10, 30+1)).astype('int') # lrk: hyper-parameters to tune
    k_list = np.clip(k_list, a_min=3, a_max=latent.shape[0]-1-2) # lrk: avoid divide by zero error in LB algorithm
    print(f'List of numbers of nearest neighbors: {k_list}')
    if if_all_methods:
        dims = estimator.fit_all_methods(latent, k_list)
        np.save(os.path.join(vars_filepath, 'intrinsic_dimension_all_methods.npy'), dims)
    else:
        dims = estimator.fit(latent, k_list)
        np.save(os.path.join(vars_filepath, 'intrinsic_dimension.npy'), dims)


if __name__ == '__main__':
    
    import yaml
    from munch import munchify
    os.system("rm logs/ID.txt")
    for tau in np.arange(0.25, 10.01, 0.25):
        print('Tau: ', tau)
        cfg = load_config(filepath='config.yaml')
        cfg = munchify(cfg)

        dims_all = []

        random_seeds = range(1, 6)
        for random_seed in random_seeds:
            cfg.seed = random_seed
            log_dir = '_'.join([cfg.log_dir+str(tau),
                                cfg.dataset,
                                cfg.model_name,
                                str(cfg.seed)])
            var_log_dir = log_dir + '/variables_val'
            if os.path.exists(var_log_dir):
                eval_id_latent(var_log_dir, if_refine=False, if_all_methods=False)
                dims = np.load(os.path.join(var_log_dir, 'intrinsic_dimension.npy'))
            dims_all.append(dims)
            dim_mean = np.mean(dims_all)
            dim_std = np.std(dims_all)

        with open("logs/ID.txt", 'a+b') as fp:
            print(f'tau[{tau}] Mean(std): ' + f'{dim_mean:.4f} (+-{dim_std:.4f})\n')
            fp.write(f'{tau}--{dim_mean:.4f}\n'.encode('utf-8'))
            fp.flush()