import os
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .common import load_config


def kNN(X, n_neighbors, n_jobs):
    neigh = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs).fit(X)
    dists, inds = neigh.kneighbors(X)
    return dists, inds


def Levina_Bickel(X, dists, k):
    m = np.log(dists[:, k:k+1] / dists[:, 1:k])
    m = (k-2) / np.sum(m, axis=1)
    # m = np.sum(m, axis=1) / (k-2)
    dim = np.mean(m)
    return dim

class ID_Estimator:
    def __init__(self, method='Levina_Bickel'):
        self.all_methods = ['Levina_Bickel', 'MiND_ML', 'MiND_KL', 'Hein', 'CD']
        self.set_method(method)
    
    def set_method(self, method='Levina_Bickel'):
        if method not in self.all_methods:
            assert False, 'Unknown method!'
        else:
            self.method = method
        
    def fit(self, X, k_list=20, n_jobs=4):
        if self.method in ['Hein', 'CD']:
            assert False, f"{self.method} not implemented!"
        else:
            if np.isscalar(k_list):
                k_list = np.array([k_list])
            else:
                k_list = np.array(k_list)
            kmax = np.max(k_list) + 2
            dists, inds = kNN(X, kmax, n_jobs)
            dims = []
            for k in k_list:
                if self.method == 'Levina_Bickel':
                    dims.append(Levina_Bickel(X, dists, k))
                elif self.method == 'MiND_ML':
                    assert False, f"{self.method} not implemented!"
                elif self.method == 'MiND_KL':
                    assert False, f"{self.method} not implemented!"
                else:
                    pass
            if len(dims) == 1:
                return dims[0]
            else:
                return np.array(dims)


def remove_duplicates(X):
    return np.unique(X, axis=0)


def eval_id_latent(vars_filepath, if_refine, if_all_methods):
    if if_refine:
        latent = np.load(os.path.join(vars_filepath, 'refine_latent.npy'))
    else:
        latent = np.load(os.path.join(vars_filepath, 'latent.npy'))
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