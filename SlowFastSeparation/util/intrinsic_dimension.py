import skdim
import numpy as np


class ID_Estimator:
    def __init__(self, method='MLE'):
        self.all_methods = ['MiND_ML', 'MLE', 'MADA', 'PCA', 'DANCo']
        self.set_method(method)
    
    def set_method(self, method='MLE'):
        if method not in self.all_methods:
            assert False, 'Unknown method!'
        else:
            self.method = method
            
    def fit(self, X, k_list=20):

        if np.isscalar(k_list):
            k_list = np.array([k_list])
        else:
            k_list = np.array(k_list)

        dims = []
        for k in k_list:
            assert k>0, "k must be larger than 0"
            if self.method == 'MiND_ML':
                dims.append(np.mean(skdim.id.MiND_ML().fit_pw(X, n_neighbors=k).dimension_pw_))
            elif self.method == 'MLE':
                dims.append(np.mean(skdim.id.MLE().fit(X, n_neighbors=k).dimension_pw_))
            elif self.method == 'MADA':
                dims.append(np.mean(skdim.id.MADA().fit(X, n_neighbors=k).dimension_pw_))
            elif self.method == 'MADA':
                dims.append(np.mean(skdim.id.MADA().fit(X, n_neighbors=k).dimension_pw_))
            elif self.method == 'PCA':
                dims.append(np.mean(skdim.id.lPCA().fit_pw(X, n_neighbors=k).dimension_pw_))
            elif self.method == 'DANCo':
                dims.append(np.mean(skdim.id.DANCo().fit_pw(X, n_neighbors=k).dimension_pw_))
            else:
                assert False, f"{self.method} not implemented!"
        if len(dims) == 1:
            return dims[0]
        else:
            return np.array(dims)


def eval_id_embedding(vars_filepath, method='MLE', is_print=False, max_point=1000, k_list=20):
    
    embedding = np.load(vars_filepath+'/embedding.npy')
    if len(embedding) > max_point: 
        embedding = embedding[np.random.choice(len(embedding), max_point, replace=False)]

    if is_print: print(f'\n[{method}] Samples (origin): {embedding.shape[0]}')
    embedding = np.unique(embedding, axis=0)
    if is_print: print(f'[{method}] Samples (unique): {embedding.shape[0]}')
    if is_print: print(f'[{method}] Numbers of nearest neighbors: {k_list}')
    
    estimator = ID_Estimator(method=method)
    dims = estimator.fit(embedding, k_list)
    np.save(vars_filepath+f'/id_{method}.npy', dims)

    if is_print: print(f'[{method}] Intrinsic dimenstion: {dims.round(1)}')

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 6))
    # plt.plot(k_list, dims, 'o-')
    # plt.xlabel('k')
    # plt.ylabel('ID')
    # plt.title(f'ID Estimation ({method})')
    # plt.savefig(f'id_{method}.png')