import skdim
import numpy as np
import matplotlib.pyplot as plt

from util.intrinsic_dimension import ID_Estimator


def eval_id_data(data_path, xdim, method='MLE', is_print=False, max_point=1000, k_list=20):
    
    data = np.load(data_path)['trace']
    data = data.reshape(data.shape[0], data.shape[1], 2*xdim)
    data = data.reshape(-1, 2*xdim)
    
    if len(data) > max_point:
        data = data[np.random.choice(len(data), max_point, replace=False)]

    if is_print: print(f'\n[{method}] Samples (origin): {data.shape[0]}')
    data = np.unique(data, axis=0)
    if is_print: print(f'[{method}] Samples (unique): {data.shape[0]}')
    if is_print: print(f'[{method}] Numbers of nearest neighbors: {k_list}')
    
    estimator = ID_Estimator(method=method)
    dims = estimator.fit(data, k_list)

    if is_print: print(f'[{method}] Intrinsic dimenstion: {dims.round(1)}')
    
    return dims
    

# calculae ID
def cal_id_data(xdim, method='MLE', is_print=False, max_point=1000, k_list=20):
    data_path = f'Data/PNAS17_xdim{xdim}_delta0.2_du0.0-random/origin/origin.npz'
    dims = eval_id_data(data_path, xdim, method=method, is_print=is_print, max_point=max_point, k_list=k_list)
    return np.mean(dims)

# k_list = np.array(range(int(0.01*len(data)), int(0.05*len(data)))).astype('int')
# k_list = np.clip(k_list, a_min=1, a_max=data.shape[0]-1)
k_list = 10
max_point = 1000
xdim_list = [1,2,3,4,5,10,20,30,40,50]

# MLE
try:
    data = np.load('MLE.npz')
    mle_mean, std = data['mean'], data['std']
except:
    mean, std = [], []
    for xdim in xdim_list:
        tmp = []
        for _ in range(10):
            tmp.append(cal_id_data(xdim, 'MLE', is_print=False, max_point=max_point, k_list=k_list))
        mean.append(np.mean(tmp))
        std.append(np.std(tmp))
    np.savez('MLE.npz', mean=mean, std=std)

# MADA
try:
    data = np.load('MADA.npz')
    mada_mean, std = data['mean'], data['std']
except:
    mean, std = [], []
    for xdim in xdim_list:
        tmp = []
        for _ in range(10):
            tmp.append(cal_id_data(xdim, 'MADA', is_print=False, max_point=max_point, k_list=k_list))
        mean.append(np.mean(tmp))
        std.append(np.std(tmp))
    np.savez('MADA.npz', mean=mean, std=std)

# MiND_ML
try:
    data = np.load('MiND.npz')
    mind_mean, std = data['mean'], data['std']
except:
    mean, std = [], []
    for xdim in xdim_list:
        tmp = []
        for _ in range(1):
            tmp.append(cal_id_data(xdim, 'MiND_ML', is_print=False, max_point=max_point, k_list=k_list))
        mean.append(np.mean(tmp))
        std.append(np.std(tmp))
    np.savez('MiND.npz', mean=mean, std=std)

plt.figure(figsize=(6,6))
plt.plot(xdim_list, mle_mean, label='MLE', marker='+', markersize=6)
plt.plot(xdim_list, mada_mean, label='MADA', marker='*', markersize=6)
plt.plot(xdim_list, mind_mean, label='MiND', marker='^', markersize=6)
plt.plot(xdim_list, xdim_list, label='ground truth')
plt.legend()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(0, 51)
plt.xlim(0, 51)
plt.xlabel('Space dimension', fontsize=18)
plt.ylabel('Intrinsic dimension', fontsize=18)
plt.savefig('ID.png', dpi=300)

exit(0)

# DANCo
k_list = 15
try:
    data = np.load('DANCo.npz')
    mean, std = data['mean'], data['std']
except:
    mean, std = [], []
    for xdim in xdim_list:
        tmp = []
        for _ in range(1):
            tmp.append(cal_id_data(xdim, 'DANCo', is_print=False, max_point=max_point, k_list=k_list))
        mean.append(np.mean(tmp))
        std.append(np.std(tmp))
    np.savez('DANCo.npz', mean=mean, std=std)
ax = plt.subplot(2, 2, 4)
ax.errorbar(xdim_list, mean, yerr=std, fmt='o')
ax.plot(xdim_list, xdim_list, label='ground truth')
ax.set_ylim(0, 51)
ax.set_xlim(0, 51)
ax.set_title('DANCo')
ax.set_xlabel('Space dimension')
ax.set_ylabel('Intrinsic dimension')

plt.savefig('ID.png', dpi=300)

# MiND_id = cal_id_data(xdim, 'MiND_ML', is_print=is_print, max_point=max_point, k_list=15)
# DANCo_id = cal_id_data(xdim, 'DANCo', is_print=is_print, max_point=max_point, k_list=15)
# print(f'MLE={MLE_id:.1f}, DANCo={DANCo_id:.1f}, MiND_ML={MiND_id:.1f}')