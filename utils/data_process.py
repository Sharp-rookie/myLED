import numpy as np
import matplotlib.pyplot as plt


def findNearestPoint(data_t, start=0, object_t=10.0):
    """Find the nearest time point to object time"""

    index = start

    if index >= len(data_t):
        return index

    while not (data_t[index] <= object_t and data_t[index+1] > object_t):
        if index < len(data_t)-2:
            index += 1
        elif index == len(data_t)-2: # last one
            index += 1
            break
    
    return index

def time_discretization(seed, total_t):
    """Time-forward NearestNeighbor interpolate to discretizate the time"""

    data = np.load(f'Data/origin/{seed}/origin.npz')
    data_t = data['t']
    data_X = data['X']
    data_Y = data['Y']
    data_Z = data['Z']

    dt = 5e-3
    current_t = 0.0
    index = 0
    t, X, Y, Z = [], [], [], []
    while current_t < total_t:
        index = findNearestPoint(data_t, start=index, object_t=current_t)
        t.append(current_t)
        X.append(data_X[index])
        Y.append(data_Y[index])
        Z.append(data_Z[index])

        current_t += dt

        if seed == 1:
            print(f'\rSeed[{seed}] interpolating {current_t:.6f}/{total_t}', end='')

    plt.figure(figsize=(16,4))
    plt.title(f'dt = {dt}')
    ax1 = plt.subplot(1,3,1)
    ax1.set_title('X')
    plt.plot(t, X, label='X')
    plt.xlabel('time / s')
    ax2 = plt.subplot(1,3,2)
    ax2.set_title('Y')
    plt.plot(t, Y, label='Y')
    plt.xlabel('time / s')
    ax3 = plt.subplot(1,3,3)
    ax3.set_title('Z')
    plt.plot(t, Z, label='Z')
    plt.xlabel('time / s')

    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.9,
        bottom=0.1,
        wspace=0.2
    )
    plt.savefig(f'Data/origin/{seed}/data.png', dpi=500)

    np.savez(f'Data/origin/{seed}/data.npz', dt=dt, t=t, X=X, Y=Y, Z=Z)