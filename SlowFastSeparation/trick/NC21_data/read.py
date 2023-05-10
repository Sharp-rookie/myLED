import os
import scipy.io
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# Convariant Matrix
def cov_dist(u0, v0):

    def distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    try:
        data = np.load('cov.npz')
        dists = data['dists']
        u0_cov = data['u0_cov']
        v0_cov = data['v0_cov']
        u0_pear = data['u0_pear']
        v0_pear = data['v0_pear']
    except:
        data_turple = []
        sample_num = 20
        for i in tqdm(np.random.choice(301**2, size=sample_num, replace=False)):
            for j in range(301**2):
                xi, yi = i // 301, i % 301
                xj, yj = j // 301, j % 301
                data_turple.append([distance(x[xi], y[yi], x[xj], y[yj]), u0[xi,yi], v0[xi,yi], u0[xj,yj], v0[xj,yj]])
        data_turple = sorted(data_turple, key=lambda x: x[0])
        data_turple = np.unique(data_turple, axis=0)

        dists = []
        u0_cov, v0_cov = [], []
        u0_pear, v0_pear = [], []
        for dist in tqdm(np.unique(data_turple[:,0])):
            samples = data_turple[data_turple[:,0]==dist]
            if len(samples) < 50: 
                continue
            dists.append(dist)
            # covariances
            u0_cov.append(np.cov(samples[:,1], samples[:,3]))
            v0_cov.append(np.cov(samples[:,2], samples[:,4]))
            # pearsonr
            u0_pear.append(pearsonr(samples[:,1], samples[:,3]))
            v0_pear.append(pearsonr(samples[:,2], samples[:,4]))
        
        np.savez('cov.npz', dists=dists, u0_cov=u0_cov, v0_cov=v0_cov, u0_pear=u0_pear, v0_pear=v0_pear)
    
    plt.figure(figsize=(16,16))
    plt.subplot(221)
    plt.plot(dists, np.array(u0_cov)[:,0,1])
    plt.xlabel('distance', fontsize=18)
    plt.ylabel('covariance', fontsize=18)
    plt.title('u0', fontsize=18)
    plt.subplot(222)
    plt.plot(dists, np.array(v0_cov)[:,0,1])
    plt.xlabel('distance', fontsize=18)
    plt.ylabel('covariance', fontsize=18)
    plt.title('v0', fontsize=18)
    plt.subplot(223)
    plt.plot(dists, np.array(u0_pear)[:,0])
    plt.xlabel('distance', fontsize=18)
    plt.ylabel('pearsonr', fontsize=18)
    plt.title('u0', fontsize=18)
    plt.subplot(224)
    plt.plot(dists, np.array(v0_pear)[:,0])
    plt.xlabel('distance', fontsize=18)
    plt.ylabel('pearsonr', fontsize=18)
    plt.title('v0', fontsize=18)
    plt.savefig('cov.png')


def map_uv(u0, v0):

    # 画出uv的分布
    plt.figure(figsize=(8,8))
    # plt.scatter(v0.flatten(), u0.flatten(), s=0.1, c='k')
    plt.scatter(v0[100], u0[100], s=1, c='k')
    plt.xlabel('v', fontsize=18)
    plt.ylabel('u', fontsize=18)
    plt.savefig('uv.png', dpi=300)

    print(pearsonr(u0.flatten(), v0.flatten()))


def plot_data(u, v, x, y, u0, v0):

    # # 直方图
    # plt.figure(figsize=(8,8))
    # plt.hist(u[:,:,0].flatten(), bins=100)
    # plt.xlabel('u', fontsize=18)
    # plt.ylabel('count', fontsize=18)
    # plt.savefig('u_hist.png', dpi=300)
    # plt.figure(figsize=(8,8))
    # plt.hist(v[:,:,0].flatten(), bins=100)
    # plt.xlabel('v', fontsize=18)
    # plt.ylabel('count', fontsize=18)
    # plt.savefig('v_hist.png', dpi=300)

    # initial u
    plt.figure(figsize=(8,8))
    x, y = np.meshgrid(x, y)
    plt.contourf(x, y, u0, cmap='jet', levels=np.linspace(-1.25,1.25,100), extend='both')
    plt.colorbar()
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.title('u(t=0)', fontsize=18)
    plt.savefig('u.png')

    # # gif u
    # import imageio.v2 as imageio
    # images = []
    # for i in range(len(t)):
    #     plt.figure(figsize=(8,8))
    #     plt.contourf(x, y, u[:,:,i], cmap='jet', levels=np.linspace(-1.25,1.25,100), extend='both')
    #     plt.colorbar()
    #     plt.xlabel('x', fontsize=18)
    #     plt.ylabel('y', fontsize=18)
    #     plt.title('u(t='+str(t[i])+')', fontsize=18)
    #     plt.savefig('u'+str(i)+'.png')
    #     plt.close()
    #     images.append(imageio.imread('u'+str(i)+'.png'))
    #     os.system('rm u'+str(i)+'.png')
    # imageio.mimsave('u.gif', images, duration=0.1)

    # initial v
    plt.figure(figsize=(8,8))
    plt.contourf(x, y, v0, cmap='jet', levels=np.linspace(-0.55,0.55,100), extend='both')
    plt.colorbar()
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.title('v(t=0)', fontsize=18)
    plt.savefig('v.png')

    # # gif v
    # images = []
    # for i in range(len(t)):
    #     plt.figure(figsize=(8,8))
    #     plt.contourf(x, y, v[:,:,i], cmap='jet', levels=np.linspace(-0.55,0.55,100), extend='both')
    #     plt.colorbar()
    #     plt.xlabel('x', fontsize=18)
    #     plt.ylabel('y', fontsize=18)
    #     plt.title('v(t='+str(t[i])+')', fontsize=18)
    #     plt.savefig('v'+str(i)+'.png')
    #     plt.close()
    #     images.append(imageio.imread('v'+str(i)+'.png'))
    #     os.system('rm v'+str(i)+'.png')
    # imageio.mimsave('v.gif', images, duration=0.1)


if __name__ == '__main__':

    data_IC1 = scipy.io.loadmat('FN_IC2_Avoid104TS.mat')
    u = np.real(data_IC1['u'])  # (301,301,484)
    v = np.real(data_IC1['v'])  # (301,301,484)
    t = np.real(data_IC1['t'].flatten())  # (484,)
    x = np.real(data_IC1['x'].flatten())  # (301,)
    y = np.real(data_IC1['y'].flatten())  # (301,)
    u0 = np.real(data_IC1['u'])[..., 0]  # (301,301)
    v0 = np.real(data_IC1['v'])[..., 0]  # (301,301)

    # plot_data(u, v, x, y, u0, v0)
    # cov_dist(u0, v0)
    map_uv(u0, v0)