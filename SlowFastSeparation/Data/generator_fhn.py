import os
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def fhn(y, t):
    
    a, b, I, epsilon = 0.7, 0.8, 0.5, 0.08

    u, v = y
    dudt = u - u**3/3 - v + I
    dvdt = epsilon * (u + a - b*v)

    return [dudt, dvdt]
    

def simulation(u_bound, v_bound, t, n_trace, u_step, v_step, dir):

    os.makedirs(dir, exist_ok=True)
    
    trace = []
    for i in range(n_trace):

        # 在给定区域内随机选取初始值
        u0, v0 = np.random.uniform(u_bound, u_bound+u_step), np.random.uniform(v_bound, v_bound+v_step)
        u0, v0 = np.clip(u0, u_bound, u_bound+u_step), np.clip(v0, v_bound, v_bound+v_step)
        y0 = [u0, v0]

        # 运行微分方程求解器，获得模拟结果
        sol = odeint(fhn, y0=y0, t=t)

        trace.append(sol)
    
    trace = np.array(trace)

    # 保存模拟结果
    np.savez(f'{dir}/origin.npz', trace=trace, u_bound=u_bound, v_bound=v_bound, u_step=u_step, v_step=v_step)

    # 绘制膜电位和恢复变量随时间演化的图形
    plt.figure(figsize=(10,12))
    for i in range(2):
        ax = plt.subplot(2,1,i+1)
        for j in range(n_trace):
            ax.plot(t, trace[j, :, i], c='b', alpha=0.05, label='trajectory' if j==0 else None)
        ax.set_xlabel('Time / s')
        ax.set_ylabel(['u', 'v'][i])
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(f'{dir}/fhn.jpg', dpi=100)
    plt.close()
    
    # 绘制相空间演化轨迹
    plt.figure(figsize=(6,6))
    ax = plt.subplot(111)
    # 绘制多条相空间演化轨迹
    for j in range(n_trace):
        sol = trace[j]
        ax.plot(sol[:, 1], sol[:, 0], c='b', alpha=0.05, label='trajectory' if j==0 else None)
    # 绘制u、v的nullcline
    u1 = np.linspace(-2.,2.,100)
    v1 = u1-u1**3/3+0.5
    v2 = np.linspace(-0.5,0.9,100)
    u2 = 1.25*v2 + 0.875
    ax.plot(v1, u1, c='r', label='u-nullcline (slow manifold))')
    ax.plot(v2, u2, c='g', label='v-nullcline(dv/dt=0)')
    ax.set_xlabel('v (slow)')
    ax.set_ylabel('u (fast)')
    ax.legend(loc='upper right')
    # 注明dudt、dvdt的正负
    ax.text(0.44, 0.55, r'$\frac{du}{dt} > 0$', transform=ax.transAxes, fontsize=14)
    ax.text(0.2, 0.75, r'$\frac{dv}{dt} < 0$', transform=ax.transAxes, fontsize=14)
    # 绘制初始范围框
    ax.add_patch(plt.Rectangle((v_bound, u_bound), v_step, u_step, fill=False, edgecolor='k', lw=1))
    # 绘制全局所有初始范围框
    for _u_bound in np.arange(-2.,2.,u_step):
        for _v_bound in np.arange(-0.5,1.5,v_step):
            ax.add_patch(plt.Rectangle((_v_bound, _u_bound), v_step, u_step, fill=False, edgecolor='k', lw=0.5, alpha=0.5))
    # 设置x、y轴范围
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-2.1, 2.1)
    ax.set_title(f'{n_trace} trajectories simulated for {int(np.max(t))}s')
    plt.savefig(f'{dir}/phase.jpg', dpi=100)
    plt.close()

    return trace


def generate_dataset_static(tau, u_bound, v_bound, total_t, dt, n_trace, u_step, v_step, data_dir):

    data_dir = data_dir + f'u0={u_bound:.2f}_v0={v_bound:.2f}'

    if os.path.exists(f'{data_dir}/origin.npz'): 
        # 导入数据
        simdata = np.load(f'{data_dir}/origin.npz')['trace']
    else:
        # 从指定范围初始化并演化微分方程
        t = np.arange(0., total_t, dt)
        simdata = simulation(u_bound, v_bound, t, n_trace, u_step, v_step, data_dir)
    
    # reshape
    simdata = simdata.reshape(n_trace, -1, 1, 2)  # (n_trace, time_length, channel=1, feature_num=2)

    # save statistic information
    data_dir = data_dir + f"/tau_{tau}"
    os.makedirs(data_dir, exist_ok=True)
    np.savetxt(data_dir + "/data_mean_static.txt", np.mean(simdata, axis=(0,1)).reshape(1,-1))
    np.savetxt(data_dir + "/data_std_static.txt", np.std(simdata, axis=(0,1)).reshape(1,-1))
    np.savetxt(data_dir + "/data_max_static.txt", np.max(simdata, axis=(0,1)).reshape(1,-1))
    np.savetxt(data_dir + "/data_min_static.txt", np.min(simdata, axis=(0,1)).reshape(1,-1))
    np.savetxt(data_dir + "/tau_static.txt", [tau])  # Save the timestep

    ##################################
    # Create [train,val,test] dataset
    ##################################
    train_num = int(0.7*n_trace)
    val_num = int(0.1*n_trace)
    test_num = int(0.2*n_trace)
    trace_list = {'train':range(train_num), 'val':range(train_num,train_num+val_num), 'test':range(train_num+val_num,train_num+val_num+test_num)}
    for item in ['train','val','test']:

        if os.path.exists(data_dir+f'/{item}_static.npz'): continue
        
        # select trace num
        data_item = simdata[trace_list[item]]

        # subsampling
        step_length = int(tau/dt) if tau!=0. else 1
        squence_length = 2 if tau!=0. else 1

        assert step_length<=data_item.shape[1], f"Tau {tau} is larger than the simulation time length {dt*data_item.shape[1]}"
        sequences = data_item[:, ::step_length]
        sequences = sequences[:, :squence_length]
        
        # save
        np.savez(data_dir+f'/{item}_static.npz', data=sequences)

        # plot
        plt.figure(figsize=(16,10))
        plt.title(f'{item.capitalize()} Data')
        plt.plot(sequences[:,0,0,0], label='u')
        plt.plot(sequences[:,0,0,1], label='v')
        plt.legend()
        plt.savefig(data_dir+f'/{item}_static_input.pdf', dpi=300)

        plt.figure(figsize=(16,10))
        plt.title(f'{item.capitalize()} Data')
        plt.plot(sequences[:,squence_length-1,0,0], label='u')
        plt.plot(sequences[:,squence_length-1,0,1], label='v')
        plt.legend()
        plt.savefig(data_dir+f'/{item}_static_target.pdf', dpi=300)

# generator_data(tau=0.2, u_bound=-2., v_bound=-0.5, total_t=20., dt=0.01, n_trace=50, u_step=0.8, v_step=0.4, data_dir='Data/FHN_grid5/')