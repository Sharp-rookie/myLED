import os
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# 定义 FHN 系统的微分方程
def fhn(y, t):
    
    a, b, I, epsilon = 0.7, 0.8, 0.5, 0.08

    u, v = y
    dudt = u - u**3/3 - v + I
    dvdt = epsilon * (u + a - b*v)

    return [dudt, dvdt]


n_trace = 100
# n_trace = 50
grid = 50
u_min, u_max = -20., 20.
v_min, v_max = -15., 15.
u_step = (u_max-u_min)/grid
v_step = (v_max-v_min)/grid
dpi = 300
# fhn_dir = 'FHN_macro'
fhn_dir = 'FHN_large'
# fhn_dir = 'FHN_micro'
# for tf in range(1, 40+1):
for tf in [300]:
    dt = 0.01
    t = np.arange(0, tf, dt)

    # 定义初值范围和轨迹数
    images = []
    flag = True
    for u_bound in np.arange(u_min, u_max, u_step):
        if flag:
            iter = np.arange(v_min, v_max, v_step)
            flag = False
        else:
            iter = np.arange(v_max-v_step, v_min-v_step, -v_step)
            flag = True
        
        for v_bound in iter:
        # for v_bound in np.arange(v_min, v_max, v_step):

            print(f'\rRunning FHN u0={u_bound:.2f} v0={v_bound:.2f}', end='')
        
            dir = fhn_dir + f'/u0={u_bound:.2f}_v0={v_bound:.2f}'
            os.makedirs(dir, exist_ok=True)
            
            results = []
            for i in range(n_trace):

                # 在给定区域内随机选取初始值
                u0, v0 = np.random.uniform(u_bound, u_bound+u_step), np.random.uniform(v_bound, v_bound+v_step)
                u0, v0 = np.clip(u0, u_bound, u_bound+u_step), np.clip(v0, v_bound, v_bound+v_step)
                y0 = [u0, v0]

                # 运行微分方程求解器，获得模拟结果
                sol = odeint(fhn, y0=y0, t=t)

                results.append(sol)
            
            results = np.array(results)

            # 绘制膜电位和恢复变量随时间演化的图形
            plt.figure(figsize=(10,12))
            for i in range(2):
                ax = plt.subplot(2,1,i+1)
                for j in range(n_trace):
                    ax.plot(t, results[j, :, i], c='b', alpha=0.05, label='trajectory' if j==0 else None)
                ax.set_xlabel('Time / s')
                ax.set_ylabel(['u', 'v'][i])
            plt.subplots_adjust(hspace=0.3)
            plt.savefig(f'{dir}/fhn.jpg', dpi=dpi)
            plt.close()
            
            # 绘制相空间演化轨迹
            plt.figure(figsize=(6,6))
            ax = plt.subplot(111)
            # 绘制多条相空间演化轨迹
            for j in range(n_trace):
                sol = results[j]
                ax.plot(sol[:, 1], sol[:, 0], c='b', alpha=0.05, label='trajectory' if j==0 else None)
            # 绘制u、v的nullcline
            u1 = np.linspace(u_min, u_max, 100)
            v1 = u1-u1**3/3+0.5
            v2 = np.linspace(v_min, v_max, 100)
            u2 = 0.8*v2 - 0.7
            ax.plot(v1, u1, c='r', label='u-nullcline (slow manifold))')
            ax.plot(v2, u2, c='g', label='v-nullcline(dv/dt=0)')
            ax.set_xlabel('v (slow)')
            ax.set_ylabel('u (fast)')
            ax.legend(loc='upper right')
            # 注明dudt、dvdt的正负
            # ax.text(0.28, 0.7, r'$\frac{du}{dt} > 0$', transform=ax.transAxes, fontsize=14)
            ax.text(0.28, 0.5, r'$\frac{du}{dt} > 0$', transform=ax.transAxes, fontsize=14)
            ax.text(0.8, 0.6, r'$\frac{dv}{dt} < 0$', transform=ax.transAxes, fontsize=14)
            # 绘制初始范围框
            ax.add_patch(plt.Rectangle((v_bound, u_bound), v_step, u_step, fill=False, edgecolor='k', lw=1))
            # 绘制全局所有初始范围框
            for _u_bound in np.arange(u_min, u_max, u_step):
                for _v_bound in np.arange(v_min, v_max, v_step):
                    ax.add_patch(plt.Rectangle((_v_bound, _u_bound), v_step, u_step, fill=False, edgecolor='k', lw=0.5, alpha=0.5))
            # 设置x、y轴范围
            ax.set_xlim(v_min-0.1, v_max+0.1)
            ax.set_ylim(u_min-0.1, u_max+0.1)
            ax.set_title(f'{n_trace} trajectories simulated for {tf}s')
            plt.savefig(f'{dir}/phase.jpg', dpi=dpi)
            plt.close()

            # 绘制整个过程的gif图
            images.append(imageio.imread(f'{dir}/phase.jpg'))

    os.makedirs(fhn_dir+'/gif/', exist_ok=True)
    imageio.mimsave(fhn_dir+f'/gif/phase_{tf:.1f}.gif', images, duration=1.)