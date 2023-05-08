import os
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio
import scienceplots
import matplotlib.pyplot as plt;plt.style.use(['science']);plt.rcParams.update({'font.size':16})
from sdeint import itoSRI2, itoEuler
import warnings;warnings.simplefilter('ignore')


class SDE_FHN():
    def __init__(self, a, epsilon, delta1, delta2):
        self.a = a
        self.epsilon = epsilon
        self.delta1 = delta1
        self.delta2 = delta2
    
    def f(self, x, t):
        return np.array([(x[0]-1/3*x[0]**3-x[1])/self.epsilon, 
                         x[0]+self.a])
    
    def g(self, x, t):
        return np.array([[self.delta1*1., 0.], 
                        [0., self.delta2*1.]])

a, epsilon, delta1, delta2 = 1.05, 0.01, 0.2, 0.1
sde = SDE_FHN(a=a, epsilon=epsilon, delta1=delta1, delta2=delta2)
n_trace = 10
tspan = np.arange(0, 0.9, 0.001)
x0 = [-3., -2.]
results = np.zeros((n_trace, len(tspan), 2))
for i in tqdm(range(n_trace)):
    sol = itoSRI2(sde.f, sde.g, x0, tspan)
    results[i, :, :] = sol

# 绘制膜电位和恢复变量随时间演化的图形
plt.figure(figsize=(10,12))
for i in range(2):
    ax = plt.subplot(2,1,i+1)
    for j in range(n_trace):
        ax.plot(tspan, results[j, :, i], c='b', alpha=0.05, label='trajectory' if j==0 else None)
    ax.set_xlabel('Time / s')
    ax.set_ylabel(['u', 'v'][i])
plt.subplots_adjust(hspace=0.3)
plt.savefig(f'fhn_delta2_{delta2}.jpg', dpi=300)
plt.close()
            
# 绘制相空间演化轨迹
plt.figure(figsize=(6,6))
ax = plt.subplot(111)
# 绘制多条相空间演化轨迹
for j in range(n_trace):
    ax.plot(results[j, :, 1], results[j, :, 0], c='b', alpha=0.05, label='trajectory' if j==0 else None)
# 绘制u、v的nullcline
u1 = np.linspace(-4.0, 4.0, 200)
v1 = u1 - u1**3/3
v2 = np.linspace(-2.5, 2.5, 200)
u2 = np.ones_like(v2)*(-a)
ax.plot(v1, u1, c='r', label='u-nullcline (slow manifold))')
ax.plot(v2, u2, c='g', label='v-nullcline(dv/dt=0)')
ax.set_xlabel('v (slow)')
ax.set_ylabel('u (fast)')
ax.legend(loc='upper right')
# 注明dudt、dvdt的正负
ax.text(0.2, 0.7, r'$\frac{du}{dt} > 0$', transform=ax.transAxes, fontsize=14)
ax.text(0.15, 0.3, r'$\frac{dv}{dt} < 0$', transform=ax.transAxes, fontsize=14)
# 设置x、y轴范围
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-4.0, 4.0)
plt.savefig(f'phase_delta2_{delta2}.jpg', dpi=300)
plt.close()