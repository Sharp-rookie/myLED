import os
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio
import scienceplots
import matplotlib.pyplot as plt;plt.style.use(['science']);plt.rcParams.update({'font.size':18})
from sdeint import itoSRI2, itoEuler
import warnings;warnings.simplefilter('ignore')


class SDE_FHN():
    def __init__(self, a, epsilon, delta1, delta2, du, xdim):
        self.a = a
        self.epsilon = epsilon
        self.delta1 = delta1
        self.delta2 = delta2
        self.du = du
        self.xdim = xdim
    
    def f(self, x, t):
        y = []

        # u
        for i in range(self.xdim):
            if i == 0:
                y.append((x[i] - 1/3*x[i]**3 - x[i+self.xdim] + self.du*(x[i+1]+x[i+self.xdim-1]-2*x[i]))/self.epsilon)
            elif i == self.xdim-1:
                y.append((x[i] - 1/3*x[i]**3 - x[i+self.xdim] + self.du*(x[i-self.xdim+1]+x[i-1]-2*x[i]))/self.epsilon)
            else:
                y.append((x[i] - 1/3*x[i]**3 - x[i+self.xdim] + self.du*(x[i+1]+x[i-1]-2*x[i]))/self.epsilon)
        
        # v
        for i in range(self.xdim):
            y.append(x[i] + self.a)

        return np.array(y)
    
    def g(self, x, t):
        return np.diag([self.delta1*1.]*self.xdim + [self.delta2*1.]*self.xdim)

xdim, t = 100, 10.0
a, epsilon, delta1, delta2, du = 1.05, 0.01, 0.2, 0.5, 1.
sde = SDE_FHN(a=a, epsilon=epsilon, delta1=delta1, delta2=delta2, du=du, xdim=xdim)
tspan = np.arange(0, t, 0.001)
x0 = [-3.]*xdim + [-2.]*xdim

try:
    sol = np.load(f'fhn_hd_delta2_{delta2}_du_{du}_xdim_{xdim}_t{t}.npz')
    u = sol['u']
    v = sol['v']
except:
    sol = itoSRI2(sde.f, sde.g, x0, tspan)
    u = sol[:, :xdim]
    v = sol[:, xdim:]
    np.savez(f'fhn_hd_delta2_{delta2}_du_{du}_xdim_{xdim}_t{t}.npz', u=u, v=v)

# heatmap
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(u[::-1], cmap='jet', vmin=-3, vmax=3, aspect='auto', extent=[0, xdim, 0, t])
ax.set_xlabel('x', fontsize=18)
ax.set_ylabel('t', fontsize=18)
ax.set_title('u', fontsize=18)
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(f'heatmap_delta2_{delta2}_du_{du}_xdim_{xdim}.png', dpi=300)
plt.close()

# sample trajectories
plt.figure(figsize=(10,12))
ax = plt.subplot(211)
ax.plot(tspan, u[:, 0])
ax.set_xlabel('t / s', fontsize=18)
ax.set_ylabel('u_x0', fontsize=18)
ax = plt.subplot(212)
ax.plot(tspan, v[:, 0])
ax.set_xlabel('t / s', fontsize=18)
ax.set_ylabel('v_x0', fontsize=18)
plt.subplots_adjust(hspace=0.3)
plt.savefig(f'fhn_delta2_{delta2}_du_{du}_xdim_{xdim}.png', dpi=300)
plt.close()