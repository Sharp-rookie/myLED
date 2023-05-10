import os
import numpy as np
from tqdm import tqdm
import scienceplots
import matplotlib.pyplot as plt;plt.style.use(['science']);plt.rcParams.update({'font.size':18})
from sdeint import itoSRI2, itoEuler
import warnings;warnings.simplefilter('ignore')
from pytorch_lightning import seed_everything

seed_everything(729)

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


def sample_gaussian_field(n, mu=0.0, sigma=1.0, l=1.0):

    def gaussian_kernel(x1, x2, sigma=1.0, l=1.0):
        return sigma**2 * np.exp(-0.5 * (x1 - x2)**2 / l**2)
        # return np.exp(np.sqrt((x1 - x2)**2) / l)

    x = np.linspace(0, 10, n)

    # covariant matrix
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i,j] = gaussian_kernel(x[i], x[j], sigma=sigma, l=l)
            K[j,i] = K[i,j]

    # Cholesky decomposition
    L = np.linalg.cholesky(K)

    # sample
    u = np.random.normal(size=n)
    f = mu + np.dot(L, u).reshape(n)

    return f


xdim, t = 2000, 0.05
l = 2.0
a, epsilon, delta1, delta2, du = 1.05, 0.01, 0.2, 0.2, 0.
init_type = 'circle'
sde = SDE_FHN(a=a, epsilon=epsilon, delta1=delta1, delta2=delta2, du=du, xdim=xdim)
tspan = np.arange(0, t, 0.001)

# initial condition
if init_type=='random':
    v0, u0 = [], []
    for _ in range(xdim):
        v0.append(np.random.uniform(-3, -2) if np.random.randint(0, 2)==0 else np.random.uniform(2, 3))
        u0.append(np.random.uniform(-3, -2) if np.random.randint(0, 2)==0 else np.random.uniform(2, 3))
elif init_type=='space':
    v0 = [np.random.uniform(-3, 3) for _ in range(xdim)]
    u0 = [np.random.uniform(-3, 3) for _ in range(xdim)]
elif init_type=='circle':
    v0, u0 = [], []
    for _ in range(xdim):
        angle = np.random.normal(loc=np.pi/2, scale=1.0) if np.random.randint(0, 2)==0 else np.random.normal(loc=-np.pi/2, scale=1.0)
        u0.append(3*np.cos(angle))
        v0.append(3*np.sin(angle))
elif init_type=='grf':
    f = sample_gaussian_field(xdim, mu=-3.0, sigma=1.0, l=5.0) if np.random.randint(0, 2)==0 else sample_gaussian_field(xdim, mu=3.0, sigma=1.0, l=5.0)
    u0 = []
    v0 = f.tolist()
    for v in f:
        func = np.poly1d([-1/3,0,-1,-v])
        for root in func.roots:
            if np.imag(root) == 0:
                u0.append(np.real(root))
                continue
x0 = u0 + v0

# initial condition
plt.figure(figsize=(16, 8))
ax = plt.subplot(121)
ax.plot(np.arange(xdim), u0)
ax.set_xlabel('x', fontsize=18)
ax.set_ylabel('u', fontsize=18)
ax.set_title('u', fontsize=18)
ax = plt.subplot(122)
ax.plot(np.arange(xdim), v0)
ax.set_xlabel('x', fontsize=18)
ax.set_ylabel('v', fontsize=18)
ax.set_title('v', fontsize=18)
plt.savefig(f'initial_delta{delta2}_du{du}.png', dpi=300)
plt.close()

try:
    sol = np.load(f'fhn_hd_delta2_{delta2}_du_{du}_xdim_{xdim}_t{t}.npz')
    u = sol['u']
    v = sol['v']
except:
    sol = itoSRI2(sde.f, sde.g, x0, tspan)
    u = sol[:, :xdim]
    v = sol[:, xdim:]
    np.savez(f'fhn_hd_delta2_{delta2}_du_{du}_xdim_{xdim}_t{t}.npz', u=u, v=v)
    
# slow manifold
u_s = np.linspace(-2.4, 2.4, 100)
v_s = -u_s**3/3 + u_s

# phase
plt.figure(figsize=(8, 8))
for i in range(xdim):
    plt.plot(v[:, i], u[:, i], alpha=0.5)
    plt.plot(v_s, u_s, 'k--')
    plt.xlabel('v', fontsize=18)
    plt.ylabel('u', fontsize=18)
plt.savefig(f'phase_delta{delta2}_du{du}.png', dpi=300)

# heatmap
plt.figure(figsize=(16, 8))
ax = plt.subplot(121)
im = ax.imshow(u[::-1], cmap='jet', vmin=-3, vmax=3, aspect='auto', extent=[0, xdim, 0, t])
ax.set_xlabel('x', fontsize=18)
ax.set_ylabel('t', fontsize=18)
ax.set_title('u', fontsize=18)
plt.colorbar(im, ax=ax)
ax = plt.subplot(122)
im = ax.imshow(v[::-1], cmap='jet', vmin=-3, vmax=3, aspect='auto', extent=[0, xdim, 0, t])
ax.set_xlabel('x', fontsize=18)
ax.set_ylabel('t', fontsize=18)
ax.set_title('v', fontsize=18)
plt.colorbar(im, ax=ax)
plt.savefig(f'heatmap_delta{delta2}_du{du}.png', dpi=300)
plt.close()

# # sample trajectories
# plt.figure(figsize=(10,12))
# ax = plt.subplot(211)
# ax.plot(tspan, u[:, 0])
# ax.set_xlabel('t / s', fontsize=18)
# ax.set_ylabel('u_x0', fontsize=18)
# ax = plt.subplot(212)
# ax.plot(tspan, v[:, 0])
# ax.set_xlabel('t / s', fontsize=18)
# ax.set_ylabel('v_x0', fontsize=18)
# plt.subplots_adjust(hspace=0.3)
# plt.savefig(f'x0trace_delta{delta2}_du{du}.png', dpi=300)
# plt.close()

# # 3D plot
# plt.figure(figsize=(10, 10))
# ax = plt.subplot(projection='3d')
# x = np.arange(0, xdim, 1)[np.newaxis].repeat(len(tspan), axis=0)
# u = u.flatten()
# v = v.flatten()
# x = x.flatten()
# ax.scatter(u, v, x, s=0.1, c=tspan.repeat(xdim), cmap='jet')
# ax.set_xlabel('u', fontsize=20)
# ax.invert_xaxis()
# ax.set_ylabel('v', fontsize=20)
# ax.set_zlabel('x', fontsize=20)
# plt.savefig(f'3d_delta{delta2}_du{du}.png', dpi=300)