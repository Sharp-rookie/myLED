import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# 定义 FHN 系统的slow manifold方程
def fhn(y, x, Du, v):

    u1, u2 = y
    du1dx = u2
    du2dx = 1/Du * (u1**3 - u1 + v)

    return [du1dx, du2dx]


# 空间间隔
grid = 100
x_space = 20
x = np.linspace(0., x_space, grid)

# 定义参数
Du = 1

# 在给定区域内随机选取初始值
u1, u2 = 0, 0.1
y0 = [u1, u2]

# 仿真 x,u,v 的演化
simdata = [[], [], []] # x, u, v
for v in np.linspace(-0.2, 0.2, 50):
    # 运行微分方程求解器，获得模拟结果
    sol = odeint(fhn, y0=y0, t=x, args=(Du, v))
    simdata[0].append(x)
    simdata[1].append(sol[:, 0])
    simdata[2].append([v for _ in range(100)])
simdata = np.array(simdata)
points = simdata[:,:,66].reshape(3,-1)
simdata = simdata.reshape(3,-1)

# 绘制3d相空间演化轨迹
plt.figure(figsize=(6,6))
ax = plt.subplot(111, projection='3d')
ax.scatter(simdata[0], simdata[1], simdata[2], c='b', alpha=0.4, label='slow manifold', s=2)
ax.scatter(points[0], points[1], points[2], c='r', alpha=1, label='x=13.2', s=2)
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_zlabel('v')
ax.legend()
plt.savefig(f'fhn_slowmanifold.jpg', dpi=300)
plt.close()