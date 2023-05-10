import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# 定义 FHN 系统的微分方程
def fhn(y, t):
    
    a, c, I = 1e5, 0.2, 1
    def G(t):
        return 1/2  *np.pi/6 * np.cos(np.pi/6*t+0.4)

    v, w, b = y
    dvdt = a * (-v*(v-1)*(v-b) - w + I)
    dwdt = v - c*w
    dbdt = G(t)

    return [dvdt, dwdt, dbdt]

# 定义初值
v0, w0, b0 = 0., 1., 0.7
y0 = [v0, w0, b0]

# 定义时间间隔
t0 = 0
tf = 24
dt = 0.01
t = np.arange(t0, tf, dt)

# 运行微分方程求解器，获得模拟结果
sol = odeint(fhn, y0=y0, t=t)

# 绘制膜电位和恢复变量随时间演化的图形
plt.figure(figsize=(9,9))
for i in range(3):
    ax = plt.subplot(3,1,i+1)
    ax.plot(t, sol[:, i])
    ax.set_xlabel('Time / s')
    ax.set_ylabel(['v', 'w', 'b'][i])
plt.savefig('fhn.jpg', dpi=300)