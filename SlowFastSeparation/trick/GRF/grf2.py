import numpy as np
import matplotlib.pyplot as plt


def sample_gaussian_field(n, mu=0.0, sigma=1.0, l=1.0):

    def gaussian_kernel(x1, x2, sigma=1.0, l=1.0):
        return sigma**2 * np.exp(-0.5 * (x1 - x2)**2 / l**2)
        # return np.exp(np.sqrt((x1 - x2)**2) / l)

    x = np.linspace(0, 200, n)

    # 计算协方差矩阵
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i,j] = gaussian_kernel(x[i], x[j], sigma=sigma, l=l)
            K[j,i] = K[i,j]

    # 对协方差矩阵进行 Cholesky 分解
    L = np.linalg.cholesky(K)

    # 生成高斯随机场
    u = np.random.normal(size=n)
    f = mu + np.dot(L, u).reshape(n)

    return f


f = sample_gaussian_field(50, mu=-3.0, sigma=1.0, l=12.0)
u = []
for v in f:
    func = np.poly1d([-1/3,0,-1,-v])
    for root in func.roots:
        if np.imag(root) == 0:
            u.append(np.real(root))
            continue

plt.figure(figsize=(8, 6))
plt.plot(f, marker='o', color='r', label='v')
plt.plot(u, marker='^', color='b', label='u')
plt.legend()
plt.xlabel('x')
plt.title('Gaussian Random Field')
plt.savefig('gaussian_field.png', dpi=300)