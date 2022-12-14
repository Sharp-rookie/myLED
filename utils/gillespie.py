#! -*- coding: utf-8 -*-

import os
import time
import numpy as np
from tqdm import tqdm
from scipy.special import comb
import matplotlib.pyplot as plt
from multiprocessing import Process


def set_seed(seed):

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    np.random.seed(seed)


class Reaction: # 封装的类，代表每一个化学反应

    def __init__(self, rate=0., num_lefts=None, num_rights=None):

        self.rate = rate # 反应速率
        assert len(num_lefts) == len(num_rights)
        self.num_lefts = np.array(num_lefts) # 反应前各个反应物的数目
        self.num_rights = np.array(num_rights) # 反应后各个反应物的数目
        self.num_diff = self.num_rights - self.num_lefts # 改变数

    def combine(self, n, s): # 算组合数
        return np.prod(comb(n, s))

    def propensity(self, n): # 算反应倾向函数
        return self.rate * self.combine(n, self.num_lefts)


class System: # 封装的类，代表多个化学反应构成的系统

    def __init__(self, num_elements):

        assert num_elements > 0
        self.num_elements = num_elements # 系统内的反应物的类别数
        self.reactions = [] # 反应集合

    def add_reaction(self, rate=0., num_lefts=None, num_rights=None):

        assert len(num_lefts) == self.num_elements
        assert len(num_rights) == self.num_elements
        self.reactions.append(Reaction(rate, num_lefts, num_rights))

    def evolute(self, steps=None, total_t=None, IC=None, seed=1): # 模拟演化

        self.t = [0] # 时间轨迹，t[0]是初始时间

        if IC is None:
            self.n = [np.array([100, 40, 2500])] # 默认初始数目
        else:
            self.n = [np.array(IC)] # 指定初始数目

        if steps is not None: # 按照固定步数仿真
            for i in tqdm(range(steps)):
                A = np.array([rec.propensity(self.n[-1])
                            for rec in self.reactions]) # 算每个反应的倾向函数
                A0 = A.sum()
                A /= A0 # 归一化得到概率分布
                t0 = -np.log(np.random.random())/A0 # 按概率选择下一个反应发生的间隔
                self.t.append(self.t[-1] + t0)
                d = np.random.choice(self.reactions, p=A) # 按概率选择其中一个反应发生
                self.n.append(self.n[-1] + d.num_diff)
        else: # 按照固定总时长仿真
            total_t = 10 if total_t is None else total_t
            while self.t[-1] < total_t:
                A = np.array([rec.propensity(self.n[-1])
                            for rec in self.reactions]) # 算每个反应的倾向函数
                A0 = A.sum()
                A /= A0 # 归一化得到概率分布
                t0 = -np.log(np.random.random())/A0 # 按概率选择下一个反应发生的间隔
                self.t.append(self.t[-1] + t0)
                d = np.random.choice(self.reactions, p=A) # 按概率选择其中一个反应发生
                self.n.append(self.n[-1] + d.num_diff)
                if seed == 1:
                    print(f'\rSeed[{seed}] time: {self.t[-1]:.3f}/{total_t}s | X={self.n[-1][0]}, Y={self.n[-1][1]}, Z={self.n[-1][2]},', end='')

    def reset(self, IC):

        self.t = [0]
        
        if IC is None:
            self.n = [np.array([100, 40, 2500])] # 默认初始数目
        else:
            self.n = [np.array(IC)] # 指定初始数目
        


def generate_origin(total_t, seed):

    time.sleep(1.0)

    set_seed(seed)
    os.makedirs(f'Data/origin/{seed}/', exist_ok=True)

    num_elements = 3
    system = System(num_elements)

    # X, Y, Z
    system.add_reaction(1000, [1, 0, 0], [1, 0, 1])
    system.add_reaction(1, [0, 1, 1], [0, 1, 0])
    system.add_reaction(40, [0, 0, 0], [0, 1, 0])
    system.add_reaction(1, [0, 1, 0], [0, 0, 0])
    system.add_reaction(1, [0, 0, 0], [1, 0, 0])

    system.evolute(total_t=total_t, seed=seed)

    t = system.t
    X = [i[0] for i in system.n]
    Y = [i[1] for i in system.n]
    Z = [i[2] for i in system.n]

    plt.figure(figsize=(16,4))
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
    plt.savefig(f'Data/origin/{seed}/origin.png', dpi=500)

    np.savez(f'Data/origin/{seed}/origin.npz', t=t, X=X, Y=Y, Z=Z)

    print(f'\nSeed[{seed}] subprocess finished!\n')


if __name__ == '__main__':

    os.makedirs('Data/origin', exist_ok=True)

    subprocess = []
    for seed in range(1, 11):
        subprocess.append(Process(target=generate_origin, args=(seed,), daemon=True))
        subprocess[-1].start()
        print(f'\rStart process[seed={seed:.3f}]' + ' '*30)
        time.sleep(0.1)

    while 1:
        pass

    # num_elements = 2
    # system = System(num_elements)

    # # X, Y
    # system.add_reaction(1, [2, 0], [0, 1])
    # system.add_reaction(100, [0, 1], [2, 0])
    # system.add_reaction(50, [0, 0], [1, 0])

    # system.evolute(IC=[100, 100])

    # t = np.array(system.t)
    # X = np.array([i[0] for i in system.n])
    # Y = np.array([i[1] for i in system.n])

    # plt.figure(figsize=(16,5))
    # ax1 = plt.subplot(1,3,1)
    # ax1.set_title('X')
    # plt.plot(t, X, label='X')
    # ax2 = plt.subplot(1,3,2)
    # ax2.set_title('Y')
    # plt.plot(t, Y, label='Y')
    # ax3 = plt.subplot(1,3,3)
    # ax3.set_title('X+2Y')
    # plt.plot(t, X+2*Y, label='X+2Y')

    # plt.subplots_adjust(
    #     left=0.05,
    #     right=0.95,
    #     top=0.95,
    #     bottom=0.05,
    #     wspace=0.2
    # )
    # plt.savefig(f'data.png', dpi=500)

    # np.savetxt(f't.txt', t)
    # np.savetxt(f'X.txt', X)
    # np.savetxt(f'Y.txt', Y)