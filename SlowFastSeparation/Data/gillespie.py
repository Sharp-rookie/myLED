#! -*- coding: utf-8 -*-

import os
import time
import numpy as np
from tqdm import tqdm
from scipy.special import comb
import matplotlib.pyplot as plt

from util import seed_everything

class Reaction:

    def __init__(self, rate=0., num_lefts=None, num_rights=None, dynamic_rate=False):

        self.dynamic_rate = dynamic_rate

        self.rate = rate

        assert len(num_lefts) == len(num_rights)
        self.num_lefts = np.array(num_lefts)
        self.num_rights = np.array(num_rights)
        self.num_diff = self.num_rights - self.num_lefts

    def combine(self, n, s):
        return np.prod(comb(n, s))

    def propensity(self, n, t=None):
        if not self.dynamic_rate:
            return self.rate * self.combine(n, self.num_lefts)
        else:
            # Epidemic SIR Model: c1=c0*(1+episilon)*sin(wt)
            return (np.abs(0.003*(1+0.2)*np.sin(np.pi/3*t))) * self.combine(n, self.num_lefts)


class System:

    def __init__(self, num_elements):

        assert num_elements > 0
        self.num_elements = num_elements
        self.reactions = []

        self.noise_t = 0

    def add_reaction(self, rate=0., num_lefts=None, num_rights=None, dynamic_rate=False):

        assert len(num_lefts) == self.num_elements
        assert len(num_rights) == self.num_elements
        self.reactions.append(Reaction(rate, num_lefts, num_rights, dynamic_rate))

    def evolute(self, steps=None, total_t=None, IC=None, seed=1, is_print=False):

        self.t = [0]

        if IC is None:
            print('IC is None!')
            return
        else:
            self.n = [np.array(IC)]

        if steps is not None:
            for i in tqdm(range(steps)):
                A = np.array([rec.propensity(self.n[-1], self.t[-1]) for rec in self.reactions])
                A0 = A.sum()
                A /= A0
                t0 = -np.log(np.random.random())/A0
                self.t.append(self.t[-1] + t0)
                d = np.random.choice(self.reactions, p=A)
                self.n.append(self.n[-1] + d.num_diff)
        else:
            total_t = 10 if total_t is None else total_t
            while self.t[-1] < total_t:
                A = np.array([rec.propensity(self.n[-1], self.t[-1]) for rec in self.reactions])
                A0 = A.sum()
                A /= A0
                t0 = -np.log(np.random.random())/A0
                self.t.append(self.t[-1] + t0)
                d = np.random.choice(self.reactions, p=A)
                self.n.append(self.n[-1] + d.num_diff)
                
                if is_print: 
                    molecules = ''
                    name = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
                    for i in range(min(10, self.num_elements)):
                        molecules += f'{name[i]}={self.n[-1][i]} '
                    print(f'\rSeed[{seed}] time: {self.t[-1]:.3f}/{total_t}s | {molecules}', end='')

    def reset(self, IC):

        self.t = [0]
        
        if IC is None:
            self.n = [np.array([100, 40, 2500])]
        else:
            self.n = [np.array(IC)]
        

def generate_1s2f_origin(total_t=None, seed=729, IC=[100,40,2500], save=True, is_print=False):

    time.sleep(1.0)

    seed_everything(seed)
    if save: os.makedirs(f'Data/1S2F/origin/{seed}/', exist_ok=True)

    num_elements = 3
    system = System(num_elements)

    # X, Y, Z
    system.add_reaction(1000, [1, 0, 0], [1, 0, 1])
    system.add_reaction(1, [0, 1, 1], [0, 1, 0])
    system.add_reaction(40, [0, 0, 0], [0, 1, 0])
    system.add_reaction(1, [0, 1, 0], [0, 0, 0])
    system.add_reaction(1, [0, 0, 0], [1, 0, 0])

    system.evolute(total_t=total_t, seed=seed, IC=IC, is_print=is_print)

    t = system.t
    X = [i[0] for i in system.n]
    Y = [i[1] for i in system.n]
    Z = [i[2] for i in system.n]

    if save:
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
            bottom=0.15,
            wspace=0.2
        )
        plt.savefig(f'Data/1S2F/origin/{seed}/origin.pdf', dpi=300)
    
        # calculate average dt
        digit = f'{np.average(np.diff(t)):.20f}'.count("0")
        avg = np.round(np.average(np.diff(t)), digit)

        np.savez(f'Data/1S2F/origin/{seed}/origin.npz', t=t, X=X, Y=Y, Z=Z, dt=avg)
    else:
        return {'t': t, 'X': X, 'Y': Y, 'Z': Z}


def generate_1s1f_origin(total_t=None, seed=729, IC=[100,100], save=True, is_print=False):

    time.sleep(1.0)

    seed_everything(seed)
    if save: os.makedirs(f'Data/1S1F/origin/{seed}/', exist_ok=True)

    num_elements = 2
    system = System(num_elements)

    # X, Y, Z
    system.add_reaction(1, [2, 0], [0, 1])
    system.add_reaction(100, [0, 1], [2, 0])
    system.add_reaction(50, [0, 0], [1, 0])

    system.evolute(total_t=total_t, seed=seed, IC=IC, is_print=is_print)

    t = system.t
    X = [i[0] for i in system.n]
    Y = [i[1] for i in system.n]

    if save:
        plt.figure(figsize=(12,4))
        ax1 = plt.subplot(1,2,1)
        ax1.set_title('X')
        plt.plot(t, X, label='X')
        plt.xlabel('time / s')
        ax2 = plt.subplot(1,2,2)
        ax2.set_title('Y')
        plt.plot(t, Y, label='Y')
        plt.xlabel('time / s')

        plt.subplots_adjust(
            left=0.05,
            right=0.95,
            top=0.9,
            bottom=0.15,
            wspace=0.2
        )
        plt.savefig(f'Data/1S1F/origin/{seed}/origin.pdf', dpi=300)
    
        # calculate average dt
        digit = f'{np.average(np.diff(t)):.20f}'.count("0")
        avg = np.round(np.average(np.diff(t)), digit)

        np.savez(f'Data/1S1F/origin/{seed}/origin.npz', t=t, X=X, Y=Y, dt=avg)
    else:
        return {'t': t, 'X': X, 'Y': Y}
    

def generate_toggle_switch_origin(total_t=None, seed=729, IC=[1,0,0,1,0,0], save=True, is_print=False):

    time.sleep(1.0)

    seed_everything(seed)
    if save: os.makedirs(f'Data/ToggleSwitch/origin/{seed}/', exist_ok=True)

    num_elements = 6
    system = System(num_elements)

    # Gx, ^Gx, Px, Gy, ^Gy, Py
    system.add_reaction(50, [1, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0])
    system.add_reaction(50, [0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1])
    system.add_reaction(1, [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0])
    system.add_reaction(1, [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0])
    system.add_reaction(1e-4, [0, 0, 2, 1, 0, 0], [0, 0, 0, 0, 1, 0])
    system.add_reaction(1e-4, [1, 0, 0, 0, 0, 2], [0, 1, 0, 0, 0, 0])
    system.add_reaction(0.1, [0, 0, 0, 0, 1, 0], [0, 0, 2, 1, 0, 0])
    system.add_reaction(0.1, [0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 2])

    system.evolute(total_t=total_t, seed=seed, IC=IC, is_print=is_print)

    t = system.t
    vars = []
    for i in range(num_elements):
        vars.append([var[i] for var in system.n])
    
    if save:
        plt.figure(figsize=(16,9))
        name = ['G_x', 'G_x2', 'P_x', 'G_y', 'G_y2', 'P_y']
        for i in range(num_elements):
            ax = plt.subplot(2,3,i+1)
            ax.set_title(rf'{name[i]}')
            ax.plot(t, vars[i])
            ax.set_xlabel('time / s')

        plt.subplots_adjust(
            left=0.05,
            right=0.95,
            top=0.9,
            bottom=0.15,
            wspace=0.15,
            hspace=0.3
        )
        plt.savefig(f'Data/ToggleSwitch/origin/{seed}/origin.pdf', dpi=300)
    
        # calculate average dt
        digit = f'{np.average(np.diff(t)):.20f}'.count("0")
        avg = np.round(np.average(np.diff(t)), digit)

        np.savez(f'Data/ToggleSwitch/origin/{seed}/origin.npz', t=t, Gx=vars[0], _Gx=vars[1], Px=vars[2], Gy=vars[3], _Gy=vars[4], Py=vars[5], dt=avg)
    else:
        return {'t': t, 'Gx': vars[0], '_Gx': vars[1], 'Px': vars[2], 'Gy': vars[3], '_Gy': vars[4], 'Py': vars[5]}


def generate_epidemicSIR_origin(total_t=None, seed=729, IC=[51,11,0], save=True, is_print=False):

    time.sleep(1.0)

    seed_everything(seed)
    if save: os.makedirs(f'Data/EpidemicSIR/origin/{seed}/', exist_ok=True)

    num_elements = 3
    system = System(num_elements)

    # S, I, R
    system.add_reaction(0, [1, 1, 0], [0, 2, 0], dynamic_rate=True)
    system.add_reaction(0.02, [0, 1, 0], [0, 0, 1])
    system.add_reaction(0.007, [0, 0, 1], [1, 0, 0])
    system.add_reaction(0.002, [1, 0, 0], [0, 0, 0])
    system.add_reaction(0.05, [0, 1, 0], [0, 0, 0])
    system.add_reaction(0.002, [0, 0, 1], [0, 0, 0])

    system.evolute(total_t=total_t, seed=seed, IC=IC, is_print=is_print)

    t = system.t
    vars = []
    for i in range(num_elements):
        vars.append([var[i] for var in system.n])
    
    if save:
        plt.figure(figsize=(6,6))
        name = ['S', 'I', 'R']
        for i in range(num_elements):
            plt.plot(t, vars[i], label=name[i])
        plt.xlabel('time / s')
        plt.legend()
        plt.savefig(f'Data/EpidemicSIR/origin/{seed}/origin.pdf', dpi=300)
    
        # calculate average dt
        digit = f'{np.average(np.diff(t)):.20f}'.count("0")
        avg = np.round(np.average(np.diff(t)), digit)

        np.savez(f'Data/EpidemicSIR/origin/{seed}/origin.npz', t=t, S=vars[0], I=vars[1], R=vars[2], dt=avg)
    else:
        return {'t': t, 'S': vars[0], 'I': vars[1], 'R': vars[2]}
    

def generate_signalling_cascade_origin(total_t=None, seed=729, IC=[0 for _ in range(4)], save=True, is_print=False):

    time.sleep(1.0)

    seed_everything(seed)
    if save: os.makedirs(f'Data/SignallingCascade/origin/{seed}/', exist_ok=True)

    num_elements = 4
    system = System(num_elements)

    # 30 reaction
    system.add_reaction(10, [0, 0, 0, 0], [1, 0, 0, 0])
    system.add_reaction(0.5, [1, 0, 0, 0], [0, 1, 0, 0])
    system.add_reaction(0.1, [0, 1, 0, 0], [0, 0, 1, 0])
    system.add_reaction(0.01, [0, 0, 1, 0], [0, 0, 0, 1])
    system.add_reaction(0.05, [1, 0, 0, 0], [0, 0, 0, 0])
    system.add_reaction(0.05, [0, 1, 0, 0], [0, 0, 0, 0])
    system.add_reaction(0.05, [0, 0, 1, 0], [0, 0, 0, 0])
    system.add_reaction(0.05, [0, 0, 0, 1], [0, 0, 0, 0])
    # system.add_reaction(0.1, [1, 0, 0, 1], [0, 0, 0, 1])

    system.evolute(total_t=total_t, seed=seed, IC=IC, is_print=is_print)

    t = system.t
    vars = []
    for i in range(num_elements):
        vars.append([var[i] for var in system.n])
    
    if save:
        plt.figure(figsize=(6,6))
        for i in range(0,num_elements):
            plt.plot(t, vars[i], label=f'X{i+1}')
        plt.xlabel('time / s')
        plt.legend()
        plt.savefig(f'Data/SignallingCascade/origin/{seed}/origin.pdf', dpi=300)
    
        # calculate average dt
        digit = f'{np.average(np.diff(t)):.20f}'.count("0")
        avg = np.round(np.average(np.diff(t)), digit)

        np.savez(f'Data/SignallingCascade/origin/{seed}/origin.npz', t=t, X1=vars[0], X2=vars[1], X3=vars[2], X4=vars[3], dt=avg)
    else:
        return {'t': t, 'X1': vars[0], 'X2': vars[1], 'X3': vars[2], 'X4': vars[3]}


def generate_self_replicator_origin(total_t=None, seed=729, IC=[3, 50, 47], save=True, is_print=False):

    time.sleep(1.0)

    seed_everything(seed)
    if save: os.makedirs(f'Data/SelfReplicator/origin/{seed}/', exist_ok=True)

    num_elements = 3
    system = System(num_elements)

    # A, D, L
    system.add_reaction(1, [1, 1, 0], [0, 2, 0])
    system.add_reaction(1, [1, 0, 1], [0, 0, 2])
    system.add_reaction(1, [1, 0, 0], [0, 1, 0])
    system.add_reaction(1, [0, 1, 0], [1, 0, 0])
    system.add_reaction(1, [1, 0, 0], [0, 0, 1])
    system.add_reaction(1, [0, 0, 1], [1, 0, 0])

    system.evolute(total_t=total_t, seed=seed, IC=IC, is_print=is_print)

    t = system.t
    vars = []
    for i in range(num_elements):
        vars.append([var[i] for var in system.n])
    
    if save:
        plt.figure(figsize=(6,6))
        name = ['A', 'D', 'L']
        for i in range(0,num_elements):
            plt.plot(t, vars[i], label=name[i])
        plt.xlabel('time / s')
        plt.legend()
        plt.savefig(f'Data/SelfReplicator/origin/{seed}/origin.pdf', dpi=300)
    
        # calculate average dt
        digit = f'{np.average(np.diff(t)):.20f}'.count("0")
        avg = np.round(np.average(np.diff(t)), digit)

        np.savez(f'Data/SelfReplicator/origin/{seed}/origin.npz', t=t, S=vars[0], I=vars[1], R=vars[2], dt=avg)
    else:
        return {'t': t, 'vars': vars}