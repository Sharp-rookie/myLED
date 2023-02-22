import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from pytorch_lightning import seed_everything
import warnings;warnings.simplefilter('ignore')


class FitzHugh_Nagumo():
    def __init__(self, a, c, I, g):
        self.a = a
        self.c = c
        self.I = I
        self.g = g

    def __call__(self, y0, t):
        
        v, w, b = y0

        dv = self.a * (-v * (v-1) * (v-b) - w + self.I)
        dw = v - self.c * w
        db = self.g(t)

        return [dv, dw, db]


def generate_original_data(trace_num, total_t=5, dt=0.01, save=True):
    
    def solve_1_trace(trace_id=0, total_t=5, dt=0.01):
        
        seed_everything(trace_id)
        
        y0 = [0., 0., 0.]

        t  =np.arange(0, total_t, dt)
        period = 12
        a, c, I = 1e7, 1., 0.2
        g = lambda x: np.pi/period * np.sin(x*2*np.pi/period)
        sol = odeint(FitzHugh_Nagumo(a,c,I,g), y0, t)

        import scienceplots
        plt.style.use(['science'])
        plt.rcParams.update({'font.size':16})
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax1.plot(t, sol[:,0], color='blue', label='v')
        ax1.plot(t, sol[:,1], color='green', label='w')
        ax2.plot(t, sol[:,2], color='red')
        ax1.legend()
        ax2.set_ylabel('b')
        ax1.set_position([0.1, 0.35, 0.8, 0.6])
        ax2.set_position([0.1, 0.1, 0.8, 0.2])

        plt.savefig(f'fhn.pdf', dpi=300)
        
        return sol
    
    if save and os.path.exists('Data/origin/origin.npz'): return
    
    trace = []
    for trace_id in tqdm(range(1, trace_num+1)):
        sol = solve_1_trace(trace_id, total_t, dt)
        trace.append(sol)
    
    if save: 
        os.makedirs('Data/origin', exist_ok=True)
        np.savez('Data/origin/origin.npz', trace=trace, dt=dt, T=total_t)

    print(f'save origin data form seed 1 to {trace_num} at Data/origin/')

    return trace
generate_original_data(1, total_t=24., dt=0.01, save=False)