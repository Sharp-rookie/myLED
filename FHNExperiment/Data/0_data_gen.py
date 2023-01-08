# -*- coding: utf-8 -*-
import os
import numpy as np


def LBM(f1_act, f_1_act, f0_act, f1_in, f_1_in, f0_in, omegas, a1, a0, epsilon, n1, dt):
    """Lattice Boltzmann Algorithm for Fitzhugh-nagumo system"""

    #############################
    # Collision terms (omega)
    #############################
    # Activator
    rho_act_t = f0_act + f1_act + f_1_act
    ome_act = omegas[0]

    omega1_act  = -ome_act * (f1_act  - n1 * rho_act_t)
    omega_1_act = -ome_act * (f_1_act - n1 * rho_act_t)
    omega0_act  = -ome_act * (f0_act  - n1 * rho_act_t)

    # Inhibitor
    rho_in_t = f0_in + f1_in + f_1_in
    ome_in = omegas[1]

    omega1_in  = -ome_in * (f1_in  - n1 * rho_in_t)
    omega_1_in = -ome_in * (f_1_in - n1 * rho_in_t)
    omega0_in  = -ome_in * (f0_in  - n1 * rho_in_t)

    #############################
    # Reaction terms (R)
    #############################
    reac1_act  = n1 * dt * (rho_act_t - np.power(rho_act_t, 3.0) - rho_in_t)
    reac_1_act = n1 * dt * (rho_act_t - np.power(rho_act_t, 3.0) - rho_in_t)
    reac0_act  = n1 * dt * (rho_act_t - np.power(rho_act_t, 3.0) - rho_in_t)

    reac1_in   = n1 * dt * epsilon * (rho_act_t - a1 * rho_in_t - a0)
    reac_1_in  = n1 * dt * epsilon * (rho_act_t - a1 * rho_in_t - a0)
    reac0_in   = n1 * dt * epsilon * (rho_act_t - a1 * rho_in_t - a0)

    #############################
    # Updating Activator terms
    #############################
    f1_act_next     = np.zeros_like(f1_act)
    f1_act_temp     = f1_act + omega1_act + reac1_act
    f1_act_next[1:] = f1_act_temp[:-1]
    f1_act_next[0]  = f1_act[0] + omega1_act[0] + reac1_act[0]

    f0_act_next     = np.zeros_like(f0_act)
    f0_act_temp     = f0_act + omega0_act + reac0_act
    f0_act_next     = f0_act_temp

    f_1_act_next    = np.zeros_like(f_1_act)
    f_1_act_temp    = f_1_act + omega_1_act + reac_1_act
    f_1_act_next[:-1]= f_1_act_temp[1:]
    f_1_act_next[-1] = f1_act_next[-1]

    #############################
    # Updating Inhibitor terms
    #############################
    f1_in_next     = np.zeros_like(f1_in)
    f1_in_temp     = f1_in + omega1_in + reac1_in
    f1_in_next[1:] = f1_in_temp[:-1]
    f1_in_next[0]  = f1_in[0] + omega1_in[0] + reac1_in[0]

    f0_in_next     = np.zeros_like(f0_in)
    f0_in_temp     = f0_in + omega0_in + reac0_in
    f0_in_next     = f0_in_temp

    f_1_in_next    = np.zeros_like(f_1_in)
    f_1_in_temp    = f_1_in + omega_1_in + reac_1_in
    f_1_in_next[:-1]= f_1_in_temp[1:]
    f_1_in_next[-1] = f1_in_next[-1]

    return f1_act_next, f_1_act_next, f0_act_next, f1_in_next, f_1_in_next, f0_in_next


def run_lb_fhn_ic(id, num, rho_act_0, rho_in_0, tf, dt):
    ###########################################
    ## Simulation of the Lattice Boltzman Method
    ## for the FitzHugh-Nagumo
    ###########################################

    N = 100
    L = 20
    x = np.linspace(0, L, N+1)
    dx = x[1]-x[0]

    Dx = 1 # Dact (activator)
    Dy = 4 # Din  (inhibitor)

    a0 = -0.03
    a1 = 2.0

    omegas = [2/(1+3*Dx*dt/(dx*dx)), 2/(1+3*Dy*dt/(dx*dx))]
    n1 = 1/3

    # Bifurcation parameter
    epsilon = 0.006

    t = 0
    it = 0

    N_T = int(np.ceil(tf/dt))+1

    # Storing the density
    rho_act = np.zeros((N_T, N+1))
    rho_in  = np.zeros((N_T, N+1))
    t_vec   = np.zeros((N_T))

    # Storing momentum terms
    mom_act = np.zeros((N_T, N+1))
    mom_in  = np.zeros((N_T, N+1))

    # Storing energy terms
    energ_act = np.zeros((N_T, N+1))
    energ_in  = np.zeros((N_T, N+1))

    # Activator
    f1_act  = 1/3*rho_act_0
    f0_act  = f1_act
    f_1_act = f1_act

    rho_act_t   = f0_act + f1_act + f_1_act
    mom_act_t   = f1_act - f_1_act
    energ_act_t = 0.5 * (f1_act + f_1_act)

    # Inhibitor
    f1_in   = 1/3*rho_in_0
    f0_in   = f1_in
    f_1_in  = f1_in

    rho_in_t   = f0_in + f1_in + f_1_in
    mom_in_t   = f1_in - f_1_in
    energ_in_t = 0.5 * (f1_in + f_1_in)

    while np.abs(t-tf)>1e-6:
        print("\r Generating Data[{:d}/{:d}]: {:.3f}/{:.2f}. {:.2f}% (dt={:.3f})".format(id, num, t, tf, t/tf*100.0, dt), end=' ')

        # Propagate the Lattice Boltzmann in time
        f1_act, f_1_act, f0_act, f1_in, f_1_in, f0_in = LBM(f1_act, f_1_act, f0_act, f1_in, f_1_in, f0_in, omegas, a1, a0, epsilon, n1, dt)

        # Updating Activator
        rho_act_t   = f0_act + f1_act + f_1_act
        mom_act_t   = f1_act - f_1_act
        energ_act_t = 0.5 * (f1_act + f_1_act)

        # Updating Inhibitor
        rho_in_t   = f0_in + f1_in + f_1_in
        mom_in_t   = f1_in - f_1_in
        energ_in_t = 0.5 * (f1_in + f_1_in)

        rho_act[it]     = rho_act_t
        rho_in[it]      = rho_in_t
        t_vec[it]       = t
        mom_act[it]     = mom_act_t
        mom_in[it]      = mom_in_t
        energ_act[it]   = energ_act_t
        energ_in[it]    = energ_in_t

        it+=1
        t +=dt

    return rho_act, rho_in, t_vec, mom_act, mom_in, energ_act, energ_in, dt, N, L, dx, x, Dx, Dy, a0, a1, n1, omegas, tf, a0


def generate_origin_data(tf=451, dt=0.001):
    # u is the Activator
    # v is the Inhibitor

    file_names = ["y00", "y01", "y02", "y03", "y04", "y05"]

    rho_act_all = []
    rho_in_all = []
    t_vec_all = []
    mom_act_all = []
    mom_in_all = []
    energ_act_all = []
    energ_in_all = []

    for f_id, file_name in enumerate(file_names):
        
        # load inital-condition file
        rho_act_0 = np.loadtxt(f"ICs/{file_name}u.txt", delimiter="\n")
        rho_in_0 = np.loadtxt(f"ICs/{file_name}v.txt", delimiter="\n")
        x = np.loadtxt("ICs/y0x.txt", delimiter="\n")
        
        # simulate by LBM
        rho_act, rho_in, t_vec, mom_act, mom_in, energ_act, energ_in, dt, N, L, dx, x, Dx, Dy, a0, a1, n1, omegas, tf, a0 = run_lb_fhn_ic(f_id, len(file_names), rho_act_0, rho_in_0, tf, dt)

        # record
        rho_act_all.append(rho_act)
        rho_in_all.append(rho_in)
        t_vec_all.append(t_vec)
        mom_act_all.append(mom_act)
        mom_in_all.append(mom_in)
        energ_act_all.append(energ_act)
        energ_in_all.append(energ_in)

    # data = {
    #     "rho_act_all":rho_act_all,
    #     "rho_in_all":rho_in_all,
    #     "t_vec_all":t_vec_all,
    #     "mom_act_all":mom_act_all,
    #     "mom_in_all":mom_in_all,
    #     "energ_act_all":energ_act_all,
    #     "energ_in_all":energ_in_all,
    #     "dt":dt,
    #     "N":N,
    #     "L":L,
    #     "dx":dx,
    #     "x":x,
    #     "Dx":Dx,
    #     "Dy":Dy,
    #     "a0":a0,
    #     "a1":a1,
    #     "n1":n1,
    #     "omegas":omegas,
    #     "tf":tf,
    #     "a0":a0,
    # }

    os.makedirs("Data/origin", exist_ok=True)
    np.savez("Data/origin/lattice_boltzmann.npz", rho_act_all=rho_act_all, rho_in_all=rho_in_all, t_vec_all=t_vec_all, dt=dt, x=x)


if __name__ == '__main__':
    
    generate_origin_data()