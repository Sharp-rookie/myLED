# -*- coding: utf-8 -*-
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
    ## u: activator, v: inhibitor
    ###########################################

    N = 100
    L = 20
    x = np.linspace(0, L, N+1)
    dx = x[1]-x[0]

    epsilon=0.006  # Bifurcation parameter
    Du = 1.
    Dv = 4.
    a1 = 2.
    a0 = -0.03

    omegas = [2/(1+3*Du*dt/(dx*dx)), 2/(1+3*Dv*dt/(dx*dx))]
    n1 = 1/3

    t = 0
    it = 0

    N_T = int(np.ceil(tf/dt))+1

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

    rho_act, rho_in, t_vec = np.zeros((N_T, N+1)), np.zeros((N_T, N+1)), np.zeros((N_T))  # u, v density
    mom_act,mom_in = np.zeros((N_T, N+1)), np.zeros((N_T, N+1))  # Momentum terms
    energ_act, energ_in = np.zeros((N_T, N+1)), np.zeros((N_T, N+1))  # Energy terms

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

        # Saving the data
        rho_act[it], rho_in[it], t_vec[it] = rho_act_t, rho_in_t, t
        mom_act[it],mom_in[it] = mom_act_t, mom_in_t
        energ_act[it], energ_in[it] = energ_act_t, energ_in_t

        it+=1
        t +=dt

    return rho_act, rho_in, t_vec, mom_act, mom_in, energ_act, energ_in, dt, N, L, dx, x, Du, Dv, a0, a1, n1, omegas, tf