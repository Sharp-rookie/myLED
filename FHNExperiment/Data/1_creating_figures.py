# -*- coding: utf-8 -*-
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
rc('text', usetex=True)

plt.rcParams["text.usetex"] = True
plt.rcParams['xtick.major.pad']='10'
plt.rcParams['ytick.major.pad']='10'
font = {'weight':'normal', 'size':16}
plt.rc('font', **font)


data = np.load("Data/origin/lattice_boltzmann.npz", allow_pickle=True)
rho_act_all = np.array(data["rho_act_all"])
rho_in_all = np.array(data["rho_in_all"])
x = np.array(data["x"])
dt_data = data["dt"]
t_vec_all = np.array(data["t_vec_all"])
print(np.shape(t_vec_all))
print(np.shape(rho_in_all))
print(np.shape(rho_act_all))

labels = ["Test", "Train", "Train", "Train", "Val", "Val"]
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))
for ic in range(np.shape(rho_act_all)[0]):
    ax1.plot(x, rho_act_all[ic,0,:], label="trace{:}".format(labels[ic]), linewidth=2)

ax1.set_ylabel(r"$u(x,t=0)$")
ax1.set_xlabel(r"$x$")
ax1.set_xlim([np.min(x), np.max(x)])
ax1.set_title("Activator")

for ic in range(np.shape(rho_in_all)[0]):
    ax2.plot(x, rho_in_all[ic,0,:], label="trace{:}".format(labels[ic]), linewidth=2)

ax2.set_ylabel(r"$v(x,t=0)$")
ax2.set_xlabel(r"$x$")
ax2.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
ax2.set_title("Inhibitor")
ax2.set_xlim([np.min(x), np.max(x)])
plt.tight_layout()

os.makedirs("Data/origin/figures/", exist_ok=True)
plt.savefig("Data/origin/figures/initial_conditions.jpg", bbox_inches="tight", dpi=300)
plt.close()


for ic in range(len(rho_act_all)):

    T_end = 451
    N_end = int(T_end / dt_data)

    rho_act = rho_act_all[ic]
    rho_in = rho_in_all[ic]
    t_vec = t_vec_all[ic]

    subsample_time = 10
    
    os.makedirs("Data/origin/figures/{:}/".format(ic), exist_ok=True)
    
    ############
    # Activator
    ############

    X = x
    Y = t_vec[:N_end]
    Y = Y[::subsample_time]

    X, Y = np.meshgrid(X, Y)

    Z = rho_act[:N_end]
    Z = Z[::subsample_time]

    fig = plt.figure(figsize=(12,10))
    ax = fig.gca(projection='3d')
    print(np.shape(X))
    print(np.shape(Y))
    print(np.shape(Z))
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rasterized=True)
    ax.set_xlabel(r"$x$", labelpad=20)
    ax.set_ylabel(r"$t$", labelpad=20)
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel(r"$u(x,t)$", rotation=0, labelpad=20)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, orientation="horizontal")
    ax.invert_xaxis()
    ax.view_init(elev=34., azim=-48.)
    plt.savefig("Data/origin/figures/{:}/surface_act.jpg".format(ic), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.gca()
    mp = ax.contourf(X, Y, Z, 100, cmap=plt.get_cmap("seismic"),zorder=-9)
    ax.set_ylabel(r"$t$")
    ax.set_xlabel(r"$x$")
    fig.colorbar(mp)
    plt.gca().set_rasterization_zorder(-1)
    plt.savefig("Data/origin/figures/{:}/contourf_act.jpg".format(ic), bbox_inches="tight", dpi=300)
    plt.close()

    ############
    # Inhibitor
    ############
    
    X = x
    Y = t_vec[:N_end]
    Y = Y[::subsample_time]

    X, Y = np.meshgrid(X, Y)

    Z = rho_in[:N_end]
    Z = Z[::subsample_time]
    
    fig = plt.figure(figsize=(12,10))
    ax = fig.gca(projection='3d')
    print(np.shape(X))
    print(np.shape(Y))
    print(np.shape(Z))
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rasterized=True)
    ax.set_xlabel(r"$x$", labelpad=20)
    ax.set_ylabel(r"$t$", labelpad=20)
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel(r"$v(x,t)$", rotation=0, labelpad=20)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, orientation="horizontal")
    ax.invert_xaxis()
    ax.view_init(elev=34., azim=-48.)
    plt.savefig("Data/origin/figures/{:}/surface_in.jpg".format(ic), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.gca()
    mp = ax.contourf(X, Y, Z, 100, cmap=plt.get_cmap("seismic"),zorder=-9)
    ax.set_ylabel(r"$t$")
    ax.set_xlabel(r"$x$")
    fig.colorbar(mp)
    plt.gca().set_rasterization_zorder(-1)
    plt.savefig("Data/origin/figures/{:}/contourf_in.jpg".format(ic), bbox_inches="tight", dpi=300)
    plt.close()