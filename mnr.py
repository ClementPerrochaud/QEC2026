# IMPORTS

from mnr_files.params import *
from mnr_files.functions import *

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


### = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


# PARAMETERS

# Set here the values of the size of the problem: code distance.

# In the SPOQC implementation of the surface code, we can consider that there are as many RUS modeles as sources.

# Note that the following parameters are assigned values by default, corresponding to the current hardware (see file params.py):
# cryostat temperature (T_cryo), 
# cooling power (cooling_power), 
# cryostat efficiency (cryo_efficiency), 
# number of detectors per cryostat (det_per_cryo), 
# non-cryogenic power consumption (extra_power_consumption), 
# source brightness at the ouptut fiber (bof), 
# efficiency of the SNSPDs (eta_snspd), 
# efficiency of the demultiplexers (eta_dmx), 
# attenuation at fiber - wave-guide interface (cfwg), 
# attenuation per MZI (cmzi).  
# The value of these parameters can be changed manually in the functions (see functions.py).

d=3
nd=num_det(d)
ns=num_sources(d)
nrus=num_rus(d)
nps=num_ps(d)

print(f'Number of PNRs={nd}')
print(f'Number of sources={ns}')


### = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


params_SPOQC(d)


### = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


# POWER CONSUMPTION

# Cryogenic power consumption 

Pcryo=P_cryo_SPOQC(nd,ns,nrus,T=T_cryo,x=cryo_efficiency)

# Total power consumption 

Ptot = P_tot_SPOQC(nd,ns,nrus,nps,T=T_cryo,x=cryo_efficiency)

print(f'Cryogenic power consumption = {Pcryo} W')
print(f'Power consumption = {Ptot} W')


### = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


# ENERGY CONSUMPTION

# To compute the energy per sample, we multiply the power consumption by the time of an error correction cycle.

en_ecc = energy_ecc(d)

print(f'Energy per error correcting cycle = {en_ecc} J')


### = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


# PLOT metric

d_min, d_max = 1, 25
T_min, T_max = 0.5, 25
t_values = [1e-11, 1e-8, 1e-5, 1e-2]

T_ranges = [(0.1, 25), (0.1, 3), (0.1, 1.5), (0.1, 1)]

M_cap = 100

plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(2, 2, figsize=(14, 12), layout='constrained')
axes = ax.flatten()

d_values = np.linspace(d_min, d_max, 500)

mesh_list = []
M_list = []
for t, (Tmin_i, Tmax_i) in zip(t_values, T_ranges):
    T_values = np.linspace(Tmin_i, Tmax_i, 500)
    d_mesh, T_mesh = np.meshgrid(d_values, T_values)
    mesh_list.append((d_mesh, T_mesh))
    M_list.append(- np.log10( np.maximum(logical_error(d_mesh, T_mesh, t), 1e-300) ))
M_min = min(M[M > 0].min() for M in M_list)
norm = mcolors.LogNorm(vmin=M_min, vmax=M_cap)

for i, t in enumerate(t_values):
    a = axes[i]
    Tmin_i, Tmax_i = T_ranges[i]
    d_mesh, T_mesh = mesh_list[i]
    mesh = a.pcolormesh(d_mesh, T_mesh, M_list[i], cmap='cubehelix', shading='auto', norm=norm)

    a.set_yscale('log')
    a.set_xlim(d_min, d_max)
    a.set_ylim(Tmin_i, Tmax_i)
    a.set_title(f't = {t:g} s')

axes[2].set_xlabel(r'Code distance $d$'); axes[3].set_xlabel(r'Code distance $d$')
axes[0].set_ylabel(r'Temperature of Cryostat $T_\text{cryo}$ (K)'); axes[2].set_ylabel(r'Temperature of Cryostat $T_\text{cryo}$ (K)')

fig.colorbar(mesh, ax=axes.tolist(), label=r"Metric: $-\log p'$", extend='max')
fig.savefig('mnr_files/contours_metric.png', bbox_inches='tight')


### = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


# PLOT: Metric Contours in Energy Heatmap

threshold = threshold_fit()[-1] 
target_metric_values = np.linspace(1,11,6)

def metric_Td(T, d, t): return - np.log10( logical_error(d, T, t) )
def energy_Td(T, d):    return energy_ecc(d, T)

contour_points_dict = {}

d_min_opt_values = [1, 3, 5, 7]

metric_norm = mcolors.LogNorm(vmin=M_min, vmax=M_cap)

d_values = np.linspace(d_min, d_max, 500)
mesh_list = []
E_list = []
for Tmin_i, Tmax_i in T_ranges:
    T_values = np.linspace(Tmin_i, Tmax_i, 500)
    d_mesh, T_mesh = np.meshgrid(d_values, T_values)
    mesh_list.append((d_mesh, T_mesh))
    E_list.append(energy_ecc(d_mesh, T_mesh))
E_min = min(E.min() for E in E_list)
E_max = max(E.max() for E in E_list)
norm = mcolors.LogNorm(vmin=E_min, vmax=E_max)

for d_min_opt in d_min_opt_values:
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(2, 2, figsize=(14, 12), layout='constrained')
    axes = ax.flatten()
    fig.suptitle(rf'Energy optimum for $d \geq {d_min_opt}$')

    for i, t in enumerate(t_values):
        a = axes[i]
        Tmin_i, Tmax_i = T_ranges[i]
        d_mesh, T_mesh = mesh_list[i]
        E_values = E_list[i]
        M_values = - np.log10( np.maximum(logical_error(d_mesh, T_mesh, t), 1e-300) )

        mesh = a.pcolormesh(d_mesh, T_mesh, E_values, cmap='rainbow', shading='auto', norm=norm)
        contour_set = a.contour(d_mesh, T_mesh, M_values, levels=target_metric_values, cmap='cubehelix', norm=metric_norm)
        a.clabel(contour_set, fontsize=8, fmt='%d')

        best_per_level = {}
        for path in contour_set.get_paths():
            v = path.vertices
            if len(v) == 0:
                continue
            d_seg, T_seg = v[:, 0], v[:, 1]
            keep = d_seg >= d_min_opt
            if not keep.any():
                continue
            d_seg, T_seg = d_seg[keep], T_seg[keep]
            E_seg = energy_ecc(d_seg, T_seg)
            k = int(np.argmin(E_seg))
            m = metric_Td(T_seg[k], d_seg[k], t)
            lvl = target_metric_values[int(np.argmin(np.abs(target_metric_values - m)))]
            prev = best_per_level.get(lvl)
            if prev is None or E_seg[k] < prev[0]:
                best_per_level[lvl] = (E_seg[k], d_seg[k], T_seg[k])

        for lvl, (E_best, d_best, T_best) in sorted(best_per_level.items()):
            a.plot(d_best, T_best, 'ko')
            contour_points_dict[(d_min_opt, lvl, t)] = (d_best, T_best)
            print(f"[d>={d_min_opt}] Min-energy point for M = {lvl:g}, t = {t}: "
                  f"T = {T_best:.3f} K, d = {d_best:.3f}, E = {E_best:.4g} J")

        if d_min_opt > d_min:
            a.axvline(d_min_opt, color='k', ls='--', lw=1)

        a.set_yscale('log')
        a.set_xlim(d_min, d_max)
        a.set_ylim(Tmin_i, Tmax_i)
        a.set_title(f't = {t:g} s')

    axes[2].set_xlabel(r'Code distance $d$'); axes[3].set_xlabel(r'Code distance $d$')
    axes[0].set_ylabel(r'Temperature of Cryostat $T_\text{cryo}$ (K)'); axes[2].set_ylabel(r'Temperature of Cryostat $T_\text{cryo}$ (K)')

    fig.colorbar(mesh, ax=axes.tolist(), label=r'Energy cost (J)')
    fig.savefig(f'mnr_files/contours_energy_dmin_{d_min_opt}.png', bbox_inches='tight')
    plt.close(fig)
