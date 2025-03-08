# Finds rho from polytropic EOS
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

l = 2
g = 2.5

Xi, Xf, Xspc = 0.01, 0.98, 50


def Rho(rho0):
    return rho0*(1-X**l)

def Pres(pre0):
    return pres0*(1-X**l)**g

X = np.linspace(Xi, Xf, num=Xspc)
rho_vals = np.array([1.00e-9, 2.50e-9, 5.00e-9])
pres_vals = np.array([1.00e-9, 2.50e-9, 5.00e-9])

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
axes = axes.flatten()

presfig, presaxes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
presaxes = presaxes.flatten()

#Plot Params
plt.rcParams['font.family'] = 'serif' #'STIXGeneral' #'serif'
matplotlib.rcParams['font.size'] = '16'
matplotlib.rcParams['ps.fonttype'] = 42 #note: fontype 42 compatible with MNRAS style file when saving figs
matplotlib.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.linewidth'] = 1.
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.numpoints'] = 1  # uses 1 symbol instead of 2
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.handletextpad'] = 0.3

for idx, rho0 in enumerate(rho_vals):
    ax = axes[idx]
    
    if idx == 0:  # Only the first column gets the y-axis label
        ax.set_ylabel(r'$\rho$', fontsize=18)
    
    ax.plot(X, Rho(rho0), lw=2, ls='-')
    
    ax.set_xlabel(r"$\chi$", fontsize = 18)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.grid()
    ax.minorticks_on()
    ax.set_aspect('auto')
    ax.tick_params(which='both', direction='in', top=True, bottom=True, left=True, right=True)

for jdx, pres0 in enumerate(pres_vals):
    presax = presaxes[jdx]
    
    if jdx == 0:  # Only the first column gets the y-axis label
        presax.set_ylabel(r'$p$', fontsize=18)
    
    presax.plot(X, Pres(pres0), lw=2, ls='-')
    
    presax.set_xlabel(r"$\chi$", fontsize = 18)
    presax.tick_params(axis='x', labelsize=15)
    presax.tick_params(axis='y', labelsize=15)
    presax.grid()
    presax.minorticks_on()
    presax.set_aspect('auto')
    presax.tick_params(which='both', direction='in', top=True, bottom=True, left=True, right=True)

for idx, rho0 in enumerate(rho_vals):
    exponent = int(np.floor(np.log10(rho0)))  # Get the exponent in scientific notation
    coeff = rho0 / (10 ** exponent)     # Get the mantissa
    fig.text(0.23 + idx * 0.31, 0.94, rf'$\rho_0 = {coeff:.1f} \times 10^{{{exponent}}}$', fontsize=18, ha='center', va='center', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
    
for jdx, pres0 in enumerate(pres_vals):
    exponent1 = int(np.floor(np.log10(pres0)))  # Get the exponent in scientific notation
    coeff1 = pres0 / (10 ** exponent1)     # Get the mantissa
    presfig.text(0.23 + jdx * 0.31, 0.94, rf'$p_0 = {coeff1:.1f} \times 10^{{{exponent1}}}$', fontsize=18, ha='center', va='center', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
    
fig.tight_layout(rect=[0.06, 0.06, 1, 0.96])
fig.savefig("PolyRho.pdf", bbox_inches='tight')
presfig.tight_layout(rect=[0.06, 0.06, 1, 0.96])
presfig.savefig("PolyPres.pdf", bbox_inches='tight')
plt.show()
