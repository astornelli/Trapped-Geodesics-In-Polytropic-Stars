# Plots normalized density, pressure (polytropic EOS) and charge for given n, l, Gamma
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

n = 3
l = 2
g = 2.5

Xi, Xf, Xspc = 0.01, 1, 50


def Rhon(X):     # normalized density
    return (1-X**l)

def Presn(X):    # normalized pressure
    return (1-X**l)**g

def Qn(X):       # normalized charge
    return X**n

X = np.linspace(Xi, Xf, num=Xspc)
rho_vals = np.array([1.00e-9])#, 2.50e-9, 5.00e-9])
pres_vals = np.array([1.00e-9, 2.50e-9, 5.00e-9])

fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True)

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


ax.plot(X, Rhon(X), lw=2, ls='-',label=r'$\rho/\rho_0$')
ax.plot(X, Presn(X), lw=2, ls='--',label=r'$p/p_0$')
ax.plot(X, Qn(X), lw=2, ls=':',label=r'$q/Q$')
ax.set_xlabel(r"$\chi$", fontsize = 15)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.grid()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.legend(loc='best')
ax.minorticks_on()
ax.set_aspect('auto')
ax.tick_params(which='both', direction='in', top=True, bottom=True, left=True, right=True)
x_ticks = ax.xaxis.get_major_ticks()
x_ticks[0].label1.set_visible(False) ## set first x tick label invisible
    
fig.tight_layout(rect=[0.06, 0.06, 1, 0.96])
fig.savefig("PolyPresRho.pdf", bbox_inches='tight')
plt.show()
