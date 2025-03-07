#Charged Massive Animation Code for Rho
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.integrate import cumulative_trapezoid

# Define global constants
Q, Rb, Lam = 1e3, 1e4, 1e-52

# Loop Constants
Xi, Xf, Xspc = 0.01, 0.98, 50
ni, nf, nspc = 1.01, 5, 30
li, lf, lspc = 0.01, 5, 30
gi, gf, gspc = 0.001, 5, 30

# Eq. 43 in paper
def phiprime(n, l, g, rho0, p0):
    fact1 = 4*np.pi*rho0*Rb
    fact12 = (l+1)/(l+3)
    fact13 = p0/rho0
    fact2 = (n-1)*Q*Q/Rb/Rb/Rb/(2*n-1)
    fact21 = Lam*Rb/3
    nume = (fact1*X*(1/3+fact12*pow(X, l)+fact13*pow(1-pow(X, l),g))
      -fact2*pow(X, 2*n-3)+fact21*X)
    denom = (1-2*fact1*Rb*pow(X,2)*(1/3-pow(X, l)/(l+3))
      -fact2*Rb*pow(X,2*n-2)/(n-1)+fact21*Rb*pow(X,2))
    return [nume/denom, denom]

# Eq. 44 in paper
def Fprime(n, l, g, rho0, p0):
    phip = phiprime(n, l, g, rho0, p0)
    num = np.exp(cumulative_trapezoid(phip[0]*Rb, X, initial=0))
    return num*Q*pow(X,n-2)/Rb/Rb/pow(phip[1],1/2)

X = np.linspace(Xi, Xf, num=Xspc)
nvals = np.linspace(ni, nf, num=nspc)
lvals = np.linspace(li, lf, num=lspc)
gvals = np.linspace(gi, gf, num=gspc)

# Rho0 Range
rho0_min, rho0_max = 1e-10, 2e-9
rho0_vals = np.linspace((rho0_min), (rho0_max), 100)

# Set up the figure and axis
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
scatter_s = ax.scatter([], [], [], marker="s", color='lightseagreen', alpha=1, rasterized=True)
scatter_us = ax.scatter([], [], [], marker="s", color='blue', alpha=1, rasterized=True)

# Axes labels and limits
ax.set_xlim(0.51, 5.51)
ax.set_ylim(-0.51, 5.5)
ax.set_zlim(0, 5.5)
ax.set_xlabel("$n$")
ax.set_ylabel("$l$")
ax.zaxis.set_rotate_label(False)
ax.set_zlabel(r'$\Gamma$', rotation=0)

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

def update(frame):
    # Current rho value
    rho0 = rho0_vals[frame]
    p0 = 0.25 * rho0

    ns, ls, gs = [], [], []
    nus, lus, gus = [], [], []
    eE = 10

    for n in nvals:
        for l in lvals:
            for g in gvals:
                if p0 / rho0 > 1 / g:
                    continue
                phip = phiprime(n, l, g, rho0, p0)
                if len(np.where(phip[1]<=0)[0])!=0:
                    continue
                F = cumulative_trapezoid(Fprime(n, l, g, rho0, p0)*Rb, X, initial=0)
                Chargemassless = (1 - X*Rb*phip[0])/(X*Rb*(phip[0]*F-Fprime(n, l, g, rho0, p0))-F)
                crossing = np.where(np.diff(np.sign(Chargemassless - 1)))[0]
                # Cutoff for discontinuities
                if len(crossing) > 1:
                    continue
                if np.any((Chargemassless[0] < eE) and (np.max(Chargemassless) >= eE)):
                    if np.any(Chargemassless[-1] < eE):
                        nus.append(n)
                        lus.append(l)
                        gus.append(g)
                    else:
                        ns.append(n)
                        ls.append(l)
                        gs.append(g)
                elif np.any((Chargemassless[0] > eE) and (Chargemassless[-1] < eE)):
                        ns.append(n)
                        ls.append(l)
                        gs.append(g)
                
    # Update scatter plot data
    scatter_s._offsets3d = (ns, ls, gs)
    scatter_us._offsets3d = (nus, lus, gus)
    ax.set_title(r"$\rho_0 = " + f"{rho0:.1e}$", fontsize=14)
    plt.suptitle(
        rf"$(p_0/\rho_0 = {p0 / rho0:.2f}, Q = 10^3, R_b = 10^4,  1/\mathcal{{E}}^2 = 1/100)$",
        fontsize=12,
        y=0.87,
    )
    return scatter_s, scatter_us

# Create animation
num_frames = len(rho0_vals)
ani = FuncAnimation(fig, update, frames=num_frames, interval=1000, blit=False)

# Save the animation as a video
writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
ani.save("0.25ChargeMassless_E=10.mp4", writer=writer)

# Display animation
plt.show()