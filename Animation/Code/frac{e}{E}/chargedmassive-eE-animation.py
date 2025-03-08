# Charged Massive Animation Code for e/E
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.integrate import cumulative_trapezoid

# Define global constants
Q, rho0, Lam = 1e3, 1e-9, 1e-52
p0 = rho0*0.25
Rb = 1e4
# Loop Constants
Xi, Xf, Xspc = 0.01, 0.98, 50
ni, nf, nspc = 1.01, 5, 30
li, lf, lspc = 0.01, 5, 30
gi, gf, gspc = 0.001, 5, 30

# Eq. 43 in paper
def phiprime(n, l, g):
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
def Fprime(n, l, g):
    phip = phiprime(n, l, g)
    num = np.exp(cumulative_trapezoid(phip[0]*Rb, X, initial=0))
    return num*Q*pow(X,n-2)/Rb/Rb/pow(phip[1],1/2)

X = np.linspace(Xi, Xf, num=Xspc)
nvals = np.linspace(ni, nf, num=nspc)
lvals = np.linspace(li, lf, num=lspc)
gvals = np.linspace(gi, gf, num=gspc)

# Rb Range
eE_min, eE_max = -1e2, 1e2
eE_vals = np.linspace(eE_min, eE_max, 200)

# Set up the figure and axis
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
scatter_s = ax.scatter([], [], [], marker="s", color='mediumpurple', alpha=1, rasterized=True)
scatter_us = ax.scatter([], [], [], marker="s", color='magenta', alpha=1, rasterized=True)

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
    eE = eE_vals[frame]
    E = 10
    
    ns, ls, gs = [], [], []
    nus, lus, gus = [], [], []

    for n in nvals:
        for l in lvals:
            for g in gvals:
                if p0 / rho0 > 1 / g:
                    continue
                phip = phiprime(n, l, g)
                Fpri = Fprime(n, l, g)
                if len(np.where(phip[1]<=0)[0])!=0:
                    continue
                F = cumulative_trapezoid(Fpri*Rb, X, initial=0)
                phi = cumulative_trapezoid(phip[0], X, initial=0)
                Chargemassive = (1 + eE*F)*np.exp(-2*phi)*pow(E,2)*((1+eE*F)*(1-X*Rb*phip[0])+eE*Fpri*X*Rb)
                CM_deriv=np.gradient(Chargemassive, X)
                crossing = np.where(np.diff(np.sign(abs(CM_deriv) - 1e10)))[0]
                # Cutoff for discontinuities
                if len(crossing) >= 1:
                    continue
                if np.any((Chargemassive[0] < 1) and (np.max(Chargemassive) >= 1)):
                    if np.any(Chargemassive[-1] < 1):
                        nus.append(n)
                        lus.append(l)
                        gus.append(g)
                    else:
                        ns.append(n)
                        ls.append(l)
                        gs.append(g)
                elif np.any((Chargemassive[0] > 1) and (Chargemassive[-1] < 1)):
                    ns.append(n)
                    ls.append(l)
                    gs.append(g)
                
    # Update scatter plot data
    scatter_s._offsets3d = (ns, ls, gs)
    scatter_us._offsets3d = (nus, lus, gus)
    fmeE = f"{eE:.1e}"
    digit, exponent = fmeE.split('e')
    fmrho = f"{rho0:.1e}"
    digit1, exponent1 = fmrho.split('e')
    ax.set_title(rf"$e/\mathcal{{E}} = {float(digit):.2f} \times 10^{{{int(exponent)}}}$", y=1.15)
    plt.suptitle(
        rf"$(\rho_0 = {float(digit1):.2f} \times 10^{{{int(exponent1)}}}, p_0/\rho_0 = {p0 / rho0:.2f}, Q = 10^3, \mathcal{{E}} = 10)$",
        fontsize=12,
        y=0.9,
    )
    return scatter_s, scatter_us

# Create animation
num_frames = len(eE_vals)
ani = FuncAnimation(fig, update, frames=num_frames, interval=1000, blit=False)

# Save the animation as a video
writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
ani.save("eE-ChargedMassive.mp4", writer=writer)

# Display animation
plt.show()
