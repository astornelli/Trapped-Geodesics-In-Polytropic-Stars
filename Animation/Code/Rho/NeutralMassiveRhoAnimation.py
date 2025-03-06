# Neutral Massive Animation Code for Rho
import numpy as np
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

def phiprime(n, l, g, rho0, p0):
    fact1 = 4*np.pi*rho0*Rb
    fact12 = (l+1)/(l+3)
    fact13 = p0/rho0
    fact2 = (n-1)*Q*Q/Rb/Rb/Rb/(2*n-1)
    fact21 = Lam*Rb/3
    num = (fact1*X*(1/3+fact12*pow(X, l)+fact13*pow(1-pow(X, l),g))
      -fact2*pow(X, 2*n-3)+fact21*X)
    denom = (1-2*fact1*Rb*pow(X,2)*(1/3-pow(X, l)/(l+3))
      -fact2*Rb*pow(X,2*n-2)/(n-1)+fact21*Rb*pow(X,2))
    return num/denom

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
scatter_s = ax.scatter([], [], [], marker="o", color='lightcoral', alpha=0.8, rasterized=True)
scatter_us = ax.scatter([], [], [], marker="o", color='red', alpha=0.8, rasterized=True)

# Axes labels and limits
ax.set_xlim(0.51, 5.51)
ax.set_ylim(-0.51, 5.5)
ax.set_zlim(0, 5.5)
ax.set_xlabel("$n$")
ax.set_ylabel("$l$")
ax.zaxis.set_rotate_label(False)
ax.set_zlabel(r'$\Gamma$', rotation=0)

def update(frame):
    # Current rho value
    rho0 = rho0_vals[frame]
    p0 = 1 * rho0

    ns, ls, gs = [], [], []
    nus, lus, gus = [], [], []
    one_over_esq = 1/100

    for n in nvals:
        for l in lvals:
            for g in gvals:
                if p0 / rho0 > 1 / g:
                    continue
                y = phiprime(n, l, g, rho0, p0)
                y_integral = cumulative_trapezoid(y, X, initial=0)
                y_combined = np.exp(-2 * y_integral) * (1 - X * Rb * y)
                if (y_combined[0] < one_over_esq) and (np.max(y_combined) >= one_over_esq):
                    if (y_combined[-1] < one_over_esq):
                        nus.append(n)
                        lus.append(l)
                        gus.append(g)
                    else:
                        ns.append(n)
                        ls.append(l)
                        gs.append(g)
                elif ((y_combined[0] > one_over_esq) and (y_combined[-1] < one_over_esq)):
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
ani.save("1NeutralMassive_E=10.mp4", writer=writer)

# Display animation
plt.show()