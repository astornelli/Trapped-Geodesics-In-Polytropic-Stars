# Neutral Massless Rho0 Animation
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Define global constants
Q, Rb = 1e3, 1e4
Xi, Xf, Xspc = 0.01, 0.98, 50
ni, nf, nspc = 1.01, 5, 30
li, lf, lspc = 0.01, 5, 30
gi, gf, gspc = 0.001, 5, 30

# Eq. 39 in paper
def func(n, l, g, rho0, p0):
    fact1 = 4 * np.pi * rho0 * Rb**2
    fact12 = (l - 1) / (l + 3)
    fact13 = p0 / rho0
    fact2 = (n - 2) * Q**2 / Rb**2 / (2 * n - 1)
    return (
        fact1 * X**2 * (1 + fact12 * X**l + fact13 * (1 - X**l)**g)
        - fact2 * X**(2 * n - 2)
    )

# Define X and ranges for the nested loops
X = np.linspace(Xi, Xf, num=Xspc)
nvals = np.linspace(ni, nf, num=nspc)
lvals = np.linspace(li, lf, num=lspc)
gvals = np.linspace(gi, gf, num=gspc)

# Rho0 Range
rho0_min, rho0_max = 1e-10, 2e-9
rho0_vals = np.linspace(rho0_min, rho0_max, 100)

# Set up the figure and axis
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
fig.suptitle(r'$n, l, \Gamma$ Parameter Space for Neutral Massive Particles', y=0.95, fontsize=16)

# Initialize scatter plots
scatter_s = ax.scatter([], [], [], marker="o", color='orange', alpha=0.8, rasterized=True)
scatter_us = ax.scatter([], [], [], marker="o", color='lawngreen', alpha=0.8, rasterized=True)

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

    for n in nvals:
        for l in lvals:
            for g in gvals:
                if n == 1 / 2:
                    continue
                if p0 / rho0 > 1 / g:
                    continue
                y = func(n, l, g, rho0, p0)
                if (y[0] < 1) and (np.max(y) >= 1):
                    if y[-1] < 1:
                        nus.append(n)
                        lus.append(l)
                        gus.append(g)
                    else:
                        ns.append(n)
                        ls.append(l)
                        gs.append(g)
                elif (y[0] > 1) and (y[-1] < 1):
                    ns.append(n)
                    ls.append(l)
                    gs.append(g)

    # Update scatter plot data
    scatter_s._offsets3d = (ns, ls, gs)
    scatter_us._offsets3d = (nus, lus, gus)
    fmrho0 = f"{rho0:.1e}"
    digit, exponent = fmrho0.split('e')
    ax.set_title(rf"$\rho_0 = {float(digit):.2f} \times 10^{{{int(exponent)}}}$", y=1.15)
    plt.suptitle(
        rf"$(p_0/\rho_0 = {p0 / rho0:.2f}, Q = 10^3, r_b = 10^4)$",
        fontsize=12,
        y=0.9,
    )
    return scatter_s, scatter_us
    
# Create animation
num_frames = len(rho0_vals)
ani = FuncAnimation(fig, update, frames=num_frames, interval=1000, blit=False)

# Save the animation as a video
writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
ani.save("NeutralMassless1.mp4", writer=writer)

# Display animation
plt.show()