# Neutral Massive
import numpy as np
from matplotlib import pyplot as plt
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
    num = (fact1*X*(1/3+fact12*pow(X, l)+fact13*pow(1-pow(X, l),g))
      -fact2*pow(X, 2*n-3)+fact21*X)
    denom = (1-2*fact1*Rb*pow(X,2)*(1/3-pow(X, l)/(l+3))
      -fact2*Rb*pow(X,2*n-2)/(n-1)+fact21*Rb*pow(X,2))
    return num/denom

# Grid of rho and p values
rho_vals = np.array([1.00e-9, 2.50e-9, 5.00e-9])  # 4 values for rho
p_ratio_vals = np.array([0.25])  # 5 values for p/rho

# Define X and ranges for the nested loops
X = np.linspace(Xi, Xf, num=Xspc)
nvals = np.linspace(ni, nf, num=nspc)
lvals = np.linspace(li, lf, num=lspc)
gvals = np.linspace(gi, gf, num=gspc)

# Create 1x3 scatter subplot
SCfig, SCaxes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})
plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.1)
SCaxes = SCaxes.flatten()

# Create 1x3 LHS subplot
LHSfig, LHSaxes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
LHSaxes = LHSaxes.flatten()

# Loop through rho and pres combinations
for idx, rho0 in enumerate(rho_vals):
    SCax = SCaxes[idx]  # Access corresponding subplot
    LHSax = LHSaxes[idx]  # Access corresponding subplot
    p0 = p_ratio_vals[0] * rho0  # Calculate pres

    ns, ls, gs = [], [], []
    nus, lus, gus = [], [], []
    one_over_esq = 1/100

    if idx == 0:  # Only the first column gets the y-axis label
        LHSax.set_ylabel(r'f$(n, l, \Gamma, \chi)$', fontsize=18)
        
# Loop through n, l, g values
    for n in nvals:
        for l in lvals:
            for g in gvals:
                if p0 / rho0 > 1 / g:
                    continue
                y = phiprime(n, l, g, rho0, p0)
                y_integral = cumulative_trapezoid(y, X, initial=0)
                y_combined = np.exp(-2 * y_integral) * (1 - X * Rb * y)
                if np.any((y_combined[0] < one_over_esq) and (np.max(y_combined) >= one_over_esq)):
                    if np.any(y_combined[-1] < one_over_esq):
                        nus.append(n)
                        lus.append(l)
                        gus.append(g)
                    else:
                        ns.append(n)
                        ls.append(l)
                        gs.append(g)
                elif np.any((y_combined[0] > one_over_esq) and (y_combined[-1] < one_over_esq)):
                    ns.append(n)
                    ls.append(l)
                    gs.append(g)
                # Makes a .txt of the scatter plot points
                #with open("NME=10.txt", "a") as f:  # Open in append mode to keep adding results
                    #f.write(f"{n} {l} {g} {rho0}\n")  # Write values separated by spaces
                        
    # Scatter plot
    SCax.scatter(ns, ls, gs, color='lightcoral', alpha = 1, marker="s", rasterized=True)
    SCax.scatter(nus, lus, gus, color='red', alpha = 1, marker="s", rasterized=True)  
    
    # Scatter plot parameters
    SCax.set_xlim(0.51, 5.51)
    SCax.set_ylim(-0.51, 5.5)
    SCax.set_zlim(0, 5.5)
    SCax.set_xlabel("$n$", fontsize=22)
    SCax.set_ylabel("$l$", fontsize=22)
    SCax.tick_params(axis='x', labelsize=15)
    SCax.tick_params(axis='y', labelsize=15)
    SCax.tick_params(axis='z', labelsize=15)
    SCax.zaxis.set_rotate_label(False)
    SCax.set_zlabel(r'$\Gamma$', rotation=0, fontsize=22)

# Determine skip intervals for plotting
    sk1 = int(len(ns) / 6) if len(ns) > 6 else 1
    sk2 = int(len(nus) / 6) if len(nus) > 6 else 1
    
    # Stable LHS
    for nv, lv, gv in zip(ns[::sk1], ls[::sk1], gs[::sk1]):
        y = phiprime(nv, lv, gv, rho0, p0)
        y_integral = cumulative_trapezoid(y, X, initial=0)
        y_combined = np.exp(-2 * y_integral) * (1 - X * Rb * y)
        y_deriv = np.gradient(y_combined, X)
        LHSax.plot(X, y_combined, lw=2, ls='-')
    
    # Unstable LHS
    for nv, lv, gv in zip(nus[::sk2], lus[::sk2], gus[::sk2]):
        y = phiprime(nv, lv, gv, rho0, p0)
        y_integral = cumulative_trapezoid(y, X, initial=0)
        y_combined = np.exp(-2 * y_integral) * (1 - X * Rb * y)
        y_deriv = np.gradient(y_combined, X)
        LHSax.plot(X, y_combined, lw=2, ls=':')

    # LHS plot parameters
    LHSax.axhline(y=one_over_esq, color='black')
    LHSax.set_xlabel(r"$\chi$", fontsize = 18)
    LHSax.set_yscale('symlog')
    LHSax.set_ylim(-1, 1)
    LHSax.tick_params(axis='x', labelsize=15)
    LHSax.tick_params(axis='y', labelsize=15)
    LHSax.grid()
    LHSax.minorticks_on()
    LHSax.set_aspect('auto')
    LHSax.tick_params(which='both', direction='in', top=True, bottom=True, left=True, right=True)

# Scatter and LHS plot: Sets labels for columns
for idx, rho0 in enumerate(rho_vals):
    exponent = int(np.floor(np.log10(rho0)))  # Get the exponent in scientific notation
    coeff = rho0 / (10 ** exponent)     # Get the mantissa
    SCfig.text(0.165 + idx * 0.321, 0.94, rf'$\rho_0 = {coeff:.1f} \times 10^{{{exponent}}}$', fontsize=18, ha='center', va='center', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
    LHSfig.text(0.255 + idx * 0.291, 0.97, rf'$\rho_0 = {coeff:.1f} \times 10^{{{exponent}}}$', fontsize=18, ha='center', va='center', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))

# Saves scatter plot
SCfig.savefig("ScNMVE=10.pdf", bbox_inches=None)
# Saves LHS plot
LHSfig.tight_layout(rect=[0.06, 0.06, 1, 0.96])
LHSfig.savefig("LHSNMVE=10.pdf", bbox_inches='tight')
plt.show()
