#Charged Massless 
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.integrate import cumulative_trapezoid

# Define global constants
Q, Rb, Lam = 1e3, 1e4, 1e-52

# Loop Constants
Xi, Xf, Xspc = 0.01, 0.98, 50
ni, nf, nspc = 1.01, 5, 30
li, lf, lspc = 0.01, 5, 30
gi, gf, gspc = 0.001, 5, 30

# Eq. 33 in paper
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

# Eq. 34 in paper
def Fprime(n, l, g, rho0, p0):
    phip = phiprime(n, l, g, rho0, p0)
    num = np.exp(cumulative_trapezoid(phip[0]*Rb, X, initial=0))
    return num*Q*pow(X,n-2)/Rb/Rb/pow(phip[1],1/2)

# Grid of rho and p values
rho_vals = np.array([8.1e-10, 1.1e-9, 2.1e-9])  # 4 values for rho
E_vals = np.array([10])    # ratio of charge to energy
p_ratio = 0.25

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

# Loop through rho and pres combinations
for jdx, eE in enumerate(E_vals):
    for idx, rho0 in enumerate(rho_vals):
        SCax = SCaxes[idx]    # Access corresponding subplot
        LHSax = LHSaxes[idx]  # Access corresponding subplot
        p0 = p_ratio * rho0   # Calculate pres    

        if idx == 0:  # Only the first column gets the y-axis label
            LHSax.set_ylabel(r'f$(n, l, \Gamma, \chi)$', fontsize=18)
        
        ns, ls, gs = [], [], []
        nus, lus, gus = [], [], []
        
        for n in nvals:
            for l in lvals:
                for g in gvals:
                    if p0 / rho0 > 1 / g:
                        continue
                    phip = phiprime(n, l, g, rho0, p0)
                    # avoiding unphysical values in the square root (in the denominator of Fprime)
                    if len(np.where(phip[1]<=0)[0])!=0:
                        continue
                    F = cumulative_trapezoid(Fprime(n, l, g, rho0, p0)*Rb, X, initial=0)
                    # lhs of eqn. 43
                    Chargemassless = (1 - X*Rb*phip[0])/(X*Rb*(phip[0]*F-Fprime(n, l, g, rho0, p0))-F)
                    # Cutoff for discontinuities, the function phiprime should not have discontinuities, sharp jumps, etc.
                    # Jump discontinuities -- checking if the left and right derivatives of the signs of the lhs match
                    # Chargemassless -1 ensures 0 values are treated as negative numbers
                    # if left and right derivatives do not match, there is jump discontinuity, and crossing is > 1, and gets discarded!
                    crossing = np.where(np.diff(np.sign(Chargemassless - 1)))[0]
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
                    # Makes a .txt of the scatter plot points
                    #with open("CMSSE=10.txt", "a") as f:  # Open in append mode to keep adding results
                        #if len(crossing) < 1:
                            #f.write(f"{n} {l} {g} {rho0}\n")  # Write values separated by spaces
        
        # Scatter plot
        SCax.scatter(ns, ls, gs, color='lightseagreen', alpha = 1, marker="s", rasterized=True)
        SCax.scatter(nus, lus, gus, color='blue', alpha = 1, marker="s", rasterized=True)  
        
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
            phip = phiprime(nv, lv, gv, rho0, p0)
            F = cumulative_trapezoid(Fprime(nv, lv, gv, rho0, p0)*Rb, X, initial=0)
            Chargemassless = (1 - X*Rb*phip[0])/(X*Rb*(phip[0]*F-Fprime(nv, lv, gv, rho0, p0))-F)
            LHSax.plot(X, Chargemassless, lw=2, ls='-')
        
        # Unstable LHS
        for nv, lv, gv in zip(nus[::sk2], lus[::sk2], gus[::sk2]):
            phip = phiprime(nv, lv, gv, rho0, p0)
            F = cumulative_trapezoid(Fprime(nv, lv, gv, rho0, p0)*Rb, X, initial=0)
            Chargemassless = (1 - X*Rb*phip[0])/(X*Rb*(phip[0]*F-Fprime(nv, lv, gv, rho0, p0))-F)
            LHSax.plot(X, Chargemassless, lw=2, ls=':')

        # LHS plot parameters
        LHSax.axhline(y=E_vals, color='black')
        LHSax.set_xlabel(r"$\chi$", fontsize = 18)
        LHSax.set_yscale('symlog')
        LHSax.set_ylim(-1e3, 1e2)
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
SCfig.savefig("ScCHMSE=10.pdf", bbox_inches=None)
# Saves LHS plot
LHSfig.tight_layout(rect=[0.06, 0.06, 1, 0.96])
LHSfig.savefig("LHSCHMSE=10.pdf", bbox_inches='tight')
plt.show()
