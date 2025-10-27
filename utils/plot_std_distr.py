import numpy as np
import matplotlib.pyplot as plt
from sys import argv


plt.figure(figsize=(8, 5))
plt.xlabel(r'$\sigma_r$')    
plt.ylabel('counts')
plt.xlim([0., 0.008])

mycolors = ['lightgray', '#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_distr(filename, label, color,  cv):
    """
    Simple function to plot smoothed histogram.
    """
    x = np.loadtxt(filename)
    mean = np.sum(x[:, 1])/x.shape[0]
    plt.hist(x[:, 1], bins=7, color=color[0], label=r'$\sigma_r$ histogram, DroPS')
    plt.axvline(mean, color=color[1], ls = "solid", lw = 2.5, label=label+r", DroPS",alpha=0.8)    
    plt.axvline(cv[0], color=color[2], ls = "dotted", lw = 2.5, label=label+r", Wolz et al. Pipeline A",alpha=0.8)
    plt.axvline(cv[1], color=color[3], ls = "dashdot", lw = 2.5, label=label+r", Wolz et al. Pipeline B",alpha=0.8)
    plt.axvline(cv[2], color=color[4], ls = "dashed", lw = 2.5, label=label+r", Wolz et al. Pipeline C",alpha=0.8)    



comp_d0s0 = np.array([2.7e-3, 2.1e-3, 3.5e-3])  #from Table 4 of 2302.04276
comp_d1s1 = np.array([2.8e-3, 2.1e-3, 3.3e-3])  #from Table 4 of 2302.04276
comp_dmsm = np.array([2.7e-3, 2.1e-3, 3.0e-3])  #from Table 4 of 2302.04276    



if(argv[1] == 'd0s0'):
    plot_distr(r'SO/r_logfile_d0s0_r0.txt', r'mean $\sigma_r$', mycolors, comp_d0s0)
elif(argv[1] == 'd1s1'):
    plot_distr(r'SO/r_logfile_d1s1_r0.txt', r'mean $\sigma_r$', mycolors, comp_d1s1)
elif(argv[1] == 'dmsm'):    
    plot_distr(r'SO/r_logfile_dmsm_r0.txt', r'mean $\sigma_r$', mycolors, comp_dmsm)
else:
    print("python plot_std_distr.py [d0s0|d1s1|dmsm]")
    exit()
plt.legend()
plt.savefig(argv[1]+'_std.png')    

plt.show()
