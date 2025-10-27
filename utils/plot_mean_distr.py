import numpy as np
import matplotlib.pyplot as plt
from sys import argv


plt.figure(figsize=(8, 5))
plt.xlabel(r'mean $r$')    
plt.ylabel('counts')
plt.xlim([-0.01, 0.01])

mycolors = ['lightgray', '#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']



def plot_distr(filename, label, color,  cv):
    """
    Simple function to plot smoothed histogram.
    """
    x = np.loadtxt(filename)
    mean = np.sum(x[:, 0])/x.shape[0]
    print("loaded ", x.shape[0], " runs")
    plt.hist(x[:, 0], bins=11, color=color[0], label="mean $r$ histogram, DroPS")
    plt.axvline(mean, color=color[1], ls = "solid", lw = 2.5, label=label+", DroPS",alpha=0.8)    
    plt.axvline(cv[0], color=color[2], ls = "dotted", lw = 2., label=label+", Wolz et al. Pipeline A",alpha=0.6)
    plt.axvline(cv[1], color=color[3], ls = (0, (6, 2, 2, 2)), lw = 2., label=label+", Wolz et al. Pipeline B",alpha=0.6)
    plt.axvline(cv[2], color=color[4], ls = "dashed", lw = 2., label=label+", Wolz et al. Pipeline C",alpha=0.6)    



comp_d0s0 = np.array([-0.5e-3, -0.1e-3, -1.7e-3])  #from Table 4 of 2302.04276
comp_d1s1 = np.array([-0.2e-3, 2.1e-3, 0.0e-3])  #from Table 4 of 2302.04276
comp_dmsm = np.array([0.3e-3, 3.8e-3, 0.2e-3])  #from Table 4 of 2302.04276    


if(argv[1] == 'd0s0'):
    plot_distr(r'SO/r_logfile_d0s0_r0.txt', r'$r$ bias', mycolors, comp_d0s0)
elif(argv[1] == 'd1s1'):
    plot_distr(r'SO/r_logfile_d1s1_r0.txt', r'$r$ bias', mycolors, comp_d1s1)
elif(argv[1] == 'dmsm'):    
    plot_distr(r'SO/r_logfile_dmsm_r0.txt', r'$r$ bias', mycolors, comp_dmsm)
else:
    print("python plot_mean_distr.py [d0s0|d1s1|dmsm]")
    exit()
plt.legend()
plt.savefig(argv[1]+'_mean.png')    

plt.show()
