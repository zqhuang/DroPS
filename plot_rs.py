import numpy as np
import matplotlib.pyplot as plt
from sys import argv
if(len(argv) < 1):
    print("python plot_rs.py r_LOGFILE_NAME")
    exit()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']    
rs = np.loadtxt(argv[1])
mean_r = np.sum(rs[:, 0])/rs.shape[0]
plt.xlabel(r"#sim")
plt.ylabel(r"$r$")
plt.errorbar(x = range(rs.shape[0]), y = rs[:, 0], yerr = rs[:, 1], fmt= 'o', capsize = 2., color=colors[0])
plt.axhline(y = mean_r, ls = "dotted", linewidth=1.5, color=colors[1], label="mean output $r$")
if(len(argv) > 2):
    r_fid = float(argv[2])
    plt.axhline(y = r_fid, ls = "dashed", linewidth=1.5, color=colors[2], label="input $r$")
plt.legend()
plt.tight_layout()
if(len(argv) > 3):
    plt.savefig(argv[3])    
plt.show()
