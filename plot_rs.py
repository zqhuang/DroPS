import numpy as np
import matplotlib.pyplot as plt
from sys import argv
if(len(argv) < 1):
    print("python plot_rs.py r_LOGFILE_NAME")
    exit()

plt.xlabel(r"#sim")
plt.ylabel(r"$r$")

def plot_data(filename, fmt, color, mean_ls, mean_label, xshift = 0.):
    rs = np.loadtxt(filename)
    mean_r = np.sum(rs[:, 0])/rs.shape[0]
    plt.errorbar(x = np.array(range(rs.shape[0]),dtype=np.float64) +xshift, y = rs[:, 0], yerr = rs[:, 1], fmt= fmt, capsize = 2., color=color)
    plt.axhline(y = mean_r, ls = mean_ls, linewidth=2., color=color, label=mean_label)

    
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

if(len(argv) > 2):
    r_fid = float(argv[2])
    plt.axhline(y = r_fid, ls = "solid", linewidth=1.5, color=colors[1], label="input")

if(len(argv)>3):
    if(argv[3] != "NULL"):
        plot_data(filename=argv[3], fmt='s', color="lightgray", mean_ls = 'dashed', mean_label=r'mean of mean', xshift = 0.15)

plot_data(filename=argv[1], fmt='o', color=colors[0], mean_ls = 'dotted', mean_label=r'mean of mean')

plt.legend()
plt.tight_layout()
if(len(argv) > 4):
    plt.savefig(argv[4])    
plt.show()
