import numpy as np
import matplotlib.pyplot as plt
from sys import argv
if(len(argv) < 1):
    print("python plot_rs.py r_LOGFILE_NAME")
    exit()

plt.xlabel(r"#sim")
plt.ylabel(r"$r$")

def plot_data(filename, fmt, color, mean_ls, mean_label, xshift = 0., max_num = None, alpha = 1.):
    xdata = np.loadtxt(filename)
    if(max_num is not None):
        if(xdata.shape[0] > max_num):
            rs = xdata[0:max_num, :]
        else:
            rs = xdata
    else:
        rs = xdata
    mean_r = np.sum(rs[:, 0])/rs.shape[0]
    plt.errorbar(x = np.array(range(1, rs.shape[0]+1),dtype=np.float64) +xshift, y = rs[:, 0], yerr = rs[:, 1], fmt= fmt, capsize = 2., color=color, alpha=alpha)
    plt.axhline(y = mean_r, ls = mean_ls, linewidth=2., color=color, label=mean_label, alpha =alpha)
    return xdata.shape[0]
    
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

if(len(argv) > 2):
    r_fid = float(argv[2])
    plt.axhline(y = r_fid, ls = "solid", linewidth=1.5, color=colors[1], label="input")

if(len(argv)>5):
    num = int(argv[5])
else:
    num = None

if(len(argv)>3):
    if(argv[3] != "NULL"):
        plot_data(filename=argv[3], fmt='s', color=colors[1], mean_ls = 'dashed', mean_label=r'Taylor', xshift = -0.15, max_num=num, alpha = 0.3)
if(len(argv)>4):
    if(argv[4] != "NULL"):
        plot_data(filename=argv[4], fmt='s', color=colors[2], mean_ls = 'dashdot', mean_label=r'None', xshift = 0.15, max_num=num, alpha = 0.3)

plot_data(filename=argv[1], fmt='o', color=colors[0], mean_ls = 'dotted', mean_label=r'mean of means', max_num = num)

plt.legend()
plt.tight_layout()
#if(len(argv) > 6):
#    plt.savefig(argv[6])    
plt.show()
