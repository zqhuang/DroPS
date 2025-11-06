import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.insert(0, os.getcwd())
from mcmc import *
import getdist
from ast import literal_eval
from getdist import plots, MCSamples, loadMCSamples
verbose  = True


mycolors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def load_file_obj(filename):
    f = open(filename, 'r') 
    obj = literal_eval(f.read())
    f.close()        
    return obj


def load_samples(root):
    return MCSamples(samples = np.load(root + r'samples.npy'), names = load_file_obj(root + 'param_keys.txt'), labels = load_file_obj(root + 'param_labels.txt'))



samples0 = load_samples(r'AliCPT/results/AliCPT_r1_d0s0_5_')
samples1 = load_samples(r'AliCPT/results_l1/AliCPT_r1_d0s0_5_')
samples2 = load_samples(r'AliCPT/results_l2/AliCPT_r1_d0s0_5_')


g = plots.get_single_plotter(width_inch = 7.2)

g.plot_2d( [samples0, samples1, samples2] , 'beta_d', 'r' ,  lims = [1.2, 1.8, -0.01, 0.03], filled=[True, False, False], alphas = 0.6, colors=mycolors[0:3], ls=['solid', 'dashed', 'dotted'])

g.add_x_marker(1.54, color="lightgray", ls='dotted', lw=1.)
g.add_y_marker(0.01, color="lightgray", ls='dotted', lw=1.)
g.add_legend([r'ell_cross_range=0', r'ell_cross_range=1', r'ell_cross_range=2'], legend_loc = 'upper left')
plt.savefig('AliCPT_ell_cross_range.png')
plt.show()
