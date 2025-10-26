import matplotlib.pyplot as plt
import numpy as np
from mcmc import *
import corner
import os
from sys import argv
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



samples0 = loadMCSamples(r'/home/zqhuang/work/CobayaRuns/chains/lcdm_PAS', settings={'ignore_rows':0.2})
samples1 = loadMCSamples(r'/home/zqhuang/work/CobayaRuns/chains/PI_PAS', settings={'ignore_rows':0.05})
samples2 = loadMCSamples(r'/home/zqhuang/work/CobayaRuns/chains/PIt_PAS', settings={'ignore_rows':0.05})

rdlabel=r'hr_d\;[\mathrm{Mpc}]'


samples0.addDerived(paramVec = samples0.getParams().rdrag * samples0.getParams().H0 / 100., name = 'hrd', label=rdlabel, comment='h time r_d', range=[50., 200.])
samples1.addDerived(paramVec = samples1.getParams().rdrag * samples1.getParams().H0 / 100., name = 'hrd', label=rdlabel, comment='h time r_d', range=[50., 200.])
samples2.addDerived(paramVec = samples2.getParams().rdrag * samples2.getParams().H0 / 100., name = 'hrd', label=rdlabel, comment='h time r_d', range=[50., 200.])
#
samples_bao = load_samples(r'lcdm_DESIDR2_')

g = plots.get_single_plotter(width_inch = 7.2)

g.plot_2d( [samples_bao, samples0, samples1, samples2] , 'hrd', 'omegam' ,  lims = [96.7, 103.5, 0.276,  0.334], filled=[True, True, False, False], alphas = 0.6, colors=mycolors[0:4], ls=['solid', 'solid', 'dashed', 'dotted'])


g.add_legend([r'BAO', r'CMB, no-step', r'CMB, $\mathrm{erf}$', r'CMB, $\tanh$'], legend_loc = 'lower left')
plt.savefig('rdomm.png')
plt.show()
