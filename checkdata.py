from beforecmb import *
import numpy as np
from sys import argv
from os import path
from mcmc import *

ana = sky_analyser(argv[1], argv[2])
ana.get_data_vector()
ana.get_covmat()

data_chisq = np.dot(ana.data_vec - ana.mean, np.dot(ana.invcov, ana.data_vec - ana.mean))/ana.fullsize
print("chi^2 = ", data_chisq)
for ifield in range(ana.num_fields):
    print('-------------' + ana.fields[ifield]+'---------------')
    for ipower in range(ana.num_powers):
        ifreq1, ifreq2 = ana.freq_indices(ipower)
        print('----'+ana.freqnames[ifreq1]+' x '+ana.freqnames[ifreq2]+'----')
        for il in range(ana.num_ells):
            if(ana.ells[il] > 60 and ana.ells[il] < 140):
                k = il*ana.blocksize + ipower*ana.num_fields + ifield
                print(ana.ells[il], ana.data_vec[k], ana.mean[k],  (ana.data_vec[k]- ana.mean[k])/np.sqrt(ana.covmat[k, k]))
