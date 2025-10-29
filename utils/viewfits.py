import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from sys import argv

try:
    m = hp.read_map(argv[1], field=[0, 1, 2])
except:
    m = hp.read_map(argv[1])

    
if(len(m.shape)>1):
    if(len(argv)>2):
        ind = int(argv[2])
    else:
        ind = 0
    sig = np.sqrt(np.sum(m[ind, :]**2)/np.count_nonzero(m[ind, :]))
    hp.mollview(m[ind, :], min = -sig*5., max = sig*5.)
else:
    sig = np.sqrt(np.sum(m**2)/np.count_nonzero(m))
    hp.mollview(m, min = -5.*sig, max = 5.*sig)
plt.show()
