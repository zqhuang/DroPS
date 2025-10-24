import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from sys import argv

mask = hp.read_map(argv[1])
npix = len(mask)
m = np.zeros((3, npix))
used_ipix = np.where(mask > 0.)[0]
try:
    m[:, used_ipix] = np.load(argv[2])
except:
    m[1:3, used_ipix] = np.load(argv[2])    
if(len(argv)>3):
    ind = int(argv[3])
else:
    ind = 0
if(len(argv)<6):
    sig = np.sqrt(np.sum(m[ind, :]**2)/np.count_nonzero(m[ind, :]))
    hp.mollview(m[ind, :], min = -sig*5., max = sig*5.)
else:
    hp.mollview(m[ind, :], min = float(argv[4]), max = float(argv[5]))    
plt.show()
