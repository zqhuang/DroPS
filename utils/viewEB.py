import numpy as np
import pymaster as nmt
import healpy as hp
import matplotlib.pyplot as plt
from sys import argv

mask = hp.read_map(argv[1])
smoothed_mask = nmt.mask_apodization(mask, 2., apotype = "C2") 
npix = len(mask)
lmin = 0

def map2scalar(m, scalar_type = 'B', lmin = lmin):  #convert the  input map into a scalar
    if(scalar_type == 'T'):
        return m[0, :]*smoothed_mask
    f2 = nmt.NmtField(smoothed_mask, m[1:3, :], purify_b = True)
    alms = f2.get_alms()
    if(lmin > 0):
        lmax = hp.Alm.getlmax(alms.shape[1])
        for l in range(lmin):
            for m in range(l+1):
                idx = hp.Alm.getidx(lmax, l, m)
                alms[:, idx] = 0. + 0j
    nside = int(np.round(np.sqrt(len(mask)/12.), 0))
    if(scalar_type == 'E'):
        return hp.alm2map(alms[0, :], nside=nside)*smoothed_mask
    elif(scalar_type == 'B'):
        return hp.alm2map(alms[1, :], nside=nside)*smoothed_mask        



m = np.zeros((3, npix))
used_ipix = np.where(mask > 0.)[0]
try:
    m[:, used_ipix] = np.load(argv[2])
except:
    m[1:3, used_ipix] = np.load(argv[2])    

ms = map2scalar(m, scalar_type = argv[3])    
if(len(argv)<6):
    sig = np.sqrt(np.sum(ms**2)/np.count_nonzero(ms))
    hp.mollview(ms, min = -sig*5., max = sig*5.)
else:
    if(len(argv) > 6):
        hp.mollview(ms, min = float(argv[4]), max = float(argv[5]), title=argv[6])
    else:
        hp.mollview(ms, min = float(argv[4]), max = float(argv[5]), title=argv[3]+' map')        
plt.show()
    
