import numpy as np
import pymaster as nmt
import healpy as hp
import matplotlib.pyplot as plt
from sys import argv


if(len(argv)<3):
    print('python viewEB.py mask_file mapfile E/B [lmin]  [title] [map_min] [map_max]')
    exit()
mask = hp.read_map(argv[1])
smoothed_mask = nmt.mask_apodization(mask, 2., apotype = "C2") 
npix = len(mask)
if(len(argv) > 4):
    lmin = int(argv[4])
else:
    lmin = 0
if(len(argv) > 5):
    title = argv[5]
else:
    title = argv[3] + ' map'

        

    
def map2scalar(m, scalar_type = 'B'):  #convert the  input map into a scalar
    if(scalar_type == 'T'):
        return m[0, :]*smoothed_mask
    f2 = nmt.NmtField(smoothed_mask, m[1:3, :], purify_b = True)
    alms = f2.get_alms()
    lmax = hp.Alm.getlmax(alms.shape[1])
    for l in range(min(lmin+1, lmax+1)):
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
if(len(argv)<8):
    sig = np.sqrt(np.sum(ms**2)/np.count_nonzero(ms))
    xmin = -sig*5.
    xmax = sig*5.
else:
    xmin = float(argv[6])
    xmax = float(argv[7])

hp.mollview(ms, rot = (180., 0., 0.), coord=('G', 'C'), min = xmin, max = xmax, title=title)
plt.show()
    
