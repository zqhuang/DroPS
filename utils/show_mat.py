from beforecmb import *
import numpy as np
from sys import argv
from os import path
import matplotlib.pyplot as plt
import healpy as hp

sim = sky_simulator(config_file=argv[1])
imap = 1
used_ipix = np.where(sim.mask_ones)[0]  
ipix = np.random.choice(used_ipix)
ifreq = 3

input_map = np.zeros((3, sim.npix))
input_map[imap, ipix] = 1.
output_map = sim.filtering.project_map(mask = sim.smoothed_mask, maps = smooth_rotate(maps = input_map, fwhm_rad=sim.fwhms_rad[ifreq]), want_wof = False)
hp.mollview(output_map[1, :])
plt.show()
