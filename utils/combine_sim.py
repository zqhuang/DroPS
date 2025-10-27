import os
import sys
sys.path.insert(0, os.getcwd())
from beforecmb import sky_simulator
from sys import argv

sim = sky_simulator(config_file=argv[1], root_overwrite=argv[2])
ind = int(argv[3])
assert(ind >= 0 and ind < sim.nmaps)
print("extracting simulation #"+argv[3])
print("writing into root = ", sim.root)
for ifreq in range(sim.num_freqs):
    cmb_map = sim.load_IQU_map(sim.cmb1f_root + sim.freqnames[ifreq] + r'_' + argv[3] + r'.npy')
    fg_map = sim.load_IQU_map(sim.fgf_root + sim.freqnames[ifreq] + r'.npy')
    for isea in range(sim.num_seasons):
        season_map = sim.load_IQU_map(sim.noisef_root + sim.freqnames[ifreq] + r'_' + argv[3] + r'_season' + str(isea) + r'.npy')
        sim.save_map(sim.root +  sim.freqnames[ifreq] + r'_season' + str(isea) + r'.npy', cmb_map + fg_map + season_map, False)
    noise_map = sim.load_IQU_map(sim.noisef_root + sim.freqnames[ifreq] + r'_' + argv[3] + r'.npy')
    sim.save_map(sim.root +  sim.freqnames[ifreq] + r'.npy', cmb_map + fg_map + noise_map, False)    

    
