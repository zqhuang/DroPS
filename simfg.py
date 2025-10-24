import numpy as np
import healpy as hp
import pymaster as nmt
import pysm3
from beforecmb import *
from sys import argv
import pickle

if(len(argv)<4):
    print("python simfg.py mask model frequency_GHz [FWHM_arcmin] [filtering] [FWHM_arcmin_after_filtering]")
    exit()
mask = argv[1]
models = []
for i in range(0, len(argv[2]), 2):
    models.append(argv[2][i:i+2])

freq = float(argv[3])

model_str = r'FGsims/FG_'+argv[2] + r'_'

bpc = band_power_calculator(mask)

sky = pysm3.Sky(nside = bpc.nside, preset_strings = models)

fgmap = sky.get_emission(freq * pysm3.units.GHz).to(pysm3.units.uK_CMB, equivalencies=pysm3.units.cmb_equivalencies(freq * pysm3.units.GHz)).value

if(len(argv) > 4):
    if(len(argv) > 5):
        smoothed_mask  = nmt.mask_apodization(bpc.binary_mask, 2., apotype = "C2")  
        f = open(argv[5], 'rb')
        filtering = pickle.load(f)
        f.close()
        if(len(argv) > 6):
            output = bpc.smooth_map(filtering.project_map(mask = smoothed_mask, maps=hp.smoothing(fgmap, fwhm = float(argv[4])*np.pi/180./60.), want_wof=False), fwhm_in = float(argv[4])*np.pi/180./60., fwhm_out=float(argv[6])*np.pi/180/60.)
        else:
            output = filtering.project_map(mask = smoothed_mask, maps=hp.smoothing(fgmap, fwhm = float(argv[4])*np.pi/180./60.), want_wof=False)      
    else:
        output = hp.smoothing(fgmap, fwhm = float(argv[4])*np.pi/180./60.)      
else:
    output = fgmap

bpc.save_map(model_str + str(int(freq))+ "GHz" + ".npy", output, overwrite=True)
