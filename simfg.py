import pysm3
import healpy as hp
import numpy as np
from beforecmb import *
from sys import argv

if(len(argv)<4):
    print("python simfg.py mask model frequency_GHz")
    exit()
mask = argv[1]
models = [argv[2]]
nside = 256
freq = float(argv[3])

model_str = 'FG_'
for model in models:
    model_str +=  model + "_"

bpc = band_power_calculator(mask)

sky = pysm3.Sky(nside = nside, preset_strings = models)

fgmap = sky.get_emission(freq * pysm3.units.GHz).to(pysm3.units.uK_CMB, equivalencies=pysm3.units.cmb_equivalencies(freq * pysm3.units.GHz)).value    

bpc.save_map(model_str + str(int(freq))+ "GHz" + ".npy", fgmap, overwrite=True)
