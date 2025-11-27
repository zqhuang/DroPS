import os
import sys
sys.path.insert(0, os.getcwd())
from beforecmb import *
from sys import argv

if(len(argv)<3):
    print("python calcpower.py mask map1 [map2]")
    exit()
bpc = band_power_calculator(argv[1], like_fields=['TT','TE',  'TB',  'EE', 'EB', 'BB'])
if(len(argv)>3):
    bp = bpc.band_power(argv[2], argv[3])
else:
    bp = bpc.band_power(argv[2])

#print(bp['BB'][0], end='')
for i in range(bpc.num_ells):
    print('band  #'+str(i), ", power spectra(TT,TE,TB,EE,EB,BB): ", bp['TT'][i],  bp['TE'][i], bp['TB'][i], bp['EE'][i], bp['EB'][i], bp['BB'][i])
#print("\n")
    
