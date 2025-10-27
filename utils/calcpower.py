import os
import sys
sys.path.insert(0, os.getcwd())
from beforecmb import *
from sys import argv

if(len(argv)<3):
    print("python calcpower.py mask map1 [map2]")
    exit()
bpc = band_power_calculator(argv[1], like_fields=['BB'])
if(len(argv)>3):
    bp = bpc.band_power(argv[2], argv[3])
else:
    bp = bpc.band_power(argv[2])

print(bp['BB'][0], end='')
for i in range(1, bpc.num_ells):
    print(' ', bp['BB'][i], end='')
print("\n")
    
