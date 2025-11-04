from beforecmb import TOD_filtering
import pickle
from os import path

def convert_string(value, target_type):
    if target_type == int:
        return int(value)
    elif target_type == float:
        return float(value)
    elif target_type == bool:
        if value.lower() in ('y', 't', 'true', '1', 'yes'):
            return True
        elif value.lower() in ('n', 'f', 'false', '0', 'no'):
            return False
        else:
            return bool(value)  
    elif target_type == str:
        return value 
    else:
        return target_type(value)
    
def read_input(prompt, default):
    inputstr = input(prompt)
    if(inputstr == ""):
        return default
    else:
        return convert_string(inputstr, type(default))

nside = read_input("enter nside (128): \n", 128)
assert(nside == 64 or nside==128 or nside==256 or nside==512 or nside==1024 or nside==2048)
output = read_input("enter output filename (filter_"+str(nside)+".pickle):\n", r"filter_" + str(nside) + r".pickle")
if(path.exists(output)):
    print('Error: file '+output+' already exists')
    exit()
lmax = nside*3 - 1
lcut = read_input("enter l_cut for TOD filtering (50): \n", 50.)
high_fac = read_input("enter suppression factor at high ells (0.999): \n", 0.999)
low_fac = read_input("enter suppression factor at low ells (0.5): \n", 0.5)
slope = read_input("enter slope (4): \n", 4.)
assert(slope > 0.)
l_mix = read_input("enter ell mixing factor (0.02): \n",  0.02)
m_mix = read_input("enter m mixing factor (0.02): \n",  0.02)

filtering = TOD_filtering(lmax = lmax, overall_factor = high_fac,  lowl = lcut, lowl_factor = low_fac, lowl_power = slope, l_mix = l_mix, m_mix = m_mix)
with open(output, 'wb') as f:
    pickle.dump(filtering, f)
