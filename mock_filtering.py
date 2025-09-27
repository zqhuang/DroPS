from beforecmb import TOD_filtering
import pickle
nsidestr = input("enter nside (128): \n")
if(nsidestr == ""):
    nside = 128
else:
    nside = int(nsidestr)
print("nside = " + str(nside))
output = input("enter output filename (filter_128.pickle):\n")
if(output == ""):
    output = r"filter_" + str(nside) + r".pickle"
print("output file: " + output)
lmax = nside*3 - 1
filtering = TOD_filtering(lmax = lmax, overall_factor = 1.,  lowl = 50., lowl_factor = 0.5, lowl_power = 4., l_mix = 0.02, m_mix = 0.02)
with open(output, 'wb') as f:
    pickle.dump(filtering, f)
