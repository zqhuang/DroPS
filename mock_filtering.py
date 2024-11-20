from beforecmb import TOD_filtering
import pickle

lmax = 256*3
filtering = TOD_filtering(lmax = lmax, overall_factor = 1.,  lowl = 50., lowl_factor = 0.5, lowl_power = 4., l_mix = 0.02, m_mix = 0.02)
with open(r'filter_opt.pickle', 'wb') as f:
    pickle.dump(filtering, f)
