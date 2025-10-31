#Map configurations
import numpy as np
import pysm3
import pymaster as nmt
import healpy as hp
from camb import CAMBparams, get_results
from os import path, mkdir
from math import ceil
from time import monotonic_ns, time
from astropy.utils.data import import_file_to_cache
import pickle
from ast import literal_eval
import matplotlib.pyplot as plt
##%%%%%%%%%%%%%%%%%%%%%%%utilities%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#\Omega_bh^22, \Omega_ch^2, 100cosmomc_theta, \tau, log(10^{10}A_s), n_s
#this model is used as the base model
cosmology_base_name = [ 'ombh2', 'omch2', 'theta', 'tau', 'logA', 'ns', 'r' ]
cosmology_base_tex = [ r'$\Omega_bh^2$', r'$\Omega_ch^2$', r'$100\theta_{\rm MC}$', r'$\tau$', r'$\ln(10^{10}A_s)$', r'$n_s$', r'$r$' ]
cosmology_base_mean = np.array([ 0.02242, 0.11933, 1.04101, 0.0561, 3.047, 0.9665, 0.])
cosmology_base_std = np.array([ 0.00014, 0.00091, 0.00029, 0.0071, 0.014, 0.0038, 0.01])
cosmology_num_params = len(cosmology_base_mean)

#computes power spectra given a pair of fields and a workspace.
def compute_master_with_workspace(f_a, f_b, wsp):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled

def time_seed():  #return an 8-digit seed
    nano_seconds = monotonic_ns() 
    if(nano_seconds % 1000000 == 0): #for some system only accurate to ms
        seed = (nano_seconds // 1000000 + int(time())) % 90000000 + 10000000
    else:
        seed = (nano_seconds + int(time())) % 90000000 + 10000000
    return seed

def set_random_seed(seed = None):
    if(seed is None):
        np.random.seed(time_seed())
    else:
        np.random.seed(seed)


def float2intstr(x,  shift = 0):
    if(shift == 0):
        return str(int(np.round(x)))
    else:
        return str(int(np.round(x*10.**shift))) + r'E-' + str(shift)

def compress_str(s, length = 8, clib="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """compress a string to a string with length up to 16; this is not one-to-one mapping of course, but practically you may ignore the probability of coincidence"""
    primes = [ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307]
    assert(length <= len(primes))
    lenlib = len(clib)
    ils = np.zeros(length, dtype=np.int32)
    k = 999
    lastic = 0
    for c in s:
        ic = ord(c)
        for i in range(length):
            ils[i] += ic * (k  % primes[i] )  + ( ic + 255) % (lastic + 1)
        k += 1
        lastic = ic
    compressed_str = ""
    for i in range(length):
        compressed_str += clib[ils[i] % lenlib]
    return compressed_str


def path23ints(mypath):
    primes = [ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307]
    lenp = len(primes)
    rpath = path.realpath(mypath)
    headtail = path.split(rpath)
    lenhead = len(headtail[0])
    lentail = len(headtail[1])
    ints = np.zeros(3, dtype=np.int64)
    k = 0
    ints[0] += (lenhead+lentail)
    ints[1] += lenhead
    ints[2] += lentail
    last_ordc = 1
    for c in rpath:
        k += 1
        ordc = ord(c)
        seed1 = primes[ordc % lenp ]
        seed2 = (ordc * k + 255 ) % (last_ordc + 1)
        seed3 = (k**2) % (ordc + 1)        
        seed4 = (last_ordc * k + 255 ) % (ordc  + 1)
        seed5 = (ordc * (ordc + 1)) % 255
        if(ordc > 47 and ordc < 58):
            r = ordc - 47
        elif(ordc > 64 and ordc < 91):
            r = ordc - 54
        elif(ordc > 96 and ordc < 123):
            r = ordc - 60
        else:
            r = 0
        seed6 = (primes[r] * k + ordc) % 397
        seed7 = (primes[62-r] * k + last_ordc) % 299
        if(k >= lenhead):
            ints[0] += (seed1 * seed2 * seed3 + seed4*seed5*seed6+ordc*last_ordc*seed7+ordc*k)
            ints[1] += (seed1+seed2+seed3+seed5+seed6+ordc)
            ints[2] += (seed4+seed5+seed6+seed7+last_ordc)
        else:
            ints[0] += (seed1 * seed2  +  seed3 * seed4 + seed5 * seed6 + seed7 * ordc + last_ordc )
            ints[1] += (seed1 + seed2 + seed3)
            ints[2] += (seed4 + seed5 + seed6)
        last_ordc = ordc
    return ints



def postfix_of(filename):
    l = len(filename)
    if(l == 0):
        return ''
    pt = l - 1
    while(filename[pt] != '.'):
        pt -= 1
        if(pt < 0):
            return ''
    return filename[pt+1:l]


def prefix_of(filename):
    l = len(filename)
    if(l == 0):
        return filename
    pt = l - 1
    while(filename[pt] != '.'):
        pt -= 1
        if(pt < 0):
            return filename
    return filename[0:pt]


def fields_str(fields):
    s = ''
    for f in fields:
        s += f
    return s

def str_fields(s):
    l = len(s)
    fields = []
    for i in range(0, l, 2):
        fields.append( s[i:i+2] )
    return fields
    
def cmb_postfix_for_r(r):
    return r'r' + float2intstr(r, 4) + r'_'


def smooth_rotate(maps, fwhm_rad, rot = None):
    if(fwhm_rad == 0. and rot is None):
        return maps
    else:
        return pysm3.apply_smoothing_and_coord_transform(maps, fwhm = (fwhm_rad*180./np.pi)*pysm3.units.deg, rot = rot)


def mkdir_if_not_exists(mydir):
    if(not (path.exists(mydir))):
       mkdir(mydir)

def mkdir_for_file(filename):
       mydir = path.dirname(filename)
       mkdir_if_not_exists(mydir)

def get_camb_cls(ells, lcdm_params = cosmology_base_mean, mnu=0.06, omk=0., fields=['BB']):  #default Planck18 best fit cosmology
    pars = CAMBparams()
    if(lcdm_params[6] > 0.):
        pars.WantTensors = True
    pars.set_cosmology(ombh2 = lcdm_params[0], omch2 = lcdm_params[1], cosmomc_theta = lcdm_params[2]/100., tau = lcdm_params[3], mnu = mnu, omk=omk)
    pars.InitPower.set_params(As = 1.e-10 * np.exp(lcdm_params[4]), ns = lcdm_params[5], r = lcdm_params[6])
    pars.set_for_lmax(ells[-1]+500, lens_potential_accuracy = 2)
    results = get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
    bp_unlensed = {}
    bp_lensed = {}
    for field in fields:
        if(field == 'TT'):
            bp_lensed[field] = powers['total'][ells, 0]
            bp_unlensed[field] = powers['unlensed_total'][ells, 0]
        elif(field == 'EE'):
            bp_lensed[field] = powers['total'][ells, 1]
            bp_unlensed[field] = powers['unlensed_total'][ells, 1]            
        elif(field == 'BB'):
            bp_lensed[field] = powers['total'][ells, 2]
            bp_unlensed[field] = powers['unlensed_total'][ells, 2]                        
        elif(field == 'TE' or field == 'ET'):
            bp_lensed[field] = powers['total'][ells, 3]
            bp_unlensed[field] = powers['unlensed_total'][ells, 3]                        
        else:
            bp_unlensed[field] = 0.
            bp_lensed[field] = 0.
    del pars
    del results
    del powers
    return bp_unlensed, bp_lensed

def corr_nb(n):
    assert(n > 1)
    if(n == 2):
        return 1.
    elif(n == 3):
        return np.sqrt(0.5)
    elif(n == 4):
        return np.sqrt(1.25)-0.5
    elif(n == 5):
        return 1./np.sqrt(3.)
    elif(n == 6):
        return 2./3.-np.sqrt(28.)/3*np.cos(np.arccos(1/np.sqrt(28.))/3+np.pi/3)
    elif(n == 7):
        return np.sqrt( (2.-np.sqrt(2.))/2.)
    else:
        return 0.5 + 2.467/n**(2+1.0439/n**1.188)

def corr_nnb(n):
    assert(n > 2)
    if(n == 3):
        return (1., 1.)
    elif(n == 4):
        return  (np.sqrt(0.75), 0.5)
    elif(n == 5):
        a = 2+np.sqrt(28./3)*np.cos((2*np.pi-np.arccos(np.sqrt(27./28)))/3.) 
        return  (np.sqrt(a), 1.-a)
    elif(n==6):
        p = np.cos(-np.arccos(17/7./np.sqrt(7.))/3. + np.pi*4/3.)*np.sqrt(7.)/3.+5./6.
        return ( np.sqrt(1.-2.*p*(1.-p)) , p )
    elif(n ==  7 ):
        return ( 0.7422271989685592 ,  0.25777280103144085 )
    elif(n ==  8 ):
        return ( 0.7453559924999305 ,  1./3.)
    elif(n ==  9 ):
        return ( 0.7396785710152316 ,  0.29430137862044004 )
    elif(n ==  10 ):
        return ( 0.7320508075688774 ,  0.26794919243112325 )
    elif(n ==  11 ):
        return ( 0.724589261019298 ,  0.24925314958886935 )
    elif(n ==  12 ):
        return ( 0.7243091650316421 ,  0.2978942568831898 )
    elif(n ==  13 ):
        return ( 0.723606797749979 ,  0.276393202250021 )
    elif(n ==  14 ):
        return ( 0.720959822006948 ,  0.25989153247414465 )
    elif(n ==  15 ):
        return ( 0.7175495863997976 ,  0.24694521498165148 )
    else:
        return (np.sqrt(0.5), 0.25 )

def total_index(num_freqs, ifreq1, ifreq2):
    imin = min(ifreq1, ifreq2)
    idiff = abs(ifreq1 -  ifreq2)
    return (2*num_freqs - idiff + 1)*idiff // 2 + imin

def sub_indices(num_freqs, ipower):
    idiff = ceil(num_freqs + 0.5 - np.sqrt((num_freqs + 0.5)**2 - 2.*ipower))
    imin = ipower - ((2*num_freqs - idiff + 1)*idiff // 2)
    if(imin < 0):
        idiff -= 1
        imin = ipower - ((2*num_freqs - idiff + 1)*idiff // 2)
    elif(imin + idiff >= num_freqs):
        idiff += 1
        imin = ipower - ((2*num_freqs - idiff + 1)*idiff // 2)
    return imin, imin + idiff

def get_weights(power_array):
    """for a fixed ell, power_array = [[D(nu1, lenstemp), D(nu2, lenstemp), ...]_sim1, [D(nu1, lenstemp), D(nu2, lenstemp), ...]_sim2, ...]"""
    num_sims = power_array.shape[0]
    num_freqs = power_array.shape[1]
    cov  = np.zeros((num_freqs, num_freqs))
    mean = np.zeros(num_freqs)
    weights = [1./num_freqs] * num_freqs
    for i in range(num_sims):
        vec = power_array[i, :] 
        mean +=  vec
        cov += vec[None, :] * vec[:, None]
    mean /= num_sims
    cov = cov/num_sims - mean[None, :] * mean[:, None]
    invcov = np.linalg.inv(cov)
    for i in range(num_freqs):
        weights[i] = np.sum(invcov[i, :])
    weights /= np.sum(weights)
    return weights


def map_to_alm(maps, lmax):
    if(len(maps.shape) == 1):
        return np.array(hp.map2alm(maps = maps, lmax = lmax))
    if(len(maps.shape) == 2):
        if(maps.shape[0] == 1 or maps.shape[0] == 3):
            return np.array(hp.map2alm(maps = maps, lmax = lmax))
        if(maps.shape[0] == 2):
            return np.array(hp.map2alm_spin(maps = maps, spin= 2, lmax = lmax))
    print('Error in map_to_alm: shape of input map must be I, IQU or QU')
    exit()

def alm_to_map(alms, nside, lmax):
    if(len(alms.shape) == 1):
        return np.array(hp.alm2map(alms = alms, nside=nside, lmax = lmax))
    if(len(alms.shape) == 2):
        if(alms.shape[0] == 1 or alms.shape[0] == 3):
            return np.array(hp.alm2map(alms=alms, nside=nside, lmax = lmax))
        if(alms.shape[0] == 2):
            return np.array(hp.alm2map_spin(alms = alms, nside=nside, spin= 2, lmax = lmax))
    print('Error in alm_to_map: shape of input alms must be I, IQU or QU')
    exit()
    
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#tod_filtering class: a toy model for filtering in harmonic space
class TOD_filtering:

    def __init__(self, lmax, overall_factor = 0.9,  lowl = 70., lowl_factor = 0.6, lowl_power = 4., l_mix = 0.05, m_mix = 0.05):
        assert(lmax > 0 and l_mix + m_mix < 0.5 and overall_factor <= 1. and overall_factor > 0. and lowl_factor <= 1. and lowl_factor >= 0. )
        self.lmax = lmax
        self.overall_factor = overall_factor
        self.lowl = lowl
        self.lowl_factor = lowl_factor
        self.lowl_power = lowl_power
        self.l_mix = l_mix
        self.m_mix = m_mix
        self.num_lms = (lmax+1)*(lmax+2) // 2
        self.transfer = np.empty( (self.num_lms, 5) )  #0: l, m,  1, 2 l+/-1, m;  3, 4 l, m+/-1;
        idx = 0
        main_mix = 1. - 2.*(l_mix + m_mix)
        self.fac = np.empty(lmax+1)
        amp = np.empty( (lmax+1, 3) )
        for l in range(lmax+1):
            self.fac[l] = overall_factor * (1. - lowl_factor * np.exp(-(l/lowl)**lowl_power))
            amp[l, 0] = np.sqrt(self.fac[l]*main_mix)
            amp[l, 1] = np.sqrt(self.fac[l]*l_mix)
            amp[l, 2] = np.sqrt(self.fac[l]*m_mix)
        sqrttwo = np.sqrt(2.)
        for m in range(lmax+1):
            for l in range(m, lmax+1):
                if( l-1 >= m and l+1 <= lmax and m-1 >= 0 and m+1 <= l):  #typical case
                    self.transfer[idx, 0] = amp[l, 0]                    
                    self.transfer[idx, 1:3] = amp[l, 1] * np.random.normal(size = 2)  
                    self.transfer[idx, 3:5] = amp[l, 2] * np.random.normal(size = 2)
                else:
                    remain = 1.
                    if(l > m):
                        self.transfer[idx, 1] = amp[l, 1] * np.random.normal()
                        self.transfer[idx, 4] = amp[l, 2] * np.random.normal()
                        remain -= (l_mix + m_mix)
                    else:
                        self.transfer[idx, 1] = 0.
                        self.transfer[idx, 4] = 0.                                   
                    if(l < lmax):
                        self.transfer[idx, 2] = amp[l, 1] * np.random.normal()
                        remain -= l_mix
                    else:
                        self.transfer[idx, 2] = 0.
                    if(m > 0):
                        self.transfer[idx, 3] = amp[l, 2] * np.random.normal()
                        remain -= m_mix
                    else:
                        self.transfer[idx, 3] = 0.
                    self.transfer[idx, 0] = np.sqrt(remain*self.fac[l])
                idx += 1
                
    def project_alms(self, alms):
        assert(self.num_lms == alms.shape[1])
        alms[:, 0] = 0.  #l = 0, m = 0, assume no monopole
        alms_copy = alms.copy()
        idx = 1
        for l in range(1, self.lmax + 1):          #m = 0, l > 0
            alms[:, idx] = alms_copy[:, idx] * self.transfer[idx, 0] + alms_copy[:, idx-1]*self.transfer[idx, 1] + alms_copy[:, idx+1] * self.transfer[idx, 2] + abs(alms_copy[:, idx + self.lmax])*self.transfer[idx, 4] 
            idx += 1
        #m > 0
        for m in range(1, self.lmax):
            for l in range(m, self.lmax+1):
                alms[:, idx] = alms_copy[:, idx] * self.transfer[idx, 0] + alms_copy[:, idx-1]*self.transfer[idx, 1] + alms_copy[:, idx+1] * self.transfer[idx, 2] + alms_copy[:, idx - self.lmax + m - 1] * self.transfer[idx, 3] + alms_copy[:, idx + self.lmax - m] * self.transfer[idx, 4]
                idx += 1
        alms[:, idx] = alms_copy[:, idx] * self.transfer[idx, 0] + alms_copy[:, idx-1]*self.transfer[idx, 1]  #l=m=lmax


    def project_map(self, mask, maps, want_wof = True):
        nside = int(np.round(np.sqrt(maps.shape[1]/12), 0))
        assert(nside==64 or nside==128 or nside == 256 or nside == 512 or nside == 1024 or nside == 2048)  #default settings for BeForeCMB, other resolutions are practically useless
        alms = map_to_alm(maps*np.tile(mask, (maps.shape[0], 1)), lmax = self.lmax)
        if(want_wof):
            maps_wof = alm_to_map(alms, nside=nside, lmax = self.lmax)
            self.project_alms(alms)
            return maps_wof, alm_to_map(alms, nside=nside, lmax = self.lmax)
        else:
            self.project_alms(alms)
            return alm_to_map(alms, nside=nside, lmax = self.lmax)
        


##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#leakage model class: summary statistics of filtering and E-B leakage 
class leakage_model:

    def __init__(self, source = None, target = None, diag = None, low2high=None, high2low = None):
        if(source is None and target is None):
            self.size = len(diag)
            assert(self.size == len(low2high)+1 and self.size == len(high2low)+1)
            self.diag = diag
            self.low2high = low2high
            self.high2low = high2low
            self.error = np.zeros(self.size)
            return
        n = source.shape[0]
        m = source.shape[1]
        assert(m > 1 and n >= m and n == target.shape[0] and m == target.shape[1])
        self.size = m
        self.diag = np.zeros(m)
        self.low2high = np.zeros(m-1)
        self.high2low = np.zeros(m-1)
        mat = np.empty( (3, 3) )
        b = np.empty(3)
        for i in range(1,m-1):
            for j in range(3):
                b[j] = sum(source[:, i+j-1]*target[:, i])
                for k in range(3):
                    mat[j, k] = sum(source[:, i+j-1]*source[:, i+k-1])
            invmat = np.linalg.inv(mat)
            x = np.matmul(invmat, b)
            self.low2high[i-1] = x[0]
            self.diag[i] = x[1]
            self.high2low[i] = x[2]
        mat = np.empty( (2, 2) )
        b = np.empty(2)
        for j in range(2):
            b[j] = sum(source[:, j]*target[:, 0])
            for k in range(2):
                mat[j, k] = sum(source[:, j] * source[:, k])
        invmat = np.linalg.inv(mat)
        x = np.matmul(invmat, b)
        self.diag[0] = x[0]
        self.high2low[0] = x[1]
        for j in range(2):
            b[j] = sum(source[:, m+j-2]*target[:, m-1])
            for k in range(2):
                mat[j, k] = sum(source[:, m+j-2] * source[:, m+k-2])
        invmat = np.linalg.inv(mat)
        x = np.matmul(invmat, b)
        self.low2high[m-2] = x[0]
        self.diag[m-1] = x[1]
        self.error = np.zeros( m )
        for i in range(n):
            tmp = self.diag * source[i, :] - target[i, :]
            tmp[1:m] += source[i, 0:m-1] * self.low2high
            tmp[0:m-1] += source[i, 1:m] * self.high2low
            self.error += tmp**2
        self.error = np.sqrt(self.error/n)
            
    def project(self, source):
        assert(len(source) == self.size)
        data = source * self.diag 
        data[1 : self.size] += source[0 : self.size - 1] * self.low2high
        data[0 : self.size-1] += source[1: self.size] * self.high2low
        return data

    def plot(self, extent = None, label=None):
        grids = np.zeros( (self.size, self.size) )
        for i in range(self.size):
            grids[i, i] = self.diag[i]
            if(i > 0):
                grids[i-1, i] = self.low2high[i-1]
            if(i<self.size-1):
                grids[i+1, i] = self.high2low[i]
        plt.imshow(grids, origin='lower', cmap = 'RdYlBu_r', extent = extent)
        plt.colorbar(label=label)

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
##FOREGROUND MODEL CLASS

class foreground_model:


    #A_dust (A_sync, corr_ds) can be a single number or an array
    def __init__(self, ells, A_dust = 10., A_sync = 1., beta_dust = 1.6, beta_sync = -3., eps2 = 0.,  alpha_eps = 0., T_CMB = 2.726, T_dust_MBB = 20., alpha_dust = 0., alpha_sync = 0., run_dust_ell = 0., run_sync_ell = 0., run_dust_freq = 0., run_sync_freq = 0., svs_dust = 0.,  svs_dust_index = 0.,  B_dust = 0., svs_sync_index = 0., svs_sync = 0., B_sync = 0., freq_sync_ref = 30., freq_dust_ref = 270., ell_ref=80., freq_decorr_model = None):
        self.freq_decorr_model = freq_decorr_model
        self.T_CMB = T_CMB   #one field only
        self.freq_sync_ref = freq_sync_ref
        self.freq_dust_ref = freq_dust_ref
        self.ell_ref = ell_ref
        self.T_dust_MBB = T_dust_MBB
        self.ells = ells
        self.num_ells = len(ells)
        assert(self.num_ells > 1)
        self.lbin_width = np.empty(self.num_ells)
        self.lbin_width[0] = (self.ells[1]-self.ells[0])
        self.lbin_width[self.num_ells-1] = (self.ells[self.num_ells-1] - self.ells[self.num_ells-2])
        for i in range(1,self.num_ells-1):
            self.lbin_width[i] = (self.ells[i+1]-self.ells[i-1])/2.
        #if A is given as a scalar, the ell spectrum is approximated by a quadratic function; if A is given as a list, the ell spectrum is a free (binned) function
        self.P_dust = A_dust * (ells/self.ell_ref)**(alpha_dust + (run_dust_ell/2.) * np.log(ells/self.ell_ref)) 
        self.P_sync = A_sync * (ells/self.ell_ref)**(alpha_sync + (run_sync_ell/2.) * np.log(ells/self.ell_ref)) 
        self.P_ds = np.sqrt(self.P_dust * self.P_sync) * eps2 * (2./ells)**alpha_eps
        #frequency spectral index
        self.beta_dust = beta_dust
        self.beta_sync = beta_sync
        self.run_dust_freq = run_dust_freq
        self.run_sync_freq = run_sync_freq
        #2011.11575, moment expansion for spatially varying SED of syncrotron
        self.B_dust = B_dust
        self.B_sync = B_sync
        self.svs_dust = svs_dust
        self.svs_sync = svs_sync
        self.svs_dust_index  = svs_dust_index
        self.svs_sync_index  = svs_sync_index        

    def dust_freq_weight(self, nu):
        hGk_t0 = 0.0479924/self.T_CMB  # h*GHz/ k_B / T_CMB
        hGk_td = 0.0479924/self.T_dust_MBB  #h GHz/k_B/T_MBB
        if(self.freq_decorr_model == "Taylor"):
            return (nu/self.freq_dust_ref)**(self.beta_dust+self.B_dust + self.run_dust_freq*np.log(nu/self.freq_dust_ref)-1.)*np.exp(hGk_t0*(self.freq_dust_ref - nu))*((np.exp(hGk_t0*nu)-1.)/(np.exp(hGk_t0*self.freq_dust_ref)-1.))**2*((np.exp(hGk_td*self.freq_dust_ref)-1.)/(np.exp(hGk_td*nu)-1.))
        else:
            return (nu/self.freq_dust_ref)**(self.beta_dust-1.)*np.exp(hGk_t0*(self.freq_dust_ref - nu))*((np.exp(hGk_t0*nu)-1.)/(np.exp(hGk_t0*self.freq_dust_ref)-1.))**2*((np.exp(hGk_td*self.freq_dust_ref)-1.)/(np.exp(hGk_td*nu)-1.))            
            

    def sync_freq_weight(self, nu):
        hGk_t0 = 0.0479924/self.T_CMB  # h*GHz/ k_B / T_CMB
        if(self.freq_decorr_model == "Taylor"):
            return (nu/self.freq_sync_ref)**(self.beta_sync+self.B_sync + self.run_sync_freq*np.log(nu/self.freq_sync_ref)-2.)*np.exp(hGk_t0*(self.freq_sync_ref - nu))*((np.exp(hGk_t0*nu)-1.)/(np.exp(hGk_t0*self.freq_sync_ref)-1.))**2
        else:
            return (nu/self.freq_sync_ref)**(self.beta_sync-2.)*np.exp(hGk_t0*(self.freq_sync_ref - nu))*((np.exp(hGk_t0*nu)-1.)/(np.exp(hGk_t0*self.freq_sync_ref)-1.))**2            

    
    
    def full_bandpower(self, freq1, freq2, weights1 = None, weights2 = None):
        if((weights1 is None) and (weights2 is None)):
            sw1 = self.sync_freq_weight(freq1)
            dw1 = self.dust_freq_weight(freq1)            
            sw2 = self.sync_freq_weight(freq2)
            dw2 = self.dust_freq_weight(freq2)
            mean_freq1 = freq1
            mean_freq2 = freq2
        else:
            nw1 = len(freq1)
            assert(len(weights1) == nw1)
            if(nw1 == 1):
                sw1 = self.sync_freq_weight(freq1[0])
                dw1 = self.dust_freq_weight(freq1[0])                
            else:
                sw1 = 0. 
                dw1 = 0. 
                for i in range(nw1):
                    sw1 += self.sync_freq_weight(freq1[i])*weights1[i]
                    dw1 += self.dust_freq_weight(freq1[i])*weights1[i]
            mean_freq1 = np.sum(freq1*weights1)/np.sum(weights1)
            nw2 = len(freq2)
            assert(len(weights2) == nw2)
            if(nw2 == 1):
                sw2 = self.sync_freq_weight(freq2[0])
                dw2 = self.dust_freq_weight(freq2[0])                
            else:
                sw2 = 0. 
                dw2 = 0. 
                for i in range(nw2):
                    sw2 += self.sync_freq_weight(freq2[i])*weights2[i]
                    dw2 += self.dust_freq_weight(freq2[i])*weights2[i]
            mean_freq2 = np.sum(freq2*weights2)/np.sum(weights2)
        if(self.freq_decorr_model == "ME"):
            power_sync = self.P_sync * sw1 * sw2 * (1. + self.B_sync * (np.log(mean_freq1/self.freq_sync_ref)**2+np.log(mean_freq2/self.freq_sync_ref)**2) +  np.log(mean_freq1/self.freq_sync_ref)*np.log(mean_freq2/self.freq_sync_ref) * self.svs_sync * (self.ells/self.ell_ref)**self.svs_sync_index )
            power_dust = self.P_dust *  dw1 * dw2 * (1. + self.B_dust * (np.log(mean_freq1/self.freq_dust_ref)**2+np.log(mean_freq2/self.freq_dust_ref)**2) +  np.log(mean_freq1/self.freq_dust_ref)*np.log(mean_freq2/self.freq_dust_ref) * self.svs_dust * (self.ells/self.ell_ref)**self.svs_dust_index )
            power_cross = self.P_ds * (sw1 * dw2 + sw2 * dw1)
        elif(self.freq_decorr_model == "Taylor"):
            power_sync = self.P_sync * sw1 * sw2 * np.exp( - (np.log(mean_freq1/mean_freq2)/self.svs_sync)**2 )
            power_dust = self.P_dust * dw1 * dw2 * np.exp( - (np.log(mean_freq1/mean_freq2)/self.svs_dust)**2)
            power_cross = self.P_ds * (sw1 * dw2 + sw2 * dw1)            
        else:
            power_sync = self.P_sync * sw1 * sw2
            power_dust = self.P_dust * dw1 * dw2
            power_cross = self.P_ds * (sw1 * dw2 + sw2 * dw1)            
        return  power_sync + power_dust + power_cross

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##########SKY SIMULATOR CLASS    
class sky_simulator:
    
            
    ##freqs in GHz, fwhms in arcmin, sensitivities in uK (sensitivity = NET / sqrt(num_detector); white noise level =  sensitifity *  sqrt(sky area in arcmin^2/ obs time in seconds) and often expressed in uK-arcmin; white noise C_l = (white noise level in uK-radian)^2 )
    def __init__(self, config_file, root_overwrite = None):
        f = open(config_file, 'r') 
        config = literal_eval(f.read())
        f.close()        
        self.cosmo_r = config.get("cosmo_r", 0.)
        self.cosmo_r1 = config.get("cosmo_r1", None)
        self.do_r1 = (self.cosmo_r1 is not None)
        if(self.do_r1):
            assert(self.cosmo_r == 0. and self.cosmo_r1 > 0. )
        self.verbose = config.get("verbose", True)
        self.path = config["path"]
        self.write_more = not ( config.get('save_disk', True))
        if(root_overwrite is None):
            self.root =  path.join(self.path, config["root"])
        else:
            self.root = path.join(self.path, root_overwrite)
        self.band_weights_root  = path.join(self.path, config.get("band_weights_root", r"band_weights/BW_"))
        self.cmb_root_raw = path.join(self.path, config.get('cmb_root', r'cmb/cmb_'))
        self.cmbf_root_raw = path.join(self.path, config.get('filtered_cmb_root', r'cmbf/cmbf_'))        
        self.cmb_root  = self.cmb_root_raw +  cmb_postfix_for_r(self.cosmo_r)
        self.cmbf_root  = self.cmbf_root_raw +  cmb_postfix_for_r(self.cosmo_r)
        if(self.do_r1):
            self.cmb1_root  = self.cmb_root_raw +  cmb_postfix_for_r(self.cosmo_r1)
            self.cmb1f_root  = self.cmbf_root_raw +  cmb_postfix_for_r(self.cosmo_r1)
        self.noise_root  = path.join(self.path, config.get('noise_root', r'noise/noise_'))
        self.noisef_root  = path.join(self.path, config.get('filtered_noise_root', r'noisef/noisef_'))
        self.fg_root = path.join(self.path, config.get('foreground_root', r'fg/fg_'))
        self.fgf_root = path.join(self.path, config.get('filtered_foreground_root', r'fgf/fgf_'))
        mkdir_for_file(self.root)
        mkdir_for_file(self.cmb_root_raw)
        mkdir_for_file(self.cmbf_root_raw)
        mkdir_for_file(self.fg_root)
        mkdir_for_file(self.fgf_root)
        mkdir_for_file(self.noise_root)
        mkdir_for_file(self.noisef_root)                
        self.cmb_spectra_file = self.cmb_root_raw + "Cls.dat"        
        self.nmaps = config["nmaps"]
        assert(self.nmaps > 1)
        self.num_seasons = config["num_seasons"]
        assert(self.num_seasons > 0)          
        self.num_season_pairs = self.num_seasons*(self.num_seasons-1) // 2
        self.season_fac = np.sqrt(float(self.num_seasons))        
        self.nside = config['nside']
        assert(self.nside == 128 or self.nside == 256 or self.nside == 512 or self.nside == 1024) #default resolutions for small aperture telescopes
        self.npix = 12*self.nside**2
        self.mask_file = config["mask_file"]
        rawmask = hp.read_map(self.mask_file, field=0, dtype=np.float64)        
        if(rawmask.shape[0] != self.npix):
            self.mask = hp.ud_grade(rawmask, self.nside, order_in="RING", order_out="RING")
        else:
            self.mask = rawmask
        self.smoothed_mask = nmt.mask_apodization(self.mask, 2., apotype = "C2")  #do 2 deg smoothing for pseudo alm calculation
        self.mask_zeros = (self.mask <= 0.)
        self.mask_ones = (self.mask > 0.)        
        self.lmax = config.get("lmax", self.nside*3-1)
#        assert(self.lmax >= self.nside * 3)  
        self.coordinate = config.get('coordinate', 'G')
        assert(self.coordinate == "G" or self.coordinate == "C" or self.coordinate == "E")
        self.freqs = np.array(config['freqs'])
        self.freqnames = [ float2intstr(freq)+'GHz' for freq in self.freqs ]
        self.num_freqs = len(self.freqs)
        assert(self.num_freqs > 0)                
        if(self.verbose):
            print("freqs [GHz] = ", self.freqs)
        self.freq_lowest = self.freqs[0]
        self.freq_highest = self.freqs[0]
        self.ifreq_lowest = 0
        self.ifreq_highest = 0
        for i in range(1, self.num_freqs):
            if(self.freqs[i] < self.freq_lowest):
                self.freq_lowest = self.freqs[i]
                self.ifreq_lowest = i
            if(self.freqs[i] > self.freq_highest):
                self.freq_highest = self.freqs[i]
                self.ifreq_highest = i            
        self.band_weights = []
        self.has_band_weights = False
        for ifreq in range(self.num_freqs):
            fname = self.band_weights_root + self.freqnames[ifreq]+r'_weights.txt'
            if(path.exists(fname)):
                weights = np.loadtxt(fname)
                weights[:, 1] /= np.sum(weights[:, 1])
                self.band_weights.append(weights)
                self.has_band_weights = True
            else:
                self.band_weights.append( np.array( [[self.freqs[ifreq],  1.]] )  )
        if(self.has_band_weights and self.verbose):
            for ifreq in range(self.num_freqs):
                print("band weights for ", self.freqnames[ifreq])
                print(self.band_weights[ifreq])            
        self.fwhms_arcmin = np.array(config['fwhms'])
        if(self.verbose):
            print("FWHMs [arcmin] = ", self.fwhms_arcmin)            
        self.fwhms_deg = self.fwhms_arcmin / 60.
        self.fwhms_rad = self.fwhms_deg * (np.pi/180.)
        self.beam_sigmas = self.fwhms_rad/np.sqrt(8.*np.log(2.))
        assert(len(self.fwhms_rad) == self.num_freqs)
        self.white_noise_arcmin = np.array(config["white_noise"])
        self.white_noise = self.white_noise_arcmin * (np.pi/180./60.)
        assert(len(self.white_noise) == self.num_freqs)
        self.delens_fac = config.get('delens_fac', 0.8)
        self.cosmo_ombh2 = config.get('cosmo_ombh2', cosmology_base_mean[0])
        self.cosmo_omch2 = config.get('cosmo_omch2', cosmology_base_mean[1])
        self.cosmo_theta = config.get('cosmo_theta', cosmology_base_mean[2])
        self.cosmo_tau = config.get('cosmo_tau', cosmology_base_mean[3])
        self.cosmo_logA = config.get('cosmo_logA', cosmology_base_mean[4])
        self.cosmo_ns = config.get('cosmo_ns', cosmology_base_mean[5])
        self.lcdm_params = np.array( [ self.cosmo_ombh2, self.cosmo_omch2, self.cosmo_theta, self.cosmo_tau, self.cosmo_logA, self.cosmo_ns, self.cosmo_r ] )
        if(self.do_r1):
            self.lcdm1_params = np.array( [ self.cosmo_ombh2, self.cosmo_omch2, self.cosmo_theta, self.cosmo_tau, self.cosmo_logA, self.cosmo_ns, self.cosmo_r1 ] )                
        f = open(config["filter_model"], 'rb')
        self.filtering = pickle.load(f)
        f.close()
        self.fg_models =config["fg_models"]
        self.l_knee_P = config["l_knee_P"]
        self.alpha_knee_P = config["alpha_knee_P"]        
        if(self.verbose):
            print('white noise (uK-arcmin): ', self.white_noise_arcmin)
        self.N_white_T =  self.white_noise **2
        self.N_white_P =  self.N_white_T * 2.
        self.N_red_P = self.N_white_P
        #for temperature set low-f component to zero
        self.N_red_T = np.zeros(self.num_freqs)  #self.N_white_T
        self.alpha_knee_T = self.alpha_knee_P
        self.l_knee_T = self.l_knee_P        
        if(self.coordinate != 'G'):
            self.rotator = hp.Rotator(coord = ['G', self.coordinate])
        else:
            self.rotator = None
        self.do_lowfreq_noise_P =  not (self.l_knee_P is None and self.alpha_knee_P is None)        
        if(self.do_lowfreq_noise_P):
            assert(len(self.l_knee_P) == self.num_freqs)
            assert(len(self.alpha_knee_P) == self.num_freqs)
                        
        

    def files_exist(self, prefix, postfix):
        for ifreq in range(self.num_freqs):
            if(not path.exists(prefix +  self.freqnames[ifreq] + postfix)):
                return False
        return True

    def save_map(self, filename, maps, overwrite):
        if((not overwrite)  and path.exists(filename)):
            print('Warning: skipping '+filename+' that already exists...')
            return
        if(filename[-5:] == ".fits"):
            hp.write_map(filename = filename, m = maps, column_units = 'uK', overwrite = overwrite, extra_header=[ ("COORDSYS", self.coordinate) ], dtype=np.float32)
        elif(filename[-4:] == ".npy"):
            np.save(filename, maps[:, self.mask_ones])
        else:
            print('Cannot save unknown map format: ' + filename)
            exit()

    def load_IQU_map(self, filename):
        if(filename[-5:] == ".fits"):
            m = hp.read_map(filename, field = [0, 1, 2], dtype = np.float64)
            m[:, self.mask_zeros] = 0.
        elif(filename[-4:] == ".npy"):
            m = np.zeros((3, self.npix))
            try:
                m[:, self.mask_ones] = np.load(filename)
            except:
                m[1:3, self.mask_ones] = np.load(filename)                
        else:
            print('unknown map format: ' + filename)
            exit()
        return m

    def load_QU_map(self, filename):
        if(filename[-5:] == ".fits"):
            m = hp.read_map(filename, field = [1, 2], dtype = np.float64)
            m[:, self.mask_zeros] = 0.
        elif(filename[-4:] == ".npy"):
            m = np.zeros((2, self.npix))
            m[:, self.mask_ones] = np.load(filename)[1:3, :]
        else:
            print('unknown map format: ' + filename)
            exit()
        return m
    
            
    def set_cosmology(self, lcdm_params = cosmology_base_mean, w = -1., wa = 0., mnu=0.06, omk=0., MOG_eta = 1., MOG_eta_slope = 0.):  #default Planck18 best fit cosmology; MOG_eta is evaluated at l = 1000
        pars = CAMBparams()
        if(lcdm_params[6] > 0.):
            pars.WantTensors = True
        pars.set_cosmology(ombh2 = lcdm_params[0], omch2 = lcdm_params[1], cosmomc_theta = lcdm_params[2]/100., tau = lcdm_params[3], mnu = mnu, omk=omk)
        pars.InitPower.set_params(As = 1.e-10*np.exp(lcdm_params[4]), ns = lcdm_params[5], r = lcdm_params[6])
        pars.set_dark_energy(w = w, wa = wa, dark_energy_model = 'ppf')
        pars.set_for_lmax(self.lmax, lens_potential_accuracy = 2)
        results = get_results(pars)
        powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
        arr = np.empty( (self.lmax-1, 8) )
        for ell in range(2, self.lmax+1):
            arr[ell-2, 0] = ell
            arr[ell-2, 1:5] = powers['unlensed_total'][ell, 0:4]
            arr[ell-2, 5:8] = powers['lens_potential'][ell, 0:3]
        if( abs(MOG_eta - 1.) > 1.e-6 ):  #modified gravity amplification of lensing potential; this only works for ell << 1000; see e.g. 1405.7004
            if(self.verbose):
                print('**Warning: doing simulation with modified gravity, eta = ', MOG_eta)
            amp = (MOG_eta+1.)/2.
            arr[:, 5] *=  ( 1. + (amp-1.) * (arr[:, 0]/1000.)**(MOG_eta_slope) ) ** 2
            arr[:, 6] *= ( 1. + (amp-1.) * (arr[:, 0]/1000.)**(MOG_eta_slope) )
            arr[:, 7] *= ( 1. + (amp-1.) * (arr[:, 0]/1000.)**(MOG_eta_slope) )            
        np.savetxt(self.cmb_spectra_file, arr, fmt = r'%d %14.7e  %14.7e %14.7e  %14.7e  %14.7e %14.7e %14.7e', header = r'   L    TT             EE             BB             TE             PP             TP             EP')
        import_file_to_cache(r'https://portal.nersc.gov/project/cmb/pysm-data/pysm_2/camb_lenspotentialCls.dat', self.cmb_spectra_file) #This is for pysm3 c1 model

    def fg_band_weights(self, beta_d, beta_s):
        fg = foreground_model(ells = np.array([10., 20.]), beta_dust = beta_d, beta_sync = beta_s, freq_dust_ref = self.freq_highest, freq_sync_ref = self.freq_lowest)
        dust_weights = np.zeros(self.num_freqs)
        sync_weights = np.zeros(self.num_freqs)
        if(self.has_band_weights):
            for ifreq  in range(self.num_freqs):
                for iw in range(self.band_weights[ifreq].shape[0]):
                    dust_weights[ifreq] += fg.dust_freq_weight(self.band_weights[ifreq][iw, 0])*self.band_weights[ifreq][iw, 1]
                    sync_weights[ifreq] += fg.sync_freq_weight(self.band_weights[ifreq][iw, 0])*self.band_weights[ifreq][iw, 1]                    
        else:
            for ifreq in range(self.num_freqs):
                dust_weights[ifreq] = fg.dust_freq_weight(self.freqs[ifreq])
                sync_weights[ifreq] = fg.sync_freq_weight(self.freqs[ifreq])
        return dust_weights, sync_weights
        

    def Noise_Power_T(self, ell, ifreq = None, deconv = True):
        if(deconv):
            if(ifreq is None):
                return  (self.N_red_T*(ell/ self.l_knee_T)**self.alpha_knee_T + self.N_white_T) * np.exp(ell* (ell+1.)*self.beam_sigmas**2) 
            else:
                return  (self.N_red_T[ifreq]*(ell/ self.l_knee_T[ifreq])**self.alpha_knee_T[ifreq] + self.N_white_T[ifreq]) * np.exp(ell* (ell+1.)*self.beam_sigmas[ifreq]**2) 
        else:
            if(ifreq is None):
                return  (self.N_red_T*(ell/ self.l_knee_T)**self.alpha_knee_T + self.N_white_T)
            else:
                return  (self.N_red_T[ifreq]*(ell/ self.l_knee_T[ifreq])**self.alpha_knee_T[ifreq] + self.N_white_T[ifreq]) 

    def Noise_Power_P(self, ell, ifreq = None, deconv = True):
        if(deconv):
            if(ifreq is None):
                return (self.N_red_P*(ell/ self.l_knee_P)**self.alpha_knee_P + self.N_white_P) * np.exp(ell* (ell+1.)*self.beam_sigmas**2)
            else:
                return (self.N_red_P[ifreq]*(ell/ self.l_knee_P[ifreq])**self.alpha_knee_P[ifreq] + self.N_white_P[ifreq]) * np.exp(ell* (ell+1.)*self.beam_sigmas[ifreq]**2) 
        else:
            if(ifreq is None):
                return (self.N_red_P*(ell/ self.l_knee_P)**self.alpha_knee_P + self.N_white_P)
            else:
                return (self.N_red_P[ifreq]*(ell/ self.l_knee_P[ifreq])**self.alpha_knee_P[ifreq] + self.N_white_P[ifreq]) 
                

    def Noise_Map(self, ifreq, seed=None, deconv = False):
        clsTT = np.empty(self.lmax+1)
        clsTE = np.zeros(self.lmax+1)
        clsEE = np.empty(self.lmax+1)
        clsBB = np.empty(self.lmax+1)                        
        clsTT[0:2] = 0.
        clsEE[0:2] = 0.
        clsBB[0:2] = 0.
        for ell in range(2, self.lmax+1):
            clsTT[ell] = self.Noise_Power_T(ell = ell, ifreq = ifreq, deconv = deconv)  #no debeam so the amp is already smoothed
            clsEE[ell] = self.Noise_Power_P(ell = ell, ifreq = ifreq, deconv = deconv)
            clsBB[ell] = clsEE[ell]
        set_random_seed(seed)
        return hp.synfast(cls = (clsTT, clsTE, clsEE, clsBB), nside=self.nside)


    def simulate_noise(self, seed = None, overwrite = False):
        set_random_seed(seed)
        for i in range(self.nmaps):
            if((not overwrite) and self.files_exist(prefix = self.noisef_root, postfix = r'_'+ str(i) + r'.npy')):
                continue                ##if all are simulated
            if(self.verbose):
                print('simulating noise #' + str(i))            
            for ifreq in range(self.num_freqs):                
                noise_sum = np.zeros( (3,self.npix))
                noisef_sum = np.zeros( (3, self.npix))
                for isea in range(self.num_seasons):
                    fn = self.noise_root + self.freqnames[ifreq] + r'_' + str(i) + r'_season' + str(isea) + r'.npy'
                    fnf = self.noisef_root + self.freqnames[ifreq] + r'_' + str(i) + r'_season' + str(isea) + r'.npy'                                    
                    noise_map = self.Noise_Map(ifreq = ifreq) * self.season_fac 
                    noisef_map = self.filtering.project_map(mask = self.smoothed_mask, maps = noise_map, want_wof = False)
                    if(self.write_more):
                        self.save_map(fn, noise_map, overwrite = True)
                    self.save_map(fnf, noisef_map, overwrite = True)
                    noise_sum += noise_map
                    noisef_sum += noisef_map
                noise_sum /= self.num_seasons
                noisef_sum /= self.num_seasons
                self.save_map(self.noise_root + self.freqnames[ifreq] + r'_' + str(i) + r'.npy', noise_sum, overwrite = True)
                self.save_map(self.noisef_root + self.freqnames[ifreq] + r'_' + str(i) + r'.npy', noisef_sum, overwrite = True)


    
    def simulate_fg(self, seed = None, overwrite=False): #only simulate one foreground map
        if((not overwrite) and self.files_exist(prefix = self.fgf_root, postfix = r'.npy')):
            return
        set_random_seed(seed)        
        sky = pysm3.Sky(nside = self.nside, preset_strings = self.fg_models)
        if(self.verbose):
            print('simulating foreground maps')
        for ifreq in range(self.num_freqs):
            fn = self.fg_root + self.freqnames[ifreq] + r'.npy'
            fnf = self.fgf_root + self.freqnames[ifreq] + r'.npy'
            if(self.has_band_weights):
                rawmap = sky.get_emission(self.band_weights[ifreq][0,0] * pysm3.units.GHz).to(pysm3.units.uK_CMB, equivalencies=pysm3.units.cmb_equivalencies(self.band_weights[ifreq][0,0] * pysm3.units.GHz))*self.band_weights[ifreq][0,1]
                for iw in range(1, self.band_weights[ifreq].shape[0]):
                    rawmap += sky.get_emission(self.band_weights[ifreq][iw,0] * pysm3.units.GHz).to(pysm3.units.uK_CMB, equivalencies=pysm3.units.cmb_equivalencies(self.band_weights[ifreq][iw,0] * pysm3.units.GHz))*self.band_weights[ifreq][iw,1]
            else:
                rawmap = sky.get_emission(self.freqs[ifreq] * pysm3.units.GHz).to(pysm3.units.uK_CMB, equivalencies=pysm3.units.cmb_equivalencies(self.freqs[ifreq] * pysm3.units.GHz))
            fg = smooth_rotate(rawmap, fwhm_rad = self.fwhms_rad[ifreq], rot = self.rotator).value
            fgf = self.filtering.project_map(mask = self.smoothed_mask, maps = fg, want_wof = False)
            if(self.write_more):
                self.save_map(fn, fg, overwrite = True)
            self.save_map(fnf, fgf, overwrite = True)                

    def simulate_cmb(self, seed = 0, overwrite=False):
        self.set_cosmology( lcdm_params = self.lcdm_params)
        for i in range(self.nmaps):
            if(not overwrite and self.files_exist(prefix = self.cmbf_root, postfix = r'_'+ str(i) + r'.npy')):
                continue
            if(self.verbose):
                print(r'simulating r=' + str(self.lcdm_params[6]) + r' cmb #'+str(i))
            cmblens = pysm3.CMBLensed(nside = self.nside, cmb_spectra = self.cmb_spectra_file, cmb_seed = i + seed, construct_map = False)  #you have to hack pysm before using this, see documentation
            map_unlensed, map_lensed = cmblens.unlensed_and_lensed_maps()
            self.save_map(self.cmb_root + 'lenstemp_' + str(i) + r'.npy', (map_lensed-map_unlensed)*self.delens_fac, overwrite = True)
            for ifreq in range(self.num_freqs):
                fn = self.cmb_root + self.freqnames[ifreq] + r'_' + str(i) + r'.npy'
                fnf = self.cmbf_root + self.freqnames[ifreq] + r'_' + str(i) + r'.npy'
                smoothed_map = smooth_rotate(map_lensed, fwhm_rad = self.fwhms_rad[ifreq])
                if(self.write_more):
                    m, mf = self.filtering.project_map(mask = self.smoothed_mask, maps = smoothed_map, want_wof = True) 
                    self.save_map(fn, m, overwrite = True)
                    self.save_map(fnf, mf, overwrite = True)                    
                else:
                    mf = self.filtering.project_map(mask = self.smoothed_mask, maps = smoothed_map, want_wof = False)        
                    self.save_map(fnf, mf, overwrite = True)                    
                    

    def simulate_cmb1(self, seed = 9999, overwrite=False): #for cmb1 we use a different default seed
        if(not self.do_r1):
            return
        self.set_cosmology( lcdm_params = self.lcdm1_params)
        for i in range(self.nmaps):
            if(not overwrite and self.files_exist(prefix = self.cmb1f_root, postfix = r'_'+ str(i) + r'.npy')):
                continue
            if(self.verbose):
                print(r'simulating r=' + str(self.lcdm1_params[cosmology_num_params-1]) + r' cmb #'+str(i))
            cmblens = pysm3.CMBLensed(nside = self.nside, cmb_spectra = self.cmb_spectra_file, cmb_seed = i + seed, construct_map = False)  #you have to hack pysm before using this, see documentation
            map_unlensed, map_lensed = cmblens.unlensed_and_lensed_maps()
            self.save_map(self.cmb1_root + 'lenstemp_' + str(i) + r'.npy', (map_lensed-map_unlensed)*self.delens_fac, overwrite = True)
            for ifreq in range(self.num_freqs):
                fnf = self.cmb1f_root + self.freqnames[ifreq] + r'_' + str(i) + r'.npy'
                smoothed_map = smooth_rotate(map_lensed, fwhm_rad = self.fwhms_rad[ifreq])
                if(self.write_more):
                    fn = self.cmb1_root + self.freqnames[ifreq] + r'_' + str(i) + r'.npy'                    
                    m, mf = self.filtering.project_map(mask = self.smoothed_mask, maps = smoothed_map, want_wof = True) 
                    self.save_map(fn, m, overwrite = True)
                    self.save_map(fnf, mf, overwrite = True)                    
                else:
                    mf = self.filtering.project_map(mask = self.smoothed_mask, maps = smoothed_map, want_wof = False)        
                    self.save_map(fnf, mf, overwrite = True)                    

    def simulate_map(self, r=None, seed = None, overwrite = False):
        lcdm_params = self.lcdm_params.copy()        
        if(r is not None):
            lcdm_params[6] = r
        if(self.verbose):
            print('simulating maps with r = ', lcdm_params[6])
        if((not overwrite) and self.files_exist(prefix =  self.root , postfix = r'.npy')):
            return
        sky = pysm3.Sky(nside = self.nside, preset_strings = self.fg_models)
        self.set_cosmology(lcdm_params = lcdm_params)
        if(seed is None):
            cmblens = pysm3.CMBLensed(nside = self.nside, cmb_spectra = self.cmb_spectra_file, cmb_seed = time_seed(), construct_map = False)  
        else:
            cmblens = pysm3.CMBLensed(nside = self.nside, cmb_spectra = self.cmb_spectra_file, cmb_seed = seed, construct_map = False)
        cmb_unlensed, cmb_lensed = cmblens.unlensed_and_lensed_maps()
        self.save_map(self.root + 'lenstemp.npy', (cmb_lensed - cmb_unlensed) * self.delens_fac, overwrite = True)
        for  ifreq in range(self.num_freqs):
            if(self.verbose):
                print('simulating frequency '+self.freqnames[ifreq])
            filename = self.root + self.freqnames[ifreq]  + r'.npy'
            #--------------
            if(self.has_band_weights):
                fgraw = sky.get_emission(self.band_weights[ifreq][0,0] * pysm3.units.GHz).to(pysm3.units.uK_CMB, equivalencies=pysm3.units.cmb_equivalencies(self.band_weights[ifreq][0,0] * pysm3.units.GHz)).value * self.band_weights[ifreq][0,1]
                for iw in range(1, self.band_weights[ifreq].shape[0]):
                    fgraw += sky.get_emission(self.band_weights[ifreq][iw,0] * pysm3.units.GHz).to(pysm3.units.uK_CMB, equivalencies=pysm3.units.cmb_equivalencies(self.band_weights[ifreq][iw,0] * pysm3.units.GHz)).value * self.band_weights[ifreq][iw,1]
            else:
                fgraw = sky.get_emission(self.freqs[ifreq] * pysm3.units.GHz).to(pysm3.units.uK_CMB, equivalencies=pysm3.units.cmb_equivalencies(self.freqs[ifreq] * pysm3.units.GHz)).value            
            fgraw = sky.get_emission(self.freqs[ifreq] * pysm3.units.GHz).to(pysm3.units.uK_CMB, equivalencies=pysm3.units.cmb_equivalencies(self.freqs[ifreq] * pysm3.units.GHz)).value
            #--------------------------
            fgcmb = smooth_rotate(fgraw+cmb_lensed, fwhm_rad = self.fwhms_rad[ifreq], rot = self.rotator)
            total_map = np.zeros( (3, self.npix) )
            for isea in range(self.num_seasons):
                noise_map = self.Noise_Map(ifreq = ifreq) * (self.season_fac)  #noise is already smoothed (and no need to rotate since it is statistically isotropic)
                season_map = self.filtering.project_map(mask = self.smoothed_mask, maps = noise_map+fgcmb, want_wof = False)
                self.save_map( self.root + self.freqnames[ifreq]  + r'_season' + str(isea) + r'.npy',  season_map, overwrite=True)
                total_map += season_map
            total_map /= self.num_seasons
            self.save_map(filename, total_map, overwrite=True)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                        
###BAND POWER CALCULATOR CLASS

class band_power_calculator:
    purify_b = True
    is_Dell = True
    
    def __init__(self, mask_file, apo_deg = 4., apo_type = "C2", like_fields = ['BB'], lmin = 21, lmax = 200, delta_ell = 20, verbose = False, coordinate = 'G'):
        self.mask_file = mask_file #mask file name
        self.verbose = verbose
        self.coordinate = coordinate
        rawmask = hp.read_map(self.mask_file, field=0, dtype=np.float64)        
        self.npix = len(rawmask)
        self.nside = int(np.round(np.sqrt(self.npix/12.), 0))
        assert(self.nside==64 or self.nside == 128 or self.nside==256 or self.nside==512 or self.nside==1024 or self.nside==2048)
        if(self.verbose):
            print('mask file = ', self.mask_file)
            print('nside = ', self.nside)
        self.like_fields = like_fields
        self.num_fields = len(self.like_fields)                
        self.BB_loc = -1
        self.EE_loc = -1
        self.EB_loc = -1
        self.TT_loc = -1
        self.TE_loc = -1
        self.TB_loc = -1                
        for ib in range(self.num_fields):
            if(self.like_fields[ib] == 'BB'):
                self.BB_loc = ib
            if(self.like_fields[ib] == 'EE'):
                self.EE_loc = ib
            if(self.like_fields[ib] == 'EB' or self.like_fields[ib] == 'BE'):
                self.EB_loc = ib
            if(self.like_fields[ib] == 'TT'):
                self.TT_loc = ib
            if(self.like_fields[ib] == 'TE'  or self.like_fields[ib] == 'ET'):
                self.TE_loc = ib
            if(self.like_fields[ib] == 'TB'  or self.like_fields[ib] == 'BT'):
                self.TB_loc = ib
        self.lmin = lmin
        self.lmax = lmax
        self.delta_ell = delta_ell
        self.apo_deg = apo_deg
        self.apo_type = apo_type
        self.lbins = nmt.NmtBin.from_edges(ell_ini = range(self.lmin, self.lmax-self.delta_ell+2,self.delta_ell),  ell_end = range(self.lmin + self.delta_ell, self.lmax+2, self.delta_ell), is_Dell = self.is_Dell)
        #self.lbins = nmt.NmtBin.from_lmax_linear(self.lmax, self.delta_ell, is_Dell=True)
        self.ells = self.lbins.get_effective_ells()
        self.num_ells = len(self.ells)
        if(rawmask.shape[0] != self.npix):
            self.binary_mask = hp.ud_grade(rawmask, self.nside, order_in="RING", order_out="RING")
        else:
            self.binary_mask = rawmask
        if(self.apo_deg > 0.):
            self.mask = nmt.mask_apodization(self.binary_mask, self.apo_deg, apotype = self.apo_type)
        else:
            self.mask = self.binary_mask
        self.mask_zeros = (self.binary_mask == 0.)
        self.mask_ones = (self.binary_mask > 0.)        
        self.mask_npix = self.mask.sum()
        self.binary_mask_npix = self.binary_mask.sum()
        self.fsky = self.mask_npix/self.npix
        self.binary_fsky = self.binary_mask_npix/self.npix
        if(self.verbose):
            print('raw fsky = ', self.binary_fsky)
            print('apodized fsky = ', self.fsky)
            print('ell-bin centers:', self.ells)
            print('likelihood uses:', self.like_fields)
        self.dofs = self.fsky  * self.delta_ell * (2. * self.ells + 1.)  ## d.o.f. in each bin, this is not exact; for complex mask caution should be taken
        self.w00_initialized = False
        self.w02_initialized = False
        self.w20_initialized = False
        self.w22_initialized = False        
        self.want_00 =  ( self.TT_loc >= 0 )
        self.want_02 =  ( self.TE_loc >= 0  or self.TB_loc >= 0)
        self.want_20 =  self.want_02
        self.want_22 =  ( self.EE_loc >= 0 or self.BB_loc >= 0 or self.EB_loc >= 0)
        if(self.want_00):
            self.w00 = nmt.NmtWorkspace()
        else:
            self.w00 = None
        if(self.want_20):
            self.w20 = nmt.NmtWorkspace()
        else:
            self.w20 = None
        if(self.want_02):
            self.w02 = nmt.NmtWorkspace()
        else:
            self.w02 = None
        if(self.want_22):
            self.w22 = nmt.NmtWorkspace()
        else:
            self.w22 = None
        self.powerbank_path = "powerbank_" + compress_str(path.realpath(self.mask_file) + str(self.lmin) + str(self.lmax) + str(self.delta_ell) + str(int(self.apo_deg*10)) + self.apo_type, length=8) 
        mkdir_if_not_exists(self.powerbank_path)
        
    def load_IQU_map(self, filename):
        if(filename[-5:] == ".fits"):
            m = hp.read_map(filename, field = [0, 1, 2], dtype = np.float64)
            m[:, self.mask_zeros] = 0.
        elif(filename[-4:] == ".npy"):
            m = np.zeros((3, self.npix))
            try:
                m[:, self.mask_ones] = np.load(filename)
            except:
                m[1:3, self.mask_ones] = np.load(filename)                
        else:
            print('unknown map format: ' + filename)
            exit()
        return m

    def smooth_map(self, m, fwhm_in, fwhm_out):
        assert(fwhm_out >= fwhm_in)        
        for i in range(3):
            m[i, :] *= self.mask
        if(fwhm_out == fwhm_in):
            return m
        else:
            return hp.smoothing(m, fwhm = 1./np.sqrt(1./fwhm_in**2 - 1./fwhm_out**2))


    def load_and_smooth(self, filename, fwhm_in, fwhm_out):
        m = self.load_IQU_map(filename)
        return self.smooth_map(m, fwhm_in, fwhm_out)
    

    def save_map(self, filename, maps, overwrite):
        if((not overwrite)  and path.exists(filename)):
            print('Warning: skipping '+filename+' that already exists...')
            return
        if(filename[-5:] == ".fits"):
            hp.write_map(filename = filename, m = maps, column_units = 'uK', overwrite = overwrite, extra_header=[ ("COORDSYS", self.coordinate) ], dtype=np.float32)
        elif(filename[-4:] == ".npy"):
            np.save(filename, maps[:, self.mask_ones])
        else:
            print('Cannot save unknown map format: ' + filename)
            exit()
    
    def get_band_power(self, map1, map2 = None):
        lmax_mask = self.lmax #self.lmax*2 + 100
        f0 = nmt.NmtField(self.mask, map1[0:1, :], lmax = self.lmax, lmax_mask = lmax_mask)
        f2 = nmt.NmtField(self.mask, map1[1:3, :], lmax = self.lmax, purify_b = self.purify_b, lmax_mask = lmax_mask)
        if(map2 is None):
            g0 = f0
            g2 = f2
        else:
            g0 = nmt.NmtField(self.mask, map2[0:1, :], lmax = self.lmax, lmax_mask =  lmax_mask)
            g2 = nmt.NmtField(self.mask, map2[1:3, :], lmax = self.lmax, purify_b = self.purify_b, lmax_mask =  lmax_mask)
        if( not self.w00_initialized and self.want_00):
            self.w00.compute_coupling_matrix(f0, g0, self.lbins)
            self.w00_initialized = True
        if( not self.w02_initialized and self.want_02):
            self.w02.compute_coupling_matrix(f0, g2, self.lbins)
            self.w02_initialized = True
        if( not self.w20_initialized and self.want_20):
            self.w20.compute_coupling_matrix(f2, g0, self.lbins)
            self.w20_initialized = True
        if( not self.w22_initialized and self.want_22):
            self.w22.compute_coupling_matrix(f2, g2, self.lbins)
            self.w22_initialized = True
        if(self.want_00):
            cls00 = compute_master_with_workspace(f0, g0, self.w00)
        else:
            cls00 = None
        if(self.want_02):
            cls02 = compute_master_with_workspace(f0, g2, self.w02)
        else:
            cls02 = None
        if(self.want_20):
            cls20 = compute_master_with_workspace(f2, g0, self.w20)
        else:
            cls20 = None
        if(self.want_22):
            cls22 = compute_master_with_workspace(f2, g2, self.w22)
        else:
            cls22 = None
        bp = {}
        i = 0
        for field in self.like_fields:
            if field == 'TT':
                bp[field] = cls00[0]
            elif (field == 'TE' or field == 'ET'):
                bp[field] = (cls02[0] + cls20[0])/2.
            elif (field == 'TB' or field == 'BT'):
                bp[field] = (cls02[1] + cls20[1])/2.
            elif field == 'EE':
                bp[field] = cls22[0]
            elif (field == 'EB' or field=='BE'):
                bp[field] = (cls22[1]+cls22[2])/2.
            elif field == 'BB':
                bp[field] = cls22[3]
            else:
                print("Unrecognized field:", field)
                exit()
            i += 1
        return bp
        


    def band_power_multiple(self, mapfiles1, mapfiles2 = None, overwrite = False, w1=1., w2=1.):
        """1st file in mapfiles1 is rescaled by w1; 1st file in mapfiles2 is rescaled by w2"""
        if(w1 != 1. or w2 != 1.):
            if(len(mapfiles1) == 1 and w1 != 1.):
                bp =  self.band_power_multiple(mapfiles1=mapfiles1, mapfiles2 = mapfiles2, overwrite = overwrite, w1=1., w2=w2)
                for field in self.like_fields:
                    bp[field] *= w1
                return bp
            if(mapfiles2 is None):
                if(len(mapfiles1) == 1 and w2 != 1.):
                    bp =  self.band_power_multiple(mapfiles1=mapfiles1, mapfiles2 = None, overwrite = overwrite, w1=w1, w2=1.)
                    for field in self.like_fields:
                        bp[field] *= w2
                    return bp
            elif(len(mapfiles2) == 1 and w2 != 1.):
                bp =  self.band_power_multiple(mapfiles1=mapfiles1, mapfiles2 = mapfiles2, overwrite = overwrite, w1=w1, w2=1.)
                for field in self.like_fields:
                    bp[field] *= w2
                return bp
        nmap1 = len(mapfiles1)
        if(mapfiles2 is None):
            nmap2 = nmap1
        else:
            nmap2 = len(mapfiles2)
        if(w1 != 1.):
            sumints1 = path23ints(mapfiles1[0]+'__'+str(int(w1*100000)))
        else:
            sumints1 = path23ints(mapfiles1[0])
        for imap in range(1, nmap1):
            sumints1 += path23ints(mapfiles1[imap])
        if(mapfiles2 is None):
            assert(w2 == w1) #same file should have same weight
            sumints2 = sumints1
        else:
            if(w2 != 1.):
                sumints2 = path23ints(mapfiles2[0]+'__'+str(int(w2*100000)))
            else:
                sumints2 = path23ints(mapfiles2[0])                
            for imap in range(1, nmap2):
                sumints2 += path23ints(mapfiles2[imap])
        prefix = path.join(self.powerbank_path, compress_str(str(sumints1[0]+sumints2[0]) + '_' + str(sumints1[1]+sumints2[1])+'_'+ str(sumints1[2]*sumints2[2]), length = 20))
        saved = True        
        if(overwrite):
            saved = False
        else:
            for field in self.like_fields:
                if(not path.exists(prefix + field + r'.npy')):
                    saved = False
                    break
        if(saved):
#            if(self.verbose):
#                print('loading saved power spectra for ', mapfiles1, mapfiles2)
            bp = {}
            for field in self.like_fields:
                try:
                    bp[field] = np.load(prefix + field + r'.npy')
                except:
                    saved = False
                    break
            if(saved):
                return bp
        map1 = self.load_IQU_map(mapfiles1[0])
        if(w1 != 1.):
            map1 *= w1
        for i in range(1, len(mapfiles1)):
            map1 += self.load_IQU_map(mapfiles1[i])            
        if(mapfiles2 is None):
            map2 = None
        else:
            map2 = self.load_IQU_map(mapfiles2[0])
            if(w2 != 1.):
                map2 *= w2
            for i in range(1, len(mapfiles2)):
                map2 += self.load_IQU_map(mapfiles2[i])            
        bp =  self.get_band_power(map1, map2)
        if(self.verbose):
            print('saving power spectra for ', mapfiles1, mapfiles2)
        for field in self.like_fields:
            np.save(prefix + field + r'.npy', bp[field])
        return bp

    def band_power(self, mapfile1, mapfile2 = None, overwrite = False, w1=1., w2=1.):
        if(isinstance(mapfile1, list)):
            mapfiles1 = mapfile1
        else:
            mapfiles1 = [ mapfile1 ]
        if((mapfile2 is None) or isinstance(mapfile2, list)):
            mapfiles2 = mapfile2
        else:
            mapfiles2 = [ mapfile2 ]
        return self.band_power_multiple(mapfiles1, mapfiles2, overwrite, w1=w1, w2=w2)    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
###SKY ANALYSER CLASS    
    
class  sky_analyser:

    def __init__(self, config_file, root_overwrite = None):
        f = open(config_file, 'r') 
        config = literal_eval(f.read())
        f.close()
        self.continue_run = config.get("continue_run", False)
        self.cosmo_r = config.get("cosmo_r", 0.)
        self.cosmo_r1 = config.get("cosmo_r1", None)
        self.do_r1 = (self.cosmo_r1 is not None)
        if(self.do_r1):
            assert(self.cosmo_r == 0. and self.cosmo_r1 > 0. )
            self.num_r_interp = 31
        self.verbose = config.get("verbose", True)
        self.mask_file = config["mask_file"]
        self.path = config["path"]
        self.mapform = r'.' + config.get("mapform", 'npy') #'npy' or 'fits'        
        if(root_overwrite is None):
            self.root =  path.join(self.path, config["root"])
        else:
            self.root = path.join(self.path, root_overwrite)
        self.r_logfile = config.get("r_logfile", path.join(self.path, "r__LOGFILE.txt"))
        self.band_weights_root  = path.join(self.path, r"band_weights/BW_")        
        self.cmb_root_raw = path.join(self.path, config.get('cmb_root', r'cmb/cmb_'))
        self.cmbf_root_raw = path.join(self.path, config.get('filtered_cmb_root', r'cmbf/cmbf_'))        
        self.cmb_root  = self.cmb_root_raw +  cmb_postfix_for_r(self.cosmo_r)
        self.cmbf_root  = self.cmbf_root_raw +  cmb_postfix_for_r(self.cosmo_r)
        if(self.do_r1):
            self.cmb1_root  = self.cmb_root_raw +  cmb_postfix_for_r(self.cosmo_r1)
            self.cmb1f_root  = self.cmbf_root_raw +  cmb_postfix_for_r(self.cosmo_r1)
        self.noise_root  = path.join(self.path, config.get('noise_root', r'noise/noise_'))
        self.noisef_root  = path.join(self.path, config.get('filtered_noise_root', r'noisef/noisef_'))
        self.fg_root = path.join(self.path, config.get('foreground_root', r'fg/fg_'))
        self.fgf_root = path.join(self.path, config.get('filtered_foreground_root', r'fgf/fgf_'))
        self.cmb_spectra_file = self.cmb_root_raw + "Cls.dat"
        self.do_delensing = config.get("do_delensing", True)
        self.nmaps = config["nmaps"]
        assert(self.nmaps > 1)
        self.num_seasons = config["num_seasons"]
        assert(self.num_seasons > 0)          
        self.num_season_pairs = self.num_seasons*(self.num_seasons-1) // 2        
        self.season_fac = np.sqrt(float(self.num_seasons))        
        self.nside = config['nside']
        assert(self.nside == 128 or self.nside == 256 or self.nside == 512 or self.nside == 1024) #default resolutions for small aperture telescopes
        self.npix = 12*self.nside**2
        self.lmax = config.get("lmax", self.nside*3-1)
#        assert(self.lmax >= self.nside * 3)  
        self.coordinate = config.get('coordinate', r'G')
        assert(self.coordinate == "G" or self.coordinate == "C" or self.coordinate == "E")
        self.freqs = np.array(config['freqs'])
        self.freqnames = [ float2intstr(freq)+'GHz' for freq in self.freqs ]
        self.fwhms_arcmin = np.array(config['fwhms'])
        self.fwhms_deg = self.fwhms_arcmin / 60.
        self.fwhms_rad = self.fwhms_deg * (np.pi/180.)
        self.beam_sigmas = self.fwhms_rad/np.sqrt(8.*np.log(2.))
        self.cosmo_ombh2 = config.get(r'cosmo_ombh2', cosmology_base_mean[0])
        self.cosmo_omch2 = config.get(r'cosmo_omch2', cosmology_base_mean[1])
        self.cosmo_theta = config.get(r'cosmo_theta', cosmology_base_mean[2])
        self.cosmo_tau = config.get(r'cosmo_tau', cosmology_base_mean[3])
        self.cosmo_logA = config.get(r'cosmo_logA', cosmology_base_mean[4])
        self.cosmo_ns = config.get(r'cosmo_ns', cosmology_base_mean[5])
        self.freq_decorr_model =config.get("freq_decorr_model", "None") #None for assuming no frequency decorrelation, ME for Moment Expansion, or Taylor for Taylor expansion, 
        self.ME_is_positive = config.get(r"ME_is_positive", True)
        self.ME_upperbound = config.get(r"ME_upperbound", 0.05)
        self.r_lowerbound = config.get(r'r_lowerbound', 0.)
        self.beta_d_prior = config.get(r"beta_d_prior", None)
        self.beta_s_prior = config.get(r"beta_s_prior", None)
        self.ds_cross_prior = config.get(r"ds_cross_prior", 0.03) #this sets the expected order of magnitude for dust x synchrotron correlation, the default value 0.03 is estimated from pysm simulations
        self.beta_prime_prior = config.get(r"beta_prime_prior", 0.02) #this sets the expected order of magnitude for beta running, the default 0.02 is estimated from pysm simulations         
        self.lcdm_params = np.array( [ self.cosmo_ombh2, self.cosmo_omch2, self.cosmo_theta, self.cosmo_tau, self.cosmo_logA, self.cosmo_ns, self.cosmo_r ] )
        if(self.do_r1):
            self.lcdm1_params = np.array( [ self.cosmo_ombh2, self.cosmo_omch2, self.cosmo_theta, self.cosmo_tau, self.cosmo_logA, self.cosmo_ns, self.cosmo_r1 ] )
            self.r_lndet_fac = config.get("r_lndet_fac", 1.)  #likelihood = exp(-chi^2/2 -ln det(C)*fac/2)
        self.camb_approx_prepared = False
        self.num_freqs = len(self.freqs)
        assert(self.num_freqs > 0)        
        assert(len(self.fwhms_rad) == self.num_freqs)
        if(self.coordinate != 'G'):
            self.rotator = hp.Rotator(coord = ['G', self.coordinate])
        else:
            self.rotator = None
        self.output_dir = config['output_dir']        
        mkdir_if_not_exists(self.output_dir)
        self.output_root = path.join(self.output_dir, path.basename(self.root))
        self.ell_cross_range = config.get('ell_cross_range', 0)
        self.analytic_fg = config.get('analytic_fg', False)
        self.analytic_ME = config.get('analytic_ME', False)
        if(self.analytic_fg):
            if(self.freq_decorr_model == "Taylor"):
                print(r'(analtyic_fg = True) is incompatible with (freq_decorr_model = "Taylor"). Please set freq_decorr_model = "ME" or "None"')
                exit()
            if(self.freq_decorr_model =="ME" and not self.analytic_ME):
                print(r'Sorry: using analytic FG but piecewise ME is nonsense, please set analytic_ME = True')
                exit()
        self.vary_cosmology = config.get('vary_cosmology', True)
        self.clean_noise_cov = config.get('clean_noise_cov', True)
        if('noise_weights' in config.keys()):
            self.noise_weights = config['noise_weights']
            assert(len(self.noise_weights) == self.num_freqs)            
            if(self.verbose):
                print("rescaliing noise maps with: ", self.noise_weights)
        else:
            self.noise_weights = None
        self.assume_parity = config.get('assume_parity', True)
        self.fields = config.get("fields", ['BB'])
        self.power_calc = band_power_calculator(lmin = config.get("band_lmin", 21), lmax = config.get("band_lmax", 201), delta_ell = config.get("band_delta_ell", 20), mask_file = self.mask_file, like_fields = self.fields, coordinate = self.coordinate)
        self.freq_lowest = self.freqs[0]
        self.freq_highest = self.freqs[0]
        self.ifreq_lowest = 0
        self.ifreq_highest = 0
        for i in range(1, self.num_freqs):
            if(self.freqs[i] < self.freq_lowest):
                self.freq_lowest = self.freqs[i]
                self.ifreq_lowest = i
            if(self.freqs[i] > self.freq_highest):
                self.freq_highest = self.freqs[i]
                self.ifreq_highest = i
        self.band_weights = []
        self.has_band_weights = False
        for ifreq in range(self.num_freqs):
            fname = self.band_weights_root + self.freqnames[ifreq]+r'_weights.txt'
            if(path.exists(fname)):
                weights = np.loadtxt(fname)
                weights[:, 1] /= np.sum(weights[:, 1])
                self.band_weights.append(weights)
                self.has_band_weights = True
            else:
                self.band_weights.append( np.array( [[self.freqs[ifreq],  1.]] )  )
        if(self.has_band_weights and self.verbose):
            for ifreq in range(self.num_freqs):
                print("band weights for ", self.freqnames[ifreq])
                print(self.band_weights[ifreq])
        self.num_powers = self.num_freqs * (self.num_freqs + 1) // 2
        self.num_ells = self.power_calc.num_ells
        if(self.ell_cross_range > self.num_ells - 1):
            self.ell_cross_range = self.num_ells - 1
        self.num_fields = self.power_calc.num_fields
        self.nside = self.power_calc.nside
        self.ells = np.empty(self.power_calc.num_ells, dtype = np.int32)
        for i in range(self.num_ells):
            self.ells[i] = int(np.round(self.power_calc.ells[i]))
        self.TT_loc = self.power_calc.TT_loc
        self.TE_loc = self.power_calc.TE_loc
        self.EE_loc = self.power_calc.EE_loc
        self.TB_loc = self.power_calc.TB_loc
        self.EB_loc = self.power_calc.EB_loc
        self.BB_loc = self.power_calc.BB_loc
        self.blocksize = self.num_fields * self.num_powers #
        self.fullsize =  self.blocksize * self.num_ells        
        self.dofs = self.power_calc.dofs.copy()
        self.model_sxn_cov = config.get("model_sxn_cov", False)
        #now work on total covariance cleaning
        self.cov_filter = np.zeros((self.fullsize, self.fullsize))
        for ib in range(self.num_ells):
            self.cov_filter[ib*self.blocksize:(ib+1)*self.blocksize, ib*self.blocksize:(ib+1)*self.blocksize] = 1.  #diagonal blocks
        if(self.ell_cross_range == 1):        #correlation between neighboring bins
            if(self.verbose):
                print('cleaning the covariance assuming only next-bin correlations')            
            q = corr_nb(self.num_ells)            
            for ib in range(self.num_ells-1):
                self.cov_filter[ib*self.blocksize:(ib+1)*self.blocksize, (ib+1)*self.blocksize:(ib+2)*self.blocksize] = q  
                self.cov_filter[(ib+1)*self.blocksize:(ib+2)*self.blocksize, ib*self.blocksize:(ib+1)*self.blocksize] = q
        elif(self.ell_cross_range == 2):
            if(self.verbose):
                print('cleaning the covariance assuming only up to next-to-next-bin correlations')                        
            q, p = corr_nnb(self.num_ells)
            for ib in range(self.num_ells-1):
                self.cov_filter[ib*self.blocksize:(ib+1)*self.blocksize, (ib+1)*self.blocksize:(ib+2)*self.blocksize] = q  
                self.cov_filter[(ib+1)*self.blocksize:(ib+2)*self.blocksize, ib*self.blocksize:(ib+1)*self.blocksize] = q
            for ib in range(self.num_ells-2):
                self.cov_filter[ib*self.blocksize:(ib+1)*self.blocksize, (ib+2)*self.blocksize:(ib+3)*self.blocksize] = p  
                self.cov_filter[(ib+2)*self.blocksize:(ib+3)*self.blocksize, ib*self.blocksize:(ib+1)*self.blocksize] = p
        elif(self.ell_cross_range > 2):
            if(self.verbose):
                print('warning: all ell-bins are assumed to be correlated, results may not be stable if you do not have sufficient number of simulations')                        
            self.cov_filter = np.ones((self.fullsize, self.fullsize))
        elif(self.verbose):
            print('cleaning the covariance by ignoring bin-to-bin correlations')            
        if(self.assume_parity):
            if(self.verbose):
                print('cleaning the covariance assuming B uncorrelated with T and E')
            if(self.BB_loc >= 0): 
                for i in range(self.num_fields):
                    if(i == self.BB_loc):
                        continue
                    self.cov_filter[i : self.fullsize : self.num_fields, self.BB_loc : self.fullsize : self.num_fields] = 0.
                    self.cov_filter[self.BB_loc : self.fullsize : self.num_fields, i : self.fullsize : self.num_fields] = 0.
            if(self.EB_loc >= 0): 
                for i in range(self.num_fields):
                    if(i == self.EB_loc or i == self.BB_loc):
                        continue
                    self.cov_filter[i : self.fullsize : self.num_fields, self.EB_loc : self.fullsize : self.num_fields] = 0.
                    self.cov_filter[self.EB_loc : self.fullsize : self.num_fields, i : self.fullsize : self.num_fields] = 0.
            if(self.TB_loc >= 0): 
                for i in range(self.num_fields):
                    if(i==self.TB_loc or i == self.EB_loc or i == self.BB_loc):
                        continue
                    self.cov_filter[i : self.fullsize : self.num_fields, self.TB_loc : self.fullsize : self.num_fields] = 0.
                    self.cov_filter[self.TB_loc : self.fullsize : self.num_fields, i : self.fullsize : self.num_fields] = 0.
        if(self.clean_noise_cov):
            if(self.verbose):
                print('cleaning noise covariance')
            #now work on noise covariance cleaning            
            self.noise_cov_filter = np.zeros((self.fullsize, self.fullsize))
            for i in range(self.blocksize):  #must be same freq pair and same field, assuming noise T, E, B are uncorrelated, and different freq noise maps are uncorrelated
                self.noise_cov_filter[i : self.fullsize : self.blocksize, i : self.fullsize : self.blocksize ] = 1.
        else:
            self.noise_cov_filter = np.ones((self.fullsize, self.fullsize))
        #now work on data mask
        self.data_mask = np.ones(self.fullsize, dtype=np.int32)            
        if(config.get("data_mask", None) is None):
            self.do_data_mask = False
        else:
            self.do_data_mask = True
            for lfs in config["data_mask"]:  # [ [ell (None), freq1 (None), freq2 (None)], ...]
                if(lfs[0] is None):
                    if(lfs[1] is None):  # mask a frequency
                        f2_loc = -1
                        for ifreq in range(self.num_freqs):
                            if(abs(lfs[2] - self.freqs[ifreq]) < 0.1):
                                f2_loc = ifreq
                                break
                        if( f2_loc >= 0):
                            for f1_loc in range(self.num_freqs):
                                ipower =  total_index(self.num_freqs, f1_loc, f2_loc)
                                for ell_loc in range(self.num_ells):
                                    self.data_mask[ell_loc * self.blocksize + ipower * self.num_fields : ell_loc * self.blocksize + (ipower+1) * self.num_fields ] = 0                    
                    else:  #mask a frequency pair
                        f1_loc = -1
                        f2_loc = -1
                        for ifreq in range(self.num_freqs):
                            if(abs(lfs[1] - self.freqs[ifreq]) < 1.):
                                f1_loc = ifreq
                            if(abs(lfs[2] - self.freqs[ifreq]) < 1.):
                                f2_loc = ifreq                    
                        if(f1_loc >= 0 and f2_loc >= 0):
                            ipower =   total_index(self.num_freqs, f1_loc, f2_loc)
                            for ell_loc in range(self.num_ells):
                                self.data_mask[ell_loc * self.blocksize + ipower * self.num_fields : ell_loc * self.blocksize + (ipower+1) * self.num_fields ] = 0                    
                elif(lfs[1] is None and lfs[2] is None):  #mask ell for all frequency pairs
                    ell_loc = -1
                    for il in range(self.num_ells):
                        if( abs(lfs[0] - self.power_calc.ells[il]) <= self.power_calc.delta_ell/2.):
                            ell_loc = il
                            break                    
                    self.data_mask[ell_loc*self.blocksize: (ell_loc+1)*self.blocksize] = 0
                else:    #mask (ell, freq1, freq2)
                    ell_loc = -1
                    f1_loc = -1
                    f2_loc = -1
                    for il in range(self.num_ells):
                        if( abs(lfs[0] - self.power_calc.ells[il]) <= self.power_calc.delta_ell/2.):
                            ell_loc = il
                            break
                    for ifreq in range(self.num_freqs):
                        if(abs(lfs[1] - self.freqs[ifreq]) < 1.):
                            f1_loc = ifreq
                        if(abs(lfs[2] - self.freqs[ifreq]) < 1.):
                            f2_loc = ifreq                    
                    if(ell_loc >= 0  and f1_loc >= 0 and f2_loc >= 0):
                        ipower =  total_index(self.num_freqs, f1_loc, f2_loc)
                        self.data_mask[ell_loc * self.blocksize + ipower * self.num_fields : ell_loc * self.blocksize + (ipower+1) * self.num_fields ] = 0
        self.used_indices = (self.data_mask == 1)
        self.ell_used_indices = np.empty( self.num_ells, dtype=bool)
        for il in range(self.num_ells):
            self.ell_used_indices[il] = any(self.used_indices[il*self.blocksize : (il+1) * self.blocksize])
        self.used_size = sum(self.data_mask)
        #now allocate the covariances for noise and total
        self.noise_cov_computed = False
        self.signal_cov_computed = False        
        self.invcov_computed = False
        self.filters_computed = False
        self.filters = None  #this is not known at initialization
        self.lens_weights = {}
        self.lens_mean = {}
        self.lens_res = np.zeros(self.fullsize)
        for field in self.fields:
            self.lens_weights[field] = np.zeros( (self.num_ells, self.num_freqs) )
            self.lens_mean[field] = np.zeros( (self.nmaps, self.num_ells) )
        if(self.do_r1):
            self.lens1_mean = {}
            self.lens1_res = np.zeros(self.fullsize)
            for field in self.fields:
                self.lens1_mean[field] = np.zeros( (self.nmaps, self.num_ells)) 
        self.lens_weights_computed = False
        self.data_product_path = config.get('data_product_path', None)
        if(self.data_product_path is not None):
            mkdir_if_not_exists(self.data_product_path)
        if(self.verbose):
            print(r'sky_analyser root = ', self.root)
            print('using fields:', self.fields)                        
            print('mask file = ', self.mask_file)
            print(r'using frequencies [GHz] :', self.freqs)
            print(r'FWHMs [arcmin] = ', self.fwhms_arcmin)
            print(r'number of simulation maps = ', self.nmaps)            
            print(r'using ells = ', self.ells[self.ell_used_indices])           
            print(r'number of power spectra :', self.num_powers)
            print(r'number of ell blocks :', self.num_ells)
            print(r'size of each block :', self.blocksize)               
            print(r'data size = ', self.fullsize)
            if(self.used_size != self.fullsize):
                print('used data size', self.used_size)


    def matrix_inv(self, mat):
        if(self.do_data_mask):
            return np.linalg.inv(mat[self.used_indices][:, self.used_indices] * self.cov_filter[self.used_indices][:, self.used_indices] )            
        if(self.ell_cross_range > 0):
            return np.linalg.inv(mat *self.cov_filter)
        matinv = np.zeros((self.fullsize, self.fullsize))
        for i in range(self.num_ells):
            base = i*self.blocksize
            matinv[base : base+self.blocksize, base : base+self.blocksize] = np.linalg.inv(mat[base : base+self.blocksize, base : base+self.blocksize] * self.cov_filter[base : base+self.blocksize, base : base+self.blocksize])
        return matinv

    def matrix_logdet(self, mat):
        if(self.do_data_mask or self.ell_cross_range > 0):
            s, logdet = np.linalg.slogdet(mat)
            if(s <= 0.):
                print('Error: covmat is not positive definite')
                exit()
            return logdet
        logdet = 0.
        for i in range(self.num_ells):
            base = i*self.blocksize            
            s, this_logdet = np.linalg.slogdet(mat[base : base+self.blocksize, base : base+self.blocksize])
            if(s <= 0.):
                print('Error: covmat is not positive definite')
                exit()
            logdet += this_logdet
        return logdet
        
    def files_exist(self, prefix, postfix):
        for ifreq in range(self.num_freqs):
            if(not path.exists(prefix +  self.freqnames[ifreq] + postfix)):
                return False
        return True
    
    def save_map(self, filename, maps, overwrite):
        if((not overwrite)  and path.exists(filename)):
            print('Warning: skipping '+filename+' that already exists...')
            return
        if(filename[-5:] == ".fits"):
            hp.write_map(filename = filename, m = maps, column_units = 'uK', overwrite = overwrite, extra_header=[ ("COORDSYS", self.coordinate) ], dtype=np.float32)
        elif(filename[-4:] == ".npy"):
            np.save(filename, maps[:, self.mask_ones])
        else:
            print('Cannot save unknown map format: ' + filename)
            exit()
                

    def field_index(self, field):
        for i in range(self.num_fields):
            if(field == self.fields[i]):
                return i
        return None
    
    def power_index(self, ifreq1, ifreq2):
        imin = min(ifreq1, ifreq2)
        idiff = abs(ifreq1 -  ifreq2)
        return (2*self.num_freqs - idiff + 1)*idiff // 2 + imin

    def freq_indices(self, ipower):
        idiff = ceil(self.num_freqs + 0.5 - np.sqrt((self.num_freqs + 0.5)**2 - 2.*ipower))
        imin = ipower - ((2*self.num_freqs - idiff + 1)*idiff // 2)
        #just be safe (round-off errors)
        if(imin < 0):
            idiff -= 1
            imin = ipower - ((2*self.num_freqs - idiff + 1)*idiff // 2)
        elif(imin + idiff >= self.num_freqs):
            idiff += 1
            imin = ipower - ((2*self.num_freqs - idiff + 1)*idiff // 2)
        return imin, imin + idiff

    def beam_filter_freq(self, ifreq):
        return np.exp(-self.power_calc.ells * (self.power_calc.ells + 1.) * (self.beam_sigmas[ifreq]**2)/2.)        
    
    def beam_filter2(self, ifreq1, ifreq2):
        return np.exp(-self.power_calc.ells * (self.power_calc.ells + 1.) * (self.beam_sigmas[ifreq1]**2 + self.beam_sigmas[ifreq2]**2)/2.)

    def beam_filter(self, ipower):
        ifreq1, ifreq2 = self.freq_indices(ipower)
        return self.beam_filter2(ifreq1, ifreq2)

    def full_vector(self, prefix, postfix = "", overwrite = False, do_seasons = [], weights = None):  
        if(isinstance(prefix, list)):
            npre = len(prefix)
            prelist = prefix
        else:
            npre = 1
            prelist = [ prefix ]
        if(isinstance(postfix, list)):
            npost = len(postfix)
            postlist = postfix
        else:
            npost = 1
            postlist = [postfix]
        if(npost < npre): #fill with ""
            nlist = npre
            for i in range(npre - npost):
                postlist.append("")
        elif(npre < npost):
            nlist = npost
            for i in range(npost - npre):
                prelist.append("")
        else:
            nlist = npre
        has_seasons = np.any(do_seasons)
        if(len(do_seasons) < nlist):
            for i in range(nlist - len(do_seasons)):
                do_seasons.append(False)                
        vec = np.zeros(self.fullsize)
        if(weights is None):
            w = np.ones(self.num_freqs)
        else:
            w = weights
        for ipower in range(self.num_powers):
            ifreq1, ifreq2 = self.freq_indices(ipower)
            if(ifreq1 == ifreq2 and has_seasons):
                for isea1 in range(self.num_seasons-1):
                    for isea2 in range(isea1+1, self.num_seasons):
                        bp = self.power_calc.band_power(mapfile1 = [ prelist[i] + self.freqnames[ifreq1] + postlist[i] + (r'_season' + str(isea1) if do_seasons[i] else "") + self.mapform for i in range(nlist) ], mapfile2 = [prelist[i] + self.freqnames[ifreq2] + postlist[i] + (r'_season' + str(isea2) if do_seasons[i] else "") + self.mapform for i in range(nlist) ], overwrite = overwrite, w1 = w[ifreq1], w2 = w[ifreq2])                            
                        for ifield in range(self.num_fields):
                            vec[ipower*self.num_fields + ifield : self.fullsize : self.blocksize ] += bp[self.fields[ifield]]
                for ifield in range(self.num_fields):
                    vec[ipower*self.num_fields + ifield : self.fullsize : self.blocksize ] /=  self.num_season_pairs
            else:
                bp = self.power_calc.band_power(mapfile1 = [ prelist[i] + self.freqnames[ifreq1] + postlist[i] + self.mapform for i in range(nlist) ], mapfile2 = [ prelist[i] + self.freqnames[ifreq2]+postlist[i] + self.mapform for i in range(nlist) ], overwrite = overwrite, w1 = w[ifreq1], w2 = w[ifreq2])                    
                for ifield in range(self.num_fields):
                    vec[ipower*self.num_fields + ifield : self.fullsize : self.blocksize ] = bp[self.fields[ifield]]
        return vec


    def cmb_vector(self, lcdm_params):
        vec_unlensed = np.empty(self.fullsize)
        vec_lensed = np.empty(self.fullsize)
        bp_unlensed, bp_lensed = get_camb_cls(self.ells, lcdm_params = lcdm_params, fields = self.fields)
        for ipower in range(self.num_powers):
            fil = self.beam_filter(ipower)
            for ifield in range(self.num_fields):
                vec_unlensed[ipower*self.num_fields + ifield : self.fullsize : self.blocksize] = bp_unlensed[self.fields[ifield]] * fil
                vec_lensed[ipower*self.num_fields + ifield : self.fullsize : self.blocksize] = bp_lensed[self.fields[ifield]] * fil
        return vec_unlensed, vec_lensed

    
    def set_camb_approx(self):
        self.cmb_unlensed_base, self.cmb_lensed_base = self.cmb_vector(lcdm_params = self.lcdm_params)
        self.cmb_unlensed_derv = np.empty( (cosmology_num_params, self.fullsize) )
        self.cmb_lensed_derv = np.empty( (cosmology_num_params, self.fullsize) )                  
        params = self.lcdm_params.copy()
        if(self.vary_cosmology):
            for i in range(cosmology_num_params-1):  #not including r
                params[i] = self.lcdm_params[i] + cosmology_base_std[i]
                unlensed_up, lensed_up = self.cmb_vector(lcdm_params = params)
                params[i] = max(self.lcdm_params[i] - cosmology_base_std[i], 0.)
                dp = self.lcdm_params[i] + cosmology_base_std[i] - params[i]
                unlensed_down, lensed_down = self.cmb_vector(lcdm_params = params)
                self.cmb_unlensed_derv[i, :] = (unlensed_up - unlensed_down)/dp
                self.cmb_lensed_derv[i, :] = (lensed_up - lensed_down)/dp
                params[i] = self.lcdm_params[i]
        else:
            for i in range(cosmology_num_params-1):  #not including r
                self.cmb_unlensed_derv[i, :] = np.zeros(self.fullsize)
                self.cmb_lensed_derv[i, :] = np.zeros(self.fullsize)
        if(self.do_r1):
            self.cmb1_unlensed_base, self.cmb1_lensed_base = self.cmb_vector(lcdm_params = self.lcdm1_params)                    
            self.cmb_unlensed_derv[cosmology_num_params-1, :] = (self.cmb1_unlensed_base - self.cmb_unlensed_base)/(self.cosmo_r1 - self.cosmo_r)
            self.cmb_lensed_derv[cosmology_num_params-1, :] = (self.cmb1_lensed_base - self.cmb_lensed_base)/(self.cosmo_r1 - self.cosmo_r )            
        else:
            if(self.cosmo_r > 0.005):
                dp =  - self.cosmo_r
            else:
                dp = 0.01
            params[cosmology_num_params-1] = self.cosmo_r + dp
            unlensed_up, lensed_up = self.cmb_vector(lcdm_params = params)
            self.cmb_unlensed_derv[cosmology_num_params-1, :] = (unlensed_up - self.cmb_unlensed_base)/dp
            self.cmb_lensed_derv[cosmology_num_params-1, :] = (lensed_up - self.cmb_lensed_base)/dp            
        self.camb_approx_prepared = True

    def cmb_vector_approx(self, lcdm_params):
        if(not self.camb_approx_prepared):
            self.set_camb_approx()
        vec_unlensed = self.cmb_unlensed_base.copy()
        vec_lensed = self.cmb_lensed_base.copy() 
        for i in range(cosmology_num_params):
            vec_unlensed += (lcdm_params[i] - self.lcdm_params[i]) * self.cmb_unlensed_derv[i, :]
            vec_lensed += (lcdm_params[i] - self.lcdm_params[i]) * self.cmb_lensed_derv[i, :]
        return vec_unlensed, vec_lensed

    def select_spectrum(self, vec, ifreq1, ifreq2, field):
        ipower = self.power_index(ifreq1, ifreq2)
        ifield = self.field_index(field)
        if(ifield is None):
            print('Error in select_spectrum: field ' + field + ' is not used!')
            exit()
        return vec[ipower*self.num_fields + ifield : self.fullsize : self.blocksize]

    def fg_band_weights(self, beta_d, beta_s):
        fg = foreground_model(ells = self.ells, beta_dust = beta_d, beta_sync = beta_s, freq_dust_ref = self.freq_highest, freq_sync_ref = self.freq_lowest)
        dust_weights = np.zeros(self.num_freqs)
        sync_weights = np.zeros(self.num_freqs)
        if(self.has_band_weights):
            for ifreq  in range(self.num_freqs):
                for iw in range(self.band_weights[ifreq].shape[0]):
                    dust_weights[ifreq] += fg.dust_freq_weight(self.band_weights[ifreq][iw, 0])*self.band_weights[ifreq][iw, 1]
                    sync_weights[ifreq] += fg.sync_freq_weight(self.band_weights[ifreq][iw, 0])*self.band_weights[ifreq][iw, 1]                    
        else:
            for ifreq in range(self.num_freqs):
                dust_weights[ifreq] = fg.dust_freq_weight(self.freqs[ifreq])
                sync_weights[ifreq] = fg.sync_freq_weight(self.freqs[ifreq])
        return dust_weights, sync_weights

        

    def fg_model_vector(self, fgs): #fgs is a list of foreground models
        vec = np.empty(self.fullsize)
        for ipower in range(self.num_powers):
            ifreq1, ifreq2 = self.freq_indices(ipower)
            if(self.has_band_weights):
                for fld in range(self.num_fields):
                    vec[ipower * self.num_fields  + fld  : self.fullsize : self.blocksize ] =  fgs[fld].full_bandpower(freq1 = self.band_weights[ifreq1][:, 0], freq2 = self.band_weights[ifreq2][:, 0], weights1 = self.band_weights[ifreq1][:, 1], weights2 = self.band_weights[ifreq2][:, 1]) * self.beam_filter(ipower)  
            else:
                for fld in range(self.num_fields):
                    vec[ipower * self.num_fields  + fld  : self.fullsize : self.blocksize ]  = fgs[fld].full_bandpower(freq1 = self.freqs[ifreq1], freq2 = self.freqs[ifreq2])*self.beam_filter(ipower) 
        return vec

    def files_exist(self, prefix, postfix):
        for ifreq in range(self.num_freqs):
            if(not path.exists(prefix +  self.freqnames[ifreq] + postfix)):
                return False
        return True

    def get_filters(self, prefix_no_filter, prefix_with_filter):
        if(self.filters_computed):
            return
        if(self.verbose):
            print('modeling TOD filtering')
        self.filters = []
        source = np.empty( (self.num_fields, self.num_powers * self.nmaps, self.num_ells) )
        target = np.empty( (self.num_fields, self.num_powers * self.nmaps, self.num_ells) )        
        for i in range(self.nmaps):
            vec_no_filter = self.full_vector( prefix = prefix_no_filter, postfix =  r'_' + str(i) )
            vec_with_filter = self.full_vector( prefix = prefix_with_filter, postfix =  r'_' + str(i) )
            for fld in range(self.num_fields):
                for ipower in range(self.num_powers):
                    source[fld, i*self.num_powers + ipower, :] = vec_no_filter[ipower * self.num_fields + fld : self.fullsize : self.blocksize]
                    target[fld, i*self.num_powers + ipower, :] = vec_with_filter[ipower * self.num_fields + fld : self.fullsize : self.blocksize]
        for fld in range(self.num_fields):
            self.filters.append(leakage_model(source[fld, :, :], target[fld, :, :]))
        if(self.BB_loc >=0 and self.EE_loc >=0):            #check if E-B leakage is making F_ell^BB insane (and approximate F_ell^BB with F_ell^EE if so)
            if( sum(self.filters[self.BB_loc].diag)/self.num_ells > 1.):
                self.filters[self.BB_loc] = self.filters[self.EE_loc]
        self.filters_computed = True

    def apply_filters(self, vec):
        if(sum(vec**2) < 1.e-14 or (self.filters is None)):  #zero vector 
            return vec
        fvec = np.empty(self.fullsize)
        for fld in range(self.num_fields):
            for ipower in range(self.num_powers):
                fvec[fld + ipower * self.num_fields: self.fullsize : self.blocksize] = self.filters[fld].project(vec[fld + ipower * self.num_fields: self.fullsize : self.blocksize])
        return fvec

    def get_lens_weights(self, overwrite = False):
        if(self.lens_weights_computed):
            return
        if(self.data_product_path is not None):
            self.lens_weights_computed = True
            for field in self.fields:
                if(path.exists(path.join(self.data_product_path, r"lens_weights_"+field+r".npy")) and path.exists(path.join(self.data_product_path, r'lens_res.npy')) and (path.exists(path.join(self.data_product_path, r'lens1_res.npy')) or not self.do_r1) ):
                    self.lens_weights[field] = np.load(path.join(self.data_product_path, r"lens_weights_"+field+r".npy"))
                else:
                    self.lens_weights_computed = False
        if(self.lens_weights_computed):
            return
        if(self.do_delensing):
            if(self.verbose):
                print('computing weights for lens templates')
            power_arrays = {}
            for field in self.fields:
                power_arrays[field] = np.empty((self.num_ells, self.nmaps, self.num_freqs))
            for i in range(self.nmaps):
                for ifreq in range(self.num_freqs):
                    bp = self.power_calc.band_power( [self.cmbf_root + self.freqnames[ifreq] + '_' + str(i) + self.mapform, self.noisef_root + self.freqnames[ifreq] + '_' + str(i) +  self.mapform, self.fgf_root + self.freqnames[ifreq] +  self.mapform ], [ self.cmb_root + 'lenstemp_' + str(i) +  self.mapform ], overwrite = overwrite )
                    for field in self.fields:
                        power_arrays[field][:, i, ifreq] = bp[field]/self.beam_filter_freq(ifreq)
            for field in self.fields:
                for l in range(self.num_ells):
                    self.lens_weights[field][l, :] = get_weights(power_arrays[field][l, :, :])
                    for imap in range(self.nmaps):
                        self.lens_mean[field][imap, l] = np.sum(power_arrays[field][l, imap, :]*self.lens_weights[field][l, :])
        if(self.data_product_path is not None):
            for field in self.fields:
                np.save(path.join(self.data_product_path, r"lens_weights_"+field+r".npy"), self.lens_weights[field])
        self.lens_res = np.zeros(self.fullsize)
        self.cmb_powers = np.empty((self.nmaps, self.fullsize))
        vec = np.empty(self.fullsize)
        for i in range(self.nmaps):
            self.cmb_powers[i, :] = self.full_vector( prefix = self.cmbf_root, postfix = r'_'+str(i) )
            if(self.do_delensing):
                for ifield in range(self.num_fields):
                    for ipower in range(self.num_powers):
                        vec[ifield + ipower*self.num_fields : self.fullsize : self.blocksize] = self.cmb_powers[i, ifield + ipower*self.num_fields : self.fullsize : self.blocksize] - self.lens_mean[self.fields[ifield]][i, :] * self.beam_filter(ipower)
                self.lens_res += vec
            else:
                self.lens_res += self.cmb_powers[i, :]
        self.lens_res /= self.nmaps
        self.lens_weights_computed = True
        if(not self.do_r1):
            return
        if(self.do_delensing):
            for i in range(self.nmaps):
                for ifreq in range(self.num_freqs):
                    bp = self.power_calc.band_power( [self.cmb1f_root + self.freqnames[ifreq] + '_' + str(i) +  self.mapform, self.noisef_root + self.freqnames[ifreq] + '_' + str(i) +  self.mapform, self.fgf_root + self.freqnames[ifreq] +  self.mapform], [ self.cmb1_root + 'lenstemp_' + str(i) +  self.mapform ], overwrite = overwrite )
                    for field in self.fields:
                        power_arrays[field][:, i, ifreq] = bp[field]/self.beam_filter_freq(ifreq)
            for field in self.fields:
                for l in range(self.num_ells):
                    for imap in range(self.nmaps):
                        self.lens1_mean[field][imap, l] = np.sum(power_arrays[field][l, imap, :]*self.lens_weights[field][l, :])
        self.lens1_res = np.zeros(self.fullsize)
        self.cmb1_powers = np.empty((self.nmaps, self.fullsize))
        vec = np.empty(self.fullsize)
        for i in range(self.nmaps):
            self.cmb1_powers[i, :] = self.full_vector( prefix = self.cmb1f_root, postfix = r'_'+str(i) )
            if(self.do_delensing):
                for ifield in range(self.num_fields):
                    for ipower in range(self.num_powers):
                        vec[ifield + ipower*self.num_fields : self.fullsize : self.blocksize] = self.cmb1_powers[i, ifield + ipower*self.num_fields : self.fullsize : self.blocksize] - self.lens1_mean[self.fields[ifield]][i, :] * self.beam_filter(ipower)
                self.lens1_res += vec
            else:
                self.lens1_res += self.cmb1_powers[i, :]
        self.lens1_res /= self.nmaps
        

    def get_data_vector(self, overwrite = False):
        if(self.verbose):
            print('computing data vector')
        if(not self.lens_weights_computed):
            self.get_lens_weights(overwrite = overwrite)
        self.data_vec = self.full_vector(prefix = self.root, do_seasons = [True], overwrite = overwrite)
        if(not self.do_delensing):
            return
        power_arrays = {}
        lens_power = {}
        for field in self.fields:
            power_arrays[field] = np.empty((self.num_ells, self.num_freqs))
            lens_power[field] = np.empty(self.num_ells)            
        for ifreq in range(self.num_freqs):
            bp = self.power_calc.band_power( self.root + self.freqnames[ifreq] +  self.mapform, self.root + 'lenstemp' +  self.mapform, overwrite = overwrite  )
            for field in self.fields:
                power_arrays[field][:, ifreq] = bp[field]/self.beam_filter_freq(ifreq)
        for field in self.fields:
            for l in range(self.num_ells):
                lens_power[field][l] = np.sum(power_arrays[field][l, :] * self.lens_weights[field][l, :])
        for ifield in range(self.num_fields):
            for ipower in range(self.num_powers):
                self.data_vec[ifield + ipower*self.num_fields : self.fullsize : self.blocksize] -= lens_power[self.fields[ifield]] * self.beam_filter(ipower)
        del power_arrays
        del lens_power

                
    def get_noise_cov(self):
        if(self.noise_cov_computed):
            return
        if(self.verbose):
            print(r'computing noise covariance...')
        self.noise_cov = np.zeros( (self.fullsize, self.fullsize) )
        self.noises = np.empty( (self.nmaps, self.fullsize))
        self.noise_mean = np.zeros(self.fullsize)
        for i in range(self.nmaps):
            self.noises[i, :] = self.full_vector(prefix = self.noisef_root, postfix =  r'_' + str(i), do_seasons = [True], weights = self.noise_weights )
        for i in range(self.fullsize):
            for j in range(i, self.fullsize):
                if(self.cov_filter[i, j] > 0. and self.noise_cov_filter[i,j] > 0.):
                    self.noise_cov[i, j] = np.sum(self.noises[:, i] * self.noises[:, j])/self.nmaps  #the mean is known to be zero
                if(i != j):
                    self.noise_cov[j, i] = self.noise_cov[i, j]
        self.noise_cov_computed = True


    def get_signal_cov(self):
        if(self.signal_cov_computed):
            return
        if(not self.lens_weights_computed):
            self.get_lens_weights()
        if(self.verbose):
            print('computing signal covariance...')
        self.signal_mean = np.zeros(self.fullsize)
        self.signal_cov = np.zeros((self.fullsize, self.fullsize))
        self.signals = np.empty((self.nmaps, self.fullsize))
        for i in range(self.nmaps):
            post = r'_' + str(i)
            self.signals[i, :] = self.full_vector(prefix = [ self.cmbf_root, self.fgf_root ], postfix =  [ post, "" ] )
            if(self.do_delensing):
                for ifield in range(self.num_fields):
                    for ipower in range(self.num_powers):
                        self.signals[i, ifield + ipower*self.num_fields : self.fullsize : self.blocksize] -= self.lens_mean[self.fields[ifield]][i, :]*self.beam_filter(ipower)
            self.signal_mean += self.signals[i, :]
        self.signal_mean /= self.nmaps
        for i in range(self.fullsize):
            for j in range(i, self.fullsize):
                if(self.cov_filter[i, j] > 0.):
                    self.signal_cov[i, j] = np.sum((self.signals[:, i] - self.signal_mean[i])*(self.signals[:, j] - self.signal_mean[j]))/(self.nmaps-1.)
                    if(i != j):
                        self.signal_cov[j, i] = self.signal_cov[i, j]
        self.signal_cov_computed = True
        if(not self.do_r1):
            return
        self.signal1_mean = np.zeros(self.fullsize)
        self.signal1_cov = np.zeros((self.fullsize, self.fullsize))
        self.signals1 = np.empty((self.nmaps, self.fullsize))
        for i in range(self.nmaps):
            post = r'_' + str(i)
            self.signals1[i, :] = self.full_vector(prefix = [ self.cmb1f_root, self.fgf_root ], postfix =  [ post, "" ] )
            if(self.do_delensing):
                for ifield in range(self.num_fields):
                    for ipower in range(self.num_powers):
                        self.signals1[i, ifield + ipower*self.num_fields : self.fullsize : self.blocksize] -= self.lens_mean[self.fields[ifield]][i, :]*self.beam_filter(ipower)
            self.signal1_mean += self.signals1[i, :]
        self.signal1_mean /= self.nmaps
        for i in range(self.fullsize):
            for j in range(i, self.fullsize):
                if(self.cov_filter[i, j] > 0.):
                    self.signal1_cov[i, j] = np.sum((self.signals1[:, i] - self.signal1_mean[i])*(self.signals1[:, j] - self.signal1_mean[j]))/(self.nmaps-1.)
                    if(i != j):
                        self.signal1_cov[j, i] = self.signal1_cov[i, j]


    def set_invcov_interp(self):
        if(not self.do_r1):
            return
        self.invcov_interp = np.empty( (self.num_r_interp, self.used_size, self.used_size) )
        self.invcov_lndet = np.zeros(self.num_r_interp)
        self.invcov_interp[0, :, :] = self.invcov
        self.invcov_lndet_base = self.matrix_logdet(self.invcov)
        for i in range(1, self.num_r_interp):
            fac = i/(self.num_r_interp-1.)            
            self.invcov_interp[i, :, :] = self.matrix_inv(self.sxn1_cov * fac + self.sxn_cov * (1.-fac) + self.signal1_cov*fac**2 + self.signal_cov*(1.-fac**2) + self.noise_cov )
        if(self.r_lndet_fac > 0.):
            for i in range(1, self.num_r_interp):
                self.invcov_lndet[i] = (self.matrix_logdet(self.invcov_interp[i, :, :]) - self.invcov_lndet_base)
            if(self.verbose):
                print("log det(Cov) corrections:")
                print(self.invcov_lndet)
            

            
    #get mean and covariance for (total Cl  - noise Cl - signal Cl)
    def get_covmat(self):
        if(self.data_product_path is not None and path.exists( path.join(self.data_product_path, r'filters.pickle')) ):  #try load from saved 
            self.lens_res = np.load(path.join(self.data_product_path, r'lens_res.npy'))
            self.mean = np.load(path.join(self.data_product_path, r'mean.npy'))
            self.noise_cov = np.load(path.join(self.data_product_path, r'noise_cov.npy'))
            self.signal_cov = np.load(path.join(self.data_product_path, r'signal_cov.npy'))
            self.sxn_cov = np.load(path.join(self.data_product_path, r'sxn_cov.npy'))
            self.covmat =  self.noise_cov  + self.signal_cov + self.sxn_cov
            self.invcov = self.matrix_inv(self.covmat)
            with open(path.join(self.data_product_path, r'filters.pickle'), 'rb') as f:
                self.filters = pickle.load(f)    
            if(self.do_r1):
                self.lens1_res = np.load(path.join(self.data_product_path, r'lens1_res.npy'))
                self.mean1 = np.load(path.join(self.data_product_path, r'mean1.npy'))
                self.signal1_cov =np.load( path.join(self.data_product_path, r'signal1_cov.npy'))
                self.sxn1_cov =np.load( path.join(self.data_product_path, r'sxn1_cov.npy'))
                self.set_invcov_interp()
            self.invcov_computed = True                
        if(self.invcov_computed):
            return
        self.get_noise_cov()        
        if(not self.lens_weights_computed):
            self.get_lens_weights()
        self.get_signal_cov()
        if(self.verbose):
            print(r'computing total covmat')            
        self.sxn_cov = np.zeros((self.fullsize, self.fullsize))
        self.totals = np.empty((self.nmaps, self.fullsize))
        self.mean = np.zeros(self.fullsize)
        for i in range(self.nmaps):
            post = r'_' + str(i)
            self.totals[i, :] = self.full_vector(prefix = [ self.noisef_root, self.cmbf_root, self.fgf_root ], postfix =  [ post, post, "" ], do_seasons = [True, False, False], weights = self.noise_weights )
            if(self.do_delensing):
                for ifield in range(self.num_fields):  
                    for ipower in range(self.num_powers):
                        self.totals[i, ifield + ipower*self.num_fields : self.fullsize : self.blocksize] -= self.lens_mean[self.fields[ifield]][i, :]*self.beam_filter(ipower)
            self.mean += self.totals[i, :]
        self.mean /= self.nmaps
        for il in range(self.num_ells):
            base = il * self.blocksize
            for i in range(base, base+self.blocksize):
                for j in range(i, base+self.blocksize):
                    self.sxn_cov[i, j] = np.sum((self.totals[:, i] - self.noises[:, i] - self.signals[:, i])*(self.totals[:, j] - self.noises[:, j] - self.signals[:, j]))/self.nmaps 
                    if(i != j):
                        self.sxn_cov[j, i] = self.sxn_cov[i, j]
        if(self.model_sxn_cov and self.BB_loc >= 0):            
            Nl = np.empty(self.num_freqs)
            for il in range(self.num_ells):
                base = il*self.blocksize
                for ifreq in range(self.num_freqs):
                    k = base+self.power_index(ifreq, ifreq)*self.num_fields+self.BB_loc
                    Nl[ifreq] = self.sxn_cov[k, k]/2./self.signal_mean[k]
                for ipower12 in range(self.num_powers):
                    ifreq1, ifreq2 = self.freq_indices(ipower12)
                    k1 = base + ipower12*self.num_fields+self.BB_loc                    
                    for ipower34 in range(self.num_powers):
                        ifreq3, ifreq4 = self.freq_indices(ipower34)
                        k2 = base + ipower34*self.num_fields+self.BB_loc                    
                    self.sxn_cov[k1, k2] = 0.
                    if(ifreq1 == ifreq3):
                        self.sxn_cov[k1, k2] += Nl[ifreq1] * self.signal_mean[base + self.power_index(ifreq2, ifreq4)*self.num_fields+self.BB_loc]
                    if(ifreq1 == ifreq4):
                        self.sxn_cov[k1, k2] += Nl[ifreq1] * self.signal_mean[base +  self.power_index(ifreq2, ifreq3)*self.num_fields+self.BB_loc]
                    if(ifreq2 == ifreq3):
                        self.sxn_cov[k1, k2] += Nl[ifreq2] * self.signal_mean[base +  self.power_index(ifreq1, ifreq4)*self.num_fields+self.BB_loc]
                    if(ifreq2 == ifreq4):
                        self.sxn_cov[k1, k2] += Nl[ifreq2] * self.signal_mean[base +  self.power_index(ifreq1, ifreq3)*self.num_fields+self.BB_loc]
        if(self.do_delensing):
            for i in range(self.fullsize):  #deapproximately take into account uncertainties of lensing residual, add to noise cov
                self.noise_cov[i, i] += self.lens_res[i]**2 * 2./self.dofs[i // self.blocksize ]
        else:  ###here only consider lensed power due to E
            for i in range(self.fullsize):
                self.noise_cov[i, i] += (self.cmb_lensed_base[i] - self.cmb_unlensed_base[i])**2 * 2./self.dofs[i // self.blocksize ]
        self.covmat = self.noise_cov  + self.signal_cov + self.sxn_cov
        self.get_filters(prefix_no_filter = self.noise_root, prefix_with_filter = self.noisef_root)
        self.invcov = self.matrix_inv(self.covmat)
        self.invcov_computed = True
        if(self.data_product_path is not None):
            np.save(path.join(self.data_product_path, r'lens_res.npy'), self.lens_res)
            np.save(path.join(self.data_product_path, r'mean.npy'), self.mean)
            np.save(path.join(self.data_product_path, r'noise_cov.npy'), self.noise_cov)
            np.save(path.join(self.data_product_path, r'signal_cov.npy'), self.signal_cov)
            np.save(path.join(self.data_product_path, r'sxn_cov.npy'), self.sxn_cov)                        
            with open(path.join(self.data_product_path, r'filters.pickle'), 'wb') as f:
                pickle.dump(self.filters, f)
        if(not self.do_r1):
            return
        self.sxn1_cov = np.zeros((self.fullsize, self.fullsize))
        self.totals1 = np.empty((self.nmaps, self.fullsize))
        self.mean1 = np.zeros(self.fullsize)
        for i in range(self.nmaps):
            post = r'_' + str(i)
            self.totals1[i, :] = self.full_vector(prefix = [ self.noisef_root, self.cmb1f_root, self.fgf_root ], postfix =  [ post, post, "" ], do_seasons = [True, False, False], weights = self.noise_weights )
            if(self.do_delensing):
                for ifield in range(self.num_fields):  
                    for ipower in range(self.num_powers):
                        self.totals1[i, ifield + ipower*self.num_fields : self.fullsize : self.blocksize] -= self.lens1_mean[self.fields[ifield]][i, :]*self.beam_filter(ipower)
            self.mean1 += self.totals1[i, :]
        self.mean1 /= self.nmaps
        for il in range(self.num_ells):
            base = il * self.blocksize
            for i in range(base, base+self.blocksize):
                for j in range(i, base+self.blocksize):
                    self.sxn1_cov[i, j] = np.sum((self.totals1[:, i] - self.noises[:, i] - self.signals1[:, i])*(self.totals1[:, j] - self.noises[:, j] - self.signals1[:, j]))/self.nmaps 
                    if(i != j):
                        self.sxn1_cov[j, i] = self.sxn1_cov[i, j]
        if(self.model_sxn_cov and self.BB_loc >= 0):            
            Nl = np.empty(self.num_freqs)
            for il in range(self.num_ells):
                base = il*self.blocksize
                for ifreq in range(self.num_freqs):
                    k = base+self.power_index(ifreq, ifreq)*self.num_fields+self.BB_loc
                    Nl[ifreq] = self.sxn1_cov[k, k]/2./self.signal1_mean[k]
                for ipower12 in range(self.num_powers):
                    ifreq1, ifreq2 = self.freq_indices(ipower12)
                    k1 = base + ipower12*self.num_fields+self.BB_loc                    
                    for ipower34 in range(self.num_powers):
                        ifreq3, ifreq4 = self.freq_indices(ipower34)
                        k2 = base + ipower34*self.num_fields+self.BB_loc                    
                    self.sxn1_cov[k1, k2] = 0.
                    if(ifreq1 == ifreq3):
                        self.sxn1_cov[k1, k2] += Nl[ifreq1] * self.signal_mean[base + self.power_index(ifreq2, ifreq4)*self.num_fields+self.BB_loc]
                    if(ifreq1 == ifreq4):
                        self.sxn1_cov[k1, k2] += Nl[ifreq1] * self.signal_mean[base +  self.power_index(ifreq2, ifreq3)*self.num_fields+self.BB_loc]
                    if(ifreq2 == ifreq3):
                        self.sxn1_cov[k1, k2] += Nl[ifreq2] * self.signal_mean[base +  self.power_index(ifreq1, ifreq4)*self.num_fields+self.BB_loc]
                    if(ifreq2 == ifreq4):
                        self.sxn1_cov[k1, k2] += Nl[ifreq2] * self.signal_mean[base +  self.power_index(ifreq1, ifreq3)*self.num_fields+self.BB_loc]                        
        if(self.data_product_path is not None):
            np.save(path.join(self.data_product_path, r'lens1_res.npy'), self.lens1_res)
            np.save(path.join(self.data_product_path, r'mean1.npy'), self.mean1)
            np.save(path.join(self.data_product_path, r'signal1_cov.npy'), self.signal1_cov)
            np.save(path.join(self.data_product_path, r'sxn1_cov.npy'), self.sxn1_cov)                        
        self.set_invcov_interp()
        
    def model_vector(self, lcdm_params, fgs):
        fg_vec = self.fg_model_vector(fgs)  
        cmb_unlensed, cmb_lensed = self.cmb_vector_approx(lcdm_params = lcdm_params)
        if(self.do_r1 ):
            fac = max(min(1., lcdm_params[cosmology_num_params-1]/self.cosmo_r1), -1.)
            if(self.do_delensing):  
                if(self.analytic_fg):
                    vec = self.apply_filters(cmb_unlensed - (self.cmb_unlensed_base*(1.-fac)+self.cmb1_unlensed_base*fac) + fg_vec)  + (self.lens_res*(1.-fac)+self.lens1_res*fac)
                else:
                    vec = self.apply_filters(cmb_unlensed - (self.cmb_unlensed_base*(1.-fac)+self.cmb1_unlensed_base*fac) ) + fg_vec + (self.lens_res*(1.-fac)+self.lens1_res*fac)
            else: 
                if(self.analytic_fg):
                    vec = self.apply_filters(cmb_lensed - (self.cmb_lensed_base*(1.-fac)+self.cmb1_lensed_base*fac) + fg_vec)  + (self.lens_res*(1.-fac)+self.lens1_res*fac)
                else:
                    vec = self.apply_filters(cmb_lensed - (self.cmb_lensed_base*(1.-fac)+self.cmb1_lensed_base*fac) ) + fg_vec + (self.lens_res*(1.-fac)+self.lens1_res*fac)
        else:
            if(self.do_delensing):  
                if(self.analytic_fg):
                    vec = self.apply_filters(cmb_unlensed - self.cmb_unlensed_base + fg_vec)  + self.lens_res             
                else:
                    vec = self.apply_filters(cmb_unlensed - self.cmb_unlensed_base ) + fg_vec + self.lens_res
            else: 
                if(self.analytic_fg):
                    vec = self.apply_filters(cmb_lensed - self.cmb_lensed_base + fg_vec)  + self.lens_res             
                else:
                    vec = self.apply_filters(cmb_lensed - self.cmb_lensed_base ) + fg_vec + self.lens_res            
        return vec
            

    
                    
    def chisq_of_vec(self, vec, r):
        if(self.do_r1):
            r_pos =  (abs(r)/self.cosmo_r1)*(self.num_r_interp-1) 
            ind_r_pos =  int(r_pos)
            if(ind_r_pos >= self.num_r_interp-1):
                if(self.do_data_mask):
                    return np.dot(np.dot(vec[self.used_indices], self.invcov_interp[self.num_r_interp-1, :, :]), vec[self.used_indices]) - self.invcov_lndet[self.num_r_interp-1]*self.r_lndet_fac
                elif(self.ell_cross_range == 0):
                    chisq = - self.invcov_lndet[self.num_r_interp-1]*self.r_lndet_fac                    
                    for i in range(self.num_ells):
                        chisq += np.dot(np.dot(vec[i*self.blocksize:(i+1)*self.blocksize], self.invcov_interp[self.num_r_interp-1, i*self.blocksize:(i+1)*self.blocksize, i*self.blocksize:(i+1)*self.blocksize]), vec[i*self.blocksize:(i+1)*self.blocksize])                         
                    return chisq
                else:
                    return np.dot(np.dot(vec, self.invcov_interp[self.num_r_interp-1, :, :]), vec) - self.invcov_lndet[self.num_r_interp-1]*self.r_lndet_fac
            else:
                r_pos = r_pos - ind_r_pos
                if(self.do_data_mask):
                    return np.dot(np.dot(vec[self.used_indices], self.invcov_interp[ind_r_pos, :, :]*(1.-r_pos)+self.invcov_interp[ind_r_pos+1, :, :]*r_pos), vec[self.used_indices]) - (self.invcov_lndet[ind_r_pos]*(1.-r_pos) + self.invcov_lndet[ind_r_pos +1]*r_pos)*self.r_lndet_fac
                elif(self.ell_cross_range == 0):
                    chisq =  - (self.invcov_lndet[ind_r_pos]*(1.-r_pos) + self.invcov_lndet[ind_r_pos +1]*r_pos)*self.r_lndet_fac
                    for i in range(self.num_ells):
                        chisq += np.dot(np.dot(vec[i*self.blocksize:(i+1)*self.blocksize], self.invcov_interp[ind_r_pos, i*self.blocksize:(i+1)*self.blocksize, i*self.blocksize:(i+1)*self.blocksize]*(1.-r_pos)+self.invcov_interp[ind_r_pos+1, i*self.blocksize:(i+1)*self.blocksize, i*self.blocksize:(i+1)*self.blocksize]*r_pos ), vec[i*self.blocksize:(i+1)*self.blocksize])                        
                    return chisq
                else:
                    return np.dot(np.dot(vec, self.invcov_interp[ind_r_pos, :, :]*(1.-r_pos)+self.invcov_interp[ind_r_pos+1, :, :]*r_pos ), vec) - (self.invcov_lndet[ind_r_pos]*(1.-r_pos) + self.invcov_lndet[ind_r_pos +1]*r_pos)*self.r_lndet_fac
        else:
            if(self.do_data_mask):
                return np.dot(np.dot(vec[self.used_indices], self.invcov), vec[self.used_indices])
            elif(self.ell_cross_range == 0):
                chisq = 0.
                for i in range(self.num_ells):
                    chisq += np.dot(np.dot(vec[i*self.blocksize:(i+1)*self.blocksize], self.invcov[i*self.blocksize:(i+1)*self.blocksize, i*self.blocksize:(i+1)*self.blocksize]), vec[i*self.blocksize:(i+1)*self.blocksize])                         
                return chisq
            else:
                return np.dot(np.dot(vec, self.invcov), vec)        
        


