from beforecmb import *
import numpy as np
from sys import argv
from os import path
import matplotlib.pyplot as plt

filedir = r'cs/'

mkdir_if_not_exists(filedir)

def complex_randn(n = None):
    if(n is None):
        random_values = np.random.randn(2)
        complex_array = random_values[0] + 1j * random_values[1]
    else:
        random_values = np.random.randn(2 * n)
        complex_array = random_values.view(np.complex128)
    return complex_array


#-----------------load configuration-----------------
if(len(argv) < 2):
    print("python compsep.py sim_config_file [root] [lmin lmax] [random_seed]")
    exit()
    
if(len(argv) > 2):
    sim = sky_simulator(config_file=argv[1], root_overwrite=argv[2])
else:
    sim = sky_simulator(config_file=argv[1])

    
#set random seed
if(len(argv) > 5):
    np.random.seed(int(argv[5]))
else:
    np.random.seed(42)



class compsep_BMH:
    
    def __init__(self, lmax = 160, vary_beta = False, snapshot=None):
        self.lmax = lmax
        self.almsize = hp.Alm.getsize(lmax = lmax)
        self.current_ind = 0
        self.n_accept = 0
        self.n_reject = 0
        self.vary_beta = vary_beta
        self.beta_d = np.array([1.54, 1.54])
        self.beta_s = np.array([-3., -3.])        
        self.fsky = np.sum(sim.smoothed_mask)/sim.npix
        self.used_ipix = np.where(sim.smoothed_mask > 0.)[0]  
        self.npix_used = len(self.used_ipix)
        self.noise_alms_theory = np.empty( (2, sim.num_freqs, 2, self.almsize), dtype=np.complex128 )
        self.noise_cls_theory = np.empty( (2, sim.num_freqs, 2, self.lmax+1) )        
        #-----------------load data and compute cl----------------
        if(path.exists(filedir + r'data_map.npy') and path.exists(filedir + r'noise_cls.npy')):
            self.data_map = np.load(filedir + r'data_map.npy')
            self.noise_cls = np.load(filedir + r'noise_cls.npy')
        else:
            self.data_map = np.empty((sim.num_freqs, 2, sim.npix))
            self.noise_cls = np.zeros((sim.num_freqs, 2, self.lmax + 1))
            for ifreq in range(sim.num_freqs):
                print("loading data with frequency "+sim.freqnames[ifreq])            
                self.data_map[ifreq, :, :] = sim.load_QU_map(sim.root +  sim.freqnames[ifreq]   + r'.npy')
                for i in range(sim.nmaps):
                    noise_map = sim.load_QU_map(sim.noisef_root + sim.freqnames[ifreq] + r'_' + str(i)  + r'.npy')
                    self.noise_cls[ifreq, :, :] += self.pseudo_cls(noise_map)
            self.noise_cls /= sim.nmaps
            np.save(filedir + r'data_map.npy', self.data_map)
            np.save(filedir + r'noise_cls.npy', self.noise_cls)
        self.alms_norm = np.empty((sim.num_freqs, 2, self.almsize))
        for l in range(self.lmax+1):
            if(l < 2):
                for m in range(l+1):                    
                    self.alms_norm[:, :, hp.Alm.getidx(self.lmax, l, m)] = 1.e99
            else:
                self.alms_norm[:, :, hp.Alm.getidx(self.lmax, l, 0)] = self.noise_cls[:, :, l]
                for m in range(1, l+1):
                    self.alms_norm[:, :, hp.Alm.getidx(self.lmax, l, m)] = self.noise_cls[:, :, l]/2.  #here <Re alm^2>  = <Im alm^2> = C_l / 2                     
        self.dust_alms = np.zeros((2, 2, self.almsize), dtype=np.complex128)
        self.sync_alms = np.zeros((2, 2, self.almsize), dtype=np.complex128)
        self.cmb_alms = np.zeros((2, 2, self.almsize), dtype=np.complex128)
        self.chisq = np.empty((2, self.lmax+1))
        self.lnlike = np.empty(2)
        self.dust_rms = np.empty((2, self.almsize))
        self.sync_rms = np.empty((2, self.almsize))
        self.cmb_rms = np.empty((2, self.almsize))
        self.mzero = np.zeros(self.almsize, dtype=np.bool_)
        for l in range(self.lmax+1):
            self.mzero[hp.Alm.getidx(self.lmax, l, 0)] = True
        self.mzero_idx = np.where(self.mzero)[0]
        if(snapshot is None):
            self.sync_alms[self.current_ind, :, :] = hp.map2alm_spin(maps = self.data_map[sim.ifreq_lowest, :, :], spin=2, lmax = self.lmax)
            dust_weights, sync_weights = sim.fg_band_weights(beta_d = 1.54, beta_s = -3.) #here just for estimation
            dust_tmp = np.zeros((2, self.lmax+1))
            sync_tmp = np.zeros((2, self.lmax+1))
            cmb_tmp = np.zeros((2, self.lmax+1))        
            for l in range(2, self.lmax+1):
                for j in range(2):
                    dust_tmp[j, l] = np.sqrt(1./np.sum(dust_weights/self.noise_cls[:, j, l])/self.fsky)
                    sync_tmp[j, l] = np.sqrt(1./np.sum(sync_weights/self.noise_cls[:, j, l])/self.fsky)
                    cmb_tmp[j, l] = np.sqrt(1./np.sum(1./self.noise_cls[:, j, l])/self.fsky)
            for i in range(self.almsize):
                l, m = hp.Alm.getlm(lmax = self.lmax,i = i)
                self.dust_rms[:, i] = dust_tmp[:, l]
                self.sync_rms[:, i] = sync_tmp[:, l]
                self.cmb_rms[:, i] = cmb_tmp[:, l]
            self.beta_d = np.array([1.54, 1.54])
            self.beta_s = np.array([-3., -3.])
            self.beta_d_rms = 2.e-3
            self.beta_s_rms = 5.e-3
            self.dust_alms[1- self.current_ind, :, :] = self.dust_alms[self.current_ind, :, :]            
            self.sync_alms[1- self.current_ind, :, :] = self.sync_alms[self.current_ind, :, :]
            self.cmb_alms[1- self.current_ind, :, :] = self.cmb_alms[self.current_ind, :, :]
            self.chisq[self.current_ind, :] = self.chisq_l()
            self.set_lnlike_from_chisq()
        else:
            print("loading snapshot...")
            self.load_snapshot(snapshot)
            self.clean_alms_mzero()
        if(len(argv) > 4):
            self.load_pp(lmin = int(argv[3]), lmax=int(argv[4]))
        else:
            self.load_pp(2, 20)            



    def load_pp(self, lmin, lmax, batch_size=200):
        print("loading pp for l between ", lmin, " and ", lmax)
        assert(lmin >1 and lmax<= self.lmax and lmin < lmax)
        self.pp_lmin = lmin
        self.pp_lmax = lmax
        self.pp_size = hp.Alm.getsize(lmax = self.pp_lmax) - hp.Alm.getsize(lmax = self.pp_lmin-1)
        assert(self.pp_size * self.almsize * sim.num_freqs < 1.35e8) 
        self.pp_batch_size = min(batch_size, self.pp_size // 2)
        self.pp_rr = np.empty((self.pp_size, 2,  sim.num_freqs, 2, self.almsize))
        self.pp_ri = np.empty((self.pp_size, 2,  sim.num_freqs, 2, self.almsize)) #input.real; output.imag
        self.pp_ir = np.empty((self.pp_size, 2,  sim.num_freqs, 2, self.almsize)) #input.imag; output.real
        self.pp_ii = np.empty((self.pp_size, 2,  sim.num_freqs, 2, self.almsize))
        self.pp_idx = np.empty(self.pp_size, dtype=np.int32)
        self.pp_mzero = np.zeros(self.pp_size, dtype=np.bool_)
        k = 0
        for l in range(self.pp_lmin, self.pp_lmax+1):
            almsize_in = hp.Alm.getsize(lmax=l)
            alms_in = np.zeros((2, almsize_in), dtype = np.complex128)                                        
            for m in range(l+1):
                self.pp_idx[k] = hp.Alm.getidx(lmax = self.lmax, l = l, m = m)
                if(m == 0):
                    self.pp_mzero[k] = True 
                if(path.exists(filedir + r'pp_rr_' + str(l) + r'_' + str(m) + r'.npy') and path.exists(filedir + r'pp_ri_' + str(l) + r'_' + str(m) + r'.npy') and path.exists(filedir + r'pp_ir_' + str(l) + r'_' + str(m) + r'.npy') and path.exists(filedir + r'pp_ii_' + str(l) + r'_' + str(m) + r'.npy')):
                    self.pp_rr[k, :, :, :, :] = np.load(filedir + r'pp_rr_' + str(l) + r'_' + str(m) + r'.npy')
                    self.pp_ri[k, :, :, :, :] = np.load(filedir + r'pp_ri_' + str(l) + r'_' + str(m) + r'.npy')
                    self.pp_ir[k, :, :, :, :] = np.load(filedir + r'pp_ir_' + str(l) + r'_' + str(m) + r'.npy')
                    self.pp_ii[k, :, :, :, :] = np.load(filedir + r'pp_ii_' + str(l) + r'_' + str(m) + r'.npy')                    
                else:           
                    print(r"computing gradient template, progress ", np.round(k*100./self.pp_size, 2), r"%, l = ", l, ", m=", m)
                    idx = hp.Alm.getidx(lmax = l, l=l, m=m)
                    for ifreq in range(sim.num_freqs):
                        for imap in range(2):
                            alms_in *= 0.
                            alms_in[imap, idx] = 1.
                            alms_out = self.smooth_and_filter_alms(ifreq = ifreq, alms_in = alms_in, lmax_in = l)
                            self.pp_rr[k, imap, ifreq, :, :] = alms_out.real
                            self.pp_ri[k, imap, ifreq, :, :] = alms_out.imag
                            if(m != 0):
                                alms_in *= 0.
                                alms_in[imap, idx] = 1j 
                                alms_out = self.smooth_and_filter_alms(ifreq = ifreq, alms_in = alms_in, lmax_in = l)
                                self.pp_ir[k, imap, ifreq, :, :] = alms_out.real
                                self.pp_ii[k, imap, ifreq, :, :] = alms_out.imag
                            else:
                                self.pp_ir[k, imap, ifreq, :, :] = 0.
                                self.pp_ii[k, imap, ifreq, :, :] = 0.
                    np.save(filedir+'pp_rr_'+str(l)+r'_'+str(m)+r'.npy', self.pp_rr[k, :, :, :, :])
                    np.save(filedir+'pp_ri_'+str(l)+r'_'+str(m)+r'.npy', self.pp_ri[k, :, :, :, :])
                    np.save(filedir+'pp_ir_'+str(l)+r'_'+str(m)+r'.npy', self.pp_ir[k, :, :, :, :])
                    np.save(filedir+'pp_ii_'+str(l)+r'_'+str(m)+r'.npy', self.pp_ii[k, :, :, :, :])
                k += 1


    def randn(self):
        rn_real = np.random.randn(2, self.almsize)
        rn_imag = np.random.randn(2, self.almsize)
        rn_imag[:, self.mzero_idx] = 0. 
        return rn_real + 1j * rn_imag
                
    #this does not do purify_b, but is linear (and much faster), good enough for low-passed (due to TOD filtering) and high-passed (due to our choice of lmax<~200) maps
    def pseudo_alms(self, maps):
        assert(maps.shape[0] == 2 and maps.shape[1] == sim.npix)
        return map_to_alm(maps = maps*np.tile(sim.smoothed_mask, (maps.shape[0], 1)), lmax=self.lmax)


    def pseudo_cls(self, maps):
        alms = self.pseudo_alms(maps)
        cls = np.empty((2, self.lmax+1))
        for i in range(2):
            cls[i, :] =  hp.alm2cl(alms[i], lmax=self.lmax)
        return cls
    
    def smooth_and_filter_alms(self, ifreq, alms_in, lmax_in):
        for i in range(2):
            hp.smoothalm(alms = alms_in[i, :], fwhm=sim.fwhms_rad[ifreq], inplace=True)
        return self.pseudo_alms(sim.filtering.project_map(mask = sim.smoothed_mask, maps = alm_to_map(alms_in, nside=sim.nside, lmax=lmax_in), want_wof = False))

    def make_freq_maps(self, ifreq, alms_in, lmax_in):
        for i in range(2):
            hp.smoothalm(alms = alms_in[i, :], fwhm=sim.fwhms_rad[ifreq], inplace=True)
        return sim.filtering.project_map(mask = sim.smoothed_mask, maps = alm_to_map(alms_in, nside=sim.nside, lmax=lmax_in), want_wof = False)
        
    
    def get_noise_alms_theory(self, dust_weights, sync_weights):
        for ifreq in range(sim.num_freqs):
            alms = self.dust_alms[self.current_ind, :, :]*dust_weights[ifreq] + self.sync_alms[self.current_ind, :, :]*sync_weights[ifreq] + self.cmb_alms[self.current_ind, :, :]
            for imap in range(2):
                hp.smoothalm(alms[imap, :], fwhm=sim.fwhms_rad[ifreq], inplace=True)
            noise_map =  sim.filtering.project_map(mask = sim.smoothed_mask, maps = alm_to_map(alms, nside=sim.nside, lmax=self.lmax), want_wof = False)
            noise_map[:, self.used_ipix] = self.data_map[ifreq, :, self.used_ipix].T - noise_map[:, self.used_ipix]
            self.noise_alms_theory[self.current_ind, ifreq, :, :] = self.pseudo_alms(noise_map)
            for imap in range(2):
                self.noise_cls_theory[self.current_ind, ifreq, imap, :] = hp.alm2cl(self.noise_alms_theory[self.current_ind, ifreq, imap, :], lmax=self.lmax)
            
    def chisq_l(self):
        chisq = np.zeros(self.lmax+1)
        dust_weights, sync_weights = sim.fg_band_weights(beta_d = self.beta_d[self.current_ind], beta_s = self.beta_s[self.current_ind])
        self.get_noise_alms_theory(dust_weights, sync_weights)
        for l in range(2, self.lmax+1):
            for ifreq in range(sim.num_freqs):            
                chisq[l] += np.sum(self.noise_cls_theory[self.current_ind, ifreq,:, l]/self.noise_cls[ifreq, :, l])
            chisq[l] *= (2.*l+1.)*self.fsky
        return chisq
        

    def sgld_gradient(self, batch_indices):
        dust_weights, sync_weights = sim.fg_band_weights(beta_d = self.beta_d[self.current_ind], beta_s = self.beta_s[self.current_ind])
        grads_real = np.zeros((3, 2, self.pp_batch_size))
        grads_imag = np.zeros((3, 2, self.pp_batch_size))
        for ifreq in range(sim.num_freqs):
            weights = np.array([dust_weights[ifreq], sync_weights[ifreq], 1.])
            for i in range(self.pp_batch_size):
                if(self.pp_mzero[batch_indices[i]]):
                    for imap in range(2):
                        grads_real[:, imap, i] += weights*np.sum((self.noise_alms_theory[self.current_ind, ifreq, :,:].real * self.pp_rr[batch_indices[i], imap, ifreq, :, :] + self.noise_alms_theory[self.current_ind, ifreq, :, :].imag * self.pp_ri[batch_indices[i], imap, ifreq, :, :])/self.alms_norm[ifreq, :, :])
                else:
                    for imap in range(2):
                        grads_real[:, imap, i] += weights*np.sum((self.noise_alms_theory[self.current_ind, ifreq, :,:].real * self.pp_rr[batch_indices[i], imap, ifreq, :, :] + self.noise_alms_theory[self.current_ind, ifreq, :, :].imag * self.pp_ri[batch_indices[i], imap, ifreq, :, :])/self.alms_norm[ifreq, :, :])
                        grads_imag[:, imap, i] += weights*np.sum((self.noise_alms_theory[self.current_ind, ifreq, :,:].real * self.pp_ir[batch_indices[i], imap, ifreq, :, :] + self.noise_alms_theory[self.current_ind, ifreq, :, :].imag * self.pp_ii[batch_indices[i], imap, ifreq, :, :])/self.alms_norm[ifreq, :, :])                                
        return (grads_real + 1j * grads_imag)*self.fsky
        
    def sgld_step(self, eps, noise_factor = 1., adaptive = False, makecopy = False):
        batch_indices = np.random.choice(self.pp_size, size= self.pp_batch_size, replace=False)
        grads = self.sgld_gradient(batch_indices)
        for imap in range(2):
            self.dust_alms[self.current_ind, imap, self.pp_idx[batch_indices]]  += (0.5*eps) * self.dust_rms[imap, self.pp_idx[batch_indices]]**2 * grads[0, imap, :]  
            self.sync_alms[self.current_ind, imap, self.pp_idx[batch_indices]]  += (0.5*eps) * self.sync_rms[imap, self.pp_idx[batch_indices]]**2 * grads[1, imap, :]  
            self.cmb_alms[self.current_ind, imap, self.pp_idx[batch_indices]]  += (0.5*eps) * self.cmb_rms[imap, self.pp_idx[batch_indices]]**2 * grads[2, imap, :]  
        if(noise_factor > 0.):
             self.dust_alms[self.current_ind, :, :] +=  (noise_factor*np.sqrt(eps)) * self.dust_rms*self.randn()
             self.sync_alms[self.current_ind, :, :] +=  (noise_factor*np.sqrt(eps)) * self.sync_rms*self.randn()
             self.cmb_alms[self.current_ind, :, :] +=  (noise_factor*np.sqrt(eps)) * self.cmb_rms*self.randn()
        self.chisq[self.current_ind, :] = self.chisq_l()
        lnlike_save = self.lnlike[self.current_ind]        
        self.set_lnlike_from_chisq()
        if(adaptive):
            if(self.lnlike[self.current_ind]-lnlike_save <= 0.): 
                self.dust_rms[:, self.pp_idx[batch_indices]] *= 0.99
                self.sync_rms[:, self.pp_idx[batch_indices]] *= 0.99
                self.cmb_rms[:, self.pp_idx[batch_indices]] *= 0.99
                eps *= 0.95
                noise_factor *= 0.8
            elif(self.lnlike[self.current_ind]-lnlike_save < abs(lnlike_save)*1.e-6):
                self.dust_rms[:, self.pp_idx[batch_indices]] *= 1.005
                self.sync_rms[:, self.pp_idx[batch_indices]] *= 1.005
                self.cmb_rms[:, self.pp_idx[batch_indices]] *= 1.005
                eps *= 1.03
            elif(self.lnlike[self.current_ind]-lnlike_save < abs(lnlike_save)*1.e-5 ):
                self.dust_rms[:, self.pp_idx[batch_indices]] *= 1.002
                self.sync_rms[:, self.pp_idx[batch_indices]] *= 1.002
                self.cmb_rms[:, self.pp_idx[batch_indices]] *= 1.002
                eps *= 1.02                
            elif(self.lnlike[self.current_ind]-lnlike_save < abs(lnlike_save)*1.e-4 ):
                self.dust_rms[:, self.pp_idx[batch_indices]] *= 1.001
                self.sync_rms[:, self.pp_idx[batch_indices]] *= 1.001
                self.cmb_rms[:, self.pp_idx[batch_indices]] *= 1.001
                eps *= 1.01                
            else:
                eps *= 1.0005
                noise_factor = min(1., noise_factor * 1.0005)
        if(makecopy):
            self.lnlike[1-self.current_ind] = self.lnlike[self.current_ind]
            self.chisq[1-self.current_ind, :] = self.chisq[self.current_ind, :]
            if(noise_factor > 0.):
                self.dust_alms[1-self.current_ind, :, :] =  self.dust_alms[self.current_ind, :, :]
                self.sync_alms[1-self.current_ind, :, :] = self.sync_alms[self.current_ind, :, :]
                self.cmb_alms[1-self.current_ind, :, :] =  self.cmb_alms[self.current_ind, :, :]
            else:
                for imap in range(2):
                    self.dust_alms[1-self.current_ind, imap, self.pp_idx[batch_indices]] =  self.dust_alms[self.current_ind, imap, self.pp_idx[batch_indices]] 
                    self.sync_alms[1-self.current_ind, imap, self.pp_idx[batch_indices]] = self.sync_alms[self.current_ind, imap, self.pp_idx[batch_indices]]
                    self.cmb_alms[1-self.current_ind, imap, self.pp_idx[batch_indices]] =  self.cmb_alms[self.current_ind, imap, self.pp_idx[batch_indices]]                     
        return eps, noise_factor


    def clean_alms_mzero(self):
        for l in range(self.lmax+1):
            idx = hp.Alm.getidx(self.lmax, l, 0)
            self.dust_alms[:, :, idx] = self.dust_alms[:, :, idx].real
            self.sync_alms[:, :, idx] = self.sync_alms[:, :, idx].real
            self.cmb_alms[:, :, idx] = self.cmb_alms[:, :, idx].real                        
            
    def check_alms_mzero(self):
        for l in range(self.lmax+1):
            idx = hp.Alm.getidx(self.lmax, l, 0)
            for imap in range(2):
                if(self.dust_alms[self.current_ind, imap, idx].imag != 0.):
                    print("Error: dust m=0 mode is not real")
                    print(l, self.dust_alms[self.current_ind, imap, idx])
                    exit()
                if(self.sync_alms[self.current_ind, imap, idx].imag != 0.):
                    print("Error: sync m=0 mode is not real")
                    print(l, self.sync_alms[self.current_ind, imap, idx])
                    exit()
                if(self.cmb_alms[self.current_ind, imap, idx].imag != 0.):
                    print("Error: cmb m=0 mode is not real")
                    print(l, self.cmb_alms[self.current_ind, imap, idx])
                    exit()
        print("m=0 components checked")


    def mcmc_step(self, step_size, batch_size, search_min = False):  #if you set search_min = True, the posterior distribution will not be correct
        batch_indices = np.random.choice(self.almsize, size= batch_size, replace=False)
        self.current_ind = 1- self.current_ind
        if(self.vary_beta):
            self.beta_d[self.current_ind] = self.beta_d[1-self.current_ind] + self.beta_d_rms*step_size*np.random.randn()
            self.beta_s[self.current_ind] = self.beta_s[1-self.current_ind] + self.beta_s_rms*step_size*np.random.randn()
        for imap in range(2):
            self.dust_alms[self.current_ind, imap,  batch_indices] += self.dust_rms[imap, batch_indices] * complex_randn(batch_size)*step_size
            self.sync_alms[self.current_ind, imap,  batch_indices] += self.sync_rms[imap, batch_indices] * complex_randn(batch_size)*step_size
            self.cmb_alms[self.current_ind, imap,  batch_indices] += self.cmb_rms[imap, batch_indices] * complex_randn(batch_size)*step_size
        self.chisq[self.current_ind, :] = self.chisq_l()
        self.set_lnlike_from_chisq()
        if(self.lnlike[self.current_ind] - self.lnlike[1-self.current_ind] > np.log(1.-np.random.rand())): #accept
            self.n_accept += 1
            if(search_min and self.lnlike[self.current_ind] > self.lnlike[1-self.current_ind]):  #try the extrapolated point 
                self.current_ind = 1- self.current_ind
                if(self.vary_beta):
                    self.beta_d[self.current_ind] = 2.*self.beta_d[1-self.current_ind] - self.beta_d[self.current_ind]
                    self.beta_s[self.current_ind] = 2.*self.beta_s[1-self.current_ind] - self.beta_s[self.current_ind] 
                for imap in range(2):
                    self.dust_alms[self.current_ind, imap,  batch_indices] = 2.*self.dust_alms[1- self.current_ind, imap,  batch_indices] - self.dust_alms[self.current_ind, imap,  batch_indices]
                    self.sync_alms[self.current_ind, imap,  batch_indices] = 2.*self.sync_alms[1- self.current_ind, imap,  batch_indices] - self.sync_alms[self.current_ind, imap,  batch_indices]
                    self.cmb_alms[self.current_ind, imap,  batch_indices] = 2.*self.cmb_alms[1- self.current_ind, imap,  batch_indices] - self.cmb_alms[self.current_ind, imap,  batch_indices]
                self.chisq[self.current_ind, :] = self.chisq_l()
                self.set_lnlike_from_chisq()
                if(self.lnlike[self.current_ind] < self.lnlike[1-self.current_ind]): #reject
                    self.current_ind = 1 - self.current_ind
        else: #reject
            if(search_min):  #try the extrapolated point
                if(self.vary_beta):
                    self.beta_d[self.current_ind] = 2.*self.beta_d[1-self.current_ind] - self.beta_d[self.current_ind]
                    self.beta_s[self.current_ind] = 2.*self.beta_s[1-self.current_ind] - self.beta_s[self.current_ind] 
                for imap in range(2):
                    self.dust_alms[self.current_ind, imap,  batch_indices] = 2.*self.dust_alms[1- self.current_ind, imap,  batch_indices] - self.dust_alms[self.current_ind, imap,  batch_indices]
                    self.sync_alms[self.current_ind, imap,  batch_indices] = 2.*self.sync_alms[1- self.current_ind, imap,  batch_indices] - self.sync_alms[self.current_ind, imap,  batch_indices]
                    self.cmb_alms[self.current_ind, imap,  batch_indices] = 2.*self.cmb_alms[1- self.current_ind, imap,  batch_indices] - self.cmb_alms[self.current_ind, imap,  batch_indices]
                self.chisq[self.current_ind, :] = self.chisq_l()
                self.set_lnlike_from_chisq()
                if(self.lnlike[self.current_ind] < self.lnlike[1-self.current_ind]): #reject
                    self.current_ind = 1 - self.current_ind
                    self.n_reject += 1
                else:  #accept
                    self.n_accept += 1
            else:
                self.current_ind = 1- self.current_ind
                self.n_reject += 1                            
        for imap in range(2):
            self.dust_alms[1-self.current_ind, imap, batch_indices] =  self.dust_alms[self.current_ind, imap, batch_indices] 
            self.sync_alms[1-self.current_ind, imap, batch_indices] = self.sync_alms[self.current_ind, imap, batch_indices]
            self.cmb_alms[1-self.current_ind, imap, batch_indices] =  self.cmb_alms[self.current_ind, imap, batch_indices] 


    def save_snapshot(self, snapshot):
        if(snapshot is None):
            return
        if(path.exists(snapshot+'_lnlike.txt')):
            x = np.loadtxt(snapshot+'_lnlike.txt')
            if(x[0] > self.lnlike[self.current_ind]):
                print('save_snapshot: last snapshot has better likelihood; skipping this snapshot...')
                return
        else:
            x = np.array([ -1.e99, -1.e99 ])    
        self.chisq[self.current_ind, :] = self.chisq_l()
        self.set_lnlike_from_chisq()
        print('saving snapshot with lnlike=', self.lnlike[self.current_ind])
        np.save(snapshot+"_dust_alms.npy", self.dust_alms[self.current_ind, :, :])
        np.save(snapshot+"_sync_alms.npy", self.sync_alms[self.current_ind, :, :])
        np.save(snapshot+"_cmb_alms.npy", self.cmb_alms[self.current_ind, :, :])
        np.save(snapshot+"_dust_rms.npy", self.dust_rms)
        np.save(snapshot+"_sync_rms.npy", self.sync_rms)
        np.save(snapshot+"_cmb_rms.npy", self.cmb_rms)
        np.save(snapshot+'_betas.npy', np.array([self.beta_d[self.current_ind], self.beta_s[self.current_ind], self.beta_d_rms, self.beta_s_rms]))
        np.savetxt(snapshot+'_lnlike.txt', np.array([self.lnlike[self.current_ind], x[0]]))
        
        

    def load_snapshot(self, snapshot):
        self.dust_alms[self.current_ind, :, :] = np.load(snapshot + r'_dust_alms.npy')
        self.sync_alms[self.current_ind, :, :] = np.load(snapshot + r'_sync_alms.npy')
        self.cmb_alms[self.current_ind, :, :] = np.load(snapshot + r'_cmb_alms.npy')
        self.dust_rms = np.load(snapshot + r'_dust_rms.npy')
        self.sync_rms =  np.load(snapshot + r'_dust_rms.npy')
        self.cmb_rms =  np.load(snapshot + r'_cmb_rms.npy')
        for l in range(2, self.lmax+1):
            fac = 1./(l+1.)**0.5
            for m in range(l+1):
                k = hp.Alm.getidx(self.lmax, l, m)
                self.dust_rms[:, k] = fac
                self.dust_rms[:, k] = fac
                self.cmb_rms[:, k] = fac    
        x = np.load(snapshot + r'_betas.npy')
        self.beta_d = np.array([x[0], x[0]])
        self.beta_s = np.array([x[1], x[1]])
        self.beta_d_rms = x[2]
        self.beta_s_rms = x[3]
        self.dust_alms[1- self.current_ind, :, :] = self.dust_alms[self.current_ind, :, :]            
        self.sync_alms[1- self.current_ind, :, :] = self.sync_alms[self.current_ind, :, :]
        self.cmb_alms[1- self.current_ind, :, :] = self.cmb_alms[self.current_ind, :, :]
        self.chisq[self.current_ind, :] = self.chisq_l()
        self.set_lnlike_from_chisq()
        print("loaded snapshot with lnlike:", self.lnlike[self.current_ind])      


    def set_lnlike_from_chisq(self):
        self.lnlike[self.current_ind] = -0.5*np.sum(self.chisq[self.current_ind, :])
        
    def sample(self, n_iter = 10000, eps = 1.e-2, noise_factor = 1.e-2, snapshot = None):
        for l in range(2, self.lmax+1):
            print(l, np.round(self.chisq[self.current_ind, l]/(2.*l+1.), 2))             
        cmb_mean = np.zeros( (2, self.almsize), dtype=np.complex128 )        
        cmb_rms = np.zeros((2, self.almsize))
        dust_mean = np.zeros( (2, self.almsize), dtype=np.complex128 )
        dust_rms = np.zeros((2, self.almsize))
        sync_mean = np.zeros( (2, self.almsize), dtype=np.complex128 )
        sync_rms = np.zeros((2, self.almsize))
        beta_d_mean = 0.
        beta_d_rms = 0.
        beta_s_mean = 0.
        beta_s_rms = 0.
        for istep in range(n_iter):
            eps, noise_factor = self.sgld_step(eps = eps, noise_factor = noise_factor, adaptive=True)
            cmb_mean += self.cmb_alms[self.current_ind,:,:]
            cmb_rms += self.cmb_alms[self.current_ind, :, :].real**2+self.cmb_alms[self.current_ind, :, :].imag**2
            dust_mean += self.dust_alms[self.current_ind,:,:]
            dust_rms += self.dust_alms[self.current_ind, :, :].real**2+self.dust_alms[self.current_ind, :, :].imag**2
            sync_mean += self.sync_alms[self.current_ind,:,:]
            sync_rms += self.sync_alms[self.current_ind, :, :].real**2+self.sync_alms[self.current_ind, :, :].imag**2
            if(self.vary_beta):
                beta_d_mean += self.beta_d[self.current_ind]
                beta_s_mean += self.beta_s[self.current_ind]
                beta_d_rms +=  self.beta_d[self.current_ind]**2
                beta_s_rms +=  self.beta_s[self.current_ind]**2
            print(istep, "lnlike: ", np.round(self.lnlike[self.current_ind], 4),"; epsilon: ", np.round(eps, 6), "; noise factor: ",np.round(noise_factor, 6))
            if(istep % 67 == 61):
                self.save_snapshot(snapshot)                
                for l in range(self.pp_lmin, self.pp_lmax+1):
                    print(l, np.round(self.chisq[self.current_ind, l]/(2.*l+1.), 2))
                lmin = np.random.randint(2, self.lmax-1)
                lmax = min(self.lmax, int(np.sqrt(lmin**2 + 3000.)))
                self.load_pp(lmin, lmax)                
        if(self.vary_beta):
            beta_d_mean /= n_iter
            beta_d_rms = np.sqrt(beta_d_rms/n_iter - beta_d_mean**2)
            beta_s_mean /= n_iter
            beta_s_rms = np.sqrt(beta_s_rms/n_iter - beta_s_mean**2)
        cmb_mean /=  n_iter
        self.cmb_rms =  np.sqrt(1.e-6 + cmb_rms/n_iter-cmb_mean.real**2-cmb_mean.imag**2)
        dust_mean /=  n_iter
        self.dust_rms =  np.sqrt(1.e-6 + dust_rms/n_iter-dust_mean.real**2-dust_mean.imag**2)
        sync_mean /=  n_iter
        self.sync_rms =  np.sqrt(1.e-6 + sync_rms/n_iter-sync_mean.real**2-sync_mean.imag**2)
        self.save_snapshot(snapshot)                
        return dust_mean, sync_mean, cmb_mean, beta_d_mean, beta_d_rms, beta_s_mean, beta_s_rms

if(path.exists(filedir + r'burn_in_lnlike.txt')):
    cs = compsep_BMH(snapshot = filedir + r'burn_in')
else:
    cs = compsep_BMH(snapshot=None)
if(len(argv) < 4):  #this is only for computing gradient templates
    print("gradient templates are done")
    exit()
dust_mean, sync_mean, cmb_mean, beta_d_mean, beta_d_rms, beta_s_mean, beta_s_rms = cs.sample(n_iter = 3000,  eps = 2.5e-5, noise_factor = 1.e-6, snapshot=filedir + r'burn_in') 
np.save(filedir + r'cmb_mean_alms.npy', cmb_mean)
np.save(filedir + r'sync_mean_alms.npy', sync_mean)
np.save(filedir + r'dust_mean_alms.npy', dust_mean)
for ifreq in range(sim.num_freqs):
    sim.save_map(filedir + r'cmb_'+str(sim.freqnames[ifreq])+r'.npy', cs.make_freq_maps(ifreq, cmb_mean, cs.lmax), True)
    sim.save_map(filedir + r'sync_'+str(sim.freqnames[ifreq])+r'.npy', cs.make_freq_maps(ifreq, sync_mean, cs.lmax), True)
    sim.save_map(filedir + r'dust_'+str(sim.freqnames[ifreq])+r'.npy', cs.make_freq_maps(ifreq, dust_mean, cs.lmax), True)    


if(cs.vary_beta):
    np.savetxt(filedir + r'beta_d_beta_s.txt', np.array([beta_d_mean, beta_s_mean, beta_d_rms, beta_s_rms]))
    print("beta_d = ", np.round(beta_d_mean, 4), "+/-", np.round(beta_d_rms, 4))
    print("beta_s = ", np.round(beta_s_mean, 4), "+/-", np.round(beta_s_rms, 4))


