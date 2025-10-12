from beforecmb import *
import numpy as np
from sys import argv
from os import path
import matplotlib.pyplot as plt


used_imap = np.array([0, 1, 2])  #set to [1, 2] if you want to skip temperature
#-----------------load configuration-----------------
if(len(argv) < 2):
    print("python compsep.py config_file [root] [random_seed]")
    exit()
if(len(argv) > 2):
    ana = sky_analyser(argv[1], argv[2])    
else:
    ana = sky_analyser(argv[1])
#set random seed
if(len(argv) > 3):
    np.random.seed(int(argv[3]))
else:
    np.random.seed(42)

used_ipix = np.where(ana.power_calc.mask_ones)[0]
npix_used = len(used_ipix)    
#-----------------load data and compute the noise rms----------------
noise_var = np.zeros( (ana.num_freqs, 3, ana.npix))
dust_var = np.zeros(3)
sync_var = np.zeros(3)
cmb_var = np.zeros(3)


data_map = np.empty( (ana.num_freqs, 3, ana.npix))
for ifreq in range(ana.num_freqs):
    data_map[ifreq, :, :] = ana.power_calc.load_IQU_map(ana.root +  ana.freqnames[ifreq]   + ana.mapform)
    for i in range(ana.nmaps):
        noise_var[ifreq, :, :]  += ana.power_calc.load_IQU_map(ana.noisef_root + ana.freqnames[ifreq] + r'_' + str(i)  + ana.mapform)**2

noise_var /= ana.nmaps
dust_map = ana.power_calc.load_IQU_map(ana.fgf_root +  ana.freqnames[ana.ifreq_highest]  + ana.mapform)
sync_map = ana.power_calc.load_IQU_map(ana.fgf_root +  ana.freqnames[ana.ifreq_lowest]  + ana.mapform)

if(ana.do_r1):
    cmb_map = ana.power_calc.load_IQU_map(ana.cmb1f_root +  ana.freqnames[0] + r'_0' + ana.mapform)    
else:
    cmb_map = ana.power_calc.load_IQU_map(ana.cmbf_root +  ana.freqnames[0] + r'_0' + ana.mapform)


for imap in range(3):
    dust_var[imap] = np.sum(dust_map[imap, used_ipix]**2)
    sync_var[imap] = np.sum(sync_map[imap, used_ipix]**2)
    cmb_var[imap] = np.sum(cmb_map[imap, used_ipix]**2)

dust_std = np.sqrt(dust_var/npix_used)
sync_std = np.sqrt(sync_var/npix_used)
cmb_std = np.sqrt(cmb_var/npix_used)

dust_norm = np.zeros( (3, ana.npix) )
sync_norm = np.zeros( (3, ana.npix) )
cmb_norm = np.zeros( (3, ana.npix) )

dust_weights, sync_weights = ana.fg_band_weights(beta_d = 1.54, beta_s = -3.) #here just for estimation
for imap in used_imap:
    for ifreq in range(ana.num_freqs):    
        dust_norm[imap, used_ipix] += dust_weights[ifreq]**2/noise_var[ifreq, imap, used_ipix]
        sync_norm[imap, used_ipix] += sync_weights[ifreq]**2/noise_var[ifreq, imap, used_ipix]        
        cmb_norm[imap, used_ipix] += 1./noise_var[ifreq, imap, used_ipix]
    dust_norm[imap, used_ipix] = np.sqrt(1./dust_norm[imap, used_ipix])
    sync_norm[imap, used_ipix] = np.sqrt(1./sync_norm[imap, used_ipix])
    cmb_norm[imap, used_ipix] = np.sqrt(1./cmb_norm[imap, used_ipix])


class compsep_sgld:
    
    def __init__(self,  batch_size=100, snapshot=None):
        self.batch_size = batch_size        
        if(snapshot is None):        
            self.sync_map = np.zeros((3, ana.npix))
            self.dust_map = np.zeros( (3, ana.npix))
            self.cmb_map = np.zeros( (3, ana.npix) )
            for imap in used_imap:
                self.sync_map[imap, used_ipix]= data_map[ana.ifreq_lowest, imap, used_ipix]/sync_norm[imap, used_ipix]
                self.dust_map[imap, used_ipix]= data_map[ana.ifreq_highest, imap, used_ipix]/dust_norm[imap, used_ipix]
            self.beta_d = 1.54
            self.beta_s = -3.
        else:
            self.sync_map = ana.power_calc.load_IQU_map(snapshot + '_sync_map.npy')
            self.dust_map = ana.power_calc.load_IQU_map(snapshot + '_dust_map.npy')
            self.cmb_map = ana.power_calc.load_IQU_map(snapshot + '_cmb_map.npy')                        
            x = np.loadtxt(snapshot + '_betas.txt')
            self.beta_d = x[0]
            self.beta_s = x[1]
        
    def log_likelihood(self):
        dust_weights, sync_weights = ana.fg_band_weights(beta_d = self.beta_d, beta_s = self.beta_s)
        chisq = 0.        
        for imap in used_imap:
            chisq += np.sum((self.dust_map[imap, used_ipix]*dust_norm[imap, used_ipix])**2)/dust_var[imap] + np.sum((self.sync_map[imap, used_ipix]*sync_norm[imap, used_ipix])**2)/sync_var[imap] + np.sum((self.cmb_map[imap, used_ipix]*cmb_norm[imap, used_ipix])**2)/cmb_var[imap]
            for ifreq in range(ana.num_freqs):
                chisq += np.sum((data_map[ifreq, imap, used_ipix] - self.dust_map[imap, used_ipix]*dust_norm[imap, used_ipix]*dust_weights[ifreq] - self.sync_map[imap, used_ipix]*sync_norm[imap, used_ipix]*sync_weights[ifreq] - self.cmb_map[imap, used_ipix]*cmb_norm[imap, used_ipix])**2/noise_var[ifreq, imap, used_ipix]) 
        return -0.5*chisq
    
    def stochastic_gradient(self, batch_indices):
        dust_grad = np.zeros((3, ana.npix))
        sync_grad = np.zeros((3, ana.npix))
        cmb_grad =  np.zeros((3, ana.npix))
        dust_weights, sync_weights = ana.fg_band_weights(beta_d = self.beta_d, beta_s = self.beta_s)        
        for imap in used_imap:
            dust_grad[imap, batch_indices] = -self.dust_map[imap, batch_indices]*dust_norm[imap, batch_indices]**2/dust_var[imap]
            sync_grad[imap, batch_indices] = -self.sync_map[imap, batch_indices]*sync_norm[imap, batch_indices]**2/sync_var[imap]
            cmb_grad[imap, batch_indices]  = -self.cmb_map[imap, batch_indices]*cmb_norm[imap, batch_indices]**2/cmb_var[imap]
            for ifreq in range(ana.num_freqs):
                dust_grad[imap, batch_indices] += dust_weights[ifreq]*dust_norm[imap, batch_indices]*(data_map[ifreq, imap, batch_indices] - self.dust_map[imap, batch_indices]*dust_norm[imap, batch_indices]*dust_weights[ifreq] - self.sync_map[imap, batch_indices]*sync_norm[imap, batch_indices]*sync_weights[ifreq] - self.cmb_map[imap, batch_indices]*cmb_norm[imap, batch_indices])/noise_var[ifreq, imap, batch_indices]
                sync_grad[imap, batch_indices] += sync_weights[ifreq]*sync_norm[imap, batch_indices]*(data_map[ifreq, imap, batch_indices] - self.dust_map[imap, batch_indices]*dust_norm[imap, batch_indices]*dust_weights[ifreq] - self.sync_map[imap, batch_indices]*sync_norm[imap, batch_indices]*sync_weights[ifreq] - self.cmb_map[imap, batch_indices]*cmb_norm[imap, batch_indices])/noise_var[ifreq, imap, batch_indices]
                cmb_grad[imap, batch_indices] += cmb_norm[imap, batch_indices]*(data_map[ifreq, imap, batch_indices] - self.dust_map[imap, batch_indices]*dust_norm[imap, batch_indices]*dust_weights[ifreq] - self.sync_map[imap, batch_indices]*sync_norm[imap, batch_indices]*sync_weights[ifreq] - self.cmb_map[imap, batch_indices]*cmb_norm[imap, batch_indices])/noise_var[ifreq, imap, batch_indices]
        return dust_grad, sync_grad, cmb_grad
    
    def sgld_step(self, epsilon):
        batch_indices = np.random.choice(used_ipix, self.batch_size, replace=False)
        dust_grad, sync_grad, cmb_grad = self.stochastic_gradient(batch_indices)
        for imap in used_imap:
            self.dust_map[imap, batch_indices]  +=  (0.5 * epsilon) * dust_grad[imap, batch_indices]
            self.dust_map[imap, used_ipix] +=  (np.sqrt(epsilon)) * np.random.randn(npix_used) 
            self.sync_map[imap, batch_indices]  +=  (0.5 * epsilon) * sync_grad[imap, batch_indices]
            self.sync_map[imap, used_ipix] +=  (np.sqrt(epsilon)) * np.random.randn(npix_used) 
            self.cmb_map[imap, batch_indices]  +=  (0.5 * epsilon)  * cmb_grad[imap, batch_indices]
            self.cmb_map[imap, used_ipix] +=  (np.sqrt(epsilon)) * np.random.randn(npix_used) 
    
    def sample(self, n_iter = 10000, target_lnlike = -1.e6, epsilon=1.e-4, snapshot = None):
        self.n_iter = n_iter
        cmb_mean = np.zeros( (3, ana.npix) )
        cmb_var = np.zeros( (3, ana.npix) ) 
        last_lnlike = self.log_likelihood()
        i = 0
        while(last_lnlike < target_lnlike):
            self.sgld_step(epsilon)
            if(i % 200 == 0):
                lnlike = self.log_likelihood()
                if(lnlike < last_lnlike and epsilon > 1.e-12):
                    epsilon *= 0.8
                else:
                    epsilon *= 1.002
                print("current loglike: ", np.round(lnlike, 1), ", current epsilon: ", np.round(epsilon, 8))  
                last_lnlike = lnlike
            i += 1
        print("burned in")
        if(i > 500 and snapshot is not None):
            ana.power_calc.save_map(snapshot+"_dust_map.npy", self.dust_map, True)
            ana.power_calc.save_map(snapshot+"_sync_map.npy", self.sync_map, True)
            ana.power_calc.save_map(snapshot+"_cmb_map.npy", self.cmb_map, True)
            np.savetxt(snapshot+'_betas.txt', [self.beta_d, self.beta_s], )
        for i in range(n_iter):
            self.sgld_step(epsilon)
            for imap in used_imap:
                cmb_mean[imap, used_ipix] += self.cmb_map[imap, used_ipix]
                cmb_var[imap, used_ipix] += self.cmb_map[imap, used_ipix]**2  
            if(i % 1000 == 0):
                lnlike = self.log_likelihood()
                epsilon *= 0.99
                print("current loglike: ", np.round(lnlike, 1), ", current epsilon: ", np.round(epsilon, 8), ", progress: "+str(np.round(100.*i/n_iter, 1))+"%")
                last_lnlike = lnlike                
        for imap in used_imap:
            cmb_mean[imap, used_ipix] /=  n_iter
            cmb_var[imap, used_ipix] /=  n_iter
            cmb_var[imap, used_ipix] -=  cmb_mean[imap, used_ipix]**2
        return cmb_mean, cmb_var

                


cs = compsep_sgld(batch_size = 500, snapshot='burn_in')
cmb_mean, cmb_var = cs.sample(n_iter = 120000, target_lnlike = -2.3e7, epsilon = 0.02, snapshot = 'burn_in')
ana.power_calc.save_map('cmb_mean_'+str(cs.n_iter)+'.fits', cmb_mean*cmb_norm, True)
ana.power_calc.save_map('cmb_rms_'+str(cs.n_iter)+'.fits', np.sqrt(cmb_var)*cmb_norm, True)


