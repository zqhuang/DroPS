from beforecmb import *
import numpy as np
from sys import argv
from os import path, system
from mcmc import *

##Usage##
##python subl.py sim_config_file ana_config_file

##this defines the parameter space you want to search
alpha_min = 1.
alpha_max = 2.2
fd_min = 0.
fd_max = 1.
r_min = 0.  #well r is always marginalized so it is not really counted as a dimension
r_max = 0.028
lower_bounds = np.array([alpha_min, fd_min, r_min])  #r must be the last one
upper_bounds = np.array([alpha_max, fd_max, r_max])
unit_vector = upper_bounds - lower_bounds
#----------------------------------------
subl_root = r'subl_workdir/subl_map_'
logfile = argv[3]
num_sims = 5000  #the number of simulations wanted
num_likeparams = 3  #alpha, fd, r
sim_data = np.zeros((num_sims, num_likeparams+2 ))  #..., r, r_output, r_std
isim = 0

mc_steps = 50000
burn_steps = 2500

use_Planck_BAO_prior = True
Planck_BAO_covmat = np.loadtxt('base_plikHM_TTTEEE_lowl_lowE_lensing_post_BAO.covmat')[0:6, 0:6]
Planck_BAO_invcov = np.linalg.inv(Planck_BAO_covmat)


if(path.exists(logfile)):
    done_data = np.loadtxt(logfile)
    isim += done_data.shape[0]
    sim_data[0:isim, :] = done_data
#-----------------------------------------

def force(isim, pos_vec):
    f =  np.random.normal( scale = 0.2, size = num_likeparams )*unit_vector #small noise to avoid lock-in
    for i in range(isim):
        f += (((sim_data[i, num_likeparams] - sim_data[i, num_likeparams-1])/sim_data[i, num_likeparams+1])**2/(1.e2*np.sum(((pos_vec - sim_data[i, 0:num_likeparams])/unit_vector)**2) + 1.e-2))**1.5 * (pos_vec - sim_data[i, 0:num_likeparams])
    return f

def move_step(isim, pos_vec, vel_vec, dt = 1.):
    vel_vec = vel_vec*np.random.rand() + force(isim, pos_vec) * dt 
    pos_vec += vel_vec * dt
    for i in range(num_likeparams):
        while(pos_vec[i] > upper_bounds[i] or pos_vec[i] < lower_bounds[i]):
            if(pos_vec[i] < lower_bounds[i]):
                pos_vec[i] = 2*lower_bounds[i] - pos_vec[i]
            else:
                pos_vec[i] = 2*upper_bounds[i] - pos_vec[i]
    return pos_vec, vel_vec


sim = sky_simulator(config_file=argv[1], root_overwrite=subl_root)
mkdir_for_file(sim.root)
ana = sky_analyser(config_file = argv[2])
ME_ubd = ana.ME_upperbound
if(ana.ME_is_positive):
    ME_lbd = 0.
    ME_ini = ME_ubd/10.
else:
    ME_lbd = - ME_ubd
    ME_ini = 0.
params = {}
params['r'] = [ r'$r$', ana.r_lowerbound, 0.2, 0.008, 0.004 ]
if(ana.vary_cosmology):
    for i in range(6):
        params[cosmology_base_name[i]] = [ cosmology_base_name[i], max(ana.lcdm_params[i] - cosmology_base_std[i]*4., 0.001), ana.lcdm_params[i] + cosmology_base_std[i]*4., ana.lcdm_params[i] ] # , cosmology_base_std[i] ]
else:
    for i in range(6):
        params[cosmology_base_name[i]] = [ cosmology_base_name[i], ana.lcdm_params[i], ana.lcdm_params[i] ]
params[r'T_dust_MBB'] =  [r'$T_{\rm MBB}$', 20., 20. ] 
params[r'beta_d'] =  [r'$\beta_d$', 1.2, 1.9, 1.54, 0.01 ]
params[r'beta_s'] =  [r'$\beta_s$', -3.5, -2.5, -3., 0.01 ]
if(ana.dust_vary_SED):
    for field in ana.fields:
        if(ana.analytic_ME):
            params[r'B_d_'+field] = [r'$B_{d,' + field + r'}$', ME_lbd, ME_ubd, ME_ini]            
            params[r'S_d_'+field] = [ r'$S_{d' + field + r'}$', ME_lbd, ME_ubd, ME_ini]
            params[r'gamma_d_'+field] = [ r'$\gamma_{d,' + field + r'}$', -5., 0., -0.5 ]
        else:
            for i in range(ana.num_ells):
                params[r'B_d_'+field+str(i)] = [r'$B_{d,'+field+r','+str(i)+r'}$', ME_lbd, ME_ubd, ME_ini]                
                params[r'S_d_'+field+str(i)] = [r'$S_{d,'+field+r','+str(i)+r'}$', ME_lbd, ME_ubd, ME_ini]
else:
    for field in ana.fields:
        if(ana.analytic_ME):
            params[r'B_d_'+field] = [r'$B_{d,' + field + r'}$', 0., 0.]            
            params[r'S_d_'+field] = [ r'$S_{d,' + field + r'}$', 0., 0.]
            params[r'gamma_d_'+field] = [ r'$\gamma_{d,' + field + r'}$', 0., 0.]        
        else:
            for i in range(ana.num_ells):
                params[r'B_d_'+field+str(i)] = [r'$B_{d,'+field+r','+str(i)+r'}$', 0., 0.]                                
                params[r'S_d_'+field+str(i)] = [r'$S_{d,' + field + r','+str(i) + r'}$', 0., 0. ]

if(ana.sync_vary_SED):
    for field in ana.fields:
        if(ana.analytic_ME):
            params[r'B_s_'+field] = [r'$B_{s,' + field + r'}$', ME_lbd, ME_ubd, ME_ini]            
            params[r'S_s_'+field] = [ r'$S_{s' + field + r'}$', ME_lbd, ME_ubd, ME_ini]
            params[r'gamma_s_'+field] = [ r'$\gamma_{s,' + field + r'}$', -5., 0., -0.5]
        else:
            for i in range(ana.num_ells):
                params[r'B_s_'+field+str(i)] = [r'$B_{s,'+field+r','+str(i)+r'}$', ME_lbd, ME_ubd, ME_ini]                                
                params[r'S_s_'+field+str(i)] = [r'$S_{s,'+field+r','+str(i)+r'}$', ME_lbd, ME_ubd, ME_ini]
else:
    for field in ana.fields:
        if(ana.analytic_ME):
            params[r'B_s_'+field] = [r'$B_{s,' + field + r'}$', 0., 0.]            
            params[r'S_s_'+field] = [ r'$S_{s,' + field + r'}$', 0., 0.]
            params[r'gamma_s_'+field] = [ r'$\gamma_{s,' + field + r'}$', 0., 0.]        
        else:
            for i in range(ana.num_ells):
                params[r'B_s_'+field+str(i)] = [r'$B_{s,'+field+r','+str(i)+r'}$', 0., 0.]                                                
                params[r'S_s_'+field+str(i)] = [r'$S_{s,' + field + r','+str(i) + r'}$', 0., 0. ]
if(ana.analytic_fg):
    for field in ana.fields:
        params[r'A_d_'+field] = [r'$A_{d, ' + field + r'}$', 0., 500., 50., 5.]
        params[r'alpha_d_'+field] = [r'$\alpha_{d, ' + field + r'}$', -4., 0., -1., 0.2]
        params[r'alpha_prime_d_'+field] = [r'$\alpha^\prime_{d, ' + field + r'}$', -1., 1., 0.]                
        params[r'A_s_'+field] = [r'$A_{s, ' + field + r'}$', 0., 50., 10., 2.]
        params[r'alpha_s_'+field] = [r'$\alpha_{s, ' + field + r'}$', -4., 0., -0.5, 0.1]
        params[r'alpha_prime_s_'+field] = [r'$\alpha^\prime_{s, ' + field + r'}$', -1., 1., 0.]
        for field in  ana.fields:
            params[r'eps2_' + field] = [r'\varepsilon_{2,' + field + r'}$', -ana.eps_upperbound, ana.eps_upperbound, 0.]
            params[r'alpha_eps_' + field] = [r'\alpha_{\varepsilon,' + field + r'}$', 0., 4., 0.5]            
    
ana.get_covmat()

def params_to_lcdm_params(x, s):
    return np.array( [ s.getp('ombh2', x), s.getp('omch2', x), s.getp('theta', x), s.getp('tau', x), s.getp('logA', x), s.getp('ns', x), s.getp('r', x) ])    

def params_to_fg_models(x, s):
    fgs = []
    if(ana.analytic_fg): #in this case we also force analytic_ME
        for field in ana.fields:
            fgs.append( foreground_model(ells = ana.power_calc.ells, T_dust_MBB = s.getp('T_dust_MBB', x), eps2 = s.getp(r'eps2_'+field, x), alpha_eps =  s.getp('alpha_eps_'+field, x), A_sync =  s.getp('A_s_'+field, x), alpha_sync = s.getp('alpha_s_'+field,x ), run_sync =  s.getp('alpha_prime_s_'+field, x), beta_sync =  s.getp('beta_s',x), A_dust =  s.getp('A_d_'+field,x), alpha_dust =  s.getp('alpha_d_'+field,x), run_dust = s.getp('alpha_prime_d_'+field,x), beta_dust = s.getp('beta_d',x), B_dust = s.getp('B_d_'+field, x), svs_dust = s.getp('S_d_'+field, x), svs_dust_index = s.getp('gamma_d_'+field, x), B_sync = s.getp('B_s_'+field, x), svs_sync = s.getp('S_s_'+field, x), svs_sync_index = s.getp('gamma_s_'+field, x), freq_sync_ref = ana.freq_lowest, freq_dust_ref = ana.freq_highest, ell_ref = 80.) )                
    elif(ana.analytic_ME):
        for field in ana.fields:
            fgs.append( foreground_model(ells = ana.power_calc.ells, T_dust_MBB = s.getp('T_dust_MBB', x),  eps2 = s.getps('eps2_'+field, ana.num_ells, x), alpha_eps = 0., A_sync =  s.getps('A_s_'+field, ana.num_ells, x), alpha_sync = 0., run_sync = 0., beta_sync =  s.getp('beta_s',x), A_dust =  s.getps('A_d_'+field, ana.num_ells, x), alpha_dust = 0., run_dust = 0.,  beta_dust = s.getp('beta_d',x), B_dust = s.getp('B_d_'+field, x), svs_dust = s.getp('S_d_'+field, x), svs_dust_index = s.getp('gamma_d_'+field, x), B_sync = s.getp('B_s_'+field, x), svs_sync = s.getp('S_s_'+field, x), svs_sync_index = s.getp('gamma_s_'+field, x), freq_sync_ref = ana.freq_lowest, freq_dust_ref = ana.freq_highest, ell_ref = 80.) )
    else:
        for field in ana.fields:
            fgs.append( foreground_model(ells = ana.power_calc.ells, T_dust_MBB = s.getp('T_dust_MBB', x),  eps2 = s.getps('eps2_'+field, ana.num_ells, x), alpha_eps = 0., A_sync =  s.getps('A_s_'+field, ana.num_ells, x), alpha_sync = 0., run_sync = 0., beta_sync =  s.getp('beta_s',x), A_dust =  s.getps('A_d_'+field, ana.num_ells, x), alpha_dust = 0., run_dust = 0.,  beta_dust = s.getp('beta_d',x), B_dust = s.getps('B_d_'+field, ana.num_ells, x), svs_dust = s.getps('S_d_'+field, ana.num_ells, x), B_sync = s.getps('B_s_'+field, ana.num_ells, x), svs_sync = s.getps('S_s_'+field, ana.num_ells, x) , freq_sync_ref = ana.freq_lowest, freq_dust_ref = ana.freq_highest, ell_ref = 80.) )
    return fgs



def cmb_loglike(x, s):
    assert(len(x) == s.num_params)    
    for i in range(s.num_params):
        if(x[i] < s.lower[i] or x[i] > s.upper[i]):
            return s.logzero
    fgs = params_to_fg_models(x, s)
    lcdm_params =  params_to_lcdm_params(x, s)
    r_value = lcdm_params[6]
    vec =  ana.model_vector(lcdm_params = lcdm_params, fgs = fgs)  - ana.data_vec
    chisq = ana.chisq_of_vec(vec, r_value)
    if(ana.vary_cosmology and use_Planck_BAO_prior):
        d_lcdm = lcdm_params[0:6] - cosmology_base_mean[0:6]
        chisq +=  np.dot(np.dot(d_lcdm, Planck_BAO_invcov), d_lcdm)  #add Planck+BAO prior
    if(ana.beta_d_prior is not None):
        chisq += ((s.getp("beta_d", x) - ana.beta_d_prior[0])/ana.beta_d_prior[1])**2
    if(ana.beta_s_prior is not None):
        chisq += ((s.getp("beta_s", x) - ana.beta_s_prior[0])/ana.beta_s_prior[1])**2
    return -chisq/2. 

if(isim == 0):
    pos_vec = (lower_bounds + upper_bounds)/2.
    vel_vec = np.random.normal(size = num_likeparams) * unit_vector
else:
    pos_vec = sim_data[isim-1, 0:num_likeparams]
    vel_vec = np.random.normal(size = num_likeparams) * unit_vector
pos_vec, vel_vec = move_step(isim, pos_vec, vel_vec, 1.)
    
while(isim < num_sims):
    print("\n########## simulation ", isim)
    print('position: ', pos_vec)
    print('velocity: ', vel_vec)
    sim.simulate_map(r=pos_vec[num_likeparams-1])  
    ana.root = sim.root + cmb_postfix_for_r(pos_vec[num_likeparams-1])
    ana.get_data_vector(overwrite = True)
    ana.r_interp_index = pos_vec[0]
    ana.r_lndet_fac = pos_vec[1]
    data_chisq = np.dot(ana.data_vec - ana.mean, np.dot(ana.invcov, ana.data_vec - ana.mean))/ana.fullsize
    print('data chi^2 = ', data_chisq)
    if(not ana.analytic_fg):
        fgm = foreground_model(ells = ana.power_calc.ells, freq_sync_ref = ana.freq_lowest, freq_dust_ref = ana.freq_highest, ell_ref = 80.)    
        for field in ana.fields:
            sync_approx = ana.select_spectrum(vec = ana.data_vec - ana.lens_res, ifreq1 = ana.ifreq_lowest, ifreq2 = ana.ifreq_lowest, field = field) / ana.beam_filter2(ifreq1 = ana.ifreq_lowest, ifreq2 = ana.ifreq_lowest) / fgm.sync_freq_weight(ana.freq_lowest)**2       
            dust_approx = ana.select_spectrum(vec = ana.data_vec  - ana.lens_res, ifreq1 = ana.ifreq_highest, ifreq2 = ana.ifreq_highest, field = field) / ana.beam_filter2(ifreq1 = ana.ifreq_highest, ifreq2 = ana.ifreq_highest) / fgm.dust_freq_weight(ana.freq_highest)**2
            eps_approx = ana.select_spectrum(vec = ana.data_vec- ana.lens_res, ifreq1 = ana.ifreq_lowest, ifreq2 = ana.ifreq_highest, field = field) / ana.beam_filter2(ifreq1 = ana.ifreq_lowest, ifreq2 = ana.ifreq_highest) / np.sqrt(abs(sync_approx * dust_approx)) / (fgm.sync_freq_weight(ana.freq_lowest) * fgm.dust_freq_weight(ana.freq_highest)+fgm.sync_freq_weight(ana.freq_highest) * fgm.dust_freq_weight(ana.freq_lowest))
            for i in range(ana.num_ells):
                dust_approx[i] = max(0.01, dust_approx[i])                
                sync_approx[i] = max(0.05, sync_approx[i])
                eps_approx[i] = max(min(eps_approx[i], ana.eps_upperbound*0.8), -ana.eps_upperbound*0.8)
                if(ana.ell_used_indices[i]):
                    params[r'A_s_' + field  + str(i)] = [ r'$A_{s, ' + field + r',' + str(i) + r'}$',  0., sync_approx[i]*4.+5., sync_approx[i], (sync_approx[i]*4.+5.)/320. ]
                    params[r'A_d_' + field  + str(i)] = [ r'$A_{d, ' + field + r',' + str(i) + r'}$',  0., dust_approx[i]*4.+20.,  dust_approx[i], (dust_approx[i]*4.+20.)/320. ]
                    params[r'eps2_' + field  + str(i)] = [ r'$\varepsilon_{2, ' + field + r',' + str(i) + r'}$',  -ana.eps_upperbound, ana.eps_upperbound,  eps_approx[i] ] #fast parameter
                else:
                    params[r'A_s_' + field  + str(i)] = [ r'$A_{s, ' + field + r',' + str(i) + r'}$',   sync_approx[i],  sync_approx[i] ]
                    params[r'A_d_' + field  + str(i)] = [ r'$A_{d, ' + field + r',' + str(i) + r'}$',  dust_approx[i], dust_approx[i] ]
                    params[r'eps2_' + field  + str(i)] = [ r'$\varepsilon_{2, ' + field + r',' + str(i) + r'}$',  0., 0. ]
    settings = mcmc_settings(burn_steps = burn_steps, mc_steps = mc_steps, verbose = True)    
    settings.add_parameters(params)
    samples, loglikes = settings.run_mcmc(cmb_loglike)
    settings.postprocess(samples = samples, loglikes = loglikes)    
    r_output = settings.mean[0]
    r_std = settings.std[0]
    write_str = ''
    for i in range(num_likeparams):
        write_str += (str(pos_vec[i]) + ' ')
    write_str += str(r_output) + ' ' + str(r_std) + "\n"
    f =  open(logfile, 'a') 
    f.write(write_str)
    f.close()
    del settings
    system('rm -f ' + ana.root + r'*.npy')
    sim_data[isim, 0:num_likeparams] = pos_vec
    sim_data[isim, num_likeparams] = r_output
    sim_data[isim, num_likeparams+1] = r_std
    pos_vec, vel_vec = move_step(isim, pos_vec, vel_vec, 0.2)    
    isim += 1
                            
