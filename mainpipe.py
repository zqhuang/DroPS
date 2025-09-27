from beforecmb import *
import numpy as np
from sys import argv
from os import path
from mcmc import *

print('-----------------------------------')
print(argv)
print('-----------------------------------')
mc_steps = 80000
burn_steps = 5000
use_Planck_BAO_prior = True
debug = False

Planck_BAO_covmat = np.loadtxt('base_plikHM_TTTEEE_lowl_lowE_lensing_post_BAO.covmat')[0:6, 0:6]
Planck_BAO_invcov = np.linalg.inv(Planck_BAO_covmat)
data_overwrite = False

if(len(argv) > 2):
    if(argv[2] == 'OVERWRITE'):
        data_overwrite = True
        ana = sky_analyser(argv[1])                
    elif(argv[2] != "NONE"):
        ana = sky_analyser(argv[1], root_overwrite = argv[2])
    else:
        ana = sky_analyser(argv[1])        
else:
    ana = sky_analyser(argv[1])


ana.get_data_vector(overwrite = data_overwrite)
ana.get_covmat()
data_chisq = np.dot(ana.data_vec - ana.mean, np.dot(ana.invcov, ana.data_vec - ana.mean))/ana.fullsize

if(ana.verbose):
    print("data chi^2=",data_chisq )
    print('BB filtering:')
    print(ana.filters[ana.BB_loc].diag)
if(data_chisq > 20.): #seesms something goes wrong
    print('Error: data seem to be very different from simulations!')
    for ifield in range(ana.num_fields):
        print('-------------' + ana.fields[ifield]+'---------------')
        for ipower in range(ana.num_powers):
            ifreq1, ifreq2 = ana.freq_indices(ipower)
            print('----'+ana.freqnames[ifreq1]+' x '+ana.freqnames[ifreq2]+'----')
            for il in range(ana.num_ells):
                k = il*ana.blocksize + ipower*ana.num_fields + ifield
                print(ana.ells[il], ana.data_vec[k], ana.mean[k],  (ana.data_vec[k]- ana.mean[k])**2/ana.covmat[k, k])
    exit()


samples_file = ana.output_root + r'samples.npy'
loglikes_file = ana.output_root + r'loglikes.npy'
bestfit_file = ana.output_root + r'bestfit.npy'
covmat_file = ana.output_root + r'covmat.npy'
slow_propose_file = ana.output_root + r'slow_propose.npy'
fast_propose_file = ana.output_root + r'fast_propose.npy'

    
        
params = {}

#cosmological parameters
if(ana.BB_loc >= 0):
    params['r'] = [ r'$r$', ana.r_lowerbound, 0.2, 0.008, 0.004 ]
else:
    params['r'] = [ r'$r$', 0., 0. ]
if(ana.vary_cosmology):
    for i in range(6):
        params[cosmology_base_name[i]] = [ cosmology_base_name[i], max(ana.lcdm_params[i] - cosmology_base_std[i]*4., 0.001), ana.lcdm_params[i] + cosmology_base_std[i]*4., ana.lcdm_params[i] ] # , cosmology_base_std[i] ]
else:
    for i in range(6):
        params[cosmology_base_name[i]] = [ cosmology_base_name[i], ana.lcdm_params[i], ana.lcdm_params[i] ]
        
#foreground parameters
params[r'T_dust_MBB'] =  [r'$T_{\rm MBB}$', 20., 20. ] 

params[r'beta_d'] =  [r'$\beta_d$', 1.2, 1.9, 1.54, 0.01 ]
params[r'beta_s'] =  [r'$\beta_s$', -3.5, -2.5, -3., 0.01 ]

ME_ubd = ana.ME_upperbound
if(ana.ME_is_positive):
    ME_lbd = 0.
    ME_ini = ME_ubd/10.
else:
    ME_lbd = - ME_ubd
    ME_ini = 0.

    

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
else:
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
                params[r'A_s_' + field  + str(i)] = [ r'$A_{s, ' + field + r',' + str(i) + r'}$',  0., sync_approx[i]*4.+5., sync_approx[i], (sync_approx[i]*4.+5.)/200. ]
                params[r'A_d_' + field  + str(i)] = [ r'$A_{d, ' + field + r',' + str(i) + r'}$',  0., dust_approx[i]*4.+20.,  dust_approx[i], (dust_approx[i]*4.+20.)/200. ]
                params[r'eps2_' + field  + str(i)] = [ r'$\varepsilon_{2, ' + field + r',' + str(i) + r'}$',  -ana.eps_upperbound, ana.eps_upperbound,  eps_approx[i] ] #fast parameter
            else:
                params[r'A_s_' + field  + str(i)] = [ r'$A_{s, ' + field + r',' + str(i) + r'}$',   sync_approx[i],  sync_approx[i] ]
                params[r'A_d_' + field  + str(i)] = [ r'$A_{d, ' + field + r',' + str(i) + r'}$',  dust_approx[i], dust_approx[i] ]
                params[r'eps2_' + field  + str(i)] = [ r'$\varepsilon_{2, ' + field + r',' + str(i) + r'}$',  0., 0. ]

if(len(argv) == 4):
    settings = mcmc_settings(burn_steps = burn_steps, mc_steps = mc_steps, verbose = ana.verbose, covmat=argv[3])        
else:
    settings = mcmc_settings(burn_steps = burn_steps, mc_steps = mc_steps, verbose = ana.verbose)    
settings.add_parameters(params)


##this is for getdist plots
if(not path.exists(ana.output_root + r'param_keys.txt')):
    with open(ana.output_root + r'param_keys.txt', 'w') as f:
        f.write(r'[')
        for i in range(settings.num_params-1):
            f.write(r'"' + settings.keys[i] + r'", ')
        f.write(r'"' + settings.keys[settings.num_params-1] + r'" ]')

if(not path.exists(ana.output_root + r'param_labels.txt')):        
    with open(ana.output_root + r'param_labels.txt', 'w') as f:
        f.write(r'[')
        for i in range(settings.num_params-1):
            f.write(r'r"' + settings.names[i].replace('$', '') + r'", ')
        f.write(r'r"' + settings.names[settings.num_params-1].replace('$', '') + r'" ]')




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


if(ana.continue_run):
    samples, loglikes = settings.run_mcmc(cmb_loglike, continue_from = ana.output_root, discard_ratio = ana.discard_ratio)    
elif(len(argv) == 5):
    samples, loglikes = settings.run_mcmc(cmb_loglike, slow_propose = argv[3], fast_propose = argv[4])    
else:
    samples, loglikes = settings.run_mcmc(cmb_loglike)

settings.postprocess(samples = samples, loglikes = loglikes)

    
#####save the results##################
np.save(samples_file, samples)
np.save(loglikes_file, loglikes)
np.save(bestfit_file, settings.global_bestfit)
np.save(covmat_file, settings.covmat)
np.save(slow_propose_file, settings.slow_propose)
np.save(fast_propose_file, settings.fast_propose)
with open(ana.r_logfile, 'a') as f:
    f.write(str(settings.mean[0]) + " " + str(settings.std[0]) + "\n")

if(ana.verbose):
    numecho = settings.num_params
else:
    numecho = 3

if(ana.verbose):
    print("-----------mean +/- standard deviation---------")    
for i in range(numecho):
    print(settings.names[i], r'=', settings.mean[i], r'\pm ', settings.std[i], r', best: ', settings.global_bestfit[i])
if(ana.verbose):
    print('best loglike = : ', np.round(settings.global_bestlike, 2))

#print('refining bestfit searching steps...')    
#settings.search_bestfit(cmb_loglike)
#print('now best loglike =  ', settings.global_bestlike)                         
    

with open(ana.output_root + r'margestat.txt', 'w') as f:
    f.write("# name     mean  std  median plus_sigma minus_sigma 95-upper 95-lower bestfit\n")    
    for i in range(settings.num_params):
        median = settings.prob_limits(i, 0.5)
        upper1sig = settings.prob_limits(i, 0.8415)
        lower1sig = settings.prob_limits(i, 0.1585)        
        upper95 = settings.prob_limits(i, 0.95)
        lower95 = settings.prob_limits(i, 0.05)                
        f.write(settings.names[i] + ' ' + str(settings.mean[i]) + ' ' + str(settings.std[i]) + ' ' + str(median) + ' ' + str(upper1sig - median) + ' ' + str(median - lower1sig) + ' '+ str(upper95) + ' ' + str(lower95) + ' ' + str(settings.global_bestfit[i]) + "\n")
    f.write("#best loglike: "+str(np.round(settings.global_bestlike, 2)))            
    

if(debug):
    fgs = params_to_fg_models(settings.global_bestfit, settings)
    vec =  ana.model_vector(lcdm_params = ana.lcdm_params, fgs = fgs)
    for ipower in range(ana.num_powers):
        ifreq1, ifreq2 = ana.freq_indices(ipower)
        print('====================' + ana.freqnames[ifreq1] + " x " + ana.freqnames[ifreq2] + '====================')
        chisq = 0.
        for i in range(ana.num_ells):
            k = ipower*ana.num_fields + ana.BB_loc + i*ana.blocksize
            dev = (vec[k] - ana.data_vec[k])/np.sqrt(ana.covmat[k, k])
            if(dev>3.):
                print(ana.ells[i], vec[k], ana.data_vec[k], np.round(dev, decimals=1))
            chisq += dev**2
        print('------chi^2 = '+str(np.round(chisq, decimals=1)))

print("\n")        

