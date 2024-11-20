import numpy as np
import emcee
import math
import os

class mcmc_settings:
    logzero = -1.e30

    ##mcmc_engine = 'emcee' or 'simple'
    def __init__(self, nwalkers = 8, burn_steps = 1000, mc_steps = 8000, verbose = True, mcmc_engine = 'emcee', covmat = ''): 
        self.num_params = 0
        self.ind = {}
        self.lower = np.array([])
        self.upper = np.array([])
        self.std = np.array([])
        self.bestfit = np.array([])                
        self.names = []
        self.keys = []
        self.nwalkers = nwalkers
        self.burn_steps = burn_steps
        self.mc_steps = mc_steps
        self.verbose = verbose
        self.mcmc_engine = mcmc_engine 
        self.default_values = {}
        self.covmat_file = covmat
        self.like_mode = 0

    def add_parameters(self, p): #p is a dictionary { name: [latex_symbol, lower, upper, initial value,  step ], ... }
        for key in p.keys():
            if key not in self.ind.keys():
                if(p[key][1] < p[key][2]):
                    self.lower = np.append(self.lower, p[key][1])
                    self.upper = np.append(self.upper, p[key][2])
                    self.ind[key] = self.num_params
                    self.names.append(p[key][0])
                    self.keys.append(key)                    
                    self.num_params += 1
                    if(len(p[key])>3):
                        if(p[key][3] < p[key][1] or p[key][3] >  p[key][2]):
                            print(r'Error in add_parameters: initial value not between lower and upper')
                            print(key, ': ', p[key])
                            exit()
                        self.default_values[key] =  p[key][3]
                    else:
                        self.default_values[key] =  (p[key][1] + p[key][2])/2.
                    self.bestfit = np.append(self.bestfit, self.default_values[key])
                    if(len(p[key])>4):
                        if(p[key][4] > 0.):
                            self.std = np.append(self.std, min(p[key][4], (p[key][2]- p[key][1])/2.))
                        else:
                            print(r'Error in add_parameters: step <= 0')
                            print(key, ' step : ', p[key][4])
                            exit()
                    else:
                        self.std = np.append( self.std, (p[key][2] - p[key][1])/6.)
                elif(p[key][1] == p[key][2]):
                    self.default_values[key] =  p[key][1]
                else:
                    print(r'Error in add_parameters: lower bound > upper bound')
                    print(key, ': ', p[key][1], ' > ', p[key][2])
                    exit()
        self.covmat = np.zeros((self.num_params, self.num_params))        
        if(self.covmat_file == ''):
            for i in range(self.num_params):
                self.covmat[i, i] = self.std[i]**2
        else:
            self.covmat = np.load(self.covmat_file)
            assert(self.covmat.shape[0] == self.num_params and self.covmat.shape[1] == self.num_params)
            for i in range(self.num_params):
                self.std[i] = np.sqrt(self.covmat[i, i])
        if(self.mcmc_engine == 'emcee'):
            if(self.nwalkers < self.num_params*2):
                self.nwalkers = self.num_params*2
        else: #simple mode
            self.nwalkers = 1
#        if(self.verbose):
#            print('list of parameters: name, [lower, upper]')
#            for i in range(self.num_params):
#                print(i, self.names[i], r'[', self.lower[i], r', ', self.upper[i], r', ' , self.bestfit[i], r']')
            

            

    def getp(self, key, x):
        if key in self.ind.keys():
            return x[self.ind[key]]
        else:
            return self.default_values[key]


    def getps(self, key_prefix, n, x):
        y = np.empty(n)
        for i in range(n):
            key = key_prefix + str(i)
            if key in self.ind.keys():
                y[i] = x[self.ind[key]]
            else:
                y[i] = self.default_values[key]
        return y

    def postprocess(self, samples, loglikes):
        assert(samples.shape[0] > 1 and samples.shape[0] == len(loglikes))
        self.mean = np.zeros(self.num_params)
        self.covmat = np.zeros((self.num_params, self.num_params))        
        ibest = -1
        self.bestlike = self.logzero
        for i in range(samples.shape[0]):
            self.mean += samples[i]
            if(loglikes[i] > self.bestlike):
                ibest = i
        self.mean /= samples.shape[0]
        self.bestfit = samples[ibest].copy()
        self.bestlike = loglikes[ibest]        
        for i in range(samples.shape[0]):
            diff = samples[i] - self.mean
            self.covmat += diff[None, :] * diff[:, None]
        self.covmat /= samples.shape[0]
        self.std = [ np.sqrt(self.covmat[i][i]) for i in range(self.num_params) ]
        self.counts = np.zeros( ( self.num_params, 1001))
        for i in range(samples.shape[1]):
            for j in range(samples.shape[0]):
                rpos = ((samples[j][i]-self.mean[i])/self.std[i] + 5.)*100. + 0.5
                npos = int(np.floor(rpos))
                if(npos >= 0 and npos < 1000):
                    self.counts[i, npos+1] +=  rpos - npos
                    self.counts[i, npos] += 1.-(rpos - npos)
                elif(npos < 0):
                    self.counts[i, 0] += 1.
                else:
                    self.counts[i, 1000] += 1.
        self.counts /= samples.shape[0]

    def prob_limits(self, i, prob):
        assert(prob > 0. and prob < 1.)
        addprob = 0.
        for j in range(1001):
            addprob += self.counts[i, j]
            if(addprob > prob):
                return self.mean[i] + self.std[i] * ((j - (addprob - prob)/self.counts[i, j]-0.5)/100 - 5.)


    def medians(self):
        m = np.empty(self.num_params)
        for i in range(self.num_params):
            m[i] = self.prob_limits(i, 0.5)
        return m

    def search_bestfit(self, loglike):
        self.bestlike = loglike(self.bestfit, self)
        step = 0.05
        while(step > 0.0005):
            x_try =  np.random.multivariate_normal(self.bestfit , self.covmat)*step
            like_try = loglike(x_try, self)
            if(like_try > self.bestlike):
                self.bestfit = x_try.copy()
                self.bestlike = like_try
                step *= 1.005
            else:
                step *= 0.98


    def run_simple(self, loglike, continue_from = None, init_values = None, discard_ratio = 0.):  #single thread, ignore nwalkers
        has_sample_cov = False
        if(continue_from is not None):
            if(os.path.exists(continue_from + r'CONTRUN_samples.npy') and os.path.exists(continue_from + r'CONTRUN_loglikes.npy')):
                old_samples = np.load(continue_from + 'CONTRUN_samples.npy')
                old_loglikes = np.load(continue_from + 'CONTRUN_loglikes.npy')               
            else:
                old_samples = np.load(continue_from + 'samples.npy')
                old_loglikes = np.load(continue_from + 'loglikes.npy')
            lold = len(old_loglikes)
            if(lold != old_samples.shape[0]):
                print('Error: ' + continue_from + r'samples.npy and loglikes.npy do not match')
                exit()
            elif(self.verbose):
                print('loaded ' + str(lold) + ' lines from ' + continue_from)
            if(lold > 0):
                self.postprocess(samples = old_samples, loglikes = old_loglikes) #this updates self.covmat
            if(os.path.exists(continue_from + 'MCMC_propose.npy')):
                sample_cov = np.load(continue_from + 'MCMC_propose.npy')
                has_sample_cov = True
        samples = np.empty( (self.mc_steps, self.num_params) )
        loglikes = np.empty(self.mc_steps)
        p_save = np.empty((2, self.num_params))
        like_save = np.empty(2)
        ind_now = 0
        if(init_values is None):
            p_save[ind_now, :] = self.bestfit + self.std * np.random.normal(0., 0.2, self.num_params)
        else:
            p_save[ind_now, :] = init_values +  self.std * np.random.normal(0., 0.03, self.num_params)
        like_save[ind_now] = loglike(p_save[ind_now, :], self)
        self.like_mode = 1
        accept = 0        
        if(not has_sample_cov):
            fac = 1.5 + 0.5 * np.sqrt(self.num_params-0.99)
            sample_cov = self.covmat/(fac-1.)   #initial guess
        while(accept*30. < self.burn_steps and not has_sample_cov):
            accept = 0                        
            for i in range(self.burn_steps):
                p_save[1-ind_now, :] =  np.random.multivariate_normal(p_save[ind_now, :], sample_cov)
                like_save[1-ind_now] = loglike(p_save[1-ind_now, :], self)
                if(like_save[1-ind_now] - like_save[ind_now] >= np.log(1.-np.random.rand()) and like_save[1-ind_now] > -1.e30):
                    ind_now = 1- ind_now
                    accept += 1
                samples[i, :] = p_save[ind_now, :]
                loglikes[i] = like_save[ind_now]
            if(accept > 0):
                self.postprocess(samples = samples[0:self.burn_steps, :], loglikes = loglikes[0:self.burn_steps])
            else:
                print('current parameters=', p_save[ind_now,:])                
                print('current loglike=', like_save[ind_now])
                print('no accepted samples! please check your likelihood')
                exit()
            sample_cov = (sample_cov + self.covmat)/fac
            if(self.verbose):
                print(r'trial accept ratio: ', accept/self.burn_steps, r', best loglike:', self.bestlike)
        self.like_mode = 2  #after initial trial set like_mode to 2, start serious running
        while(accept*4.2 < self.burn_steps and not has_sample_cov):
            accept = 0
            for i in range(self.burn_steps):
                p_save[1-ind_now, :] =  np.random.multivariate_normal(p_save[ind_now, :], sample_cov)
                like_save[1-ind_now] = loglike(p_save[1-ind_now, :], self)
                if(like_save[1-ind_now] - like_save[ind_now] >= np.log(1.-np.random.rand())  and like_save[1-ind_now] > -1.e30):
                    ind_now = 1- ind_now
                    accept += 1
                samples[i, :] = p_save[ind_now, :]
                loglikes[i] = like_save[ind_now]
            if(accept > 0):
                self.postprocess(samples = samples[0:self.burn_steps, :], loglikes = loglikes[0:self.burn_steps])
            else:
                print('current parameters=', p_save[ind_now,:])                                
                print('current loglike=', like_save[ind_now])                
                print('no accepted samples! please check your likelihood')
                exit()
            fac *= 1.2            
            sample_cov = (sample_cov + self.covmat)/fac                
            if(self.verbose):
                print(r'trial* accept ratio: ', accept/self.burn_steps, r', best loglike:', self.bestlike)
        accept = 0
        for i in range(self.burn_steps*2):
            p_save[1-ind_now, :] =  np.random.multivariate_normal(p_save[ind_now, :], sample_cov)
            like_save[1-ind_now] = loglike(p_save[1-ind_now, :], self)
            if(like_save[1-ind_now] - like_save[ind_now] >= np.log(1.-np.random.rand())  and like_save[1-ind_now] > -1.e30):
                ind_now = 1- ind_now
                accept += 1
        if(self.verbose):
            print(r'burned in, now accept ratio: ', accept/(self.burn_steps*2.), r', initial values:', p_save[ind_now, :])
        if(continue_from is not None and not has_sample_cov):
            np.save(continue_from+"MCMC_propose.npy", sample_cov)
            has_sample_cov = True
        for i in range(self.mc_steps):
            if(self.verbose and (i % 1000 == 999)):
                print(r'MCMC step #', i+1, ' / ', self.mc_steps)
            p_save[1-ind_now, :] =  np.random.multivariate_normal(p_save[ind_now, :], sample_cov)
            like_save[1-ind_now] = loglike(p_save[1-ind_now, :], self)
            if(like_save[1-ind_now] - like_save[ind_now] >= np.log(1.-np.random.rand()) and like_save[1-ind_now] > -1.e30):
                ind_now = 1- ind_now
            samples[i, :] = p_save[ind_now, :]
            loglikes[i] = like_save[ind_now]            
        if(continue_from is None or discard_ratio > 0.99):
            return samples, loglikes
        else:
            istart = math.ceil(lold*discard_ratio)
            return np.concatenate( (old_samples[istart:lold, :], samples), axis = 0), np.concatenate( (old_loglikes[istart:lold], loglikes), axis = None ) 
               

    def run_emcee(self, loglike, continue_from = None, init_values = None, discard_ratio = 0.):
        p0 = np.random.rand(self.nwalkers, self.num_params)            
        if(continue_from is None):
            if(init_values is None):
                self.search_bestfit(loglike)
                for j in range(self.nwalkers):    
                    p0[j] = self.bestfit * 0.95 + (p0[j]*(self.upper-self.lower)+self.lower)*0.05
            else:
                for j in range(0, self.nwalkers):
                    p0[j] = init_values * 0.95 + (p0[j]*(self.upper-self.lower)+self.lower)*0.05
        else:
            old_samples = np.load(continue_from + '_samples.npy')
            old_loglikes = np.load(continue_from + '_loglikes.npy')
            lold = len(old_loglikes)
            if(lold != old_samples.shape[0]):
                print('Error: ' + continue_from + r'_samples.npy and _loglikes.npy do not match')
                exit()
            elif(self.verbose):
                print('loaded ' + str(lold) + ' lines from ' + continue_from)
            self.postprocess(samples = old_samples, loglikes = old_loglikes)                
            for j in range(0, self.nwalkers):
                p0[j] = np.random.multivariate_normal(old_samples[lold-1, :], self.covmat)
                while(loglike(p0[j], self) == self.logzero):
                    p0[j] = np.random.multivariate_normal(old_samples[lold-1, :], self.covmat)                    
        if(self.verbose):
            for j in range(0, self.nwalkers):
                print('Initial loglike on node ', j, ' = ', loglike(p0[j], self))
        sampler = emcee.EnsembleSampler(self.nwalkers, self.num_params, loglike, args = [ self ])
        burn = sampler.run_mcmc(p0, self.burn_steps)
        if(self.verbose):
            burn_loglikes = sampler.get_log_prob(flat = True)            
            print('max loglike after burn in:', burn_loglikes.max())
            sampler.reset()
        sampler.run_mcmc(burn, self.mc_steps)
        samples = sampler.get_chain(flat=True)
        loglikes = sampler.get_log_prob(flat = True)
        if(continue_from is None):
            return samples, loglikes
        else:
            istart = math.ceil(lold*discard_ratio)
            return np.concatenate( (old_samples[istart:lold, :], samples), axis = 0), np.concatenate( (old_loglikes[istart:lold], loglikes), axis = None ) 


    def run_mcmc(self, loglike, continue_from = None, init_values = None, discard_ratio = 0.3):
        if(self.mcmc_engine == 'simple'):
            return self.run_simple(loglike, continue_from, init_values, discard_ratio)
        elif(self.mcmc_engine == 'emcee'):
            return self.run_emcee(loglike, continue_from, init_values, discard_ratio)
        else:
            print(self.mcmc_engine + ' is an unknown mcmc engine')
            exit()
