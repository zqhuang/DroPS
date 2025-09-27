import numpy as np
import math
import os

class mcmc_settings:
    logzero = -1.e30

    def __init__(self, burn_steps = 2000, mc_steps = 50000, verbose = True, covmat = ''): 
        self.num_params = 0
        self.ind = {}
        self.lower = np.array([])
        self.upper = np.array([])
        self.is_slow = []
        self.slow_indices = []
        self.fast_indices = []
        self.std = np.array([])
        self.bestfit = np.array([])
        self.global_bestlike = self.logzero
        self.bestlike = self.logzero
        self.names = []
        self.keys = []
        self.burn_steps = burn_steps
        self.mc_steps = mc_steps
        assert(self.mc_steps >= self.burn_steps*3) #for good convergence 
        self.verbose = verbose
        self.default_values = {}
        self.covmat_file = covmat
        self.like_mode = 0
        self.num_counts = 500

    def add_parameters(self, p): #p is a dictionary { name: [latex_symbol, lower, upper, initial value,  step ], ... }
        for key in p.keys():
            if key not in self.ind.keys():
                if(p[key][1] < p[key][2]):
                    self.lower = np.append(self.lower, p[key][1])
                    self.upper = np.append(self.upper, p[key][2])
                    self.ind[key] = self.num_params
                    self.names.append(p[key][0])
                    self.keys.append(key)                    
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
                        self.is_slow.append(True)
                        self.slow_indices.append(self.num_params)
                        if(p[key][4] > 0.):
                            self.std = np.append(self.std, min(p[key][4], (p[key][2]- p[key][1])/2.))
                        else:
                            print(r'Error in add_parameters: step <= 0')
                            print(key, ' step : ', p[key][4])
                            exit()
                    else:
                        self.is_slow.append(False)
                        self.fast_indices.append(self.num_params)                        
                        self.std = np.append( self.std, (p[key][2] - p[key][1])/20.)
                    self.num_params += 1                    
                elif(p[key][1] == p[key][2]):
                    self.default_values[key] =  p[key][1]
                else:
                    print(r'Error in add_parameters: lower bound > upper bound')
                    print(key, ': ', p[key][1], ' > ', p[key][2])
                    exit()
        self.num_slow = len(self.slow_indices)
        self.num_fast = self.num_params - self.num_slow
        self.covmat = np.zeros((self.num_params, self.num_params))
        self.slow_covmat = np.empty((self.num_slow, self.num_slow))
        self.fast_std = np.empty( self.num_fast )
        if(self.covmat_file == ''):
            for i in range(self.num_params):
                self.covmat[i, i] = self.std[i]**2
        else:
            self.covmat = np.load(self.covmat_file)
            assert(self.covmat.shape[0] == self.num_params and self.covmat.shape[1] == self.num_params)
            for i in range(self.num_params):
                self.std[i] = np.sqrt(self.covmat[i, i])
        self.slow_covmat = self.covmat[self.slow_indices][:, self.slow_indices]
        self.fast_std = self.std[self.fast_indices]

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

    def postprocess(self, samples, loglikes, get_std = True, get_counts = True):
        assert(samples.shape[0] > 1 and samples.shape[0] == len(loglikes))
        assert(samples.shape[1] == self.num_params)
        self.mean = np.zeros(self.num_params)
        self.covmat = np.zeros((self.num_params, self.num_params))        
        ibest = -1
        self.bestlike = self.logzero
        for i in range(samples.shape[0]):
            self.mean += samples[i]
            if(loglikes[i] > self.bestlike):
                ibest = i
                self.bestlike = loglikes[i]
        self.mean /= samples.shape[0]
        self.bestfit = samples[ibest].copy()
        if(self.bestlike > self.global_bestlike):
            self.global_bestlike = self.bestlike
            self.global_bestfit = self.bestfit.copy()
        for i in range(samples.shape[0]):
            diff = samples[i] - self.mean
            self.covmat += diff[None, :] * diff[:, None]
        self.covmat /= samples.shape[0]
        self.slow_covmat = self.covmat[self.slow_indices][:, self.slow_indices]
        if(not get_std):
            return
        self.std = np.array([ np.sqrt(self.covmat[i][i]) for i in range(self.num_params) ])
        self.fast_std = self.std[self.fast_indices]
        if(not get_counts):
            return
        nbm = self.num_counts * 2         
        self.counts = np.zeros( ( self.num_params, nbm + 1))
        for i in range(samples.shape[1]):
            for j in range(samples.shape[0]):
                rpos = ((samples[j][i]-self.mean[i])/self.std[i]/10. + 0.5)*nbm + 0.5
                npos = int(np.floor(rpos))
                if(npos >= 0 and npos < nbm):
                    self.counts[i, npos+1] +=  (rpos - npos)
                    self.counts[i, npos] += 1.-(rpos - npos)
                elif(npos < 0):
                    self.counts[i, 0] += 1.
                else:
                    self.counts[i, nbm] += 1.
        self.counts /= samples.shape[0]

    def prob_limits(self, i, prob):
        assert(prob > 0. and prob < 1.)
        addprob = 0.
        for j in range(2*self.num_counts + 1):
            addprob += self.counts[i, j]
            if(addprob >= prob):
                return self.mean[i] + self.std[i] * ((j - (addprob - prob)/self.counts[i, j]-0.5)*5./self.num_counts - 5.)
            
    def medians(self):
        m = np.empty(self.num_params)
        for i in range(self.num_params):
            m[i] = self.prob_limits(i, 0.5)
        return m

    def search_bestfit(self, loglike):
        step = 0.05
        while(step > 1.e-5):
            x_try =  np.random.multivariate_normal(self.global_bestfit , self.covmat*step)
            like_try = loglike(x_try, self)
            if(like_try > self.global_bestlike):
                self.global_bestfit = x_try.copy()
                self.global_bestlike = like_try
                step *= 1.005
            else:
                step *= 0.975

    def run_mcmc(self, loglike, continue_from = None, init_values = None, discard_ratio = 0., slow_propose = None, fast_propose = None):  #single thread, ignore nwalkers
        try_propose = True
        if(continue_from is not None):
            old_samples = np.load(continue_from + 'samples.npy')
            old_loglikes = np.load(continue_from + 'loglikes.npy')
            lold = len(old_loglikes)
            if(lold != old_samples.shape[0]):
                print('Error: ' + continue_from + r'samples.npy and loglikes.npy do not match')
                fix = input("fix the loglikes file? (Y/N)")
                if(fix == "Y"):
                    nskip = 0                    
                    old_loglikes = np.empty(old_samples.shape[0])
                    old_loglikes[0] = loglike(old_samples[0, :], self)
                    if(old_loglikes[0] <= self.logzero):
                        print("Error: get logZero for parameters: ", old_samples[0, :], old_loglikes[0])                        
                        exit()
                    i=1
                    skip = False
                    while(i < old_samples.shape[0] - nskip):
                        if(np.sum(abs(old_samples[i, :] - old_samples[i-1, :]))<1.e-12):
                            old_loglikes[i] = old_loglikes[i-1]
                            i += 1
                        else:
                            old_loglikes[i] = loglike(old_samples[i, :], self)
                            if(old_loglikes[i] > self.logzero):
                                i += 1
                            else:
                                print("Error: get logZero for parameters: ", old_samples[i, :], old_loglikes[i])
                                if(nskip == 0):
                                    skip = (input("skip this line? (Y/N)") == "Y")
                                if(skip):
                                    nskip += 1
                                    old_samples[i, :] = old_samples[old_samples.shape[0]-nskip, :].copy()
                                    
                                else:
                                    exit()
                        if(i % 100 == 0):
                            print("progress: ", np.round(i*100./old_samples.shape[0], 2), r"%, skipped lines: ", nskip)
                    print("finished, skipped lines: ", nskip)
                    if(nskip == 0):
                        np.save(continue_from + 'loglikes.npy', old_loglikes)
                    else:
                        np.save(continue_from + 'loglikes.npy', old_loglikes[0:old_samples.shape[0]-nskip])
                        np.save(continue_from + 'samples.npy', old_samples[0:old_samples.shape[0]-nskip, :])                                                
                exit()
            elif(self.verbose):
                print('loaded ' + str(lold) + ' lines from ' + continue_from)
            if(lold > 0):
                self.postprocess(samples = old_samples, loglikes = old_loglikes, get_std = True, get_counts=False) #this updates self.covmat
            if(os.path.exists(continue_from + 'slow_propose.npy') and os.path.exists(continue_from + 'fast_propose.npy')):
                self.slow_propose = np.load(continue_from + 'slow_propose.npy')
                assert( self.slow_propose.shape[0] == self.num_slow and self.slow_propose.shape[1] == self.num_slow)
                self.fast_propose = np.load(continue_from + 'fast_propose.npy')
                assert( len(self.fast_propose) == self.num_fast )
                try_propose = False
        elif((slow_propose is not None) and (fast_propose is not None)):
            self.slow_propose = np.load(slow_propose)
            assert( self.slow_propose.shape[0] == self.num_slow and self.slow_propose.shape[1] == self.num_slow)            
            self.fast_propose = np.load(fast_propose)
            assert( len(self.fast_propose) == self.num_fast )            
            try_propose = False
        samples = np.empty( (self.mc_steps, self.num_params) )
        loglikes = np.empty(self.mc_steps)
        p_save = np.empty((2, self.num_params))
        like_save = np.empty(2)
        ind_now = 0
        if(init_values is None):
            p_save[ind_now, self.slow_indices] = self.bestfit[self.slow_indices] + self.std[self.slow_indices] * np.random.normal(0., 0.2, self.num_slow)
            p_save[ind_now, self.fast_indices] = self.bestfit[self.fast_indices]
        else:
            p_save[ind_now, :] = init_values 
        like_save[ind_now] = loglike(p_save[ind_now, :], self)
        self.global_bestlike = like_save[ind_now]
        self.global_bestfit = p_save[ind_now, :].copy()
        p_save[1-ind_now, self.fast_indices ] =p_save[ind_now, self.fast_indices]        
        self.like_mode = 1
        accept = 0
        if(try_propose):  
            fac = 2. + np.sqrt(self.num_params-0.99)
            self.slow_propose = self.slow_covmat/(fac-1.)   #initial guess
        miansi = 5
        while(try_propose and accept*30. < self.burn_steps):
            accept = 0                        
            for i in range(self.burn_steps):
                p_save[1-ind_now, self.slow_indices] =  np.random.multivariate_normal(p_save[ind_now, self.slow_indices], self.slow_propose)
                #p_save[1-ind_now, self.fast_indices] = p_save[ind_now, self.fast_indices] + self.fast_propose*np.random.normal(size=self.num_fast)
                like_save[1-ind_now] = loglike(p_save[1-ind_now, :], self)
                if(like_save[1-ind_now] - like_save[ind_now] >= np.log(1.-np.random.rand()) and like_save[1-ind_now] > self.logzero):
                    ind_now = 1- ind_now
                    accept += 1
                samples[i, :] = p_save[ind_now, :]
                loglikes[i] = like_save[ind_now]
            if(accept > 0):
                self.postprocess(samples = samples[0:self.burn_steps, :], loglikes = loglikes[0:self.burn_steps], get_std = False)
                self.slow_propose = (self.slow_propose + self.slow_covmat)/fac                
            else:
                if(miansi > 0):
                    self.slow_propose /= 5.
                    miansi -= 1
                else:
                    print('current parameters=', p_save[ind_now,:])                
                    print('current loglike=', like_save[ind_now])
                    print('no accepted samples! please check your likelihood')
                    exit()
            if(self.verbose):
                print(r'trial accept ratio: ', np.round(accept/self.burn_steps, 4), r', best loglike:', np.round(self.bestlike, 5))
        self.like_mode = 2  #after initial trial set like_mode to 2, start serious running
        while(try_propose and accept*3. < self.burn_steps):  
            accept = 0
            for i in range(self.burn_steps):
                p_save[1-ind_now, self.slow_indices] =  np.random.multivariate_normal(p_save[ind_now, self.slow_indices], self.slow_propose)
               # p_save[1-ind_now, self.fast_indices] = p_save[ind_now, self.fast_indices] + self.fast_propose*np.random.normal(size=self.num_fast)
                like_save[1-ind_now] = loglike(p_save[1-ind_now, :], self)
                if(like_save[1-ind_now] - like_save[ind_now] >= np.log(1.-np.random.rand())  and like_save[1-ind_now] > self.logzero):
                    ind_now = 1- ind_now
                    accept += 1
                samples[i, :] = p_save[ind_now, :]
                loglikes[i] = like_save[ind_now]
            if(accept > 0):
                self.postprocess(samples = samples[0:self.burn_steps, :], loglikes = loglikes[0:self.burn_steps], get_std=False)
            else:
                print('current parameters=', p_save[ind_now,:])                                
                print('current loglike=', like_save[ind_now])                
                print(' no accepted samples! please check your likelihood')
                exit()
            fac *= 1.2
            self.slow_propose = (self.slow_propose + self.slow_covmat)/fac            
            if(self.verbose):
                print(r'trial* accept ratio: ', accept/self.burn_steps, r', best loglike:', self.bestlike)
        if(try_propose): #in this long trial loop we start to vary fast parameters
            fast_scale = 0.005
            accept = 0
            for i in range(self.burn_steps*3):  
                p_save[1-ind_now, self.slow_indices] =  np.random.multivariate_normal(p_save[ind_now, self.slow_indices], self.slow_propose)
                p_save[1-ind_now, self.fast_indices] = p_save[ind_now, self.fast_indices] + self.fast_std*np.random.normal(scale=fast_scale, size=self.num_fast)
                like_save[1-ind_now] = loglike(p_save[1-ind_now, :], self)
                if(like_save[1-ind_now] - like_save[ind_now] >= np.log(1.-np.random.rand())  and like_save[1-ind_now] > self.logzero):
                    ind_now = 1- ind_now
                    accept += 1
                    if( fast_scale  < 1.):
                        fast_scale *= 1.2
                else:
                    fast_scale *= 0.98
                samples[i, :] = p_save[ind_now, :]
                loglikes[i] = like_save[ind_now]
            if(accept > 0):
                self.postprocess(samples = samples[0:self.burn_steps*3, :], loglikes = loglikes[0:self.burn_steps*3], get_std=True, get_counts=False)
            else:
                print('current parameters=', p_save[ind_now,:])                                
                print('current loglike=', like_save[ind_now])                
                print('  no accepted samples! please check your likelihood')
                exit()
            if(self.verbose):
                print(r'burned in, now accept ratio: ', np.round(accept/(self.burn_steps*3.), 4), r';best loglike: ', np.round(self.bestlike, 5), '; fast_scale: ', np.round(fast_scale, 5))
            self.fast_propose = self.fast_std  / np.sqrt((1.+self.num_fast))/2.
        accept = 0
        for i in range(self.mc_steps // 20):  #throw away 5% samples (when propose matrix is fixed)
            p_save[1-ind_now, self.slow_indices] =  np.random.multivariate_normal(p_save[ind_now, self.slow_indices], self.slow_propose)
            if(i%7 == 0): #in case locked in local minimum
                p_save[1-ind_now, self.fast_indices] = p_save[ind_now, self.fast_indices] + self.fast_propose*np.random.normal(scale=4., size=self.num_fast)                
            else:
                p_save[1-ind_now, self.fast_indices] = p_save[ind_now, self.fast_indices] + self.fast_propose*np.random.normal(scale=1., size=self.num_fast)
            like_save[1-ind_now] = loglike(p_save[1-ind_now, :], self)
            if(like_save[1-ind_now] - like_save[ind_now] >= np.log(1.-np.random.rand())  and like_save[1-ind_now] > self.logzero):
                ind_now = 1- ind_now
                accept += 1
                if(like_save[ind_now] > self.global_bestlike):
                    self.global_bestlike = like_save[ind_now]
                    self.global_bestfit = p_save[ind_now, :].copy()
        if(accept == 0):
            print('current parameters=', p_save[ind_now,:])                                
            print('current loglike=', like_save[ind_now])                
            print('no accepted samples! please check your likelihood')
            exit()            
        accept = 0
        self.bestlike = like_save[ind_now]
        if(self.verbose):
            print('current loglike:', np.round(self.bestlike, 5), '; global best loglike:', np.round(self.global_bestlike, 5) )
        for i in range(self.mc_steps):  #key ingredient here is that I am forcing the run to have like >= global_bestlike
            if(self.verbose and (i % 1000 == 999)):
                print(r'MCstep #', i+1, '/', self.mc_steps, r'; accept: ', np.round(accept/i, 4), r'; bestlike: ', np.round(self.bestlike, 5), r'; now like:', np.round(like_save[ind_now], 5))
            p_save[1-ind_now, self.slow_indices] =  np.random.multivariate_normal(p_save[ind_now, self.slow_indices], self.slow_propose)
            if(i%7 == 0): #in case locked in local minimum
                p_save[1-ind_now, self.fast_indices] = p_save[ind_now, self.fast_indices] + self.fast_propose*np.random.normal(scale=4., size=self.num_fast)                
            else:
                p_save[1-ind_now, self.fast_indices] = p_save[ind_now, self.fast_indices] + self.fast_propose*np.random.normal(scale=1., size=self.num_fast)
            like_save[1-ind_now] = loglike(p_save[1-ind_now, :], self)
            if(like_save[1-ind_now] - like_save[ind_now] >= np.log(1.-np.random.rand()) and like_save[1-ind_now] > self.logzero):
                ind_now = 1- ind_now
                if(like_save[ind_now] > self.bestlike):
                    self.bestlike = like_save[ind_now]
                    if(self.bestlike > self.global_bestlike):
                        self.global_bestlike = self.bestlike
                        self.global_bestfit = p_save[ind_now, :].copy()
                accept += 1
            samples[i, :] = p_save[ind_now, :]
            loglikes[i] = like_save[ind_now]
        if(self.verbose):
            print(r'MCMC done; accept ratio: ', np.round(accept/(self.mc_steps), 4), r';best loglike: ', np.round(self.global_bestlike, 5))
        if(continue_from is None or discard_ratio > 0.99):
            return samples, loglikes
        else:
            istart = math.ceil(lold*discard_ratio)
            return np.concatenate( (old_samples[istart:lold, :], samples), axis = 0), np.concatenate( (old_loglikes[istart:lold], loglikes), axis = None ) 
               


