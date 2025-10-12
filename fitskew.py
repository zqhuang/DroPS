#find the skew-normal distribution for  median + upper_sigma - lower_sigma 
#by Zhiqi Huang
#huangzhq25@mail.sysu.edu.cn

import numpy as np
from scipy.stats import skewnorm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.integrate import quad


#generalized skew normal distribution
class general_skewnorm:
    def __init__(self, median, sig_upper, sig_lower):
        self.median = median
        self.sig_upper = sig_upper
        self.sig_lower = sig_lower
        if(sig_upper > sig_lower):
            self.sign = 1.
            self.sig_small = sig_lower
            self.sig_big = sig_upper
        else:
            self.sign = -1.
            self.sig_small = sig_upper
            self.sig_big = sig_lower          
        self.nonlinear_map = False
        sigrat = self.sig_big / self.sig_small
        #compute a 
        if(sigrat == 1.):
            self.a =  0.
        elif(sigrat < 1.001):
            self.a = ((sigrat - 1.)/0.0726)**(1./3.)
        elif(sigrat < 1.488):
            x = np.log(sigrat - 1.)
            self.a =  np.exp(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*2.83269244e-07+1.75642561e-05)+4.77035076e-04)+7.46879829e-03)+7.46585387e-02)+4.98582710e-01)+2.26225149e+00)+6.95544240e+00)+1.41932241e+01)+1.84154004e+01)+1.43915426e+01)+6.20830294e+00)
        elif(sigrat <= 1.54985):
            self.a =  ( -np.log(1.54985043696967-sigrat)/0.455) ** (0.9 - 0.25*(1.54985043696967-sigrat))
        else:
            self.a = 22.744027206558968
            if(sigrat > 1.5498504):
                self.nonlinear_map = True
        if(self.nonlinear_map):         #compute lambda 
            self.rs_median = 0.6744897501960816
            self.rs_sig_lower = 0.47431608753370647
            self.rs_sig_upper = 0.7351189590973726
            self.scale = self.rs_sig_lower / self.sig_small
            mu = self.rs_sig_upper/self.sig_big/self.scale  # mu < 1
            self.lam = 1. - mu*(1.-np.tanh(1./mu))
            for i in range(50):
                self.lam = 1. - mu*(1.-np.tanh(self.lam/mu))
            if(self.lam >= 1.):
                self.lam = 1.-1.e-14                
            self.lamscale = self.lam * self.scale 
            self.tanhscale = self.lamscale / self.rs_sig_upper
            self.llscale = (1.-self.lam)*self.scale                
            self.narr = 1024
            self.y_arr = np.linspace( 0., 4./self.tanhscale , self.narr)
            self.rsy_arr = np.array( [ self.llscale * y + self.rs_sig_upper * np.tanh(self.tanhscale * y) for y in self.y_arr])
            self.y_of_rsy = interp1d(self.rsy_arr, self.y_arr, copy=False, assume_sorted=True)
        else:
            self.rs_median = skewnorm.median(self.a)
            self.rs_sig_lower = self.rs_median - skewnorm.ppf(0.15865525393145707, self.a)
            self.rs_sig_upper = skewnorm.ppf(0.8413447460685429, self.a) - self.rs_median 
            self.scale = self.rs_sig_lower / self.sig_small
        
    def rsx_of_x(self, x):  #convert x to skewnorm variable
        y = (x - self.median)*self.sign
        if(y > 0. and self.nonlinear_map):
            return self.llscale * y + self.rs_sig_upper * np.tanh(self.tanhscale * y) + self.rs_median               
        else:
            return y*self.scale + self.rs_median

    def rsx_drsx_of_x(self, x):  #convert x to skewnorm variable
        y = (x - self.median)*self.sign
        if(y > 0. and self.nonlinear_map):
            rsx = self.llscale * y + self.rs_sig_upper * np.tanh(self.tanhscale * y) + self.rs_median
            drsx = self.llscale + self.lamscale/ np.cosh(self.tanhscale * y)**2
        else:
            rsx =  y*self.scale + self.rs_median
            drsx = self.scale
        return rsx, drsx

    def nonlinear_y_of_rsy(self, rsy):  #here I assume nonlinear_map = True
        if(rsy > 0.):
            if(rsy < self.rsy_arr[self.narr-1]):
                return self.y_of_rsy(rsy)
            else:
                 return (rsy - self.rs_sig_upper * np.tanh(self.tanhscale * ((rsy - self.rs_sig_upper) / self.llscale))) / self.llscale
        else:
            return rsy/self.scale
        
        
        
    def full_y_of_rsy(self, rsy):
        if(rsy > 0. and self.nonlinear_map):
            if(rsy < self.rsy_arr[self.narr-1]):
                return self.y_of_rsy(rsy)
            else:
                return (rsy - self.rs_sig_upper * np.tanh(self.tanhscale * ((rsy - self.rs_sig_upper) / self.llscale))) / self.llscale                
        else:
            return rsy/self.scale
        
    def x_of_rsx(self, rsx):
        return self.full_y_of_rsy(rsx - self.rs_median)*self.sign + self.median

    def cdf(self, x):
        if(self.sign > 0.):
            return skewnorm.cdf(self.rsx_of_x(x), self.a)
        else:
            return 1.-skewnorm.cdf(self.rsx_of_x(x), self.a)
        
    def ppf(self, p):
        if(self.sign > 0.):
            rsx =  skewnorm.ppf(p, self.a)
        else:
            rsx =  skewnorm.ppf(1.-p, self.a)
        return self.x_of_rsx(rsx)
        
    def pdf(self, x):
        rsx, drsx = self.rsx_drsx_of_x(x)
        return skewnorm.pdf(rsx, self.a) * drsx


    def rvs(self, size=1):
        rsy = skewnorm.rvs(a = self.a, size = size) - self.rs_median
        if(self.nonlinear_map):
            for i in range(size):
                rsy[i] = self.nonlinear_y_of_rsy(rsy[i])
            return rsy*self.sign + self.median
        else:
            return rsy*(self.sign/self.scale) + self.median


#-------------- below is an example --------------------------


#x_median = 10.
#x_sig_upper = 1.2
#x_sig_lower = 0.9

#get the distribution
#gs = general_skewnorm(x_median, x_sig_upper, x_sig_lower)


#plt.xlabel(r'$x$')
#sample the distribution
#x = gs.rvs(1000000)
#normalized histogram
#plt.hist(x, bins=200, density=True, range = (x_median - x_sig_lower*5.,  x_median + x_sig_upper*5.))
#compare with the pdf
#n = 256
#x_arr = np.linspace(x_median - x_sig_lower*5.,  x_median + x_sig_upper*5., n )
#pdf_arr = np.array( [ gs.pdf( x) for x in x_arr ] )
#cdf_arr = np.array( [ gs.cdf( x) for x in x_arr ] )
#ppf_arr = np.array( [ gs.ppf(y) for y in cdf_arr ])
#plt.plot(x_arr, pdf_arr, lw=2.)
#plt.show()

