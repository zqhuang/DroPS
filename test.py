import numpy as np
from sys import argv

nu_ref=300.
hGk_t0 = 0.0479924/2.72585  # h*GHz/ k_B / T_CMB
hGk_td = 0.0479924/19.6  #h GHz/k_B/T_MBB

def dust_weight(nu):
    return (nu_ref/nu)*np.exp(hGk_t0*(nu_ref - nu))*((np.exp(hGk_t0*nu)-1.)/(np.exp(hGk_t0*nu_ref)-1.))**2*((np.exp(hGk_td*nu_ref)-1.)/(np.exp(hGk_td*nu)-1.))

def sync_weight(nu):
    return (nu_ref/nu)**2*np.exp(hGk_t0*(nu_ref - nu))*((np.exp(hGk_t0*nu)-1.)/(np.exp(hGk_t0*nu_ref)-1.))**2


def print_vec(v):
    for x in v:
        print(x, end= ' ')
    print("\n")

nu1 = float(argv[1])
nu2 = float(argv[2])

if(argv[3][0:1] == 's'):
    w1 = sync_weight(nu1)
    w2 = sync_weight(nu2)    
else:    
    w1 = dust_weight(nu1)
    w2 = dust_weight(nu2)

    
x = np.loadtxt('powers.txt')
print("beta = ")
print_vec(np.log(x[0, :]/x[1, :]*(w2/w1)**2)/np.log(nu1/nu2)/2.)
print("eps = ")
print_vec(x[2, :]/np.sqrt(x[0, :]*x[1, :]))
