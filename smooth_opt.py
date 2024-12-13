import numpy as np
import matplotlib.pyplot as plt
x = np.loadtxt(r'CMBS4/opt_workdir/opt_map_OPTIMIZE.txt')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$f_d$')
ns = x.shape[0]
alpha = x[:, 0]
fd = x[:, 1]
dev = (x[:, 3] - x[:,2])/x[:, 4]
alpha_min = np.min(alpha)
alpha_max = np.max(alpha)
fd_min = np.min(fd)
fd_max  = np.max(fd)
n1 = 6
n2 = 6
d_alpha = (alpha_max - alpha_min)/n1
d_fd = (fd_max - fd_min)/n2
x_alpha = np.linspace(alpha_min+d_alpha/2., alpha_max-d_alpha/2., n1)
x_fd =  np.linspace(fd_min+d_fd/2., fd_max-d_fd/2., n2)
x_dev = np.zeros((n1, n2))
w = np.zeros((n1, n2))
for i in range(n1):
    for j in range(n2):
        for k in range(ns):
            thisw = np.exp(- ((alpha[k]-x_alpha[i])/d_alpha)**2 - ((fd[k]-x_fd[j])/d_fd)**2)
            x_dev[j, i] += dev[k] * thisw
            w[j, i] += thisw
        if(w[j, i] > 1.):
            x_dev[j, i] /= w[j, i]
        else:
            x_dev[j, i] = np.Infinity
plt.imshow(x_dev, origin = 'lower', extent = (alpha_min, alpha_max, fd_min, fd_max) , cmap='bwr', vmax = 1., vmin=-1.)
plt.plot(x[:, 0], x[:, 1])
plt.scatter(x = x[:, 0], y = x[:, 1], c=(x[:, 3] - x[:,2])/x[:, 4], cmap='bwr', vmin = -2., vmax = 2.)
plt.savefig(r'biasrun.png')
plt.show()
