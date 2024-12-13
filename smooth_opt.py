import numpy as np
import matplotlib.pyplot as plt
x = np.loadtxt(r'CMBS4/opt_workdir/opt_map_OPTIMIZE.txt')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$f_d$')
ns = x.shape[0]
alpha = x[:, 0]
fd = x[:, 1]
dev = (x[:, 2] - x[:,3])/x[:, 4]
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
            x_dev[i, j] += dev[k] * thisw
            w[i, j] += thisw
        if(w[i, j] > 0.):
            x_dev[i, j] /= w[i, j]
        else:
            x_dev[i, j] = 2.
plt.imshow(x_dev, origin = 'lower', extent = (alpha_min, alpha_max, fd_min, fd_max) , cmap='rainbow')

plt.show()
