import numpy as np
from sys import argv
import matplotlib.pyplot as plt
x = np.loadtxt(argv[1])
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$f_d$')
ns = x.shape[0]
alpha = x[:, 0]
fd = x[:, 1]
dev = (x[:, 3] - x[:,2])/x[:, 4]
alpha_min = 1.
alpha_max = 2.2
fd_min = 0.
fd_max  = 1.
n1 = 20
n2 = 20
d_alpha = 0.15 #(alpha_max - alpha_min)/n1
d_fd = 0.15 #(fd_max - fd_min)/n2
x_alpha = np.linspace(alpha_min+d_alpha/2., alpha_max-d_alpha/2., n1)
grid_alpha = x_alpha[1] - x_alpha[0]
x_fd =  np.linspace(fd_min+d_fd/2., fd_max-d_fd/2., n2)
grid_fd = x_fd[1] - x_fd[0]
x_dev = np.zeros((n1, n2))
w = np.zeros((n1, n2))
range_alpha = int(d_alpha/grid_alpha*3.)
range_fd = int(d_fd/grid_fd*3.)
for k in range(ns):
    pos_alpha = int(np.floor((alpha[k]-alpha_min)/grid_alpha))
    pos_fd = int(np.floor((fd[k] - fd_min)/grid_fd))
    for i in range(max(pos_alpha-range_alpha, 0), min(pos_alpha+range_alpha+1, n1)):
        for j in range(max(pos_fd-range_fd, 0), min(pos_fd + range_fd+1, n2)):
            thisw = np.exp(- ((alpha[k]-x_alpha[i])/d_alpha)**2 - ((fd[k]-x_fd[j])/d_fd)**2)
            x_dev[j, i] += dev[k] * thisw
            w[j, i] += thisw

x_dev /= w
plt.imshow(x_dev, origin = 'lower', extent = (alpha_min, alpha_max, fd_min, fd_max) , cmap='bwr', vmax = 0.25, vmin=-0.25)
plt.plot(x[ns-10:ns, 0], x[ns-10:ns, 1], color="black", alpha=0.3)
plt.scatter(x = x[:, 0], y = x[:, 1], c=(x[:, 3] - x[:,2])/x[:, 4], cmap='rainbow', vmin = -3., vmax = 3., alpha=0.5)
plt.savefig(r'biasrun.png')
plt.show()
