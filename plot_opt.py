import numpy as np
import matplotlib.pyplot as plt
x = np.loadtxt(r'CMBS4/opt_workdir/opt_map_OPTIMIZE.txt')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$f_d$')
plt.plot(x[:, 0], x[:, 1])
plt.scatter(x = x[:, 0], y = x[:, 1], c=((x[:, 2] - x[:,3])/x[:, 4])**2/4., cmap='rainbow', vmin = 0., vmax = 1.)
plt.show()

