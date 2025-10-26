import numpy as np

def set_x(x):
    x[0] = 3.
    
x = np.empty((2, 3))
x[0, :] = [1., 2., 3.]
x[1, :] = x[0, :]
set_x(x[0, :])
print(x[0, :])
print(x[1, :])
