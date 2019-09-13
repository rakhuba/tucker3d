#
# Computing Newton potential of a Slater function
#

import sys
sys.path.append('../../')

import numpy as np
import time
from math import pi
import tucker3d as tuck
import timeit

np.random.seed(1)

N = 2**9
M = 2*N-1
print('N= %s' % N)

eps = 1e-7
print('Accuracy: %s' % eps)

a = -5.
b = abs(a)
h = (b-a)/(N-1)

newton_const = 1./(4*pi)* h**3

x = np.zeros(N)
for i in range(N):
    x[i] = a + i*h

def slater_func(ind):
    return np.exp(-(x[ind[0]]**2 + x[ind[1]]**2 + x[ind[2]]**2)**0.5)

def newton_func(ind):
    return newton_const * 1./(x[ind[0]]**2 + x[ind[1]]**2 + x[ind[2]]**2)**0.5

# Tucker representaition of the slater and newton functions

print('WARNING!')
print('Very slow part (cross approximation of Slater and Newton funs).')
print('It does not influence cross-conv time performance.')
print('In process...')
slater = tuck.cross.cross3d(slater_func, N, eps, delta_add=1e-5)
newton = tuck.cross.cross3d(newton_func, N, eps, delta_add=1e-5)
newton_ext = tuck.cross.toepl2circ(newton)

# Convolution part

start = time.time()
start_conv = time.time()
conv = tuck.cross.conv(newton_ext, slater, eps)
end_conv = time.time()
print('Cross-conv Time: %s' % (end_conv - start_conv))

