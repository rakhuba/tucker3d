#
# Some linear algebra operations
#

import sys
sys.path.append('../')
import numpy as np
from core import *



N = 2**12
M = 2*N-1
print 'N = %s' % N

eps = 1e-5
print 'Accuracy: %s' % eps

a = -5.
b = abs(a)
h = (b-a)/(N-1)


h=(b-a)/(N-1)

x = np.zeros(N)
for i in xrange(N):
    x[i] = a + i*h

def slater_fun((i,j,k)):
    return np.exp(-(x[i]**2 + x[j]**2 + x[k]**2)**0.5)


def gaussian_fun((i,j,k)):
    return  np.exp(-(x[i]**2 + x[j]**2 + x[k]**2))



print 'Converting tensors in the Tucker format...'
print '(significantly faster version to be updated soon)'
a = cross(slater_fun, N, eps)
b = cross(gaussian_fun, N, eps)
print 'Converting is done'

print 'tensor a: %s' % (a)
print 'tensor b: %s' % (b)
print 'tensor 2a: %s' % (2*a)
print 'tensor a+a: %s' % (a + a)
print 'tensor a+a after rounding: %s' % (tensor_round(a+a, eps))
print 'relative Frobenius norm of a-a: %s' % (tensor_norm(a-a)/tensor_norm(a))

# Warning! for big mode sizes (N >~ 256) full tensors may be out of memory.
# So, use tensor_full function carefully.








