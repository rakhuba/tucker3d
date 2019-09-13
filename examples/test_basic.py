# Example: basic linear algebra operations
import sys
sys.path.append('../../')
import numpy as np
import tucker3d as tuck

N = 2**12
M = 2*N-1
print('N = %s' % N)

eps = 1e-5
print('Accuracy: %s' % eps)

a = -5.
b = abs(a)
h = (b-a)/(N-1)

x = np.zeros(N)
for i in range(N):
    x[i] = a + i*h

def slater_fun(ind):
    return np.exp(-(x[ind[0]]**2 + x[ind[1]]**2 + x[ind[2]]**2)**0.5)

def gaussian_fun(ind):
    return  np.exp(-(x[ind[0]]**2 + x[ind[1]]**2 + x[ind[2]]**2))

print('Converting tensors in the Tucker format via cross approximation \n (may be slow due to python loops)...')
a = tuck.cross.cross3d(slater_fun, N, eps)
b = tuck.cross.cross3d(gaussian_fun, N, eps)
print('Converting is done')
print('tensor a: \n%s' % (a))
print('tensor b: \n%s' % (b))
print('tensor 2a: \n%s' % (2*a))
print('tensor a+a: \n%s' % (a + a))
print('tensor a+a after rounding: \n%s' % (tuck.round(a+a, eps)))
print('relative Frobenius norm of a-a: \n%s' % (tuck.norm(a-a)/tuck.norm(a)))

# Warning! for big mode sizes (N >~ 256) full tensors may be out of memory.
# So, use tensor_full function carefully.








