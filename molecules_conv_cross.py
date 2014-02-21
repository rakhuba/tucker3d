
import numpy as np
import time
from math import pi
from core import *
import timeit

np.random.seed(4)

N = 5121
M = 2*N-1

molecule = 'ch4'  # ch4 c2h6 c2h5oh gly
eps_char = '1e-5'
eps = float(eps_char)



a = -5.
b = abs(a)

h = (b-a)/(N-1)

print 'Molecule: %s' % molecule
print 'Precision: %s' % eps



####################### Reading functions ########################


def newton_reading(precision): # rank = 19, 27, 34
    
    newt = tensor()
    
    if precision == '1e-5':
        rank = 19
    if precision == '1e-7':
        rank = 27
    if precision == '1e-9':
        rank = 34
    
    f = open('data/' + 'newton/' + precision + '.u', "rb")
    newt.U[0] = np.zeros((M, rank), dtype=np.complex128)
    for i in xrange(M):
        for j in xrange(rank):
            newt.U[0][i, j] = np.fromfile(f, dtype='float64', count=1)
    f.close()
    
    f = open('data/' + 'newton/' + precision + '.v', "rb")
    newt.U[1] = np.zeros((M, rank), dtype=np.complex128)
    for i in xrange(M):
        for j in xrange(rank):
            newt.U[1][i, j] = np.fromfile(f, dtype='float64', count=1)
    f.close()
    
    f = open('data/' + 'newton/' + precision + '.w', "rb")
    newt.U[2] = np.zeros((M, rank), dtype=np.complex128)
    for i in xrange(M):
        for j in xrange(rank):
            newt.U[2][i, j] = np.fromfile(f, dtype='float64', count=1)
    f.close()
    
    f = open('data/' + 'newton/' + precision + '.g', "rb")
    newt.G = np.zeros((rank, rank, rank))
    for i in xrange(rank):
        for j in xrange(rank):
            for k in xrange(rank):
                newt.G[i, j, k] = np.fromfile(f, dtype='float64', count=1)
    f.close()

    newt.r = (rank, rank, rank)
    newt.n = (M, M, M)

    return newt

def molecule_reading(molecule, N = 5121, M = 2*N - 1):
    
    mol = tensor()
    
    f = open('data/' + 'molecules' + '/' + molecule + '/' + molecule + '.u', "rb")
    mol.r[0] = int( np.fromfile(f, dtype='int32', count=1)/(8*5121) )
    mol.U[0] = np.zeros((N, mol.r[0]), dtype=np.complex128)
    for j in xrange(mol.r[0]):
        for i in xrange(N):
            mol.U[0][i, j] = np.fromfile(f, dtype='float64', count=1)
    f.close()
    
    
    f = open('data/' + 'molecules' + '/' + molecule + '/' + molecule + '.v', "rb")
    mol.r[1] = int( np.fromfile(f, dtype='int32', count=1)/(8*5121) )
    mol.U[1] = np.zeros((N, mol.r[1]), dtype=np.complex128)
    for j in xrange(mol.r[1]):
        for i in xrange(N):
            mol.U[1][i, j] = np.fromfile(f, dtype='float64', count=1)
    f.close()
    
    
    f = open('data/' + 'molecules' + '/' + molecule + '/' + molecule + '.w', "rb")
    mol.r[2] = int( np.fromfile(f, dtype='int32', count=1)/(8*5121) )
    mol.U[2] = np.zeros((N, mol.r[2]), dtype=np.complex128)
    for j in xrange(mol.r[2]):
        for i in xrange(N):
            mol.U[2][i, j] = np.fromfile(f, dtype='float64', count=1)
    f.close()
    
    
    f = open('data/' + 'molecules' + '/' + molecule + '/' + molecule + '.g', "rb")
    np.fromfile(f, dtype='int32', count=1)
    mol.G = np.zeros(mol.r)
    for k in xrange(mol.r[2]):
        for j in xrange(mol.r[1]):
            for i in xrange(mol.r[0]):
                mol.G[i, j, k] = np.fromfile(f, dtype='float64', count=1)
    f.close()

    mol.n = (N, N, N)
    
    return mol

################################################################################


print 'Data reading...'
newt = newton_reading(eps_char)
mol = molecule_reading(molecule)

print 'Data rounding...'
#newt = tensor_round(newt, eps)
mol = tensor_round(mol, eps)

start = time.time()
start_conv = time.time()
conv = cross_conv(newt, mol, eps, (2,4,4)) # One can vary (r1,r2,r3) for better time performance
end_conv = time.time()
print 'Cross-conv Time: %s' % (end_conv - start_conv)

