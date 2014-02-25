import numpy as np
import time
from math import pi
import maxvol as mv
import copy
from scipy.special import erf
from tucker import *
from cross_multifun import cross_multifun


def cross_conv(c_g, f, delta_cross, (r1_0, r2_0, r3_0) = (4, 4, 4)):
    # convolution of g and f tensors
    # c_g - generating a circulant subtensor (for symmetric g use lr_circulant func)
    
    aa = tensor_fft(c_g)
    bb = tensor_fft(expand(f))
    
    ab = cross_multifun([aa, bb], delta_cross, lambda (a,b): a*b, (r1_0, r2_0, r3_0))
    
    ab = tensor_ifft(ab)
    
    conv = copy.copy(ab)
    conv.n = [(ab.n[0]+1)/2, (ab.n[1]+1)/2, (ab.n[2]+1)/2]
    conv.U[0] = ab.U[0][:conv.n[0], :]
    conv.U[1] = ab.U[1][:conv.n[1], :]
    conv.U[2] = ab.U[2][:conv.n[2], :]
    
    return conv


def newton_galerkin(x, eps, ind):

    # galerkin tensor for convolution as a hartree potential

    if ind == 6:
        a, b, r = -15., 10, 80
    elif ind == 8:
        a, b, r = -20., 15, 145
    elif ind == 10:
        a, b, r = -25., 20, 220
    elif ind == 12:
        a, b, r = -30., 25, 320
    else:
        raise Exception("wrong ind parameter")
    
        
    N = x.shape

    hr = (b-a)/(r - 1)
    h = x[1]-x[0]

    s = np.array(range(r), dtype = np.complex128)
    s = a + hr * (s - 1)

    w = np.zeros(r, dtype = np.complex128)
    for alpha in xrange(r):
        w[alpha] = 2*hr * np.exp(s[alpha]) / np.sqrt(pi)
    w[0]   = w[0]/2
    w[r-1] = w[r-1]/2


    U = np.zeros((N[0], r), dtype = np.complex128)
    for alpha in xrange(r):
        U[:, alpha] = (  func_int(x-h/2, x[0]-h/2, np.exp(2*s[alpha])) -
                         func_int(x+h/2, x[0]-h/2, np.exp(2*s[alpha])) +
                         func_int(x+h/2, x[0]+h/2, np.exp(2*s[alpha])) -
                         func_int(x-h/2, x[0]+h/2, np.exp(2*s[alpha]))  )

    newton = can2tuck(w, U, U, U)
    newton = tensor_round(newton, eps)
    
    return (1./h**3) * newton


def lr_circulant(T):
    
    # expands T - first columns of a multilevel
    # Toeplitz matrix to first columns of a multilevel circulant

    C = expand(T)

    U1 = T.U[0][1:, :]
    U2 = T.U[1][1:, :]
    U3 = T.U[2][1:, :]
    
    C.U[0][T.n[0] :, :] = U1[::-1, :]
    C.U[1][T.n[1] :, :] = U2[::-1, :]
    C.U[2][T.n[2] :, :] = U3[::-1, :]

    return C


def func_int(x, y, a):
    
    if (a*(2*np.max(x))**2 > 1e-10):
        f = -(np.exp(-a*(x-y)**2)-1)/(2*a) + np.sqrt(pi/a)/2 * (
            (y - x) * erf(np.sqrt(a) * (x-y))  )
    else:
        f = (-(x-y)**2/2) 
    return f    



def expand(a):
    b = tensor()
    b.n = [2*a.n[0] -1, 2*a.n[1] -1, 2*a.n[2] -1]
    b.r = a.r
    b.U[0] = np.zeros((b.n[0], b.r[0]), dtype=np.complex128)
    b.U[1] = np.zeros((b.n[1], b.r[1]), dtype=np.complex128)
    b.U[2] = np.zeros((b.n[2], b.r[2]), dtype=np.complex128)
    b.U[0][:a.n[0], :] = a.U[0]
    b.U[1][:a.n[1], :] = a.U[1]
    b.U[2][:a.n[2], :] = a.U[2] 
    b.G = a.G

    return b