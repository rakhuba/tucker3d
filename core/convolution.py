import numpy as np
import time
from math import pi
import maxvol as mv
import copy
from scipy.special import erf
from tucker import *
#from numba.decorators import jit
#from numba import int32, double

#def lr_newton(f, x, eps, (r1_0, r2_0, r3_0) = (4, 4, 4)):

#    T = newton_galerkin(x, eps)
#    newton = convolution_cross(T, f, eps, (r1_0, r2_0, r3_0))

#    return newton
    

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

def cross_conv(c_g, f, delta_cross, (r1_0, r2_0, r3_0) = (4, 4, 4)):
    # convolution of g and f tensors
    # c_g - generating a circulant subtensor (for symmetric g use lr_circulant func) 

    aa = tensor_fft(c_g)
    bb = tensor_fft(expand(f))

    ab = lr_func(aa, bb, delta_cross, lambda a, b: a*b, (r1_0, r2_0, r3_0))

    ab = tensor_ifft(ab)

    conv = copy.copy(ab)
    conv.n = [(ab.n[0]+1)/2, (ab.n[1]+1)/2, (ab.n[2]+1)/2]
    conv.U[0] = ab.U[0][:conv.n[0], :]
    conv.U[1] = ab.U[1][:conv.n[1], :]
    conv.U[2] = ab.U[2][:conv.n[2], :]
    
    return conv




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
    
def schur_comp(A, A11, A12):
    r, r0 = A12.shape
    R = r + r0

    #print np.linalg.solve(A11.T, A[:,:r].T).T
    S_hat = np.zeros((R,r0), dtype=np.complex128)

    S_hat[:r, :] = np.dot(np.linalg.pinv(A11), -A12)#np.linalg.solve(A11, -A12)
    S_hat[r:, :] = np.identity(r0)

    #print A[:,:]
    #uu, ss, vv = np.linalg.svd(np.dot(A, S_hat))
    #'ss:', ss


    Q, R = np.linalg.qr(np.dot(A, S_hat))
    #Q , trash1, trash2 = round_matrix(np.dot(A, S_hat), delta_tucker)

    return Q

def mod(X,Y):
    return int(X/Y), X%Y

def maxvol_update(A, ind):
    # finds new r0 good rows
    # [ A11 A12]
    # [ A21 A22] => S = A22 - A21 A11^(-1) A12

    N, R = A.shape
    r = len(ind)
    r0 = R - r

    S_hat = np.zeros((R, r0),dtype=np.complex128)

    S_hat[:r, :] = np.linalg.solve(A[ind, :r], -A[ind, r:])
    S_hat[r:, :] = np.identity(r0)
    Q, R = np.linalg.qr(np.dot(A, S_hat))

    ind_update = mv.maxvol(Q)


    return ind_update 


def column_update(UU, u, ind):

    S = u - np.dot(UU, u[ind,:])
    ind_add = mv.maxvol(S)

    SS = np.dot(np.linalg.pinv(S[ind_add, :].T), S.T).T # WARNING! pinv instead of solve!
    #np.linalg.solve(S[ind_add, :].T, S.T).T#np.dot(np.linalg.pinv(S[ind_add, :].T), S.T).T
    
    U1 = UU - np.dot(SS, UU[ind_add])
    U2 = SS

    return np.concatenate((U1, U2), 1), ind_add



def lr_func(a, b, delta_cross, func, (r1_0, r2_0, r3_0) = (4, 4, 4)):
    #print 'cross starting'
    M = a.n[0]
    N = (M+1)/2

    C_Tucker = copy.copy(a)
    Q_Tucker = copy.copy(b)



    r1 = r1_0
    r2 = r2_0
    r3 = r3_0

    GG = np.zeros((r1,r2,r3), dtype=np.complex128)

    U1 = np.zeros((M,r1), dtype=np.complex128)
    U2 = np.zeros((M,r2), dtype=np.complex128)
    U3 = np.zeros((M,r3), dtype=np.complex128)

    U1[:N,:] = np.random.random((N,r1))
    U2[:N,:] = np.random.random((N,r2))
    U3[:N,:] = np.random.random((N,r3))

    U1, R = np.linalg.qr(U1)
    U2, R = np.linalg.qr(U2)
    U3, R = np.linalg.qr(U3)


    eps_cross = 1



    row_order_U1 = mv.maxvol(U1)
    row_order_U2 = mv.maxvol(U2)
    row_order_U3 = mv.maxvol(U3)

    
    AC = np.dot(C_Tucker.G, np.transpose(C_Tucker.U[2][row_order_U3,:]))
    AC = np.dot(np.transpose(AC,[2,0,1]), np.transpose(C_Tucker.U[1][row_order_U2,:]))
    AC = np.dot(np.transpose(AC,[0,2,1]), np.transpose(C_Tucker.U[0][row_order_U1,:]))
    AC = np.transpose(AC,[2,1,0])
    AQ = np.dot(Q_Tucker.G,np.transpose(Q_Tucker.U[2][row_order_U3,:]))
    AQ = np.dot(np.transpose(AQ,[2,0,1]),np.transpose(Q_Tucker.U[1][row_order_U2,:]))
    AQ = np.dot(np.transpose(AQ,[0,2,1]),np.transpose(Q_Tucker.U[0][row_order_U1,:]))
    AQ = np.transpose(AQ,[2,1,0])
    Ar = func(AC, AQ)

    A1 = np.reshape(Ar, [r1,-1], order='f')
    A1 = np.transpose(A1)
    column_order_U1 = mv.maxvol(A1)
    A1_11 = A1[column_order_U1, :]


    A2 = np.reshape(np.transpose(Ar, [1,0,2]), [r2,-1], order='f')
    A2 = np.transpose(A2)
    column_order_U2 = mv.maxvol(A2)
    A2_11 = A2[column_order_U2, :]


    A3 = np.reshape(np.transpose(Ar, [2,0,1]), [r3,-1], order='f')
    A3 = np.transpose(A3)
    column_order_U3 = mv.maxvol(A3)
    A3_11 = A3[column_order_U3, :]


    u1 = np.zeros((M, r1), dtype=np.complex128)
    for i in xrange(r1):
        k1_order, j1_order = mod(column_order_U1[i], r2)
        AC = np.dot(C_Tucker.G,np.transpose(C_Tucker.U[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
        AC = np.dot(np.transpose(AC,[2,0,1]),np.transpose(C_Tucker.U[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
        AC = np.dot(np.transpose(AC,[0,2,1]),np.transpose(C_Tucker.U[0]))
        AC = np.transpose(AC,[2,1,0])
        AQ = np.dot(Q_Tucker.G,np.transpose(Q_Tucker.U[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
        AQ = np.dot(np.transpose(AQ,[2,0,1]),np.transpose(Q_Tucker.U[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
        AQ = np.dot(np.transpose(AQ,[0,2,1]),np.transpose(Q_Tucker.U[0]))
        AQ = np.transpose(AQ,[2,1,0])
        u1[:,i] = func(AC[:,0,0], AQ[:,0,0])


    u2 = np.zeros((M, r2), dtype=np.complex128)
    for j in xrange(r2):
        k1_order, i1_order = mod(column_order_U2[j], r1)
        AC = np.dot(C_Tucker.G,np.transpose(C_Tucker.U[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
        AC = np.dot(np.transpose(AC,[2,1,0]),np.transpose(C_Tucker.U[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
        AC = np.dot(np.transpose(AC,[0,2,1]),np.transpose(C_Tucker.U[1]))
        AC = np.transpose(AC,[1,2,0])
        AQ = np.dot(Q_Tucker.G,np.transpose(Q_Tucker.U[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
        AQ = np.dot(np.transpose(AQ,[2,1,0]),np.transpose(Q_Tucker.U[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
        AQ = np.dot(np.transpose(AQ,[0,2,1]),np.transpose(Q_Tucker.U[1]))
        AQ = np.transpose(AQ,[1,2,0])
        u2[:,j] = func(AC[0,:,0],AQ[0,:,0])


    u3 = np.zeros((M, r3), dtype=np.complex128)
    for k in xrange(r3):
        j1_order, i1_order = mod(column_order_U3[k], r1)
        AC = np.dot(np.transpose(C_Tucker.G,[2,1,0]),np.transpose(C_Tucker.U[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
        AC = np.dot(np.transpose(AC,[0,2,1]),np.transpose(C_Tucker.U[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
        AC = np.dot(np.transpose(AC,[1,2,0]),np.transpose(C_Tucker.U[2]))
        AQ = np.dot(np.transpose(Q_Tucker.G,[2,1,0]),np.transpose(Q_Tucker.U[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
        AQ = np.dot(np.transpose(AQ,[0,2,1]),np.transpose(Q_Tucker.U[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
        AQ = np.dot(np.transpose(AQ,[1,2,0]),np.transpose(Q_Tucker.U[2]))
        u3[:,k] = func(AC[0,0,:], AQ[0,0,:])


    U1_hat = np.linalg.solve(U1[row_order_U1, :].T, U1.T).T
    U2_hat = np.linalg.solve(U2[row_order_U2, :].T, U2.T).T
    U3_hat = np.linalg.solve(U3[row_order_U3, :].T, U3.T).T

    UU1, ind_update_1 = column_update(U1_hat, u1, row_order_U1)
    UU2, ind_update_2 = column_update(U2_hat, u2, row_order_U2)
    UU3, ind_update_3 = column_update(U3_hat, u3, row_order_U3)

    U1 = np.concatenate((U1, u1), 1)
    U2 = np.concatenate((U2, u2), 1)
    U3 = np.concatenate((U3, u3), 1)

    A1_12 = np.zeros((r1, r1_0),dtype=np.complex128)
    for ii in xrange(r1):
        k1_order, j1_order = mod(column_order_U1[ii], r2)
        AC = np.dot(C_Tucker.G,np.transpose(C_Tucker.U[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
        AC = np.dot(np.transpose(AC,[2,0,1]),np.transpose(C_Tucker.U[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
        AC = np.dot(np.transpose(AC,[0,2,1]),np.transpose(C_Tucker.U[0][ind_update_1, :]))
        AC = np.transpose(AC,[2,1,0])
        AQ = np.dot(Q_Tucker.G,np.transpose(Q_Tucker.U[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
        AQ = np.dot(np.transpose(AQ,[2,0,1]),np.transpose(Q_Tucker.U[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
        AQ = np.dot(np.transpose(AQ,[0,2,1]),np.transpose(Q_Tucker.U[0][ind_update_1, :]))
        AQ = np.transpose(AQ,[2,1,0])
        A1_12[ii,:] = func(AC[:,0,0], AQ[:,0,0])


    A2_12 = np.zeros((r2, r2_0),dtype=np.complex128)
    for ii in xrange(r2):
        k1_order, i1_order = mod(column_order_U2[ii], r1)
        AC = np.dot(C_Tucker.G,np.transpose(C_Tucker.U[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
        AC = np.dot(np.transpose(AC,[2,1,0]),np.transpose(C_Tucker.U[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
        AC = np.dot(np.transpose(AC,[0,2,1]),np.transpose(C_Tucker.U[1][ind_update_2, :]))
        AC = np.transpose(AC,[1,2,0])
        AQ = np.dot(Q_Tucker.G,np.transpose(Q_Tucker.U[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
        AQ = np.dot(np.transpose(AQ,[2,1,0]),np.transpose(Q_Tucker.U[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
        AQ = np.dot(np.transpose(AQ,[0,2,1]),np.transpose(Q_Tucker.U[1][ind_update_2, :]))
        AQ = np.transpose(AQ,[1,2,0])
        A2_12[ii, :] = func(AC[0,:,0], AQ[0,:,0])


    A3_12 = np.zeros((r3, r3_0),dtype=np.complex128)
    for ii in xrange(r3):
        j1_order, i1_order = mod(column_order_U3[ii], r1)
        AC = np.dot(np.transpose(C_Tucker.G,[2,1,0]),np.transpose(C_Tucker.U[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
        AC = np.dot(np.transpose(AC,[0,2,1]),np.transpose(C_Tucker.U[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
        AC = np.dot(np.transpose(AC,[1,2,0]),np.transpose(C_Tucker.U[2][ind_update_3, :]))
        AQ = np.dot(np.transpose(Q_Tucker.G,[2,1,0]),np.transpose(Q_Tucker.U[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
        AQ = np.dot(np.transpose(AQ,[0,2,1]),np.transpose(Q_Tucker.U[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
        AQ = np.dot(np.transpose(AQ,[1,2,0]),np.transpose(Q_Tucker.U[2][ind_update_3, :]))
        A3_12[ii, :] = func(AC[0,0,:], AQ[0,0,:])


    r1 = r1+r1_0
    r2 = r2+r2_0
    r3 = r3+r3_0



    while True:

        AC = np.dot(np.transpose(C_Tucker.G,[2,1,0]), np.transpose(C_Tucker.U[0][ind_update_1,:]))
        AC = np.dot(np.transpose(AC,[0,2,1]), np.transpose(C_Tucker.U[1][row_order_U2,:]))
        AC = np.dot(np.transpose(AC,[1,2,0]), np.transpose(C_Tucker.U[2][row_order_U3,:]))
        AQ = np.dot(np.transpose(Q_Tucker.G,[2,1,0]), np.transpose(Q_Tucker.U[0][ind_update_1,:]))
        AQ = np.dot(np.transpose(AQ,[0,2,1]), np.transpose(Q_Tucker.U[1][row_order_U2,:]))
        AQ = np.dot(np.transpose(AQ,[1,2,0]),np.transpose(Q_Tucker.U[2][row_order_U3,:]))
        Ar_1 = np.concatenate((Ar, func(AC, AQ)), 0)

        row_order_U1 = np.concatenate((row_order_U1, ind_update_1))

        AC = np.dot(np.transpose(C_Tucker.G, [0,2,1]), np.transpose(C_Tucker.U[1][ind_update_2,:]))
        AC = np.dot(np.transpose(AC,[0,2,1]), np.transpose(C_Tucker.U[2][row_order_U3,:]))
        AC = np.dot(np.transpose(AC,[2,1,0]),np.transpose(C_Tucker.U[0][row_order_U1,:]))
        AC = np.transpose(AC,[2,1,0])
        AQ = np.dot(np.transpose(Q_Tucker.G, [0,2,1]), np.transpose(Q_Tucker.U[1][ind_update_2,:]))
        AQ = np.dot(np.transpose(AQ,[0,2,1]), np.transpose(Q_Tucker.U[2][row_order_U3,:]))
        AQ = np.dot(np.transpose(AQ,[2,1,0]),np.transpose(Q_Tucker.U[0][row_order_U1,:]))
        AQ = np.transpose(AQ,[2,1,0])
        Ar_2 = np.concatenate((Ar_1, func(AC, AQ)), 1)

        row_order_U2 = np.concatenate((row_order_U2, ind_update_2))

        AC = np.dot(C_Tucker.G,np.transpose(C_Tucker.U[2][ind_update_3,:]))
        AC = np.dot(np.transpose(AC,[2,0,1]),np.transpose(C_Tucker.U[1][row_order_U2,:]))
        AC = np.dot(np.transpose(AC,[0,2,1]),np.transpose(C_Tucker.U[0][row_order_U1,:]))
        AC = np.transpose(AC,[2,1,0])
        AQ = np.dot(Q_Tucker.G,np.transpose(Q_Tucker.U[2][ind_update_3,:]))
        AQ = np.dot(np.transpose(AQ,[2,0,1]),np.transpose(Q_Tucker.U[1][row_order_U2,:]))
        AQ = np.dot(np.transpose(AQ,[0,2,1]),np.transpose(Q_Tucker.U[0][row_order_U1,:]))
        AQ = np.transpose(AQ,[2,1,0])
        Ar= np.concatenate((Ar_2, func(AC, AQ)), 2)

        row_order_U3 = np.concatenate((row_order_U3, ind_update_3))



        A1 = np.reshape(Ar, [r1,-1], order='f')
        A1 = np.transpose(A1)
        column_order_update_U1 = mv.maxvol( schur_comp(A1, A1_11, A1_12) )
        r1_0 = len(column_order_update_U1)

        A2 = np.reshape(np.transpose(Ar, [1,0,2]), [r2,-1], order='f')
        A2 = np.transpose(A2)
        column_order_update_U2 = mv.maxvol( schur_comp(A2, A2_11, A2_12) )
        r2_0 = len(column_order_update_U2)

        A3 = np.reshape(np.transpose(Ar, [2,0,1]), [r3,-1], order='f')
        A3 = np.transpose(A3)
        column_order_update_U3 = mv.maxvol( schur_comp(A3, A3_11, A3_12) )
        r3_0 = len(column_order_update_U3)



        u1_approx = np.zeros((M, r1_0), dtype=np.complex128)
        u1 = np.zeros((M, r1_0), dtype=np.complex128)
        for i in xrange(r1_0):
            k1_order, j1_order = mod(column_order_update_U1[i], r2)
            AC = np.dot(C_Tucker.G,np.transpose(C_Tucker.U[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
            AC = np.dot(np.transpose(AC,[2,0,1]),np.transpose(C_Tucker.U[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
            AC = np.dot(np.transpose(AC,[0,2,1]),np.transpose(C_Tucker.U[0]))
            AC = np.transpose(AC,[2,1,0])
            AQ = np.dot(Q_Tucker.G,np.transpose(Q_Tucker.U[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
            AQ = np.dot(np.transpose(AQ,[2,0,1]),np.transpose(Q_Tucker.U[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
            AQ = np.dot(np.transpose(AQ,[0,2,1]),np.transpose(Q_Tucker.U[0]))
            AQ = np.transpose(AQ,[2,1,0])
            u1[:,i] = func(AC[:,0,0], AQ[:,0,0])

            u1_approx_i = np.dot(Ar, np.transpose(UU3[row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
            u1_approx_i = np.dot(np.transpose(u1_approx_i,[2,0,1]),np.transpose(UU2[row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
            u1_approx_i = np.dot(np.transpose(u1_approx_i,[0,2,1]),np.transpose(UU1))
            u1_approx_i = np.transpose(u1_approx_i,[2,1,0])
            u1_approx[:,i] = u1_approx_i[:, 0, 0]


        u2_approx = np.zeros((M, r2_0), dtype=np.complex128)
        u2 = np.zeros((M, r2_0), dtype=np.complex128)
        for j in xrange(r2_0):
            k1_order, i1_order = mod(column_order_update_U2[j], r1)
            AC = np.dot(C_Tucker.G,np.transpose(C_Tucker.U[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
            AC = np.dot(np.transpose(AC,[2,1,0]),np.transpose(C_Tucker.U[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
            AC = np.dot(np.transpose(AC,[0,2,1]),np.transpose(C_Tucker.U[1]))
            AC = np.transpose(AC,[1,2,0])
            AQ = np.dot(Q_Tucker.G,np.transpose(Q_Tucker.U[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
            AQ = np.dot(np.transpose(AQ,[2,1,0]),np.transpose(Q_Tucker.U[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
            AQ = np.dot(np.transpose(AQ,[0,2,1]),np.transpose(Q_Tucker.U[1]))
            AQ = np.transpose(AQ,[1,2,0])
            u2[:,j] = func(AC[0,:,0], AQ[0,:,0])

            u2_approx_j = np.dot(Ar,np.transpose(UU3[row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
            u2_approx_j = np.dot(np.transpose(u2_approx_j,[2,1,0]),np.transpose(UU1[row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
            u2_approx_j = np.dot(np.transpose(u2_approx_j,[0,2,1]),np.transpose(UU2))
            u2_approx[:,j] = u2_approx_j[0, 0, :]

        u3_approx = np.zeros((M, r3_0), dtype=np.complex128)
        u3 = np.zeros((M, r3_0), dtype=np.complex128)
        for k in xrange(r3_0):
            j1_order, i1_order = mod(column_order_update_U3[k], r1)
            AC = np.dot(np.transpose(C_Tucker.G,[2,1,0]),np.transpose(C_Tucker.U[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
            AC = np.dot(np.transpose(AC,[0,2,1]),np.transpose(C_Tucker.U[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
            AC = np.dot(np.transpose(AC,[1,2,0]),np.transpose(C_Tucker.U[2]))
            AQ = np.dot(np.transpose(Q_Tucker.G,[2,1,0]),np.transpose(Q_Tucker.U[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
            AQ = np.dot(np.transpose(AQ,[0,2,1]),np.transpose(Q_Tucker.U[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
            AQ = np.dot(np.transpose(AQ,[1,2,0]),np.transpose(Q_Tucker.U[2]))
            u3[:,k] = func(AC[0,0,:], AQ[0,0,:])

            u3_approx_k = np.dot(np.transpose(Ar,[2,1,0]),np.transpose(UU1[row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
            u3_approx_k = np.dot(np.transpose(u3_approx_k,[0,2,1]),np.transpose(UU2[row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
            u3_approx_k = np.dot(np.transpose(u3_approx_k,[1,2,0]),np.transpose(UU3))
            u3_approx[:,k] = u3_approx_k[0, 0, :]


        eps_cross = 1./3*(  np.linalg.norm(u1_approx - u1)/ np.linalg.norm(u1) +
                            np.linalg.norm(u2_approx - u2)/ np.linalg.norm(u2) +
                            np.linalg.norm(u3_approx - u3)/ np.linalg.norm(u3)   )
            #print 'relative accuracy = %s' % (eps_cross), 'ranks = %s' % r1, r2, r3

        if eps_cross<delta_cross:
            break

        #print np.linalg.norm( full(G, U1, U2, U3) - C_toch )/np.linalg.norm(C_toch)


        UU1, ind_update_1 = column_update(UU1, u1, row_order_U1)
        UU2, ind_update_2 = column_update(UU2, u2, row_order_U2)
        UU3, ind_update_3 = column_update(UU3, u3, row_order_U3)


        U1 = np.concatenate((U1, u1), 1)
        U2 = np.concatenate((U2, u2), 1)
        U3 = np.concatenate((U3, u3), 1)


        A1_11 = np.concatenate((A1_11, A1_12), 1)
        A1_11 = np.concatenate((A1_11, A1[column_order_update_U1,:]) )

        A2_11 = np.concatenate((A2_11, A2_12), 1)
        A2_11 = np.concatenate((A2_11, A2[column_order_update_U2,:]) )

        A3_11 = np.concatenate((A3_11, A3_12), 1)
        A3_11 = np.concatenate((A3_11, A3[column_order_update_U3,:]) )

        A1_12 = U1[ind_update_1, r1_0:].T
        A2_12 = U2[ind_update_2, r2_0:].T
        A3_12 = U3[ind_update_3, r3_0:].T

        r1 = r1+r1_0
        r2 = r2+r2_0
        r3 = r3+r3_0


        #print r1, r2, r3


    #print r1, r2, r3
    U1, R1 = np.linalg.qr(UU1)
    U2, R2 = np.linalg.qr(UU2)
    U3, R3 = np.linalg.qr(UU3)


    GG = np.dot(np.transpose(Ar,[2,1,0]),np.transpose(R1))
    GG = np.dot(np.transpose(GG,[0,2,1]),np.transpose(R2))
    GG = np.transpose(GG,[1,2,0])
    G = np.dot(GG,np.transpose(R3))

    G_Tucker = tensor(G, delta_cross)
    #print 'ranks after rounding = %s' % G_Tucker.r[0], G_Tucker.r[1], G_Tucker.r[2]


    prod = tensor()
    prod.G = G_Tucker.G
    prod.U[0] = np.dot(U1, G_Tucker.U[0])
    prod.U[1] = np.dot(U2, G_Tucker.U[1])
    prod.U[2] = np.dot(U3, G_Tucker.U[2])
    prod.r =  G_Tucker.r
    prod.n = (M, M, M)
   
    #print 'cross ending'
    return prod
