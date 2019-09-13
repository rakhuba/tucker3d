import numpy as np
import time
from math import pi
import copy
import tucker3d as tuck

#
#                      !!WARNING!!
#
# Very slow version of the Cross3D (will be updated).
# Use cross_multifun to find a function of tensors in the Tucker format.

def cross3d(func, M, eps_init, delta_add = 1e-5):

    N = int((M+1)/2)

    r1 = 2
    r2 = 2
    r3 = 2

    GG = np.zeros((r1,r2,r3),dtype=np.complex128)

    U1 = np.zeros((M,r1),dtype=np.complex128)
    U2 = np.zeros((M,r2),dtype=np.complex128)
    U3 = np.zeros((M,r3),dtype=np.complex128)

    U1[:N,:] = np.random.random((N,r1))
    U2[:N,:] = np.random.random((N,r2))
    U3[:N,:] = np.random.random((N,r3))

    U1, R = np.linalg.qr(U1)
    U2, R = np.linalg.qr(U2)
    U3, R = np.linalg.qr(U3)

    eps_cross = 1

    while True:

        row_order_U1 = tuck.mv.maxvol(U1)
        row_order_U2 = tuck.mv.maxvol(U2)
        row_order_U3 = tuck.mv.maxvol(U3)

        Ar = np.zeros((r1,r2,r3),dtype=np.complex128)

        for i in range(r1):
            for j in range(r2):
                for k in range(r3):
                    Ar[i,j,k] = func((row_order_U1[i],row_order_U2[j],row_order_U3[k]))


        U1_r = U1[row_order_U1,:]
        U2_r = U2[row_order_U2,:]
        U3_r = U3[row_order_U3,:]

        G_UV = np.linalg.solve(U3_r,np.reshape(np.transpose(Ar,[2,0,1]),(r3,r1*r2),order='f'))
        G_UV = np.reshape(G_UV,(r3,r1,r2),order='f')
        G_UV = np.transpose(G_UV,[1,2,0])

        G_U = np.linalg.solve(U2_r,np.reshape(np.transpose(G_UV,[1,2,0]),(r2,r1*r3),order='f'))
        G_U = np.reshape(G_U,(r2,r3,r1),order='f')
        G_U = np.transpose(G_U,[2,0,1])

        G = np.linalg.solve(U1_r,np.reshape(G_U,(r1,r2*r3),order='f'))
        G = np.reshape(G,(r1,r2,r3),order='f')

        norm = np.linalg.norm(G)
        eps_cross = (np.linalg.norm(GG-G))/norm
    #print 'relative accuracy = %s' % (eps_cross), 'ranks = %s' % r1, r2, r3
        G_Tucker = tuck.tensor(G, eps_init/10)

        G = G_Tucker.core

        U1 = np.dot(U1, G_Tucker.u[0])
        U2 = np.dot(U2, G_Tucker.u[1])
        U3 = np.dot(U3, G_Tucker.u[2])

        (r1, r2, r3) = G_Tucker.r

        if eps_cross < eps_init:
            break

        row_order_U1 = tuck.mv.maxvol(U1)
        row_order_U2 = tuck.mv.maxvol(U2)
        row_order_U3 = tuck.mv.maxvol(U3)


        Ar = np.zeros((r1,r2,r3),dtype=np.complex128)

        for i in range(r1):
            for j in range(r2):
                for k in range(r3):
                    Ar[i, j, k] = func((row_order_U1[i], row_order_U2[j], row_order_U3[k]))


        A1 = np.reshape(Ar, [r1,-1], order='f')
        A1_r = np.transpose(A1)
        A1_r,R = np.linalg.qr(A1_r)
        column_order_U1 = tuck.mv.maxvol(A1_r)


        A2 = np.reshape(np.transpose(Ar, [1,0,2]), [r2,-1], order='f')
        A2_r = np.transpose(A2)
        A2_r,R = np.linalg.qr(A2_r)
        column_order_U2 = tuck.mv.maxvol(A2_r)


        A3 = np.reshape(np.transpose(Ar, [2,0,1]), [r3,-1], order='f')
        A3_r = np.transpose(A3)
        A3_r,R = np.linalg.qr(A3_r)
        column_order_U3 = tuck.mv.maxvol(A3_r)


        u1 = np.zeros((M, r1), dtype=np.complex128)
        for i in range(r1):
            for ii in range(M):
                k1_order, j1_order = mod(column_order_U1[i], r2)
                u1[ii,i] = func((ii, row_order_U2[j1_order], row_order_U3[k1_order]))

        u2 = np.zeros((M, r2), dtype=np.complex128)
        for j in range(r2):
            for jj in range(M):
                k1_order, i1_order = mod(column_order_U2[j], r1)
                u2[jj,j] = func((row_order_U1[i1_order], jj, row_order_U3[k1_order]))

        u3 = np.zeros((M, r3), dtype=np.complex128)
        for k in range(r3):
            for kk in range(M):
                j1_order, i1_order = mod(column_order_U3[k], r1)
                u3[kk,k] = func((row_order_U1[i1_order], row_order_U2[j1_order], kk))


        u1, v, r11 = round_matrix(u1, delta_add)
        u2, v, r22 = round_matrix(u2, delta_add)
        u3, v, r33 = round_matrix(u3, delta_add)

        u1 = u1[:,:r11]
        u2 = u2[:,:r22]
        u3 = u3[:,:r33]

        U1_0 = np.zeros((M,r1+r11),dtype=np.complex128)
        U2_0 = np.zeros((M,r2+r22),dtype=np.complex128)
        U3_0 = np.zeros((M,r3+r33),dtype=np.complex128)

        U1_0[:,:r1] = U1.copy()
        U2_0[:,:r2] = U2.copy()
        U3_0[:,:r3] = U3.copy()

        U1_0[:,r1:r1+r11] = u1
        U2_0[:,r2:r2+r22] = u2
        U3_0[:,r3:r3+r33] = u3


        U1 = U1_0.copy()
        U2 = U2_0.copy()
        U3 = U3_0.copy()

        r1 = r1+r11
        r2 = r2+r22
        r3 = r3+r33


        U1, R1 = np.linalg.qr(U1)
        U2, R2 = np.linalg.qr(U2)
        U3, R3 = np.linalg.qr(U3)


        GG = np.zeros((r1,r2,r3),dtype=np.complex128)
        GG[:(r1-r11),:(r2-r22),:(r3-r33)] = G.copy()


        GG = np.dot(np.transpose(GG,[2,1,0]),np.transpose(R1))
        GG = np.dot(np.transpose(GG,[0,2,1]),np.transpose(R2))
        GG = np.transpose(GG,[1,2,0])
        GG = np.dot(GG,np.transpose(R3))

            #print 'ranks after rounding = %s' % r1, r2, r3
    G_Tucker.n = (M, M, M)
    G_Tucker.u[0] = U1
    G_Tucker.u[1] = U2
    G_Tucker.u[2] = U3
    G_Tucker.r = [r1, r2, r3]


    return tuck.round(G_Tucker, eps_init)


def mod(X,Y):
    return int(X/Y), X%Y


def round_matrix(A, eps):

    u, s, v = np.linalg.svd(np.array(A), full_matrices = False)

    N1, N2 = A.shape
    
    eps_svd = eps*s[0]/np.sqrt(3)
    r = min(N1, N2)
    for i in range(min(N1, N2)):
        if s[i] <= eps_svd:
            r = i
            break
    #print s/s[0]
    u = u[:,:r].copy()
    v = v[:r,:].copy()
    s = s[:r].copy()
    
    return u, H(v), r

def H(A):
    return np.transpose(np.conjugate(A))
