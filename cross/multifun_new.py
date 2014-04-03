import numpy as np
import time
from math import pi
import copy
from scipy.special import erf
import tucker3d as tuck




def multifun_new(X, delta_cross, fun, r0 = 4, y0 = None, pr = None):

    # For X = [X_1,...,X_d], where X_i - tensors in the Tucker format
    # cross_func computes func(X) == func(X_1,...,X_d) in the Tucker format by using cross3d
    #
    # delta_cross - accuracy for cross3D
    # r0 - number of computed columns on each iteration of cross3d. May be used to improve time performing.
    
    d = len(X)
    if type(r0) == int:
        r0 = [r0, r0, r0]
    elif type(r0) == list:
            None
    else:
        raise Exception('r0 must be of type int or list of len = 3')

    if pr <> None
        print 'cross multifun... \n'

    r = r0
    
    M = X[0].n[0]
    N = int((M+1)/2)



    GG = np.zeros(r, dtype=np.complex128)

    U1 = np.zeros((M, r[1]), dtype=np.complex128)
    U2 = np.zeros((M, r[2]), dtype=np.complex128)
    U3 = np.zeros((M, r[3]), dtype=np.complex128)

    U1[:N,:] = np.random.random((N,r[1]))
    U2[:N,:] = np.random.random((N,r[2]))
    U3[:N,:] = np.random.random((N,r[3]))

    U1, R = np.linalg.qr(U1)
    U2, R = np.linalg.qr(U2)
    U3, R = np.linalg.qr(U3)

    row_order_U1 = tuck.mv.maxvol(U1)
    row_order_U2 = tuck.mv.maxvol(U2)
    row_order_U3 = tuck.mv.maxvol(U3)

    eps_cross = 1

    A = [None]*d
    
    for alpha in xrange(d):
        A[alpha] = np.dot(X[alpha].G, np.transpose(X[alpha].U[2][row_order_U3,:]))
        A[alpha] = np.dot(np.transpose(A[alpha], [2,0,1]), np.transpose(X[alpha].U[1][row_order_U2,:]))
        A[alpha] = np.dot(np.transpose(A[alpha], [0,2,1]), np.transpose(X[alpha].U[0][row_order_U1,:]))
        A[alpha] = np.transpose(A[alpha], [2,1,0])
    Ar = fun(A)
    A1 = np.reshape(Ar, [r[1],-1], order='f')
    A1 = np.transpose(A1)

    Q_A1, R = np.linalg.qr(A1)
    column_order_U1 = tuck.mv.maxvol(Q_A1)
    A1_11 = A1[column_order_U1, :]

    
    A2 = np.reshape(np.transpose(Ar, [1,0,2]), [r[2],-1], order='f')
    A2 = np.transpose(A2)
    Q_A2, R = np.linalg.qr(A2)
    column_order_U2 = tuck.mv.maxvol(Q_A2)
    A2_11 = A2[column_order_U2, :]


    A3 = np.reshape(np.transpose(Ar, [2,0,1]), [r[3],-1], order='f')
    A3 = np.transpose(A3)
    Q_A3, R = np.linalg.qr(A3)
    column_order_U3 = tuck.mv.maxvol(Q_A3)
    A3_11 = A3[column_order_U3, :]


    u1 = np.zeros((M, r[1]), dtype=np.complex128)
    for i in xrange(r[1]):
        for alpha in xrange(d):
            k1_order, j1_order = mod(column_order_U1[i], r[2])
            A[alpha] = np.dot(X[alpha].G,np.transpose(X[alpha].U[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [2,0,1]), np.transpose(X[alpha].U[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [0,2,1]), np.transpose(X[alpha].U[0]))
            A[alpha] = np.transpose(A[alpha], [2,1,0])[:, 0, 0]
        u1[:,i] = fun(A)


    u2 = np.zeros((M, r[2]), dtype=np.complex128)
    for j in xrange(r[2]):
        for alpha in xrange(d):
            k1_order, i1_order = mod(column_order_U2[j], r[1])
            A[alpha] = np.dot(X[alpha].G, np.transpose(X[alpha].U[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [2,1,0]),np.transpose(X[alpha].U[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [0,2,1]),np.transpose(X[alpha].U[1]))
            A[alpha] = np.transpose(A[alpha], [1,2,0])[0, :, 0]
        u2[:,j] = fun(A)


    u3 = np.zeros((M, r[3]), dtype=np.complex128)
    for k in xrange(r[3]):
        for alpha in xrange(d):
            j1_order, i1_order = mod(column_order_U3[k], r[1])
            A[alpha] = np.dot(np.transpose(X[alpha].G, [2,1,0]),np.transpose(X[alpha].U[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [0,2,1]),np.transpose(X[alpha].U[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [1,2,0]),np.transpose(X[alpha].U[2]))[0,0,:]
        u3[:,k] = fun(A)


    U1_hat = np.linalg.solve(U1[row_order_U1, :].T, U1.T).T
    U2_hat = np.linalg.solve(U2[row_order_U2, :].T, U2.T).T
    U3_hat = np.linalg.solve(U3[row_order_U3, :].T, U3.T).T

    UU1, ind_update_1 = column_update(U1_hat, u1, row_order_U1)
    UU2, ind_update_2 = column_update(U2_hat, u2, row_order_U2)
    UU3, ind_update_3 = column_update(U3_hat, u3, row_order_U3)

    U1 = np.concatenate((U1, u1), 1)
    U2 = np.concatenate((U2, u2), 1)
    U3 = np.concatenate((U3, u3), 1)

    A1_12 = np.zeros((r[1], r0[1]),dtype=np.complex128)
    for ii in xrange(r[1]):
        for alpha in xrange(d):
            k1_order, j1_order = mod(column_order_U1[ii], r[2])
            A[alpha] = np.dot(X[alpha].G, np.transpose(X[alpha].U[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [2,0,1]),np.transpose(X[alpha].U[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [0,2,1]),np.transpose(X[alpha].U[0][ind_update_1, :]))
            A[alpha] = np.transpose(A[alpha], [2,1,0])[:,0,0]
        A1_12[ii,:] = fun(A)


    A2_12 = np.zeros((r[2], r0[2]),dtype=np.complex128)
    for ii in xrange(r[2]):
        for alpha in xrange(d):
            k1_order, i1_order = mod(column_order_U2[ii], r[1])
            A[alpha] = np.dot(X[alpha].G, np.transpose(X[alpha].U[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [2,1,0]), np.transpose(X[alpha].U[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [0,2,1]), np.transpose(X[alpha].U[1][ind_update_2, :]))
            A[alpha] = np.transpose(A[alpha], [1,2,0])[0,:,0]
        A2_12[ii, :] = fun(A)


    A3_12 = np.zeros((r[3], r0[3]),dtype=np.complex128)
    for ii in xrange(r[3]):
        for alpha in xrange(d):
            j1_order, i1_order = mod(column_order_U3[ii], r[1])
            A[alpha] = np.dot(np.transpose(X[alpha].G, [2,1,0]),np.transpose(X[alpha].U[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [0,2,1]),np.transpose(X[alpha].U[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [1,2,0]),np.transpose(X[alpha].U[2][ind_update_3, :]))[0,0,:]
        A3_12[ii, :] = fun(A)


    r[1] = r[1]+r0[1]
    r[2] = r[2]+r0[2]
    r[3] = r[3]+r0[3]



    while True:
    
        for alpha in xrange(d):
            A[alpha] = np.dot(np.transpose(X[alpha].G, [2,1,0]), np.transpose(X[alpha].U[0][ind_update_1,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [0,2,1]), np.transpose(X[alpha].U[1][row_order_U2,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [1,2,0]), np.transpose(X[alpha].U[2][row_order_U3,:]))
        Ar_1 = np.concatenate((Ar, fun(A)), 0)

        row_order_U1 = np.concatenate((row_order_U1, ind_update_1))

        for alpha in xrange(d):
            A[alpha] = np.dot(np.transpose(X[alpha].G, [0,2,1]), np.transpose(X[alpha].U[1][ind_update_2,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [0,2,1]), np.transpose(X[alpha].U[2][row_order_U3,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [2,1,0]), np.transpose(X[alpha].U[0][row_order_U1,:]))
            A[alpha] = np.transpose(A[alpha], [2,1,0])
        Ar_2 = np.concatenate((Ar_1, fun(A)), 1)

        row_order_U2 = np.concatenate((row_order_U2, ind_update_2))

        for alpha in xrange(d):
            A[alpha] = np.dot(X[alpha].G, np.transpose(X[alpha].U[2][ind_update_3,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [2,0,1]),np.transpose(X[alpha].U[1][row_order_U2,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [0,2,1]),np.transpose(X[alpha].U[0][row_order_U1,:]))
            A[alpha] = np.transpose(A[alpha], [2,1,0])
        Ar = np.concatenate((Ar_2, fun(A)), 2)

        row_order_U3 = np.concatenate((row_order_U3, ind_update_3))



        A1 = np.reshape(Ar, [r[1],-1], order='f')
        A1 = np.transpose(A1)
        column_order_update_U1 = tuck.mv.maxvol( schur_comp(A1, A1_11, A1_12) )
        r0[1] = len(column_order_update_U1)

        A2 = np.reshape(np.transpose(Ar, [1,0,2]), [r[2],-1], order='f')
        A2 = np.transpose(A2)
        column_order_update_U2 = tuck.mv.maxvol( schur_comp(A2, A2_11, A2_12) )
        r0[2] = len(column_order_update_U2)

        A3 = np.reshape(np.transpose(Ar, [2,0,1]), [r[3],-1], order='f')
        A3 = np.transpose(A3)
        column_order_update_U3 = tuck.mv.maxvol( schur_comp(A3, A3_11, A3_12) )
        r0[3] = len(column_order_update_U3)



        u1_approx = np.zeros((M, r0[1]), dtype=np.complex128)
        u1 = np.zeros((M, r0[1]), dtype=np.complex128)
        for i in xrange(r0[1]):
            for alpha in xrange(d):
                k1_order, j1_order = mod(column_order_update_U1[i], r[2])
                A[alpha] = np.dot(X[alpha].G, np.transpose(X[alpha].U[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
                A[alpha] = np.dot(np.transpose(A[alpha], [2,0,1]),np.transpose(X[alpha].U[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
                A[alpha] = np.dot(np.transpose(A[alpha], [0,2,1]),np.transpose(X[alpha].U[0]))
                A[alpha] = np.transpose(A[alpha], [2,1,0])[:,0,0]
            u1[:,i] = fun(A)

            u1_approx_i = np.dot(Ar, np.transpose(UU3[row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
            u1_approx_i = np.dot(np.transpose(u1_approx_i,[2,0,1]),np.transpose(UU2[row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
            u1_approx_i = np.dot(np.transpose(u1_approx_i,[0,2,1]),np.transpose(UU1))
            u1_approx_i = np.transpose(u1_approx_i,[2,1,0])
            u1_approx[:,i] = u1_approx_i[:, 0, 0]


        u2_approx = np.zeros((M, r0[2]), dtype=np.complex128)
        u2 = np.zeros((M, r0[2]), dtype=np.complex128)
        for j in xrange(r0[2]):
            for alpha in xrange(d):
                k1_order, i1_order = mod(column_order_update_U2[j], r[1])
                A[alpha] = np.dot(X[alpha].G, np.transpose(X[alpha].U[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
                A[alpha] = np.dot(np.transpose(A[alpha], [2,1,0]), np.transpose(X[alpha].U[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
                A[alpha] = np.dot(np.transpose(A[alpha], [0,2,1]), np.transpose(X[alpha].U[1]))
                A[alpha] = np.transpose(A[alpha], [1,2,0])[0,:,0]
            u2[:,j] = fun(A)

            u2_approx_j = np.dot(Ar,np.transpose(UU3[row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
            u2_approx_j = np.dot(np.transpose(u2_approx_j,[2,1,0]),np.transpose(UU1[row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
            u2_approx_j = np.dot(np.transpose(u2_approx_j,[0,2,1]),np.transpose(UU2))
            u2_approx[:,j] = u2_approx_j[0, 0, :]

        u3_approx = np.zeros((M, r0[3]), dtype=np.complex128)
        u3 = np.zeros((M, r0[3]), dtype=np.complex128)
        for k in xrange(r0[3]):
            for alpha in xrange(d):
                j1_order, i1_order = mod(column_order_update_U3[k], r[1])
                A[alpha] = np.dot(np.transpose(X[alpha].G, [2,1,0]),np.transpose(X[alpha].U[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
                A[alpha] = np.dot(np.transpose(A[alpha], [0,2,1]),np.transpose(X[alpha].U[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
                A[alpha] = np.dot(np.transpose(A[alpha], [1,2,0]),np.transpose(X[alpha].U[2]))[0,0,:]
            u3[:,k] = fun(A)

            u3_approx_k = np.dot(np.transpose(Ar,[2,1,0]),np.transpose(UU1[row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
            u3_approx_k = np.dot(np.transpose(u3_approx_k,[0,2,1]),np.transpose(UU2[row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
            u3_approx_k = np.dot(np.transpose(u3_approx_k,[1,2,0]),np.transpose(UU3))
            u3_approx[:,k] = u3_approx_k[0, 0, :]


        eps_cross = 1./3*(  np.linalg.norm(u1_approx - u1)/ np.linalg.norm(u1) +
                            np.linalg.norm(u2_approx - u2)/ np.linalg.norm(u2) +
                            np.linalg.norm(u3_approx - u3)/ np.linalg.norm(u3)   )
        #print 'relative accuracy = %s' % (eps_cross), 'ranks = %s' % r

        if eps_cross < delta_cross:
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

        A1_12 = U1[ind_update_1, r0[1]:].T
        A2_12 = U2[ind_update_2, r0[2]:].T
        A3_12 = U3[ind_update_3, r0[3]:].T

        r[1] = r[1]+r0[1]
        r[2] = r[2]+r0[2]
        r[3] = r[3]+r0[3]


        #print r


    #print r
    U1, R1 = np.linalg.qr(UU1)
    U2, R2 = np.linalg.qr(UU2)
    U3, R3 = np.linalg.qr(UU3)


    GG = np.dot(np.transpose(Ar,[2,1,0]),np.transpose(R1))
    GG = np.dot(np.transpose(GG,[0,2,1]),np.transpose(R2))
    GG = np.transpose(GG,[1,2,0])
    G = np.dot(GG,np.transpose(R3))

    G_Tucker = tuck.tensor(G, delta_cross)
    #print 'ranks after rounding = %s' % G_Tucker.r[0], G_Tucker.r[1], G_Tucker.r[2]


    fun = tuck.tensor()
    fun.G = G_Tucker.G
    fun.U[0] = np.dot(U1, G_Tucker.U[0])
    fun.U[1] = np.dot(U2, G_Tucker.U[1])
    fun.U[2] = np.dot(U3, G_Tucker.U[2])
    fun.r =  G_Tucker.r
    fun.n = (M, M, M)
   
    #print 'cross ending'
    return fun



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
    
    ind_update = tuck.mv.maxvol(Q)
    
    
    return ind_update


def column_update(UU, u, ind):
    
    S = u - np.dot(UU, u[ind,:])
    ind_add = tuck.mv.maxvol(S)
    
    SS = np.dot(np.linalg.pinv(S[ind_add, :].T), S.T).T # WARNING! pinv instead of solve!
    #np.linalg.solve(S[ind_add, :].T, S.T).T#np.dot(np.linalg.pinv(S[ind_add, :].T), S.T).T
    
    U1 = UU - np.dot(SS, UU[ind_add])
    U2 = SS
    
    return np.concatenate((U1, U2), 1), ind_add

def H(A):
    return np.transpose(np.conjugate(A))
