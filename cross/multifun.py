import numpy as np
import time
from math import pi
import copy
from scipy.special import erf
import tucker3d as tuck


def pinv(A):
    try:
        return np.linalg.pinv(A)
    except: #LinAlgError
        try:
            print "PINV failded"
            return np.linalg.pinv(A + 1e-12*np.linalg.norm(A, 1))
        except:
            print "PINV failded twice"
            return np.linalg.pinv(A + 1e-8*np.linalg.norm(A, 1))


#from numba import autojit
#@autojit
def multifun(X, delta_cross, fun, r_add=4, y0=None, rmax=100, pr=None):
    
    # For X = [X_1,...,X_d], where X_i - tensors in the Tucker format
    # cross_func computes y = func(X) == func(x_1,...,x_d) in the Tucker format by using cross3d
    #
    # delta_cross - accuracy for cross3D
    # r_add - number of computed columns on each iteration of cross3d. May be used to improve time performing.
    
    d = len(X)
    if type(r_add) == int:
        r_add = [r_add, r_add, r_add]
    elif len(r_add) == 3:
        None
    else:
        raise Exception('r_add must be of type int or list of len = 3')

    eps_cross = 1
    
    if pr <> None:
        print 'cross multifun... \n'
    
    r = copy.copy(r_add)


    n = X[0].n
    N = int((min(n)+1)/2)

    # Type check
    list = [X[i].u[0][0,0] for i in xrange(len(X))]
    if type(np.sum(list)) is np.complex128:
        dtype = np.complex128
    else:
        dtype = np.float64

    if pr <> None:
        print 'data type is', dtype

    # if there is initial guess
    if y0 <> None:
############################################################
############################################################
############################################################

        Q1, R = np.linalg.qr(y0.u[0]);
        row_order_U1 = np.sort(tuck.mv.maxvol(Q1));
        Q2, R = np.linalg.qr(y0.u[1]);
        row_order_U2 = np.sort(tuck.mv.maxvol(Q2));
        Q3, R = np.linalg.qr(y0.u[2]);
        row_order_U3 = np.sort(tuck.mv.maxvol(Q3));

        r0 = [len(row_order_U1), len(row_order_U2), len(row_order_U3)]

        A = [None]*d


        for alpha in xrange(d):
            A[alpha] = np.tensordot(X[alpha].core, np.transpose(X[alpha].u[2][row_order_U3,:]), (2, 0))
            A[alpha] = np.tensordot(np.transpose(A[alpha], [2,0,1]), np.transpose(X[alpha].u[1][row_order_U2,:]), (2, 0))
            A[alpha] = np.tensordot(np.transpose(A[alpha], [0,2,1]), np.transpose(X[alpha].u[0][row_order_U1,:]), (2, 0))
            A[alpha] = np.transpose(A[alpha], [2,1,0])
        
        Ar = fun(A)


        A1 = np.reshape(Ar, [r0[0],-1], order='f')
        A1 = np.transpose(A1)
        Q_A1, R = np.linalg.qr(A1)
        column_order_U1 = tuck.mv.maxvol(Q_A1)
        A1_11 = A1[column_order_U1, :]


        A2 = np.reshape(np.transpose(Ar, [1,0,2]), [r0[1],-1], order='f')
        A2 = np.transpose(A2)
        Q_A2, R = np.linalg.qr(A2)
        column_order_U2 = tuck.mv.maxvol(Q_A2)
        A2_11 = A2[column_order_U2, :]


        A3 = np.reshape(np.transpose(Ar, [2,0,1]), [r0[2],-1], order='f')
        A3 = np.transpose(A3)
        Q_A3, R = np.linalg.qr(A3)
        column_order_U3 = tuck.mv.maxvol(Q_A3)
        A3_11 = A3[column_order_U3, :]


        u1 = np.zeros((n[0], r0[0]), dtype=dtype)
        for i in xrange(r0[0]):
            for alpha in xrange(d):
                k1_order, j1_order = mod(column_order_U1[i], r0[1])
                A[alpha] = np.dot(X[alpha].core,np.transpose(X[alpha].u[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
                A[alpha] = np.dot(np.transpose(A[alpha], [2,0,1]), np.transpose(X[alpha].u[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
                A[alpha] = np.tensordot(np.transpose(A[alpha], [0,2,1]), np.transpose(X[alpha].u[0]), (2,0))
                A[alpha] = np.transpose(A[alpha], [2,1,0])[:, 0, 0]
            u1[:,i] = fun(A)


        u2 = np.zeros((n[1], r0[1]), dtype=dtype)
        for j in xrange(r0[1]):
            for alpha in xrange(d):
                k1_order, i1_order = mod(column_order_U2[j], r0[0])
                A[alpha] = np.dot(X[alpha].core, np.transpose(X[alpha].u[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
                A[alpha] = np.dot(np.transpose(A[alpha], [2,1,0]),np.transpose(X[alpha].u[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
                A[alpha] = np.tensordot(np.transpose(A[alpha], [0,2,1]),np.transpose(X[alpha].u[1]), (2, 0))
                A[alpha] = np.transpose(A[alpha], [1,2,0])[0, :, 0]
            u2[:,j] = fun(A)


        u3 = np.zeros((n[2], r0[2]), dtype=dtype)
        for k in xrange(r0[2]):
            for alpha in xrange(d):
                j1_order, i1_order = mod(column_order_U3[k], r0[0])
                A[alpha] = np.dot(np.transpose(X[alpha].core, [2,1,0]),np.transpose(X[alpha].u[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
                A[alpha] = np.dot(np.transpose(A[alpha], [0,2,1]),np.transpose(X[alpha].u[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
                A[alpha] = np.tensordot(np.transpose(A[alpha], [1,2,0]),np.transpose(X[alpha].u[2]), (2, 0))[0,0,:]
            u3[:,k] = fun(A)



    else:
############################################################
############################################################
############################################################
    
    
    
        GG = np.zeros(r, dtype=dtype)
    
        u1 = np.zeros((n[0], r_add[0]), dtype=dtype)
        u2 = np.zeros((n[1], r_add[1]), dtype=dtype)
        u3 = np.zeros((n[2], r_add[2]), dtype=dtype)
    
        u1[:N,:] = np.random.random((N,r_add[0]))
        u2[:N,:] = np.random.random((N,r_add[1]))
        u3[:N,:] = np.random.random((N,r_add[2]))
    
        u1, R = np.linalg.qr(u1)
        u2, R = np.linalg.qr(u2)
        u3, R = np.linalg.qr(u3)
    
        row_order_U1 = tuck.mv.maxvol(u1)
        row_order_U2 = tuck.mv.maxvol(u2)
        row_order_U3 = tuck.mv.maxvol(u3)

        r0 = [len(row_order_U1), len(row_order_U1), len(row_order_U1)]
        
        
        A = [None]*d

        for alpha in xrange(d):
            A[alpha] = np.tensordot(X[alpha].core, np.transpose(X[alpha].u[2][row_order_U3,:]), (2, 0))
            A[alpha] = np.tensordot(np.transpose(A[alpha], [2,0,1]), np.transpose(X[alpha].u[1][row_order_U2,:]), (2, 0))
            A[alpha] = np.tensordot(np.transpose(A[alpha], [0,2,1]), np.transpose(X[alpha].u[0][row_order_U1,:]), (2, 0))
            A[alpha] = np.transpose(A[alpha], [2,1,0])
        Ar = fun(A)

        A1 = np.reshape(Ar, [r0[0],-1], order='f')
        A1 = np.transpose(A1)    
        Q_A1, R = np.linalg.qr(A1)
        column_order_U1 = tuck.mv.maxvol(Q_A1)
        A1_11 = A1[column_order_U1, :]


        A2 = np.reshape(np.transpose(Ar, [1,0,2]), [r0[1],-1], order='f')
        A2 = np.transpose(A2)
        Q_A2, R = np.linalg.qr(A2)
        column_order_U2 = tuck.mv.maxvol(Q_A2)
        A2_11 = A2[column_order_U2, :]


        A3 = np.reshape(np.transpose(Ar, [2,0,1]), [r0[2],-1], order='f')
        A3 = np.transpose(A3)
        Q_A3, R = np.linalg.qr(A3)
        column_order_U3 = tuck.mv.maxvol(Q_A3)
        A3_11 = A3[column_order_U3, :]

#################################################################################


    U1 = u1
    U2 = u2
    U3 = u3

    U1_hat = np.linalg.solve(U1[row_order_U1, :].T, U1.T).T
    U2_hat = np.linalg.solve(U2[row_order_U2, :].T, U2.T).T
    U3_hat = np.linalg.solve(U3[row_order_U3, :].T, U3.T).T

    u1 = np.random.random((n[0],r_add[0]))
    u2 = np.random.random((n[1],r_add[1]))
    u3 = np.random.random((n[2],r_add[2]))
  

    UU1, ind_update_1 = column_update(U1_hat, u1, row_order_U1)
    UU2, ind_update_2 = column_update(U2_hat, u2, row_order_U2)
    UU3, ind_update_3 = column_update(U3_hat, u3, row_order_U3)

    U1 = np.concatenate((U1, u1), 1)
    U2 = np.concatenate((U2, u2), 1)
    U3 = np.concatenate((U3, u3), 1)

    A1_12 = np.zeros((r0[0], r_add[0]),dtype=dtype)
    for ii in xrange(r0[0]):
        for alpha in xrange(d):
            k1_order, j1_order = mod(column_order_U1[ii], r0[1])
            A[alpha] = np.dot(X[alpha].core, np.transpose(X[alpha].u[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [2,0,1]),np.transpose(X[alpha].u[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
            A[alpha] = np.tensordot(np.transpose(A[alpha], [0,2,1]),np.transpose(X[alpha].u[0][ind_update_1, :]), (2, 0))
            A[alpha] = np.transpose(A[alpha], [2,1,0])[:,0,0]
        A1_12[ii,:] = fun(A)


    A2_12 = np.zeros((r0[1], r_add[1]),dtype=dtype)
    for ii in xrange(r0[1]):
        for alpha in xrange(d):
            k1_order, i1_order = mod(column_order_U2[ii], r0[0])
            A[alpha] = np.dot(X[alpha].core, np.transpose(X[alpha].u[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [2,1,0]), np.transpose(X[alpha].u[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
            A[alpha] = np.tensordot(np.transpose(A[alpha], [0,2,1]), np.transpose(X[alpha].u[1][ind_update_2, :]), (2, 0))
            A[alpha] = np.transpose(A[alpha], [1,2,0])[0,:,0]
        A2_12[ii, :] = fun(A)


    A3_12 = np.zeros((r0[2], r_add[2]),dtype=dtype)
    for ii in xrange(r0[2]):
        for alpha in xrange(d):
            j1_order, i1_order = mod(column_order_U3[ii], r0[0])
            A[alpha] = np.dot(np.transpose(X[alpha].core, [2,1,0]),np.transpose(X[alpha].u[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [0,2,1]),np.transpose(X[alpha].u[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
            A[alpha] = np.tensordot(np.transpose(A[alpha], [1,2,0]),np.transpose(X[alpha].u[2][ind_update_3, :]), (2, 0))[0,0,:]
        A3_12[ii, :] = fun(A)


    r[0] = r0[0]+r_add[0]
    r[1] = r0[1]+r_add[1]
    r[2] = r0[2]+r_add[2]
    
    
    
    while True:
        
        for alpha in xrange(d):
            A[alpha] = np.dot(np.transpose(X[alpha].core, [2,1,0]), np.transpose(X[alpha].u[0][ind_update_1,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [0,2,1]), np.transpose(X[alpha].u[1][row_order_U2,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [1,2,0]), np.transpose(X[alpha].u[2][row_order_U3,:]))
        Ar_1 = np.concatenate((Ar, fun(A)), 0)
        
        row_order_U1 = np.concatenate((row_order_U1, ind_update_1))
        
        for alpha in xrange(d):
            A[alpha] = np.dot(np.transpose(X[alpha].core, [0,2,1]), np.transpose(X[alpha].u[1][ind_update_2,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [0,2,1]), np.transpose(X[alpha].u[2][row_order_U3,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [2,1,0]), np.transpose(X[alpha].u[0][row_order_U1,:]))
            A[alpha] = np.transpose(A[alpha], [2,1,0])
        Ar_2 = np.concatenate((Ar_1, fun(A)), 1)
        
        row_order_U2 = np.concatenate((row_order_U2, ind_update_2))
        
        for alpha in xrange(d):
            A[alpha] = np.dot(X[alpha].core, np.transpose(X[alpha].u[2][ind_update_3,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [2,0,1]),np.transpose(X[alpha].u[1][row_order_U2,:]))
            A[alpha] = np.dot(np.transpose(A[alpha], [0,2,1]),np.transpose(X[alpha].u[0][row_order_U1,:]))
            A[alpha] = np.transpose(A[alpha], [2,1,0])
        Ar = np.concatenate((Ar_2, fun(A)), 2)
        
        row_order_U3 = np.concatenate((row_order_U3, ind_update_3))
        
        
        
        A1 = np.reshape(Ar, [r[0],-1], order='f')
        A1 = np.transpose(A1)
        column_order_update_U1 = tuck.mv.maxvol( schur_comp(A1, A1_11, A1_12, dtype) )
        r_add[0] = len(column_order_update_U1)
        
        A2 = np.reshape(np.transpose(Ar, [1,0,2]), [r[1],-1], order='f')
        A2 = np.transpose(A2)
        column_order_update_U2 = tuck.mv.maxvol( schur_comp(A2, A2_11, A2_12, dtype) )
        r_add[1] = len(column_order_update_U2)
        
        A3 = np.reshape(np.transpose(Ar, [2,0,1]), [r[2],-1], order='f')
        A3 = np.transpose(A3)
        column_order_update_U3 = tuck.mv.maxvol( schur_comp(A3, A3_11, A3_12, dtype) )
        r_add[2] = len(column_order_update_U3)
        
        
        
        u1_approx = np.zeros((n[0], r_add[0]), dtype=dtype)
        u1 = np.zeros((n[0], r_add[0]), dtype=dtype)
        for i in xrange(r_add[0]):
            for alpha in xrange(d):
                k1_order, j1_order = mod(column_order_update_U1[i], r[1])
                A[alpha] = np.dot(X[alpha].core, np.transpose(X[alpha].u[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
                A[alpha] = np.dot(np.transpose(A[alpha], [2,0,1]),np.transpose(X[alpha].u[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
                A[alpha] = np.tensordot(np.transpose(A[alpha], [0,2,1]),np.transpose(X[alpha].u[0]), (2, 0))
                A[alpha] = np.transpose(A[alpha], [2,1,0])[:,0,0]
            u1[:,i] = fun(A)
            
            u1_approx_i = np.dot(Ar, np.transpose(UU3[row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
            u1_approx_i = np.dot(np.transpose(u1_approx_i,[2,0,1]),np.transpose(UU2[row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
            u1_approx_i = np.tensordot(np.transpose(u1_approx_i,[0,2,1]),np.transpose(UU1), (2, 0))
            u1_approx_i = np.transpose(u1_approx_i,[2,1,0])
            u1_approx[:,i] = u1_approx_i[:, 0, 0]
        
        
        u2_approx = np.zeros((n[1], r_add[1]), dtype=dtype)
        u2 = np.zeros((n[1], r_add[1]), dtype=dtype)
        for j in xrange(r_add[1]):
            for alpha in xrange(d):
                k1_order, i1_order = mod(column_order_update_U2[j], r[0])
                A[alpha] = np.dot(X[alpha].core, np.transpose(X[alpha].u[2][row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
                A[alpha] = np.dot(np.transpose(A[alpha], [2,1,0]), np.transpose(X[alpha].u[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
                A[alpha] = np.tensordot(np.transpose(A[alpha], [0,2,1]), np.transpose(X[alpha].u[1]), (2, 0))
                A[alpha] = np.transpose(A[alpha], [1,2,0])[0,:,0]
            u2[:,j] = fun(A)
            
            u2_approx_j = np.dot(Ar,np.transpose(UU3[row_order_U3[k1_order]:row_order_U3[k1_order]+1,:]))
            u2_approx_j = np.dot(np.transpose(u2_approx_j,[2,1,0]),np.transpose(UU1[row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
            u2_approx_j = np.tensordot(np.transpose(u2_approx_j,[0,2,1]),np.transpose(UU2), (2, 0))
            u2_approx[:,j] = u2_approx_j[0, 0, :]
        
        u3_approx = np.zeros((n[2], r_add[2]), dtype=dtype)
        u3 = np.zeros((n[2], r_add[2]), dtype=dtype)
        for k in xrange(r_add[2]):
            for alpha in xrange(d):
                j1_order, i1_order = mod(column_order_update_U3[k], r[0])
                A[alpha] = np.dot(np.transpose(X[alpha].core, [2,1,0]),np.transpose(X[alpha].u[0][row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
                A[alpha] = np.dot(np.transpose(A[alpha], [0,2,1]),np.transpose(X[alpha].u[1][row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
                A[alpha] = np.tensordot(np.transpose(A[alpha], [1,2,0]),np.transpose(X[alpha].u[2]), (2, 0))[0,0,:]
            u3[:,k] = fun(A)
            
            u3_approx_k = np.dot(np.transpose(Ar,[2,1,0]),np.transpose(UU1[row_order_U1[i1_order]:row_order_U1[i1_order]+1,:]))
            u3_approx_k = np.dot(np.transpose(u3_approx_k,[0,2,1]),np.transpose(UU2[row_order_U2[j1_order]:row_order_U2[j1_order]+1,:]))
            u3_approx_k = np.tensordot(np.transpose(u3_approx_k,[1,2,0]),np.transpose(UU3), (2, 0))
            u3_approx[:,k] = u3_approx_k[0, 0, :]
        
        
        eps_cross = 1./3*(  np.linalg.norm(u1_approx - u1)/ np.linalg.norm(u1) +
                          np.linalg.norm(u2_approx - u2)/ np.linalg.norm(u2) +
                          np.linalg.norm(u3_approx - u3)/ np.linalg.norm(u3)   )
        if pr <> None:
            print 'relative accuracy = %s' % (eps_cross), 'ranks = %s' % r
        
        if eps_cross < delta_cross:
            break
        elif r[0]>rmax:
            print 'Rank has exceeded rmax value'
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
        
        A1_12 = U1[ind_update_1, r_add[0]:].T
        A2_12 = U2[ind_update_2, r_add[1]:].T
        A3_12 = U3[ind_update_3, r_add[2]:].T
        
        r[0] = r[0]+r_add[0]
        r[1] = r[1]+r_add[1]
        r[2] = r[2]+r_add[2]
    
    

    U1, R1 = np.linalg.qr(UU1)
    U2, R2 = np.linalg.qr(UU2)
    U3, R3 = np.linalg.qr(UU3)
    
    
    GG = np.tensordot(np.transpose(Ar,[2,1,0]),np.transpose(R1), (2, 0))
    GG = np.tensordot(np.transpose(GG,[0,2,1]),np.transpose(R2), (2, 0))
    GG = np.transpose(GG,[1,2,0])
    G = np.tensordot(GG,np.transpose(R3), (2, 0))

    G_Tucker = tuck.tensor(G, delta_cross)
    if pr <> None:
        print 'ranks after rounding = %s' % G_Tucker.r[0], G_Tucker.r[1], G_Tucker.r[2]
    
    
    fun = tuck.tensor()
    fun.core = G_Tucker.core
    fun.u[0] = np.dot(U1, G_Tucker.u[0])
    fun.u[1] = np.dot(U2, G_Tucker.u[1])
    fun.u[2] = np.dot(U3, G_Tucker.u[2])
    fun.r =  G_Tucker.r
    fun.n = n

    return fun



def schur_comp(A, A11, A12, dtype):
    r, r0 = A12.shape
    R = r + r0
    
    #print np.linalg.solve(A11.T, A[:,:r].T).T
    S_hat = np.zeros((R,r0), dtype=dtype)
    
    S_hat[:r, :] = np.dot(pinv(A11), -A12)#np.linalg.solve(A11, -A12)
    S_hat[r:, :] = np.identity(r0)
    
    #print A[:,:]
    #uu, ss, vv = np.linalg.svd(np.dot(A, S_hat))
    #'ss:', ss
    
    
    Q, R = np.linalg.qr(np.dot(A, S_hat))
    #Q , trash1, trash2 = round_matrix(np.dot(A, S_hat), delta_tucker)
    
    return Q

def mod(X,Y):
    return int(X/Y), X%Y

def maxvol_update(A, ind, dtype):
    # finds new r0 good rows
    # [ A11 A12]
    # [ A21 A22] => S = A22 - A21 A11^(-1) A12
    
    N, R = A.shape
    r = len(ind)
    r0 = R - r
    
    S_hat = np.zeros((R, r0),dtype=dtype)
    
    S_hat[:r, :] = np.linalg.solve(A[ind, :r], -A[ind, r:])
    S_hat[r:, :] = np.identity(r0)
    Q, R = np.linalg.qr(np.dot(A, S_hat))
    
    ind_update = tuck.mv.maxvol(Q)
    
    
    return ind_update


def column_update(UU, u, ind):
    
    S = u - np.dot(UU, u[ind,:])
    ind_add = tuck.mv.maxvol(S)
    
    SS = np.dot(pinv(S[ind_add, :].T), S.T).T # WARNING! pinv instead of solve!
    #np.linalg.solve(S[ind_add, :].T, S.T).T#np.dot(np.linalg.pinv(S[ind_add, :].T), S.T).T
    
    U1 = UU - np.dot(SS, UU[ind_add])
    U2 = SS
    
    return np.concatenate((U1, U2), 1), ind_add

def H(A):
    return np.transpose(np.conjugate(A))
