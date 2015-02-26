import numpy as np
import copy
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

def cross2d_full(func, eps, r0 = 4, rmax = 100):
    
    
    #A = np.zeros((M1, M2), dtype = np.complex128)
    #for i in xrange(M1):
    #    for j in xrange(M2):
    #        A[i, j] = func(i, j)
    
    M1, M2 = func.shape
    
    r1_0 = r0
    r2_0 = r0
    
    r1 = r1_0
    r2 = r2_0
    
    U1 = np.zeros((M1, r1), dtype = np.complex128)
    U2 = np.zeros((M2, r2), dtype = np.complex128)
    
    U1[:, :] = np.random.random((M1,r1))
    U2[:, :] = np.random.random((M2,r2))
    
    U1, R = np.linalg.qr(U1)
    U2, R = np.linalg.qr(U2)
    
    eps_cross = 1
    
    
    row_order_U1 = tuck.mv.maxvol(U1)
    row_order_U2 = tuck.mv.maxvol(U2)
    
    
    
    u1 = func[:, row_order_U2]
    u2 = func[row_order_U1, :].T
    
    U1_hat = np.linalg.solve(U1[row_order_U1, :].T, U1.T).T
    U2_hat = np.linalg.solve(U2[row_order_U2, :].T, U2.T).T
    
    UU1, ind_update_1 = column_update(U1_hat, u1, row_order_U1)
    UU2, ind_update_2 = column_update(U2_hat, u2, row_order_U2)
    
    U1 = np.concatenate((U1, u1), 1)
    U2 = np.concatenate((U2, u2), 1)
    
    r1 = r1 + r1_0
    r2 = r2 + r2_0
    
    
    while True:
        
        
        row_order_U1 = np.concatenate((row_order_U1, ind_update_1))
        row_order_U2 = np.concatenate((row_order_U2, ind_update_2))
        
        
        Ar = func[row_order_U1, :][:, row_order_U2]
        
        
        u1 = func[:, ind_update_2]
        u2 = func[ind_update_1, :].T
        
        u1_approx = np.dot(np.dot(UU1, Ar), H(UU2[ind_update_2, :]))
        u2_approx = np.dot(np.dot(UU1[ind_update_1, :], Ar), H(UU2)).T
        
        
        
        #A_appr = np.dot(np.dot(UU1, Ar), H(UU2))
        
        eps_cross = 1./2*(  np.linalg.norm(u1_approx - u1)/ np.linalg.norm(u1)  +  np.linalg.norm(u2_approx - u2)/ np.linalg.norm(u2))
        #print eps_cross, np.linalg.norm(A_appr - func)/np.linalg.norm(func)
        
        if eps_cross < eps:
            break
        if r1>rmax:
            print 'Rank has exceeded rmax value'
            break
        
        
        UU1, ind_update_1 = column_update(UU1, u1, row_order_U1)
        UU2, ind_update_2 = column_update(UU2, u2, row_order_U2)
        
        U1 = np.concatenate((U1, u1), 1)
        U2 = np.concatenate((U2, u2), 1)
        
        r1 = r1 + r1_0
        r2 = r2 + r2_0
    
    #print r1, r2
    
    U1, R1 = np.linalg.qr(UU1)
    U2, R2 = np.linalg.qr(UU2)
    
    G = np.dot(np.dot(R1, Ar), H(R2))
    
    u1, s, u2 = round_matrix(G, eps)
    #print s.shape
    
    U1 = np.dot(U1, u1)
    U2 = np.dot(U2, u2)
    
    return U1, U2
#print  np.linalg.norm(A - np.dot(U1,H(U2)))/np.linalg.norm(A)

def H(A):
    return np.transpose(np.conjugate(A))

def round_matrix(A, eps):
    
    u, s, v = np.linalg.svd(A, full_matrices=False)
    
    N, M = A.shape
    r = 0 # r=rank
    
    # rank
    for i in range(min(N, M)):
        if s[i]>eps*s[0]:
            r+=1
    
    
    return u[:, :r], np.diag(s[:r]), H( np.dot(np.diag(s[:r]), v[:r, :]) )


def column_update(UU, u, ind):
    
    S = u - np.dot(UU, u[ind,:])
    ind_add = tuck.mv.maxvol(S)
    
    SS = np.dot(pinv(S[ind_add, :].T), S.T).T # WARNING! pinv instead of solve!
    #np.linalg.solve(S[ind_add, :].T, S.T).T#np.dot(np.linalg.pinv(S[ind_add, :].T), S.T).T
    
    U1 = UU - np.dot(SS, UU[ind_add])
    U2 = SS
    
    return np.concatenate((U1, U2), 1), ind_add