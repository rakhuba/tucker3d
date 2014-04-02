import numpy as np
import time
from math import pi
import maxvol as mv
import copy

def svd(A):
    try:
        return np.linalg.svd(A, full_matrices = False)
    except: #LinAlgError
        try:
            print "SVD failded"
            return np.linalg.svd(A + 1e-12*np.linalg.norm(A, 1), full_matrices = False)
        except:
            print "SVD failded twice"
            return np.linalg.svd(A + 1e-8*np.linalg.norm(A, 1), full_matrices = False)



class tensor:

    def __init__(self, A = None, eps = 1e-14):

        if A is None:
            self.G = 0
            self.U = [0, 0, 0]
            self.n = [0, 0, 0]
            self.r = [0, 0, 0]
            return

        N1, N2, N3 = A.shape

        B1 = np.reshape(A, (N1, -1), order='F')
        B2 = np.reshape(np.transpose(A, [1, 0, 2]), (N2, -1), order='F' )
        B3 = np.reshape(np.transpose(A, [2, 0, 1]), (N3, -1), order='F')

        U1, V1, r1 = svd_trunc(B1, eps)
        U2, V2, r2 = svd_trunc(B2, eps)
        U3, V3, r3 = svd_trunc(B3, eps)

        G = np.dot(A, np.conjugate(U3))
        G = np.transpose(G, [2, 0, 1])
        G = np.dot(G, np.conjugate(U2))
        G = np.transpose(G, [0, 2, 1])
        G = np.dot(G, np.conjugate(U1))
        G = np.transpose(G, [2, 1, 0])

        self.n = [N1, N2, N3]
        self.r = G.shape
        self.U = [U1, U2, U3]
        self.G = G

    def __getitem__(self, index):
        return [self.U[0], self.U[1], self.U[2], self.G][index]
        
    
    def __repr__(self):
        
        res = "This is a 3D tensor in the Tucker format with \n"
        r = self.r
        n = self.n
        for i in xrange(3):
            res = res + ("r(%d)=%d, n(%d)=%d \n" % (i, r[i], i, n[i]))
        
        return res

    def __add__(self, other):

        #if self.n != other.n:
        #print self.n, other.n
        #  raise Exception('mode sizes must agree')
        
        c = tensor()
        c.r = [self.r[0] + other.r[0], self.r[1] + other.r[1], self.r[2] + other.r[2] ]
        c.n = self.n
        c.U[0] = np.concatenate((self.U[0], other.U[0]), axis = 1)
        c.U[1] = np.concatenate((self.U[1], other.U[1]), axis = 1)
        c.U[2] = np.concatenate((self.U[2], other.U[2]), axis = 1)
        c.G = np.zeros(c.r, dtype=np.complex128)
        c.G[:self.r[0], :self.r[1], :self.r[2]] = self.G
        c.G[self.r[0]:, self.r[1]:, self.r[2]:] = other.G

        return c

    def __rmul__(self, const): # only scalar by tensor product!
        mult = copy.copy(self)
        mult.G = const * self.G
        return mult

    def __neg__(self):
        neg = copy.copy(self)
        neg.G = (-1.) * neg.G
        return neg

    def __sub__(self, other):
        a = copy.copy(self)
        b = copy.copy(other)
        b.G = (-1.) * b.G
        sub = a + b
        return sub
        
                
    def full(self):

        A = np.dot(self.G, np.transpose(self.U[2]))
        A = np.transpose(A, [2,0,1])
        A = np.dot(A, np.transpose(self.U[1]))
        A = np.transpose(A, [0,2,1])
        A = np.dot(A, np.transpose(self.U[0]))
        A = np.transpose(A, [2,1,0])

        return A

def can2tuck(g, U1, U2, U3):

    a = tensor()
    
    n, r1 = U1.shape
    n, r2 = U2.shape
    n, r3 = U3.shape

    if r1<>r2 or r2<>r3 or r1<>r3:
        raise Exception("Wrong factor sizes")
    
    r = r1
    G = np.zeros((r, r, r), dtype = np.complex128)
    for i in xrange(r):
        G[i, i, i] = g[i]

    a.r = (r1, r2, r3)
    a.n = (n, n, n)
    
    a.G = G
    a.U[0] = U1.copy()
    a.U[1] = U2.copy()
    a.U[2] = U3.copy()
    
    return a

def real(a): # doubled ranks!

    b = tensor()
    
    b.n = a.n
    b.U[0] = np.concatenate((np.real(a.U[0]), np.imag(a.U[0])), 1)
    b.U[1] = np.concatenate((np.real(a.U[1]), np.imag(a.U[1])), 1)
    b.U[2] = np.concatenate((np.real(a.U[2]), np.imag(a.U[2])), 1)

    R1 = np.zeros((2*a.r[0], a.r[0]), dtype = np.complex128)
    R2 = np.zeros((2*a.r[1], a.r[1]), dtype = np.complex128)
    R3 = np.zeros((2*a.r[2], a.r[2]), dtype = np.complex128)

    R1[:a.r[0], :] = np.identity(a.r[0])
    R1[a.r[0]:, :] = 1j*np.identity(a.r[0])
    R2[:a.r[1], :] = np.identity(a.r[1])
    R2[a.r[1]:, :] = 1j*np.identity(a.r[1])
    R3[:a.r[2], :] = np.identity(a.r[2])
    R3[a.r[2]:, :] = 1j*np.identity(a.r[2])
    
    
    GG = np.dot(np.transpose(a.G,[2,1,0]),np.transpose(R1))
    GG = np.dot(np.transpose(GG,[0,2,1]),np.transpose(R2))
    GG = np.transpose(GG,[1,2,0])
    b.G = np.real(np.dot(GG,np.transpose(R3)))

    b.r = b.G.shape
    
    return b
    
def full(a, ind = None):
    if ind == None:
        return a.full()
    else:
        b = tensor()
        b.r = a.r
        b.G = a.G
        b.U[0] = a.U[0][ind[0], :]
        b.U[1] = a.U[1][ind[1], :]
        b.U[2] = a.U[2][ind[2], :]
        b.n[0] = len(ind[0])
        b.n[1] = len(ind[1])
        b.n[2] = len(ind[2])
        return b.full()

def qr(a):

    b = tensor()

    b.G = a.G
    b.r = a.r
    b.n = a.n
    
    b.U[0], R1 = np.linalg.qr(a.U[0])
    b.U[1], R2 = np.linalg.qr(a.U[1])
    b.U[2], R3 = np.linalg.qr(a.U[2])

    GG = np.dot(np.transpose(a.G,[2,1,0]),np.transpose(R1))
    GG = np.dot(np.transpose(GG,[0,2,1]),np.transpose(R2))
    GG = np.transpose(GG,[1,2,0])
    b.G = np.dot(GG,np.transpose(R3))

    return b

def round(a, eps):

    a = qr(a)

    b = tensor()
    b.n = a.n
    core = tensor(a.G, eps)
    b.G = core.G
    b.r = b.G.shape
    b.U[0] = np.dot(a.U[0], core.U[0])
    b.U[1] = np.dot(a.U[1], core.U[1])
    b.U[2] = np.dot(a.U[2], core.U[2])

    return b

def conj(a):
    b = copy.copy(a)
    b.U[0] = np.conjugate(a.U[0])
    b.U[1] = np.conjugate(a.U[1])
    b.U[2] = np.conjugate(a.U[2])
    b.G = np.conjugate(a.G)

    return b

def dot(a, b):

    U0 = np.dot(H(a.U[0]), b.U[0]) # Gram matrices (size ra * rb)
    U1 = np.dot(H(a.U[1]), b.U[1])
    U2 = np.dot(H(a.U[2]), b.U[2])

    G = np.dot(b.G, U2.T) # b0 b1 a2
    G = np.transpose(G, [0, 2, 1]) # b0 a2 b1
    G = np.dot(G, U1.T) # b0 a2 a1
    G = np.transpose(G, [2, 1, 0]) # a1 a2 b0
    G = np.dot(G, U0.T) # a1 a2 a0
    G = np.transpose(G, [2, 0, 1])

    G = np.conjugate(a.G) * G
    return sum(sum(sum(G)))

def norm(a): # need correction
    return np.sqrt(dot(a, a))

def fft(a):

    b = tensor()

    b.G = a.G
    b.r = a.r
    b.n = a.n
    b.U[0] = np.fft.fft(a[0], axis = 0)
    b.U[1] = np.fft.fft(a[1], axis = 0)
    b.U[2] = np.fft.fft(a[2], axis = 0)

    return b

def ifft(a):

    b = tensor()

    b.G = a.G
    b.r = a.r
    b.n = a.n
    b.U[0] = np.fft.ifft(a[0], axis = 0)
    b.U[1] = np.fft.ifft(a[1], axis = 0)
    b.U[2] = np.fft.ifft(a[2], axis = 0)

    return b

def dst(a):
    
    b = tensor()

    b.G = a.G
    b.r = a.r
    b.n = a.n
    b.U[0] = dst1D(np.real(a[0])) + 1j * dst1D(np.imag(a[0]))
    b.U[1] = dst1D(np.real(a[1])) + 1j * dst1D(np.imag(a[1]))
    b.U[2] = dst1D(np.real(a[2])) + 1j * dst1D(np.imag(a[2]))

    return b

def idst(a):
    
    b = tensor()

    b.G = a.G
    b.r = a.r
    b.n = a.n
    b.U[0] = idst1D(np.real(a[0])) + 1j * idst1D(np.imag(a[0]))
    b.U[1] = idst1D(np.real(a[1])) + 1j * idst1D(np.imag(a[1]))
    b.U[2] = idst1D(np.real(a[2])) + 1j * idst1D(np.imag(a[2]))

    return b

def svd_trunc(A, eps = 1e-14):
    
    u, s, v = svd(A)

    N1, N2 = A.shape
    
    eps_svd = eps*s[0]/np.sqrt(3)
    r = min(N1, N2)
    for i in xrange(min(N1, N2)):
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


def ones((n1, n2, n3)):
    a = tensor()
    
    a.U[0] = np.ones((n1, 1), dtype = np.complex128)
    a.U[1] = np.ones((n2, 1), dtype = np.complex128)
    a.U[2] = np.ones((n3, 1), dtype = np.complex128)
    a.G = np.ones((1, 1, 1), dtype = np.complex128)
    a.r = (1, 1, 1)
    a.n = (n1, n2, n3)
    
    return a

def zeros((n1, n2, n3)):
    a = tensor()
    
    a.U[0] = np.zeros((n1, 1), dtype = np.complex128)
    a.U[1] = np.zeros((n2, 1), dtype = np.complex128)
    a.U[2] = np.zeros((n3, 1), dtype = np.complex128)
    a.G = np.ones((1, 1, 1), dtype = np.complex128)
    a.r = (1, 1, 1)
    a.n = (n1, n2, n3)
    
    return a


def dst1D(A):

    n = np.array(A.shape)
    new_size = n.copy()
    new_size[0] = 2*(n[0]+1)
    X = np.zeros(new_size, dtype = np.complex128)

    X[1: n[0] + 1, :] = A
    X = np.imag(np.fft.fft(X, axis = 0))
    return -X[1: n[0] + 1, :] * np.sqrt(2./(n[0] + 1))

def idst1D(A):

    n = np.array(A.shape)
    new_size = n.copy()
    new_size[0] = 2*(n[0]+1)
    X = np.zeros(new_size, dtype = np.complex128)

    X[1: n[0] + 1, :] = A
    X = (2*(n[0]+1))* np.imag(np.fft.ifft(X, axis = 0))
    return X[1: n[0] + 1, :] * np.sqrt(2./(n[0] + 1))

def dst3D(A):

    X = dst1D(A) # 0 1 2
    X = dst1D(np.transpose(X, [2, 0, 1])) # 2 0 1
    X = dst1D(np.transpose(X, [2, 1, 0])) # 1 0 2
    X = np.transpose(X, [1, 0, 2])

    return X
