# hybrid mimo ofdm

import numpy as np
import matplotlib.pyplot as plt
import timeit
#from numpy.linalg import norm
#import scipy.linalg

def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * np.pi * 1j / N )
    W = np.power( omega, i * j ) / np.sqrt(N)
    return W

# Make some shorcuts for transpose,hermitian:
#    num.transpose(A) --> T(A)
#    hermitian(A) --> H(A)
def hermitian(A, **kwargs):
    return np.transpose(A,**kwargs).conj()
T = np.transpose
H = hermitian

def mutual_coherence(A):
    A_n = np.linalg.norm(A, axis=0, ord=2)
    A /= A_n
    return ( np.absolute( np.dot ( H(A), A ) ) - np.eye(A.shape[1]) ).max()

def precoder(Nt, Lt):
    sets = [1, -1, 1j, -1j]
    return 1/np.sqrt(Nt) * np.random.choice( sets, (Nt, Lt) )

N = 50
Nt = 128
Lt = 50
M = 128
L = 50

Times = 10
mu_A = []
mu_B = []
mu_C = []
for t in range(Times):

    start = timeit.default_timer()

    X = np.eye(Lt, N)
    P = precoder(Nt, Lt)
    A_T = 1 / np.sqrt(Nt) * DFT_matrix(Nt)
    C = np.dot( np.dot ( T(X), T(P) ) , A_T )
    mu_C.append( mutual_coherence(C) )
    print ( mutual_coherence(C) )

    F = DFT_matrix(M)
    n = np.random.choice(M, N)
    F_L = F[:, 0:L]
    B = F_L[n, :]
    mu_B.append( mutual_coherence(B) )
    print ( mutual_coherence(B) )

    A = np.zeros((N, L * Nt) , dtype=complex )
    for i in range(N):
        A[i, :] = np.kron(B[i, :], C[i, :]) 

    mu_A.append( mutual_coherence(A) )
    print ( mutual_coherence(A) )

    stop = timeit.default_timer()
    print ( stop - start  )


# plt.subplot(2, 1, 1)
plt.plot( range(Times), mu_A)
plt.plot( range(Times), mu_B)
plt.plot( range(Times), mu_C)
plt.xlabel('times')
plt.ylabel('mutual coherence')
plt.title('Analysis')
plt.legend(['joint', 'ofdm', 'precoder'])
plt.grid()
# plt.subplot(2, 1, 2)
# plt.plot(x, y_sin)
# plt.plot(x, y_cos)
# plt.xlabel('x axis label')
# plt.ylabel('y axis label')
# plt.title('Sine and Cosine')
# plt.legend(['Sine', 'Cosine'])
# plt.grid()
# plt.show()



