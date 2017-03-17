# hybrid mimo ofdm

import numpy as np
import matplotlib.pyplot as plt
import itertools
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
	#return max( [ np.absolute( np.dot( A[:,i], A[:,j]) ) for i, j in itertools.product(range( A.shape[1] ), range(A.shape[1])) if i != j ] )

def precoder(Nt, Lt):
	sets = [1, -1, 1j, -1j]
	return 1/np.sqrt(Nt) * np.random.choice( sets, (Nt, Lt) )

N = 50
Nt = 128
Lt = 50
M = 128
L = 50

X = np.eye(Lt, N)
P = precoder(Nt, Lt)
A_T = 1 / np.sqrt(Nt) * DFT_matrix(Nt)

sets = [1, -1, 1j, -1j]
#P = np.random.choice( sets, (Nt, Lt) )
#print T(X).shape 
C = np.dot( np.dot ( T(X), T(P) ) , A_T )
#np.kron()
print mutual_coherence(C)

F = DFT_matrix(M)
n = np.random.choice(M, N)
F_L = F[:, 0:L]
B = F_L[n, :]
print mutual_coherence(B)

print B.shape
print C.shape

A = np.zeros((N, L * Nt) , dtype=complex )
for i in range(N):
	A[i, :] = np.kron(B[i, :], C[i, :]) 

# print A[0,1:3]
# print A[1,1:3]
print A.shape
print mutual_coherence(A)


# plt.subplot(2, 1, 1)
# plt.plot(x, y_sin)
# plt.plot(x, y_cos)
# plt.xlabel('x axis label')
# plt.ylabel('y axis label')
# plt.title('Sine and Cosine')
# plt.legend(['Sine', 'Cosine'])
# plt.grid()
# plt.subplot(2, 1, 2)
# plt.plot(x, y_sin)
# plt.plot(x, y_cos)
# plt.xlabel('x axis label')
# plt.ylabel('y axis label')
# plt.title('Sine and Cosine')
# plt.legend(['Sine', 'Cosine'])
# plt.grid()
# plt.show()



