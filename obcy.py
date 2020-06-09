import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

startProg = datetime.now()
J1 = 0.5 * np.pi
J2 = 0.5 * np.pi
J3 = 0.4 * np.pi
M = 1

lengthLocal = 0.2
K1 = np.linspace(0, 2 * np.pi, 121)

N2 = 100
s1 = np.matrix([[0, 1], [1, 0]], dtype=np.complex)
s2 = np.matrix([[0, -1j], [1j, 0]], dtype=np.complex)
s3 = np.matrix([[1, 0], [0, -1]], dtype=np.complex)
s0 = np.matrix([[1, 0], [0, 1]], dtype=np.complex)
n2Dim = 2 * N2
# V is a 3d array holding eigenvector matrix for each momentum value
V = np.zeros((K1.size, n2Dim, n2Dim), dtype=np.complex)
# LM is a 2d array holding the localization length of each eigenvector for each momentum value
LM = np.zeros((K1.size, n2Dim))
# LC is a 2d array holding the localization center of each eigenvector for each momentum value
LC = np.zeros((K1.size, n2Dim))
# E is a 2d matrix holding the phases in ascending order for each momentum value
E = np.zeros((K1.size, n2Dim))
# ELE is a dict holding left edge state phases
ELE = dict()
# ERE is a dict holding right edge state phases
ERE = dict()
# EE is a dict holding bulk state phases
EE = dict()
# VL is a dict holding corresponding left edge state eigenvectors
VL = dict()
# VR is a dict holding corresponding right edge state eigenvectors
VR = dict()


def expMat(H):
    '''
    This si a function calculating exp(-iH)
    :param H:
    :return: exp(-iH)
    '''
    # EigH is a 1d array holding eigenvalues of H, VH is the corresponding eigenvector array
    EigH, VH = np.linalg.eigh(H)
    # initialization, in order to calculate each exp(-i*EigH[a])
    EH = np.zeros((n2Dim, n2Dim), dtype=complex)
    for a in range(0, n2Dim):
        EH[a, a] = np.exp(-1j * EigH[a])
    # UH=exp(-iH)=VH*EH*VH^{-1}
    UH = np.matmul(VH, EH)
    UH = np.matmul(UH, np.linalg.inv(VH))

    return UH


# asemble matrix H1, except for coef sin(K1[k1]), including j1

H1tmp = np.eye(N2, dtype=np.complex)
H1tmp = np.kron(H1tmp, s1)
H1tmp *= J1

# assemble matrix H2, including coef J2/2i

H2 = np.zeros((N2, N2), dtype=np.complex)
for a in range(1, N2):
    H2[a - 1, a] = 1
    H2[a, a - 1] = -1
H2 = np.kron(H2, s2)
H2 *= J2 / (2 * 1j)
U2 = expMat(H2)


# assemble H3, term1, except for coef M+cos(K1[k1]), including J3

H31tmp = np.eye(N2, dtype=np.complex)
H31tmp = np.kron(H31tmp, s3)
H31tmp *= J3

# assemble H3, term2, including coef J2/2

H32tmp = np.zeros((N2, N2), dtype=np.complex)
for a in range(1, N2):
    H32tmp[a - 1, a] = 1
    H32tmp[a, a - 1] = 1
H32tmp = np.kron(H32tmp, s3)
H32tmp *= J3 / 2

for k1 in range(0, K1.size):

    H1 = H1tmp * np.sin(K1[k1])

    H31 = H31tmp * (M + np.cos(K1[k1]))
    H32 = H32tmp
    H3 = H31 + H32
    U1 = expMat(H1)

    U3 = expMat(H3)
    # U=U3*U2*U1
    U = np.matmul(U3, U2)
    U = np.matmul(U, U1)

    # U is not Hermitian in general
    EigU, VecU = np.linalg.eig(U)
    # phases of eigenvalues EigU
    angVal = np.angle(EigU)
    # sort phases in ascending order, return indices
    ind = np.argsort(angVal)
    # sorted phases stored in k1th row of matrix E
    E[k1, :] = angVal[ind]
    # sort eigenvectors by ind
    VecU = VecU[:, ind]
    # sorted eigenvector stored in k1th page of 3d matrix V
    V[k1, :, :] = VecU
    for m in range(0, n2Dim):
        # this is a metric of localization by Raffaele Resta
        tmpList = [np.exp(1j * 2 * np.pi * (a + 1) / n2Dim) * np.abs(VecU[a, m]) ** 2 for a in range(0, n2Dim)]
        tmp = sum(tmpList)
        # localization length of momentum K1[k1], mth eigenvector
        LM[k1, m] = -np.log(np.abs(tmp))
        # localization center of momentum K1[k1], mth eigenvector
        LC[k1, m] = np.mod(n2Dim * np.imag(np.log(tmp)) / (2 * np.pi), n2Dim)

# separate states
indAll = range(0, n2Dim)
for k1 in range(0, K1.size):
    indEdge = [i for i, val in enumerate(LM[k1, :]) if val <= lengthLocal]
    indLeft = [a for a in indEdge if LC[k1, a] < n2Dim / 2]
    indRight = list(set(indEdge) - set(indLeft))
    indBulk = list(set(indAll) - set(indEdge))
    if len(indLeft) != 0:
        ELE[k1] = E[k1, indLeft]
        VL[k1] = V[k1, :, indLeft]
    if len(indRight) != 0:
        ERE[k1] = E[k1, indRight]
        VR[k1] = V[k1, :, indRight]
    if len(indBulk) != 0:
        EE[k1] = E[k1, indBulk]

# plot spectrum
plt.figure()
bulkScatter = None
leftScatter = None
rightScatter = None
# plot phases of  bulk states
for k in EE.keys():
    elems = EE[k]
    if len(elems) != 0:
        bulkScatter = plt.scatter([K1[k] / np.pi] * len(elems), elems / np.pi, marker='.', c='black')

# plot phaes of left states
for k in ELE.keys():
    elems = ELE[k]
    if len(elems) != 0:
        leftScatter = plt.scatter([K1[k] / np.pi] * len(elems), elems / np.pi, marker='.', c='red')

# plot phases of right states
for k in ERE.keys():
    elems = ERE[k]
    if len(elems) != 0:
        rightScatter = plt.scatter([K1[k] / np.pi] * len(elems), elems / np.pi, marker='.', c='green')
# plot spectrum
plt.xlabel('$K_{x}/\pi$')
plt.ylabel('$E/\pi$')
plt.ylim((-1.5, 1.5))
plt.title('OBC along y, PBC along x')
plt.legend((bulkScatter, leftScatter, rightScatter), ('Bulk state', 'Left edge state', 'right edge state'), loc='best')
plt.savefig('spec_obcy.png')

# plot proportion of left eigenvectors
cell = np.arange(0, n2Dim)
lTmp = np.zeros((1, n2Dim))
for k in VL.keys():
    lTmp += np.power(np.abs(VL[k]), 2)
lTmp /= max(max(lTmp))
plt.figure()
plt.bar(cell, lTmp[0], color='red')
plt.title('Left edge state proportion')
plt.savefig('leftBar_obcy.png')

# plot proportion of right eigenvectors
rTmp = np.zeros((1, n2Dim))
for k in VR.keys():
    rTmp += np.power(np.abs(VR[k]), 2)

rTmp /= max(max(rTmp))
plt.figure()
plt.bar(cell, rTmp[0], color='green')
plt.title('Right edge state proportion')
plt.savefig('rightBar_obcy.png')
endProg = datetime.now()
print('Time: ' + str(endProg - startProg))
