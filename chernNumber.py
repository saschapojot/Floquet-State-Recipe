import numpy as np


s1 = np.array([[0, 1], [1, 0]], dtype=complex)
s2 = np.array([[0, -1j], [1j, 0]])
s3 = np.array([[1, 0], [0, -1]], dtype=complex)
J1 = 0.5 * np.pi
J2 = 3 * np.pi
J3 = 2.5 * np.pi
M = 1
gridNum = 3*258
Kx = np.linspace(0, 2 * np.pi, gridNum)
Ky = np.linspace(0, 2 * np.pi, gridNum)


# Eigensystem



def U1(k1):
    '''
    U1 = exp(-i * H1), H1 = J1 * sin(px) * s1
    :param k1:
    :return: U1
    '''
    tmp = J1 * np.sin(Kx[k1])
    return np.array([[np.cos(tmp), -1j * np.sin(tmp)], [-1j * np.sin(tmp), np.cos(tmp)]], dtype=np.complex)


def U2(k2):
    '''
    U2 = exp(-i * H2), H2 = J2 * sin(py) * s2
    :param k2:
    :return: U2
    '''
    tmp = J2 * np.sin(Ky[k2])
    return np.array([[np.cos(tmp), -np.sin(tmp)], [np.sin(tmp), np.cos(tmp)]], dtype=np.complex)


def U3(k1, k2):
    '''
    U3 = exp(-i * H3), H3=J3(M + cos(px) + cos(py))
    :param k1:
    :param k2:
    :return:
    '''
    tmp = J3 * (M + np.cos(Kx[k1]) + np.cos(Ky[k2]))
    return np.array([[np.cos(tmp) - 1j * np.sin(tmp), 0], [0, np.cos(tmp) + 1j * np.sin(tmp)]], dtype=np.complex)


def eigVec(k1, k2):
    '''

    :param k1:
    :param k2:
    :return: eigenvector of U=U3*u2*u1, in ascending order of phase
    '''
    U = np.matmul(U3(k1, k2), U2(k2))
    U = np.matmul(U, U1(k1))
    e, v = np.linalg.eig(U)
    ind = np.argsort(np.angle(e))
    return v[:, ind]


'''
def constructPlaqutte():
    # V1 is an array holding the eigenvectors with smaller phase,col V1[i*Kx.size+j,:] is (i,j)th eigenvector
    V1 = np.zeros((2, Kx.size * Ky.size), dtype=np.complex)
    # V2 is an array holding the eigenvectors with larger phasse, col V2[i*Kx.size+j,:] is (i,j)th eigenvector
    V2 = np.zeros((2, Kx.size * Ky.size), dtype=np.complex)
    wvNum = [(k1, k2) for k1 in range(0, Kx.size) for k2 in range(0, Ky.size)]
    for wv in wvNum:
        i = wv[0]
        j = wv[1]
        vecTmp = eigVec(i, j)
        V1[:, i * Kx.size + j] = vecTmp[:, 0]
        V2[:, i * Ky.size + j] = vecTmp[:, 1]
    return V1, V2

'''


def constructPlaquette():
    # V1  is a dict holding the eigenvectors with smaller phase
    V1 = dict()

    # V2 is a dict holding the eigenvectors with larger phase
    V2 = dict()
    wvNum = [(k1, k2) for k1 in range(0, Kx.size) for k2 in range(0, Ky.size)]
    for wv in wvNum:
        i = wv[0]
        j = wv[1]
        vecTmp = eigVec(i, j)
        V1[(i, j)] = vecTmp[:, 0]
        V2[(i, j)] = vecTmp[:, 1]

    return V1, V2


def plaquettePhase(V, k1, k2):
    '''
    This is an algorithm computing a plaquette of Berry phase by Resta, it has convergence problems
    :param V:dict
    :param k1:
    :param k2:
    :return: phase of plaquette (k1,k2), (k1+1,k2), (k1+1,k2+1),(k1+1,k2)
    '''
    '''
    v1Tmp = V[:, k1 * Kx.size + k2]
    v2Tmp = V[:, (k1 + 1) * Kx.size + k2]
    v3Tmp = V[:, (k1 + 1) * Kx.size + k2 + 1]
    v4Tmp = V[:, k1 * Kx.size + k2 + 1]
    '''
    v1Tmp = V[(k1, k2)]
    v2Tmp = V[(k1 + 1, k2)]
    v3Tmp = V[(k1 + 1, k2 + 1)]
    v4Tmp = V[(k1, k2 + 1)]

    retTmp = np.dot(v1Tmp.conj().T, v2Tmp)
    retTmp *= np.dot(v2Tmp.conj().T, v3Tmp)
    retTmp *= np.dot(v3Tmp.conj().T, v4Tmp)
    retTmp *= np.dot(v4Tmp.conj().T, v1Tmp)

    return -np.imag(np.log(retTmp))


def chernNumber():
    V1, V2 = constructPlaquette()

    c1 = 0
    c2 = 0
    wvNum = [(k1, k2) for k1 in range(0, Kx.size - 1) for k2 in range(0, Ky.size - 1)]
    #sum up all the phases from each plaquette
    for wv in wvNum:
        c1 += plaquettePhase(V1, wv[0], wv[1])
        c2 += plaquettePhase(V2, wv[0], wv[1])

    print(c1 / (2 * np.pi))
    print(c2 / (2 * np.pi))


chernNumber()
