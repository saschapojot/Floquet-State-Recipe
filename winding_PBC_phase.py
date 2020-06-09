import numpy as np

from datetime import datetime


def U1Operator_PBC(J1, px, time):
    '''

    :param J1:
    :param p1:
    :param time:
    :return: exp(-iH1*time), H1=J1*sin(px)sx
    '''
    tmp = J1 * np.sin(px) * time
    return np.array([[np.cos(tmp), -1j * np.sin(tmp)], [-1j * np.sin(tmp), np.cos(tmp)]], dtype=np.complex)


def U2Operator_PBC(J2, py, time):
    '''

    :param J2:
    :param py:
    :param time:
    :return: exp(-iH2*time), H2=J2*sin(py)*s2
    '''
    tmp = J2 * np.sin(py) * time
    return np.array([[np.cos(tmp), -np.sin(tmp)], [np.sin(tmp), np.cos(tmp)]], dtype=np.complex)


def U3Operator_PBC(J3, M, px, py, time):
    '''

    :param J3:
    :param M:
    :param px:
    :param py:
    :param time:
    :return: exp(-iH3*time), H3=J3*(M+cos(px)+cos(py))*s3
    '''
    tmp = J3 * (M + np.cos(px) + np.cos(py)) * time
    return np.array([[np.cos(tmp) - 1j * np.sin(tmp), 0], [0, np.cos(tmp) + 1j * np.sin(tmp)]], dtype=np.complex)

def retMod(x,m):
    if(m!=0):
        return np.mod(x,m)
    else:
        return x
def UOperator_PBC(J1, J2, J3, M, px, py, T1, t, cut):
    # U=np.zeros((2,2),dtype=np.complex)
    if t <= T1[0]:
        U = U1Operator_PBC(J1, px, t)
    elif (t > T1[0]) and (t <= T1[1]):
        U1 = U1Operator_PBC(J1, px, 1 / 3)
        U2 = U2Operator_PBC(J2, py, 1 / 3)
        U = np.matmul(U2, U1)

    elif (t > T1[1]) and (t <= T1[2]):
        U1 = U1Operator_PBC(J1, px, 1 / 3)
        U2 = U2Operator_PBC(J2, py, 1 / 3)
        U3 = U3Operator_PBC(J3, M, px, py, t - 2 / 3)
        U = np.matmul(U3, U2)
        U = np.matmul(U, U1)

    else:
        U1 = U1Operator_PBC(J1, px, 1 / 3)
        U2 = U2Operator_PBC(J2, py, 1 / 3)
        U3 = U3Operator_PBC(J3, M, px, py, 1 / 3)
        U = np.matmul(U3, U2)
        U = np.matmul(U, U1)
        EigU, VecU = np.linalg.eig(U)
        E = np.zeros(VecU.shape, dtype=np.complex)
        for i in range(0, EigU.size):
            phaseCut = retMod(-np.imag(np.log(EigU[i])), cut)
            E[i, i] = np.exp(-1j * phaseCut * (1 - (t - 1)))
        UNew = np.matmul(VecU, E)
        U = np.matmul(UNew, np.linalg.inv(VecU))
    return U


J1 = 3*0.5 * np.pi
J2 = 3*0.5 * np.pi
J3 = 3*0.2 * np.pi
M = 1
#number of grid points
GN=129
Kx = np.linspace(0, 2 * np.pi, GN)
Ky = np.linspace(0, 2 * np.pi, GN)
T1 = np.array([1 / 3, 2 / 3, 3 / 3])
T = np.linspace(0, 2, GN)
CUT = -2*np.pi


def commutator(A, B):
    return (np.matmul(A, B) - np.matmul(B, A))


start=datetime.now()
WList = []
points = [(a, b, c) for a in range(0, Kx.size-1) for b in range(0, Ky.size-1) for c in range(0, T.size-1)]
for elem in points:
    a = elem[0]
    b = elem[1]
    c = elem[2]


    U = UOperator_PBC(J1, J2, J3, M, Kx[a], Ky[b], T1, T[c], CUT)
    Uxp = UOperator_PBC(J1, J2, J3, M, Kx[a + 1], Ky[b], T1, T[c], CUT)
    Uyp = UOperator_PBC(J1, J2, J3, M, Kx[a], Ky[b + 1], T1, T[c], CUT)
    Utp = UOperator_PBC(J1, J2, J3, M, Kx[a], Ky[b], T1, T[c + 1], CUT)
    Uinv = np.linalg.inv(U)
    tmpt = np.matmul(Uinv, Utp )
    tmpkx = np.matmul(Uinv, Uxp)
    tmpky = np.matmul(Uinv, Uyp )
    wTmp = np.trace(np.matmul(tmpt, commutator(tmpkx, tmpky)))
    WList.append(wTmp)

W = sum(WList) / (8 * np.pi ** 2)
end=datetime.now()

print(W)
print('Time: '+str(end-start))

