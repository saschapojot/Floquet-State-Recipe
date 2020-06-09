import numpy as np
import matplotlib.pyplot as plt



class obcxSolver:
    def __init__(self):
        # parameters
        self.J1 = 0.5 * np.pi
        self.J2 = 2.5 * np.pi
        self.J3 = 0.2 * np.pi
        self.M = 1
        self.dx = 1 / 60
        self.lengthLocal = 0.2
        self.K2 = np.arange(0, 2 + self.dx, self.dx)
        self.K2 *= np.pi
        self.N1 = 100
        self.s1 = np.matrix([[0, 1], [1, 0]], dtype=np.complex)
        self.s2 = np.matrix([[0, -1j], [1j, 0]])
        self.s3 = np.matrix([[1, 0], [0, -1]], dtype=np.complex)
        self.s0 = np.matrix([[1, 0], [0, 1]], dtype=np.complex)
        self.n1Dim = 2 * self.N1
        # V is a 3d array holding eigenvector matrix for each momentum value
        self.V = np.zeros((self.K2.size, self.n1Dim, self.n1Dim), dtype=np.complex)
        # LM is a 2d array holding the localization length of each eigenvector for each momentum value
        self.LM = np.zeros((self.K2.size, self.n1Dim))
        # LC is a 2d array holding the localization center of each eigenvector for each momentum value
        self.LC = np.zeros((self.K2.size, self.n1Dim))
        #E is a 2d matrix holding the phases in ascending order for each momentum value
        self.E = np.zeros((self.K2.size, self.n1Dim))
        #ELE is a dict holding left edge state phases
        self.ELE = dict()
        #ERE is a dict holding right edge state phases
        self.ERE = dict()
        #EE is a dict holding bulk state phases
        self.EE = dict()
        #VL is a dict holding corresponding left edge state eigenvectors
        self.VL = dict()
        #VR is a dict holding corresponding right edge state eigenvectors
        self.VR = dict()

    def expMat(self, H):
        '''
        This is a function calculating exp(-iH)
        :param H: an Hermitian matrix
        :return: exp(-iH)
        '''
        # EigH is a 1d array holding eigenvalues of H, VH is the corresponding eigenvector array
        EigH, VH = np.linalg.eigh(H)
        # initialization, in order to calculate each exp(-i*EigH[a])
        EH = np.zeros((self.n1Dim, self.n1Dim), dtype=np.complex)
        for a in range(0, self.n1Dim):
            EH[a, a] = np.exp(-1j * EigH[a])

        # UH=exp(-iH)=VH*EH*VH^{-1}
        UH = np.matmul(VH, EH)
        UH = np.matmul(UH, np.linalg.inv(VH))
        return UH

    def calculatePhases(self):
        '''
        This function calculates the spectrum and eigenvectors of operator U=U3*U2*U1, localization length and localization center

        :return: None
        '''
        # assemble matrix H1
        H1 = np.zeros((self.n1Dim, self.n1Dim), dtype=np.complex)
        for a in range(0, self.n1Dim - 3, 2):
            H1[a, a + 3] = 1
        for a in range(1, self.n1Dim - 1, 2):
            H1[a, a + 1] = 1
        for a in range(2, self.n1Dim, 2):
            H1[a, a - 1] = -1

        for a in range(3, self.n1Dim, 2):
            H1[a, a - 3] = -1
        H1 *= -1j * self.J1 / 2
        # H1 is hermitian
        U1 = self.expMat(H1)
        # assemble H2, except for the coef J2sin(ky)
        H2Init = np.zeros((self.n1Dim, self.n1Dim), dtype=np.complex)
        for a in range(1, self.n1Dim, 2):
            H2Init[a - 1, a] = -1j
            H2Init[a, a - 1] = 1j
        # assemble H3, term1, except for the coef J3(M+cos(ky))
        H31Init = np.zeros((self.n1Dim, self.n1Dim), dtype=np.complex)
        for a in range(0, self.n1Dim, 2):
            H31Init[a, a] = 1
        for a in range(1, self.n1Dim, 2):
            H31Init[a, a] = -1

        # assemble H3, term2 including coef J3/2
        H32Init = np.zeros((self.n1Dim, self.n1Dim), dtype=np.complex)
        for a in range(2, self.n1Dim, 2):
            H32Init[a - 2, a] = 1
        for a in range(3, self.n1Dim, 2):
            H32Init[a - 2, a] = -1
        for a in range(2, self.n1Dim, 2):
            H32Init[a, a - 2] = 1

        for a in range(3, self.n1Dim, 2):
            H32Init[a, a - 2] = -1
        H32Init *= self.J3 / 2  # coef does not depend on ky
        for k2 in range(0, self.K2.size):
            H2 = H2Init * self.J2 * np.sin(self.K2[k2])
            U2 = self.expMat(H2)

            H3 = H31Init * self.J3 * (self.M + np.cos(self.K2[k2])) + H32Init
            U3 = self.expMat(H3)
            # U=U3*U2*U1
            U = np.matmul(U3, U2)
            U = np.matmul(U, U1)
            # U is not hermitian in general
            EigU, VecU = np.linalg.eig(U)
            # phases of eigenvalues EigU
            angVal = np.angle(EigU)
            # sort phases in ascending order, return indices
            ind = np.argsort(angVal)
            # sorted phases stored in k2th row of matrix E
            self.E[k2, :] = angVal[ind]
            # sort eigenvectors by ind
            VecU = VecU[:, ind]
            # sorted eigenvector stored in k2th page of 3d matrix V
            self.V[k2, :, :] = VecU

            for m in range(0, self.n1Dim):
                # this is a metric of localization by Raffaele Resta
                tmpList = [np.exp(1j * 2 * np.pi * a / (self.n1Dim)) * np.abs(VecU[a, m]) ** 2 for a in
                           range(0, self.n1Dim)]
                tmp = sum(tmpList)
                # localization length of momentum K2[k2], mth eigenvector
                self.LM[k2, m] = -np.log(np.abs(tmp))
                # localization center of momentum K2[k2], mth eigenvector
                self.LC[k2, m] = np.mod(self.n1Dim * np.imag(np.log(tmp)) / (2 * np.pi), self.n1Dim)

    def separateStates(self):
        '''
        This function separate left and right edge states from bulk states
        :return: None
        '''
        indAll = range(0, self.n1Dim)
        for k2 in range(0, self.K2.size):
            indEdge = [i for i, val in enumerate(self.LM[k2, :]) if val <= self.lengthLocal]
            indLeft = [a for a in indEdge if self.LC[k2, a] < self.n1Dim / 2]
            indRight = list(set(indEdge) - set(indLeft))
            indBulk = list(set(indAll) - set(indEdge))
            if len(indLeft) != 0:
                self.ELE[k2] = self.E[k2, indLeft]
                self.VL[k2] = self.V[k2, :, indLeft]
            if len(indRight) != 0:
                self.ERE[k2] = self.E[k2, indRight]
                self.VR[k2] = self.V[k2, :, indRight]
            if len(indBulk) != 0:
                self.EE[k2] = self.E[k2, indBulk]

    def plotSpectrum(self):
        plt.figure()
        bulkScatter = None
        leftScatter = None
        rightScatter = None
        #plot phases of  bulk states
        for k in self.EE.keys():
            elems = self.EE[k]
            if len(elems)!=0:
                bulkScatter = plt.scatter([self.K2[k] / np.pi] * len(elems), elems / np.pi, marker='.', c='black')
        #plot phases of left states
        for k in self.ELE.keys():
            elems = self.ELE[k]
            if len(elems)!=0:
                leftScatter = plt.scatter([self.K2[k] / np.pi] * len(elems), elems / np.pi, marker='.', c='red')
        #plot phases of rights states
        for k in self.ERE.keys():
            elems = self.ERE[k]
            if len(elems)!=0:
                rightScatter = plt.scatter([self.K2[k] / np.pi] * len(elems), elems / np.pi, marker='.', c='green')
        plt.xlabel('$K_{y}/\pi$')
        plt.ylabel('$E/\pi$')
        plt.ylim((-1.5, 1.5))
        plt.title('OBC along x, PBC along y')

        plt.legend((bulkScatter, leftScatter, rightScatter), ('Bulk state', 'left edge state', 'right edge state'),
                   loc='best')
        plt.savefig('spec_obcx.png')

    def plotEdgeStates(self):

        lTmp = np.zeros((1, self.n1Dim))
        for k in self.VL.keys():
            lTmp += np.power(np.abs(self.VL[k]), 2)

        lTmp /= max(max(lTmp))
        # lTmp/=len(self.VL.keys())
        # lTmp=np.power(lTmp,1/2)

        cell = np.arange(0, self.n1Dim)
        #plot proportion of left eigenvectors
        plt.figure()
        plt.bar(cell, lTmp[0], color='red')
        plt.title('Left edge state proportion')
        plt.savefig('leftBar_obcx.png')

        rTmp = np.zeros((1, self.n1Dim))
        for k in self.VR.keys():
            rTmp += np.power(np.abs(self.VR[k]), 2)

        rTmp /= max(max(rTmp))
        # rTmp/=len(self.VR.keys())
        # rTmp=np.power(rTmp,1/2)
        # plot proportion of right eigenvectors
        plt.figure()
        plt.bar(cell, rTmp[0], color='green')
        plt.title('Right edge state proportion')

        plt.savefig('rightBar_obcx.png')
