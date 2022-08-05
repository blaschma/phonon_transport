import numpy as np
from utils import constants

class ModelSystems:
    def __init__(self):
        self.dimension = -1
        pass
    def build_hessian(self):
        pass

class Chain1D(ModelSystems):

    def __init__(self, k, N):
        super().__init__()
        self.k = k
        self.N = N
        self.hessian = self.build_hessian()
        self.dimension = 1

    def build_hessian(self):
        hessian = np.zeros((self.N,self.N), dtype=float)

        for i in range(0,self.N):
            #offdiagonals
            if(i<self.N-1):
                print(i)
                hessian[i,i+1] = -self.k
            if(i>0):
                hessian[i, i - 1] = -self.k
            #diagonals
            hessian[i,i] = -np.sum(hessian[i,:])
        # check sum rule
        assert np.sum(hessian) == 0, "Sum rule not correct"
        return hessian

class FiniteLattice2D(ModelSystems):

    def __init__(self):
        super().__init__()
        self.dimension = 2
        self.N_y = 3
        self.N_x = 2
        self.k_x = 0.1 * (constants.eV2hartree / constants.ang2bohr ** 2) * 0
        self.k_y = 0.1 * (constants.eV2hartree / constants.ang2bohr ** 2) * 1
        self.hessian = self.build_hessian()

    def build_hessian(self):

        def build_H_NN():
            H_NN = np.zeros((2*self.N_y, 2*self.N_y), dtype=float)
            for i in range(0,H_NN.shape[0]):
                # x components
                if(i%2 == 0):
                    H_NN[i,i] = 2*self.k_x
                # y components
                else:
                    # offdiagonals
                    if (i < 2*self.N_y - 1):
                        H_NN[i, i + 2] = -self.k_y
                    if (i > 1):
                        H_NN[i, i - 2] = -self.k_y
                    # diagonals
                    H_NN[i, i] = -np.sum(H_NN[i, 1::2])
            return H_NN
        def build_H_00():
            H_00 = np.zeros((2 * self.N_y, 2 * self.N_y), dtype=float)
            for i in range(0, H_00.shape[0]):
                # x components
                if (i % 2 == 0 ):
                    if(self.N_x > 1):
                        H_00[i, i] = self.k_x
                # y components
                else:
                    # offdiagonals
                    if (i < 2 * self.N_y - 1):
                        H_00[i, i + 2] = -self.k_y
                    if (i > 1):
                        H_00[i, i - 2] = -self.k_y
                    # diagonals
                    H_00[i, i] = -np.sum(H_00[i, 1::2])
            return H_00

        def build_H_01():
            H_01 = np.zeros((2 * self.N_y, 2 * self.N_y), dtype=float)
            for i in range(0, H_01.shape[0]):
                #x components
                if(i%2==0):
                    #H_01[i,i+2] = -self.k_x
                    H_01[i, i] = -self.k_x
                if (i > 1):
                    #H_01[i, i - 2] = -self.k_y
                    pass
            return H_01

        H_NN = build_H_NN()
        H_00 = build_H_00()
        H_01 = build_H_01()
        hessian = np.zeros((self.N_y * 2 * self.N_x, self.N_y * 2 * self.N_x), dtype=float)
        for i in range(0,self.N_x):
            #surface element beginning
            if(i == 0):
                hessian[0:H_00.shape[0], 0:H_00.shape[0]] = H_00
            #surface element end
            elif(i==self.N_x-1 and self.N_x>1):
                hessian[i * 2 * self.N_y: i * 2 * self.N_y + H_00.shape[0],
                i * 2 * self.N_y: i * 2 * self.N_y + H_00.shape[0]] = H_00
            #intermediate layers
            elif(self.N_x>1):
                hessian[i * 2 * self.N_y: i * 2 * self.N_y + H_00.shape[0],
                i * 2 * self.N_y: i * 2 * self.N_y + H_00.shape[0]] = H_NN
            # couple everything
            if(i<self.N_x-1 and self.N_x>1):
                hessian[i * 2 * self.N_y: i * 2 * self.N_y + H_00.shape[0],
                (i+1) * 2 * self.N_y: (i + 1) * 2 * self.N_y + H_00.shape[0]] = H_01
                hessian[(i + 1) * 2 * self.N_y: (i + 1) * 2 * self.N_y + H_00.shape[0],
                i * 2 * self.N_y: i * 2 * self.N_y + H_00.shape[0]] = H_01
        #check sum rule
        assert np.sum(hessian) == 0, "Sum rule not correct"
        return hessian


if __name__ == '__main__':
    #model_system = Chain1D(k=0.1, N =3)
    #hessian = model_system.hessian

    model_system = FiniteLattice2D()
    hessian = model_system.hessian
    print(hessian)



