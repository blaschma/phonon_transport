__docformat__ = "google"
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
    """
    2D finite lattice
            N_x
        <------->
            k_x
    A...-o-o-o-o-o
    | ...| | | |k_y|
    N_y -o-o-o-o-o
    | ...| | | | |
    V ...-o-o-o-o-o

    """

    def __init__(self):
        super().__init__()
        self.dimension = 2
        self.N_y = 3
        self.N_x = 1
        self.k_x = 0.1 * (constants.eV2hartree / constants.ang2bohr ** 2) * 1
        self.k_y = 0.1 * (constants.eV2hartree / constants.ang2bohr ** 2) * 1
        self.k_xy = 0.1 * (constants.eV2hartree / constants.ang2bohr ** 2) * 0
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
        def build_H_NN_spacial():
            H_NN = np.zeros((2*self.N_y, 2*self.N_y), dtype=float)
            for i in range(0,H_NN.shape[0]):
                # x components
                if(i%2 == 0):
                    H_NN[i, i + 1] = -self.k_xy
                    H_NN[i + 1, i] = -self.k_xy
                    H_NN[i, i] = 2*self.k_x + self.k_xy
                # y components
                else:
                    # offdiagonals
                    if (i < 2*self.N_y - 1):
                        H_NN[i, i + 2] = -self.k_y
                    if (i > 1):
                        H_NN[i, i - 2] = -self.k_y
                    # diagonals
                    # vielleicht ist das falsch
                    H_NN[i, i] = -np.sum(H_NN[i, :])
                    # H_NN[i, i] = -np.sum(H_NN[i, 1::2])
            return H_NN
        def build_H_00_spacial():
            H_00 = np.zeros((2 * self.N_y, 2 * self.N_y), dtype=float)
            for i in range(0, H_00.shape[0]):
                # x components (and xy components)
                if (i % 2 == 0):
                    H_00[i, i + 1] = -self.k_xy
                    H_00[i + 1, i] = -self.k_xy
                    H_00[i, i] = self.k_x + self.k_xy
                # y components
                else:
                    # offdiagonals
                    if (i < 2 * self.N_y - 1):
                        H_00[i, i + 2] = -self.k_y
                    if (i > 1):
                        H_00[i, i - 2] = -self.k_y
                    # diagonals
                    #vielleicht ist das falsch
                    H_00[i, i] = -np.sum(H_00[i, :])
                    #H_00[i, i] = -np.sum(H_00[i, 1::2])
            return H_00

        def build_H_01_spacial():

            H_01 = np.zeros((2 * self.N_y, 2 * self.N_y), dtype=float)
            for i in range(0, H_01.shape[0]):
                #x components
                if(i%2==0):
                    #H_01[i,i+2] = -self.k_x
                    H_01[i, i] = -self.k_x
                    #H_01[i, i+1] = -self.k_xy
                    #H_01[i+1, i] = -self.k_xy

                if (i > 1):
                    #H_01[i, i - 2] = -self.k_y
                    pass
            return H_01

        #H_NN = build_H_NN()
        #H_00 = build_H_00()
        #H_01 = build_H_01()

        H_NN = build_H_NN_spacial()
        H_00 = build_H_00_spacial()
        H_01 = build_H_01_spacial()

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
            #special case just one single layer
            if(self.N_x==1):
                hessian = hessian + H_01
        #check sum rule
        delta = 1E-12
        assert np.abs(np.sum(hessian)) < delta, "Sum rule not correct"
        return hessian

    def dimension_match(self, hessian, N_y_new):
        #Attention: In this version just limited to N_x = 1
        """
        This method places the hessian in zero matrix to match the (greater) N_y of a electrode
        Args:
            hessian:
            N_y_new:

        Returns:

        """
        assert hessian.shape[0]/2<N_y_new*2, "I can only add zeros"
        hessian_new = np.zeros((N_y_new * 2 * self.N_x, N_y_new * 2 * self.N_x), dtype=float)
        lower_index = int((N_y_new - self.N_y)/2)
        hessian_new[lower_index:lower_index+hessian.shape[0], lower_index:lower_index+hessian.shape[0]] = hessian
        self.N_y = N_y_new
        return hessian_new

if __name__ == '__main__':
    #model_system = Chain1D(k=0.1, N =3)
    #hessian = model_system.hessian

    model_system = FiniteLattice2D()
    hessian = model_system.hessian
    hessian_new = model_system.dimension_match(hessian,3)
    print(hessian_new)



