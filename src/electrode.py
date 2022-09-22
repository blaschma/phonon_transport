__docformat__ = "google"
import codecs
import configparser
from functools import partial

import numpy as np
import scipy
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.integrate import simps
import ray

from utils import constants

class Electrode():
    """Class for the definition of a electrode """

    def __init__(self,w, config_path, model=-1):
        self.w = w
        self.dimension = -1
        self.cfg = configparser.ConfigParser()
        self.cfg.read_file(codecs.open(config_path, "r", "utf8"))
        self.data_path = str(self.cfg.get('Data Input', 'data_path'))
        #set for different calculations
        self.model = model

    def calcalculate_g0(self):
        return

    def calculate_g(self, g_0):
        """Calculates surface greens of 2d half infinite square lattic (nearest neighbor coupling)

        Args:
        g_0 (array_like): Uncoupled surface greens function

        Returns:
        g	(array_like) Surface greens function coupled by dyson equation
        """

        gamma_hb = -self.k_c
        gamma_prime = gamma_hb
        #g = np.dot(g_0, np.linalg.inv(np.identity(g_0.shape[0]) + np.dot(gamma_prime,g_0)))
        g = g_0 / (1 + gamma_prime * g_0)


        return g

    def plot_g0(self):
        fig, ax1 = plt.subplots()
        # """
        #print((np.imag(self.g0)))
        ax1.plot(self.w * constants.unit2SI * constants.h_bar * constants.J2meV, np.imag(self.g0) * self.k_c, color="red", label="Im(g0)")
        ax1.plot(self.w * constants.unit2SI * constants.h_bar * constants.J2meV, np.real(self.g0) * self.k_c, color="green", label="Re(g0)")
        #ax1.set_yscale("log")
        #ax1.set_ylim(-5, 5)
        plt.xlabel("Energy (meV)")
        plt.ylabel(r"g $(1/k_c)$")
        plt.grid()
        plt.legend()
        plt.title(r"$g_0$")
        plt.show()
        plt.savefig(self.data_path + "/g0_from_electrode.pdf", bbox_inches='tight')

    def plot_g(self):
        fig, ax1 = plt.subplots()
        ax1.plot(self.w * constants.unit2SI * constants.h_bar * constants.J2meV, np.imag(self.g) * self.k_c, color="red", label="Im(g0)")
        ax1.plot(self.w * constants.unit2SI * constants.h_bar * constants.J2meV, np.real(self.g) * self.k_c, color="green", label="Re(g0)")
        #ax1.plot(np.imag(self.g) * self.k_c, color="red", label="Im(g0)")
        #ax1.plot(np.real(self.g) * self.k_c, color="green", label="Re(g0)")
        ax1.set_ylim(-1.5, 1.5)
        #ax1.set_ylim(-50, 50)
        plt.xlabel("Energy (meV)")
        plt.ylabel(r"g $(1/k_c)$")
        plt.grid()
        plt.legend()
        plt.title(r"$g_{\mathrm{surf}}}$")
        plt.savefig(self.data_path + "/g_from_electrode.pdf", bbox_inches='tight')
        plt.show()






class DebeyeModel(Electrode):

    def __init__(self, w, config_path, model, k_c, w_D):
        super().__init__(w, config_path, model)
        self.k_c = k_c * (constants.eV2hartree / constants.ang2bohr ** 2)
        self.w_D = w_D
        
        #self.load_model_parameters()
        #check! dimension might be other as well
        self.dimension = 1
        self.g0 = self.calculate_g0(w, w_D)
        self.g = self.calculate_g(self.g0)

    def calculate_g0(self, w, w_D):
        """Calculates surface greens function according to Markussen, T. (2013). Phonon interference effects in molecular junctions. The Journal of chemical physics, 139(24), 244101 (https://doi.org/10.1063/1.4849178).

        Args:
        w (array_like): Frequency where g0 is calculated
        w_D (float): Debeye frequency

        Returns:
        g0	(array_like) Surface greens function g0
        """

        def im_g(w):
            if (w <= w_D):
                Im_g = -np.pi * 3.0 * w / (2 * w_D ** 3)
            else:
                Im_g = 0
            return Im_g

        Im_g = map(im_g, w)
        Im_g = np.asarray(list(Im_g))
        Re_g = -np.asarray(np.imag(scipy.signal.hilbert(Im_g)))
        g0 = np.asarray((Re_g + 1.j * Im_g), complex)

        return g0

    def calculate_g(self, g_0):
        """

        Args:
        g_0 (array_like): Uncoupled surface greens function

        Returns:
        g	(array_like) Surface greens function coupled by dyson equation
        """

        gamma_hb = -self.k_c
        gamma_prime = gamma_hb
        #g = np.dot(g_0, np.linalg.inv(np.identity(g_0.shape[0]) + np.dot(gamma_prime,g_0)))
        g = g_0 / (1 + gamma_prime * g_0)


        return g

    def load_model_parameters(self):
        self.E_D = float(self.cfg.get('Electrode', 'E_D'))
        # convert to J
        E_D = self.E_D * constants.meV2J
        # convert to 1/s
        w_D = E_D / constants.h_bar
        # convert to har*s/(bohr**2*u)
        self.w_D = w_D / constants.unit2SI



class Chain1D(Electrode):

    def __init__(self, w, config_path):
        super().__init__(w, config_path)
        #self.load_model_parameters()
        self.dimension = 1
        self.k = 0.1*(constants.eV2hartree / constants.ang2bohr ** 2)
        self.k_c = 0.1* (constants.eV2hartree / constants.ang2bohr ** 2)
        self.g0 = self.calculate_g0(self.w, self.k)
        self.g = self.calculate_g(self.g0)

    def calculate_g0(self, w, k):
        """Calculates surface greens of one-dimensional chain (nearest neighbor coupling) with coupling parameter k

        Args:
        w (array_like): Frequency where g0 is calculated
        k (float): NN-coupling constant

        Returns:
        g0	(array_like) Surface greens function g0
        """

        g_0 = 1/(2*k*w)*(w-np.sqrt(w**2-4*k, dtype=complex))

        return g_0

    def load_model_parameters(self):
        self.k = float(self.cfg.get('Electrode', 'k'))
        assert self.k >= 0, "Due to the sign in the dyson equation, k must be negative"



class Square2d(Electrode):

    def __init__(self, w, config_path, model):
        super().__init__(w, config_path, model)
        #self.w = w + 1E-32 * 1.j
        #self.load_model_parameters()
        self.k_x = 0.1 * (constants.eV2hartree / constants.ang2bohr ** 2)
        self.k_y = 0.002 * (constants.eV2hartree / constants.ang2bohr ** 2)*1
        self.dimension = 1
        self.N_q = 1000
        self.k_c = 0.1 * (constants.eV2hartree / constants.ang2bohr ** 2)
        self.g0 = self.calculate_g0(self.w)
        self.g = self.calculate_g(self.g0)

    def calculate_g0(self, w):
        """Calculates surface greens of 2d half infinite square lattice (nearest neighbor coupling)

        Args:
        w (array_like): Frequency where g0 is calculated
        k (float): NN-coupling constant

        Returns:
        g0	(array_like) Surface greens function g0
        """

        k_x = self.k_x
        k_y = self.k_y
        N_q = self.N_q
        q = np.linspace(0,2.*np.pi, N_q)
        #q = np.linspace(0, 2*np.pi, N_q)
        def g0_w(w):
            y = k_y * np.sin(q) ** 2
            w = np.real(w)
            g0_q = 2 * ((w**2 - 4*y) + np.sqrt((w**2-4*y)*(w**2-4*k_x-4*y), dtype=complex))**(-1)
            #g0_q = 2 * (w**2-y-np.sqrt((w**2-y)*(w**2-y-4*self.k_x), dtype = complex))/(4*self.k_x)

            g0 = 1/((2*np.pi))*np.trapz(g0_q, q)
            return g0
        g_0 = map(g0_w, w)
        g_0_ = list()
        for item in g_0:
            g_0_.append(item)
        g_0 = np.array(g_0_)

        return g_0


    def load_model_parameters(self):
        """
        Loads model parameters from config file. Units are converted
        Returns:

        """
        model = str(self.cfg.get('Electrode', 'model'))
        assert model == "Square2d"
        self.k_x = float(self.cfg.get('Electrode', 'k_x')) * (constants.eV2hartree / constants.ang2bohr ** 2)
        self.k_y = float(self.cfg.get('Electrode', 'k_y')) * (constants.eV2hartree / constants.ang2bohr ** 2)
        self.N_qy = float(self.cfg.get('Electrode', 'N_qy'))
        self.gamma = float(self.cfg.get('Calculation', 'gamma'))

class Lattice3d(Electrode):

    def __init__(self, w, config_path):
        super().__init__(w, config_path)

        #self.load_model_parameters()
        self.k_x = 0.1 * (constants.eV2hartree / constants.ang2bohr ** 2)
        self.k_y = 0.1 * (constants.eV2hartree / constants.ang2bohr ** 2)
        self.k_y = 0.1 * (constants.eV2hartree / constants.ang2bohr ** 2)
        self.k_z = 0.1 * (constants.eV2hartree / constants.ang2bohr ** 2)
        self.dimension = 3
        self.N_q = 800
        self.k_c = 0.1 * (constants.eV2hartree / constants.ang2bohr ** 2)
        self.g0 = self.calculate_g0(self.w)
        self.g = self.calculate_g(self.g0)

    def calculate_g0(self, w):
        """Calculates surface greens of 3d half infinite square lattice (nearest neighbor coupling)

        Args:
        w (array_like): Frequency where g0 is calculated
        k (float): NN-coupling constant

        Returns:
        g0	(array_like) Surface greens function g0
        """

        k_x = self.k_x
        k_y = self.k_y
        k_z = self.k_z
        N_q = self.N_q
        q_y = np.linspace(0, 2. * np.pi, N_q)
        q_z = np.linspace(0, 2 * np.pi, N_q)
        ray.init()

        #@ray.remote
        def g0_w(w):
            def integrand(w, q_y, q_z):
                y = k_y * np.sin(q_y) ** 2 + k_z * np.sin(q_z) ** 2
                w = np.real(w)
                g0_q = 2 * ((w ** 2 - 4 * y) + np.sqrt((w ** 2 - 4 * y) * (w ** 2 - 4 * k_x - 4 * y),
                                                       dtype=complex)) ** (-1)
                return g0_q

            tmp = integrand(w, q_y.reshape(-1, 1), q_z.reshape(1, -1))
            g0_ = 1 / ((2 * np.pi) ** 2) * simps([simps(zz_y, q_y) for zz_y in tmp], q_z)

            return g0_
        """
        def worker(f, list):
            return [f.remote(x) for x in list]

        keine_ahnung = worker(g0_w, w)
        results = ray.get(keine_ahnung)
        """
        #"""
        g_0 = map(g0_w, w)

        g_0 = map(g0_w, w)
        g_0_ = list()
        for item in g_0:
            g_0_.append(item)
        g_0 = np.array(g_0_)

        return g_0


    def load_model_parameters(self):
        """
        Loads model parameters from config file. Units are converted
        Returns:

        """

        model = str(self.cfg.get('Electrode', 'model'))
        assert model == "Lattice3d"
        self.k_x = float(self.cfg.get('Electrode', 'k_x'))* (constants.eV2hartree / constants.ang2bohr ** 2)
        self.k_y = float(self.cfg.get('Electrode', 'k_y'))* (constants.eV2hartree / constants.ang2bohr ** 2)
        self.k_z = float(self.cfg.get('Electrode', 'k_z'))* (constants.eV2hartree / constants.ang2bohr ** 2)
        self.N_qy = float(self.cfg.get('Electrode', 'N_qy'))
        self.N_qz = float(self.cfg.get('Electrode', 'N_qy'))
        self.k_c = float(self.cfg.get('Calculation', 'gamma'))

class Ribbon2D(Electrode):

    def __init__(self, w, config_path, model, N_y, k_x, k_y, k_xy, k_c):
        super().__init__(w, config_path, model)
        self.N_y = N_y
        self.k_x = k_x * (constants.eV2hartree / constants.ang2bohr ** 2) * 1
        self.k_y = k_y * (constants.eV2hartree / constants.ang2bohr ** 2) * 1
        self.k_xy = k_xy * (constants.eV2hartree / constants.ang2bohr ** 2) * 0
        self.k_c = k_c * (constants.eV2hartree / constants.ang2bohr ** 2)
        self.eps = 1E-50
        self.g0 = self.calculate_g0(w)
        self.g = self.calculate_g(self.g0)

    def calculate_g0(self, w):
        """Calculates surface greens 2d half infinite square lattice with finite width N_y. The greens function is
        calculated using the procedure from Guinea, F., et al. "Effective two-dimensional Hamiltonian at surfaces."
        Physical Review B 28.8 (1983): 4397.

        Args:
        w (array_like): Frequency where g0 is calculated

        Returns:
        g0	(array_like) Surface greens function g0
        """

        def build_H_NN():
            H_NN = np.zeros((self.N_y,self.N_y), dtype=float)
            for i in range(0, self.N_y):
                # offdiagonals
                if (i < self.N_y - 1):
                    H_NN[i, i + 1] = -self.k_y
                if (i > 0):
                    H_NN[i, i - 1] = -self.k_y
                # diagonals
                H_NN[i, i] = -np.sum(H_NN[i, :]) + 2*self.k_x
            return  H_NN
        def build_H_00():
            H_00 = build_H_NN()
            H_00[0,0] -= self.k_x
            if (self.N_y > 1 ):
                H_00[self.N_y-1, self.N_y-1] -= self.k_x
            return H_00
        def build_H_01():
            H_01 = np.identity(self.N_y) * (-self.k_x)
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

        H_01_dagger = np.transpose(np.conj(H_01))

        assert np.sum(H_00+H_01)==0 and np.sum(H_NN+2*H_01)==0, "sum rule cannot be fullfilled"

        def calc_g0_w(w):
            w = np.identity(H_NN.shape[0]) * (w **2 +(1.j*1E-24))
            g = np.linalg.inv(w-H_NN)
            alpha_i = np.dot(np.dot(H_01, g), H_01)
            beta_i = np.dot(np.dot(H_01_dagger, g), H_01_dagger)
            epsilon_is = H_00 + np.dot(np.dot(H_01,g),H_01_dagger)
            epsilon_i = H_NN + np.dot(np.dot(H_01,g),H_01_dagger) + np.dot(np.dot(H_01_dagger,g),H_01)
            delta = np.abs(2*np.trace(alpha_i))
            deltas = list()
            deltas.append(delta)
            counter = 0
            terminated = False
            while(delta>self.eps):
                counter +=1
                if(counter > 10000):
                    terminated = True
                    break
                g = np.linalg.inv(w-epsilon_i)
                epsilon_i = epsilon_i + np.dot(np.dot(alpha_i, g), beta_i) + np.dot(np.dot(beta_i, g), alpha_i)
                epsilon_is = epsilon_is + np.dot(np.dot(alpha_i, g), beta_i)
                alpha_i = np.dot(np.dot(alpha_i, g), alpha_i)
                beta_i = np.dot(np.dot(beta_i, g), beta_i)
                delta = np.abs(2*np.trace(alpha_i))
                deltas.append(delta)
            if(delta>=self.eps or terminated):
                print("warning")

            g_0 = np.linalg.inv(w-epsilon_is)
            #return np.trace(g_0[int(H_NN.shape[0]/2):int(H_NN.shape[0]/2)+2,int(H_NN.shape[0]/2):int(H_NN.shape[0]/2)+2])
            #return (g_0[int(H_NN.shape[0] / 2)-1, int(H_NN.shape[0] / 2)-1])
            #return (g_0[0, 0])
            return g_0

        g_0 = map(calc_g0_w, w)
        g_0_ = list()
        for item in g_0:
            g_0_.append(item)

        g_0 = np.array(g_0_)
        return  g_0
        #"""

    def calculate_g(self, g_0):
        """Calculates surface greens of 2d half infinite square lattice (nearest neighbor coupling)

        Args:
        g_0 (array_like): Uncoupled surface greens function

        Returns:
        g	(array_like) Surface greens function coupled by dyson equation
        """

        if(self.model==1):
            # This is for point contact only x connected
            gamma_hb = np.zeros(g_0[0].shape)
            gamma_hb[int(gamma_hb.shape[0] / 2) - 1, int(gamma_hb.shape[0] / 2) - 1] = -self.k_c
        if (self.model == 3):
            # This is for point contact x&y connected
            gamma_hb = np.zeros(g_0[0].shape)
            gamma_hb[int(gamma_hb.shape[0] / 2) - 1, int(gamma_hb.shape[0] / 2) - 1] = -self.k_c
            gamma_hb[int(gamma_hb.shape[0] / 2), int(gamma_hb.shape[0] / 2)] = -self.k_c
        if (self.model == 5):
            gamma_hb = np.zeros(g_0[0].shape)
            #couple only x
            for u in range(0,gamma_hb.shape[0]):
                if(u%2==0):
                    gamma_hb[u, u] = -self.k_c



        #This is for 2d contact only x components connected
        """
        gamma_hb = -self.k_c * np.identity(g_0[0].shape[0])
        for i in range(0, gamma_hb.shape[0]):
            if(i%2==1):
                gamma_hb[i,i] = 0
                pass
        """
        # This is for 2d contact x and y components connected
        """
        gamma_hb = -self.k_c * np.identity(g_0[0].shape[0])
        """
        #this is for 2d contact x and y connected but dimension mismatch (just for electrode N_y=5 contact N_y=3)
        """
        gamma_hb = -self.k_c * np.identity(g_0[0].shape[0])
        gamma_hb[0,0] = 0
        gamma_hb[1, 1] = 0
        gamma_hb[gamma_hb.shape[0]-1, gamma_hb.shape[0]-1] = 0
        gamma_hb[gamma_hb.shape[0] -2, gamma_hb.shape[0] - 2] = 0
        """



        # This is for point contact x and y connected
        """
        gamma_hb = np.zeros(g_0[0].shape)
        gamma_hb[int(gamma_hb.shape[0] / 2) - 1, int(gamma_hb.shape[0] / 2) - 1] = -self.k_c
        gamma_hb[int(gamma_hb.shape[0] / 2) , int(gamma_hb.shape[0] / 2) ] = -self.k_c

        """

        def worker(g_0):
            gamma_prime = gamma_hb
            #dyson equation
            g = np.dot(g_0, np.linalg.inv(np.identity(g_0.shape[0]) + np.dot(gamma_prime , g_0)))

            #return g[1,1]
            if(self.model == 1):
                return g[int(g_0.shape[0] / 2)-1, int(g_0.shape[0] / 2)-1]
            elif(self.model == 3):
                return g[int(g_0.shape[0] / 2)-1:int(g_0.shape[0] / 2)+1, int(g_0.shape[0] / 2)-1:int(g_0.shape[0] / 2)+1]
            #return g[int(g_0.shape[0] / 2), int(g_0.shape[0] / 2)]
            #return np.trace(g_0[int(g_0.shape[0] / 2):int(g_0.shape[0] / 2) + 2,int(g_0.shape[0] / 2):int(g_0.shape[0] / 2) + 2])
            elif(self.model == 5):
                return g
            return g

        g = map(worker, g_0)
        g_ = list()
        for item in g:
            g_.append(item)
        g = np.array(g_)
        return g

if __name__ == '__main__':

    N = 750
    E_D = 50
    config_path = "../../1D_test/phonon_config"
    # convert to J
    E_D = E_D * constants.meV2J
    # convert to 1/s
    w_D = E_D / constants.h_bar
    # convert to har*s/(bohr**2*u)
    w_D = w_D / constants.unit2SI
    w = np.linspace(w_D*1E-12, w_D * 1.1, N)

    """
    electrode = Chain1D(w, config_path)
    electrode.plot_g0()
    electrode.plot_g()
    """

    """
    electrode = Square2d(w, config_path)
    electrode.plot_g0()
    electrode.plot_g()
    """

    """
    electrode = Lattice3d(w, config_path)
    electrode.plot_g0()
    electrode.plot_g()
    """

    electrode = Ribbon2D(w, config_path)
    #electrode.plot_g0()
    electrode.plot_g()