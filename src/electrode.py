import codecs
import configparser
from functools import partial

import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.integrate import quad

from utils import constants

class Electrode():
    """Class for the definition of a electrode """

    def __init__(self,w, config_path):
        self.w = w
        self.dimension = -1
        self.cfg = configparser.ConfigParser()
        self.cfg.read_file(codecs.open(config_path, "r", "utf8"))

    def calcalculate_g0(self):
        return





class DebeyeModel(Electrode):

    def __init__(self, w, config_path):
        super().__init__(w, config_path)
        #self.load_model_parameters()
        #check! dimension might be other as well
        self.dimension = 3

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
        #self.w = w + 1E-26 * 1.j
        #self.load_model_parameters()
        self.dimension = 1
        self.k = 0.1*(constants.eV2hartree / constants.ang2bohr ** 2)
        self.k_c = 0.1* (constants.eV2hartree / constants.ang2bohr ** 2)
        self.g0 = self.calculate_g0(self.w, self.k)
        self.g = self.calcalculate_g(self.w)

    def calculate_g0(self, w, k):
        """Calculates surface greens of one-dimensional chain (nearest neighbor coupling) with coupling parameter k

        Args:
        w (array_like): Frequency where g0 is calculated
        k (float): NN-coupling constant

        Returns:
        g0	(array_like) Surface greens function g0
        """

        g_0 = 1/(2*k*w)*(w-np.sqrt(w**2-4*k, dtype=complex))
        """
        gamma_hb = -self.k_c
        gamma_prime = gamma_hb
        g_0 = g_0 / (1 + gamma_prime * g_0)
        """

        return g_0


    def calcalculate_g(self, w):
        g = 0.5*(w**2-2*self.k_c-w*np.sqrt(w**2-4*self.k, dtype=complex))/(self.w**2*(self.k-self.k_c)+ self.k_c**2)
        return g

    def load_model_parameters(self):
        self.k = float(self.cfg.get('Electrode', 'k'))
        assert self.k >= 0, "Due to the sign in the dyson equation, k must be negative"

    def plot_g0(self):
        fig, ax1 = plt.subplots()
        # """
        ax1.plot(w * constants.unit2SI * constants.h_bar * constants.J2meV, np.imag(self.g0)*self.k_c, color="red", label="Im(g0)")
        ax1.plot(w * constants.unit2SI * constants.h_bar * constants.J2meV, np.real(self.g0)*self.k_c, color="green", label="Re(g0)")
        #ax1.set_yscale("log")
        #ax1.set_ylim(-100, 100)
        plt.xlabel("Energy (meV)")
        plt.ylabel(r"g $(1/k_c)$")
        plt.grid()
        plt.legend()
        plt.show()
        #plt.savefig(self.data_path + "/g0_1d.pdf", bbox_inches='tight')

    def plot_g(self):
        fig, ax1 = plt.subplots()
        # """
        ax1.plot(w * constants.unit2SI * constants.h_bar * constants.J2meV, np.imag(self.g)*self.k_c, color="red", label="Im(g0)")
        ax1.plot(w * constants.unit2SI * constants.h_bar * constants.J2meV, np.real(self.g)*self.k_c, color="green", label="Re(g0)")
        #ax1.set_yscale("log")
        #ax1.set_ylim(-100, 100)
        plt.xlabel("Energy (meV)")
        plt.ylabel(r"g $(1/k_c)$")
        plt.grid()
        plt.legend()
        plt.show()
        #plt.savefig(self.data_path + "/g0_1d.pdf", bbox_inches='tight')

class Square2d(Electrode):

    def __init__(self, w, config_path):
        super().__init__(w, config_path)
        #self.w = w + 1E-32 * 1.j
        #self.load_model_parameters()
        self.k_x = 0.1 * (constants.eV2hartree / constants.ang2bohr ** 2)
        self.k_y = 0.002 * (constants.eV2hartree / constants.ang2bohr ** 2) * 1E-12
        self.dimension = 1
        self.N_q = 51101
        self.k_c = 0.1 * (constants.eV2hartree / constants.ang2bohr ** 2)
        self.g0 = self.calculate_g0(self.w)
        self.g = self.calculate_g(self.w)

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

        """
        gamma_hb = -self.k_c
        gamma_prime = gamma_hb
        g_0 = g_0 / (1 + gamma_prime * g_0)
        """

        return g_0

    def calculate_g(self, w):
        """Calculates surface greens of 2d half infinite square lattic (nearest neighbor coupling)

        Args:
        w (array_like): Frequency where g0 is calculated
        k (float): NN-coupling constant

        Returns:
        g0	(array_like) Surface greens function g0
        """


        q = np.linspace(-np.pi,np.pi, self.N_q)
        #q = np.linspace(0, 2*np.pi, N_q)
        def g0_w(w):
            #y = k_y * np.sin(q*a/2)**2
            y = self.k_y * np.sin(q) ** 2

            #python cannot square complex numbers properly
            w = np.real(w)
            g0_q = 2 * ((w**2-4*y)-np.sqrt((w**2-4*y)*(w**2-4*y-4*self.k_x), dtype = complex))/((w**2-4*y)*4*self.k_x)


            g0 = 1 / ((2 * np.pi)) * np.trapz(g0_q, q)
            return g0

        g_0 = map(g0_w, w)
        g_0_ = list()
        for item in g_0:
            g_0_.append(item)
        g_0 = np.array(g_0_)


        gamma_hb = -self.k_c
        gamma_prime = gamma_hb
        g_0 = g_0 / (1 + gamma_prime * g_0)


        return g_0

    def plot_g0(self):
        fig, ax1 = plt.subplots()
        # """
        ax1.plot(w* constants.unit2SI * constants.h_bar * constants.J2meV, np.imag(self.g0)*self.k_c, color="red", label="Im(g0)")
        ax1.plot(w* constants.unit2SI * constants.h_bar * constants.J2meV, np.real(self.g0)*self.k_c, color="green", label="Re(g0)")
        #ax1.set_yscale("log")
        #ax1.set_ylim(-5, 5)
        plt.xlabel("Energy (meV)")
        plt.ylabel(r"g $(1/k_c)$")
        plt.grid()
        plt.legend()
        plt.show()
        plt.savefig(self.data_path + "/g0_2d.pdf", bbox_inches='tight')

    def plot_g(self):
        fig, ax1 = plt.subplots()
        # """
        ax1.plot(w* constants.unit2SI * constants.h_bar * constants.J2meV, np.imag(self.g)*self.k_c, color="red", label="Im(g0)")
        ax1.plot(w* constants.unit2SI * constants.h_bar * constants.J2meV, np.real(self.g)*self.k_c, color="green", label="Re(g0)")
        #ax1.set_yscale("log")
        #ax1.set_ylim(-5, 5)
        plt.xlabel("Energy (meV)")
        plt.ylabel(r"g $(1/k_c)$")
        plt.grid()
        plt.legend()
        plt.show()
        #plt.savefig(self.data_path + "/g0_2d.pdf", bbox_inches='tight')

    def load_model_parameters(self):
        self.k = float(self.cfg.get('Electrode', 'k'))
        assert self.k >= 0, "Due to the sign in the dyson equation, k must be negative"


if __name__ == '__main__':

    N = 2000
    E_D = 50
    config_path = "../../1D_test/phonon_config"
    # convert to J
    E_D = E_D * constants.meV2J
    # convert to 1/s
    w_D = E_D / constants.h_bar
    # convert to har*s/(bohr**2*u)
    w_D = w_D / constants.unit2SI
    w = np.linspace(0, w_D * 1.1, N)

    #"""
    #w = w + 1E-3*1.j
    #w = w + 1.j * 1E-128
    electrode = Square2d(w, config_path)
    #electrode.plot_g0()
    electrode.plot_g()
    #"""

    #"""
    #w = w + 1E-12 * 1.j
    electrode = Chain1D(w, config_path)
    #electrode.plot_g0()
    electrode.plot_g()
    #"""