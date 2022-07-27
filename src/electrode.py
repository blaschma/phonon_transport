import codecs
import configparser

import numpy as np
import scipy
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
        #self.load_model_parameters()
        self.dimension = 1
        self.k = 0.1*(constants.eV2hartree / constants.ang2bohr ** 2)
        self.g0 = self.calculate_g0(self.w, self.k)

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

