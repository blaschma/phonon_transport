'''
	File name: phonon_transport.py
	Author: Matthias Blaschke
	Python Version: 3.9
'''
import codecs
import copy
import os.path
import sys
import json

import numpy as np
import matplotlib
from model_systems import Chain1D

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from turbomoleOutputProcessing import turbomoleOutputProcessing as top
import scipy.signal
import configparser
from optparse import OptionParser
import calculate_kappa as ck
from utils import eigenchannel_utils as eu
import electrode as el
from utils import constants


# h_bar in Js
h_bar = 1.0545718 * 10 ** (-34)
eV2hartree = 0.0367493
ang2bohr = 1.88973
har2J = 4.35974E-18
bohr2m = 5.29177E-11
u2kg = 1.66054E-27
har2pJ = 4.35974e-6
J2meV = 6.24150934190e+21
meV2J = 1.60217656535E-22
#This unit is calulated from the greens function: w**2 and D have to have the same unit
unit2SI = np.sqrt(9.375821464623672e+29)
#unit2SI = 9.375821464623672e+29
#s2au = 2.4188843265857E-17
#unit2SI = 1/s2au



class PhononTransport:
	"""Class for phonon transport calculations

	This class can be used for phonon transport calculations. It follows the


	"""

	def __init__(self, data_path, coord_path, n_l, n_r, gamma, E_D, M_L, M_C, N, T_min, T_max, kappa_grid_points, in_plane, eigenchannel, every_nth,
				 channel_max):
		self.data_path = data_path
		self.coord_path = coord_path
		self.n_l = n_l
		self.n_r = n_r
		#pfusch
		N_chain = 1
		self.n_r =[N_chain-1]
		self.gamma = gamma
		self.in_plane = in_plane
		self.eigenchannel = eigenchannel
		self.every_nth = every_nth
		self.channel_max = channel_max
		self.M_L = M_L
		self.M_C = M_C
		#pfusch
		self.M_L = "test"
		self.M_C = "test"
		self.N = N
		self.E_D = E_D
		# convert to J
		E_D = E_D * meV2J
		# convert to 1/s
		w_D = E_D / h_bar
		# convert to har*s/(bohr**2*u)
		self.w_D = w_D / unit2SI
		self.temperature = np.linspace(T_min, T_max, kappa_grid_points)

		# pfusch!!
		self.dimension = 1



		self.w = np.linspace(0, self.w_D * 1.1, N)
		self.w = self.w + 1.j*1E-12
		self.E = self.w * unit2SI * h_bar * J2meV
		self.i = np.linspace(0, self.N, self.N, False, dtype=int)
		print("setting up electrode")
		#self.electrode = el.Chain1D(self.w, sys.argv[1])
		#self.g0 = self.electrode.g0

		#self.electrode = el.Square2d(self.w, sys.argv[1])
		#self.g0 = self.electrode.g0

		#self.electrode = el.Lattice3d(self.w, sys.argv[1])
		#self.g0 = self.electrode.g0

		self.electrode = el.Ribbon2D(self.w, sys.argv[1])
		self.g0 = self.electrode.g0
		self.g = self.electrode.g
		self.electrode.plot_g()

		print("electrode done")


		self.Sigma = self.calculate_Sigma(self.w, self.g0, gamma, self.M_L, self.M_C)
		# set up dynamical matrix K
		#self.D = top.create_dynamical_matrix(filename_hessian, self.coord_path, t2SI=False, dimensions=self.dimension)
		self.D = Chain1D(0.1*(constants.eV2hartree / constants.ang2bohr ** 2), N_chain).hessian
		self.coord = top.read_coord_file(self.coord_path)

		self._G_cc = np.ones((N,self.D.shape[0], self.D.shape[1]), dtype=complex)
		self._trans_prob_matrix = np.ones((N, self.D.shape[0], self.D.shape[1]), dtype=complex)
		self._T = np.ones(N)*-1
		self._kappa = np.ones(N) * -1
		self.T_channel_vals = np.ones((N,self.channel_max), dtype=complex)


	@property
	def G_cc(self):
		if np.all(self._G_cc == 1):
			raise ValueError("Calculate G_cc first!")
		else:
			return self._G_cc

	@G_cc.setter
	def G_cc(self, G_cc_):
		self._G_cc = G_cc_

	@property
	def kappa(self):
		if np.all(self._kappa == -1):
			raise ValueError("Calculate kappa first!")
		else:
			return self._kappa

	@kappa.setter
	def kappa(self, kappa):
		self._kappa = kappa

	@property
	def T(self):
		if np.all(self._T == -1):
			raise ValueError("Calculate T first!")
		else:
			return self._T

	@T.setter
	def T(self, T):
		self._T = T

	@property
	def trans_prob_matrix(self):
		if np.all(self._trans_prob_matrix == -1):
			raise ValueError("Calculate trans_prob_matrix first!")
		else:
			return self._T

	@trans_prob_matrix.setter
	def trans_prob_matrix(self, trans_prob_matrix):
		self._trans_prob_matrix = trans_prob_matrix


	def calculate_Sigma(self, w, g0, gamma, M_L, M_C):
		"""Calculates self energy according to Markussen, T. (2013). Phonon interference effects in molecular junctions. The Journal of chemical physics, 139(24), 244101  (https://doi.org/10.1063/1.4849178).

		Args:
		w (np.array): frequency
		g0 (np.array): g0
		gamma (float): gamma
		M_L (str): M_L atom type in reservoir
		M_C (str): M_C atom type coupled to reservoir

		Returns:
		sigma_nu (array_like) self energy term
		"""

		# convert to hartree/Bohr**2
		gamma_hb = gamma * (eV2hartree / ang2bohr ** 2)

		M_L = top.atom_weight(M_L, u2kg=False)
		M_C = top.atom_weight(M_C, u2kg=False)

		gamma_prime = gamma_hb / np.sqrt(M_C * M_L)

		g = self.g
		print("transport", g)
		#self.g = g

		fig, ax1 = plt.subplots()
		ax1.plot(self.E, np.imag(g)*gamma_hb, color="red", label="Im(g0)")
		ax1.plot(self.E, np.real(g)*gamma_hb, color="green", label="Re(g0)")
		#ax1.set_ylim(-1E1,10)
		plt.grid()
		plt.legend()
		plt.show()
		plt.savefig(self.data_path + "/g_produced.pdf", bbox_inches='tight')
		#"""
		sigma_nu = gamma_prime ** 2 * g

		return sigma_nu

	def calculate_G_cc(self):

		# calculate Greens function for central part
		G_cc_ = map(self.calculate_G_cc_i, self.i)
		for j, item in enumerate(G_cc_):
			self._G_cc[j] = item

	def calculate_G_cc_i(self, i):
		"""Calculates Greens Function with given parameters at given frequency w.

		Args:
			i: (int): frequency index
			para: (tuple): frequency w (array), self energy sigma (complex), filename_hessian (str), filename_coord (str), left atom for transport calculation n_l (int), right atom for transport calculation n_r (int), coupling constant Gamma (complex), in_plane (boolean)

		Returns:
			T (array_like): phonon transmission
		"""

		w = self.w
		sigma = self.Sigma
		n_l = self.n_l
		n_r = self.n_r
		gamma = self.gamma
		in_plane = self.in_plane
		D = self.D
		D = copy.copy(D)

		n_atoms = int(D.shape[0] / self.dimension)

		# set up self energies
		sigma_L = np.zeros((n_atoms * self.dimension, n_atoms * self.dimension), complex)
		sigma_R = np.zeros((n_atoms * self.dimension, n_atoms * self.dimension), complex)
		if (in_plane == True):
			lower = 2
		else:
			lower = 0

		for n_l_ in n_l:
			for u in range(lower, self.dimension):
				sigma_L[n_l_ * self.dimension + u, n_l_ * self.dimension + u] = sigma[i]
		for n_r_ in n_r:
			for u in range(lower, self.dimension):
				sigma_R[n_r_ * self.dimension + u, n_r_ * self.dimension + u] = sigma[i]


		# correct momentum conservation
		# convert to hartree/Bohr**2
		gamma_hb = gamma * eV2hartree / ang2bohr ** 2

		for u in range(lower, self.dimension):
			for n_l_ in n_l:
				# remove mass weighting
				K_ = D[n_l_ * self.dimension + u][n_l_ * self.dimension + u] * top.atom_weight(self.M_C)
				# correct momentum
				K_ = K_ - gamma_hb
				# add mass weighting again
				D_ = K_ / top.atom_weight(self.M_C)
				D[n_l_ * self.dimension + u][n_l_ * self.dimension + u] = D_

			for n_r_ in n_r:
				# remove mass weighting
				K_ = D[n_r_ * self.dimension + u][n_r_ * self.dimension + u] * top.atom_weight(self.M_C)
				# correct momentum
				K_ = K_ - gamma_hb
				# add mass weighting again
				D_ = K_ / top.atom_weight(self.M_C)
				D[n_r_ * self.dimension + u][n_r_ * self.dimension + u] = D_

		# calculate greens function
		G = np.linalg.inv(self.w[i] ** 2 * np.identity(self.dimension * n_atoms) - D - sigma_L - sigma_R)
		return G

	def calculate_T(self):
		# calculate Transmission
		T_ = map(self.calculate_T_i, self.i)
		if(self.eigenchannel == False):
			for j, item in enumerate(T_):
				self._T[j] = float(item)
		else:
			for j, (item1, item2) in enumerate(T_):
				self._T[j] = float(item1)
				self.T_channel_vals[j,:] = np.array([float(v) for v in item2])



	def calculate_T_i(self, i):
		data_path = self.data_path
		coord = self.coord
		eigenchannel = self.eigenchannel
		every_nth = self.every_nth
		channel_max = self.channel_max

		trans_prob_matrix = self.calc_trans_prob_matrix_i(i)

		if (eigenchannel == True):
			write_out = False
			energy = -1
			if (i % every_nth == 0 and every_nth != -1):
				write_out = True

			T, T_channel = self.calc_eigenchannel_i(i, write_out)
			return T, T_channel
		else:
			T = np.real(np.trace(trans_prob_matrix))
			return T

	def calc_eigenchannel(self, trans_prob_matrix, calc_path, channel_max, coord, write_out, energy):
		"""
		Calculates phonon transmission eigenchannels according to Klöckner, J. C., Cuevas, J. C., & Pauly, F. (2018). Transmission eigenchannels for coherent phonon transport. Physical Review B, 97(15), 155432 (https://doi.org/10.1103/PhysRevB.97.155432)
		Args:
			trans_prob_matrix (np.ndarray): Transmission prob matrix (eq 25 in ref)
			calc_path (String): path to calculation
			channel_max (int): number of stored eigenvaues
			coord (array): coord file loaded with top
			write_out (bool): write channel information
			energy (float): Phonon energy in meV (for filename)

		Returns: T, T_vals: Total transmission, Contribution of each channel (up to channel_max)

		"""

		eigenvalues, eigenvectors = np.linalg.eigh(trans_prob_matrix + 0*np.ones(trans_prob_matrix.shape) * (1.j * 1E-15))
		# sort eigenvalues and eigenvecors
		idx = eigenvalues.argsort()[::-1]
		eigenvalues = eigenvalues[idx]
		eigenvectors = eigenvectors[:, idx]

		def calc_displacement(z_value):
			# z_value = eigenvectors[i, j]
			z_abs = np.abs(z_value)
			if (z_abs) > 0:
				phase = np.arccos(np.real(z_value) / z_abs)
				if (np.imag(z_value) <= 0):
					phase = 2.0 * np.pi - phase
			else:
				phase = 0
			# real part
			displacement = z_abs * np.cos(phase)

			return displacement

		# calc displacement
		calc_func = np.vectorize(calc_displacement)
		displacement_matrix = calc_func(eigenvectors)
		if (write_out == True):
			if (os.path.exists(calc_path + "/eigenchannels") == False):
				os.mkdir(f"{calc_path}/eigenchannels")
			eu.write_nmd_file(f"{calc_path}/eigenchannels/eigenchannel_{energy}.nmd", coord, displacement_matrix,
							  channel_max)

		# calculate Transmission
		T = np.sum(eigenvalues)
		return T, eigenvalues[0:channel_max]

	def calc_trans_prob_matrix_i(self, i):
		"""
		Calculates transmission prob matrix at w[i].
		Args:
			i (int): index in frequency array

		Returns: trans_prob_matrix

		"""

		sigma = self.Sigma
		n_l = self.n_l
		n_r = self.n_r
		in_plane = self.in_plane
		D = self.D
		D = copy.copy(D)

		n_atoms = int(D.shape[0] / self.dimension)

		# set up self energies
		sigma_L = np.zeros((n_atoms * self.dimension, n_atoms * self.dimension), complex)
		sigma_R = np.zeros((n_atoms * self.dimension, n_atoms * self.dimension), complex)
		if (in_plane == True):
			lower = 2
		else:
			lower = 0

		for n_l_ in n_l:
			for u in range(lower, self.dimension):
				sigma_L[n_l_ * self.dimension + u, n_l_ * self.dimension + u] = sigma[i]
		for n_r_ in n_r:
			for u in range(lower, self.dimension):
				sigma_R[n_r_ * self.dimension + u, n_r_ * self.dimension + u] = sigma[i]

		Gamma_L = -2 * np.imag(sigma_L)
		Gamma_R = -2 * np.imag(sigma_R)
		#trans_prob_matrix = np.dot(np.dot(Gamma_L, self.G_cc[i]), np.dot(Gamma_R, np.conj(np.transpose(self.G_cc[i]))))
		trans_prob_matrix = np.dot(np.dot(self.G_cc[i], Gamma_L), np.dot(np.conj(np.transpose(self.G_cc[i])), Gamma_R, ))
		return trans_prob_matrix

	def calc_eigenchannel(self, E):
		"""
		Calculates Tranmssion Eigenchannel at given Energy E
		Args:
			E (float): Energy in meV

		Returns:

		"""
		#convert
		w = E / (unit2SI * h_bar / (meV2J))
		#find index
		index = np.argmin(np.abs(w-self.w))
		#prepare and calculate
		self._G_cc[index] = self.calculate_G_cc_i(index)
		self.calc_eigenchannel_i(index, write_out=True)


	def calc_eigenchannel_i(self, i, write_out):
		"""
		Calculates phonon transmission eigenchannels at w[i] according to Klöckner, J. C., Cuevas, J. C., & Pauly, F. (2018). Transmission eigenchannels for coherent phonon transport. Physical Review B, 97(15), 155432 (https://doi.org/10.1103/PhysRevB.97.155432)
		Args:
			trans_prob_matrix (np.ndarray): Transmission prob matrix (eq 25 in ref)
			calc_path (String): path to calculation
			channel_max (int): number of stored eigenvaues
			coord (array): coord file loaded with top
			write_out (bool): write channel information
			energy (float): Phonon energy in meV (for filename)

		Returns: T, T_vals: Total transmission, Contribution of each channel (up to channel_max)

		"""
		energy = np.round(self.w[i] * unit2SI * h_bar / (meV2J), 3)
		trans_prob_matrix = self.calc_trans_prob_matrix_i(i)
		eigenvalues, eigenvectors = np.linalg.eigh(trans_prob_matrix + 0*np.ones(trans_prob_matrix.shape) * (1.j * 1E-25))
		# sort eigenvalues and eigenvecors
		idx = eigenvalues.argsort()[::-1]
		eigenvalues = eigenvalues[idx]
		eigenvectors = eigenvectors[:, idx]

		def calc_displacement(z_value):
			# z_value = eigenvectors[i, j]
			z_abs = np.abs(z_value)
			if (z_abs) > 0:
				phase = np.arccos(np.real(z_value) / z_abs)
				if (np.imag(z_value) <= 0):
					phase = 2.0 * np.pi - phase
			else:
				phase = 0
			# real part
			displacement = z_abs * np.cos(phase)

			return displacement

		# calc displacement
		calc_func = np.vectorize(calc_displacement)
		displacement_matrix = calc_func(eigenvectors)
		if (write_out == True):
			if (os.path.exists(self.data_path + "/eigenchannels") == False):
				os.mkdir(f"{self.data_path}/eigenchannels")
			eu.write_nmd_file(f"{self.data_path}/eigenchannels/eigenchannel_{energy}.nmd", self.coord, displacement_matrix,
							  channel_max, self.dimension)

		# calculate Transmission
		T = np.sum(eigenvalues)
		eigenvalues[0] = eigenvalues[0]+eigenvalues[-1]
		return T, np.asarray(eigenvalues[0:channel_max], dtype=float)

	def calc_kappa(self):
		kappa = list()
		# w to SI
		w_kappa = self.w * unit2SI
		E = h_bar * w_kappa
		# joule to hartree
		E = E / har2J
		for j in range(0, len(self.temperature)):
			kappa.append(ck.calculate_kappa(self.T[1:len(self.T)], E[1:len(E)], self.temperature[j]) * har2pJ)
		self._kappa = kappa

	def plot_eigenchannels(self):

		# top.write_plot_data(data_path + "/transmission_channels.dat", (T, T_val_tuple), "T (K), T_c")
		fig, ax = plt.subplots()
		for i in range(self.T_channel_vals.shape[1]):
			ax.plot(self.E, self.T_channel_vals[:, i], label=i + 1)
		ax.set_yscale('log')
		ax.set_xlabel('Phonon Energy ($\mathrm{meV}$)', fontsize=12)
		ax.set_ylabel(r'Transmission $\tau_{\mathrm{ph}}$', fontsize=12)
		ax.axvline(self.w_D * unit2SI * h_bar / (meV2J), ls="--", color="black")
		ax.axhline(1, ls="--", color="black")
		ax.set_ylim(1E-4, 2)
		plt.rc('xtick', labelsize=12)
		plt.rc('ytick', labelsize=12)
		plt.legend(fontsize=12)
		plt.savefig(self.data_path + "/transport_channels.pdf", bbox_inches='tight')

	def plot_transport(self):
		print(self.T)
		fig, (ax1, ax2) = plt.subplots(2, 1)
		fig.tight_layout()
		ax1.plot(self.E, self.T)
		#ax1.set_yscale('log')
		ax1.set_xlabel('Phonon Energy ($\mathrm{meV}$)', fontsize=12)
		ax1.set_ylabel(r'Transmission $\tau_{\mathrm{ph}}$', fontsize=12)
		ax1.axvline(self.w_D * unit2SI * h_bar / (meV2J), ls="--", color="black")
		ax1.axhline(1, ls="--", color="black")
		ax1.set_ylim(0, 1.5)
		ax1.set_xlim(0, self.E_D)
		ax1.grid()
		#""""
		#analytic solution
		k_c  = -self.gamma * (eV2hartree / ang2bohr ** 2)
		transp = 4*k_c**4*np.imag(self.g)**2/((self.w**2-2*k_c-2*k_c**2*np.real(self.g))**2 + 4*k_c**4*np.imag(self.g)**2)
		ax1.plot(self.E, transp, lw=4, alpha = 0.5)
		#"""

		ax2.plot(self.temperature, self.kappa)
		ax2.set_xlabel('Temperature ($K$)', fontsize=12)
		ax2.set_ylabel(r'Thermal Conductance $\mathrm{pw/K}$', fontsize=12)
		plt.rc('xtick', labelsize=12)
		plt.rc('ytick', labelsize=12)
		plt.savefig(data_path + "/transport.pdf", bbox_inches='tight')
		plt.clf()


	def tranport_calc(self):
		self.calculate_G_cc()
		self.calculate_T()
		self.calc_kappa()
		self.plot_transport()
		print(self.eigenchannel)
		if(self.eigenchannel==True):
			self.plot_eigenchannels()
			data = list([self.w])
			for j in range(0,self.channel_max):
				data.append(self.T_channel_vals[:,j])
			top.write_plot_data(self.data_path + "/phonon_trans_channel.dat", data, "T (K), kappa (pW/K)")

		top.write_plot_data(self.data_path + "/phonon_trans.dat", (self.w, self.T), "w (sqrt(har/(bohr**2*u))), T_vals")
		top.write_plot_data(self.data_path + "/kappa.dat", (self.temperature, self.kappa), "T (K), kappa (pW/K)")




if __name__ == '__main__':

	parser = OptionParser()
	parser.add_option("--eigenchannel", dest="eigenchannel_calc", action="store_true", help="Calculation of Eigenchannel at given energy")
	(opts, args) = parser.parse_args()
	print(opts, args)

	config_path = sys.argv[1]
	cfg = configparser.ConfigParser()
	cfg.read_file(codecs.open(config_path, "r", "utf8"))

	try:
		data_path = str(cfg.get('Data Input', 'data_path'))
		hessian_name = str(cfg.get('Data Input', 'hessian_name'))
		coord_name = str(cfg.get('Data Input', 'coord_name'))
		filename_hessian = data_path + "/" + hessian_name
		filename_coord = data_path + "/" + coord_name

		# atoms which are coupled to the electrodes -> self energy
		n_l = np.asarray(str(cfg.get('Calculation', 'n_l')).split(','), dtype=int)
		n_r = np.asarray(str(cfg.get('Calculation', 'n_r')).split(','), dtype=int)

		# atom type in resevoir M_L and molecule M_C
		M_L = str(cfg.get('Calculation', 'M_L'))
		M_C = str(cfg.get('Calculation', 'M_C'))
		# coupling force constant resevoir in eV/Ang**2
		gamma = float(cfg.get('Calculation', 'gamma'))

		# Debeye energy in meV
		E_D = float(cfg.get('Calculation', 'E_D'))
		# Number of grid points
		N = int(cfg.get('Calculation', 'N'))
		# only in plane motion (-> set x and y coupling to zero)
		in_plane = json.loads(str(cfg.get('Calculation', 'in_plane')).lower())

		# for thermal conducatance
		T_min = float(cfg.get('Calculation', 'T_min'))
		T_max = float(cfg.get('Calculation', 'T_max'))
		kappa_grid_points = int(cfg.get('Calculation', 'kappa_grid_points'))

		# check if eigenchannel should be calculated. But first check if section exists -> backward compatibility
		if (cfg.has_option('Eigenchannel', 'eigenchannel')):
			eigenchannel = json.loads(str(cfg.get('Eigenchannel', 'eigenchannel')).lower())
			every_nth = int(cfg.get('Eigenchannel', 'every_nth'))
			channel_max = int(cfg.get('Eigenchannel', 'channel_max'))
			#TODO Handle input -1
		else:
			eigenchannel = False
			every_nth = None

		# check if g0 should be plotted
		plot_g0 = json.loads(str(cfg.get('Data Output', 'plot_g')).lower())




	except configparser.NoOptionError as e:
		print(f"Missing option in config file. Check config file! {e}")
		exit(-1)
	except ValueError:
		print("Wrong value in config file. Check config file!")
		exit(-1)



	PT = PhononTransport(data_path, filename_coord, n_l, n_r, gamma,E_D, M_L, M_C, N, T_min, T_max, kappa_grid_points, in_plane, eigenchannel, every_nth,
	channel_max)

	if(opts.eigenchannel_calc == True):
		#TODO Proper assignment
		PT.calc_eigenchannel(float(args[1]))

	else:
		PT.tranport_calc()



	#top.write_plot_data(data_path + "/phonon_trans.dat", (w, T_vals), "w (sqrt(har/(bohr**2*u))), P_vals")
	#top.write_plot_data(data_path + "/kappa.dat", (T, kappa), "T (K), kappa (pW/K)")



