import numpy as np
import pywt
from pywt import wavedec2
from pyrtools.pyramids import SteerablePyramidSpace as SPyr
import cv2
 
import os
import sys
sys.path.append("/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity")
from functions.img_tools import moments, im2col


class Compute_VIF():
	def __init__(self,
		wavelet="steerable",
		num_levels=4,
		M=3,
		sigma_n_squared=0.1
	) -> None:
		"""
		References:
		- H. R. Sheikh and A. C. Bovik, "Image information and visual quality," in IEEE Transactions on Image Processing, vol. 15, no. 2, pp. 430-444, Feb. 2006, doi: 10.1109/TIP.2005.859378.
		- https://github.com/abhinaukumar/vif

		Args:
			wavelet (str): Wavelet Domain to which images are transformed to.
			num_levels (int): No.of levels of steerable-pyramidal/wavelet decomposation of images.
			M (int): Size of non-overlapping block of coefficients.
			sigma_n_squared (float): Values of variance of noise considered for HVS model.
		"""
		assert wavelet=="steerable" or wavelet in pywt.wavelist(kind='discrete'), "Invalid Choice of Wavelet Domain"
		
		self.wavelet = wavelet
		self.num_levels = num_levels
		self.M = M
		self.sigma_n_squared=sigma_n_squared


	def GSM_Model(self, pyr, subband_keys):
		M = self.M
		tolerance = 1e-15

		S_squared_all = []
		EigenValues_all = []

		for subband_key in subband_keys:
			y = pyr[subband_key]
			y_size = (int(y.shape[0]/M)*M, int(y.shape[1]/M)*M)
			y = y[:y_size[0], :y_size[1]]
			
			# Estimation of Covariance Matrix C_{U}
			"""
			References: 
			- Scale mixtures of gaussians and the statistics of natural images
			- Image Information and Visual Quality
			"""
			y_vectors = im2col(img=y, k=M, stride=1)
			Covariance = np.cov(y_vectors)
			eigen_values, V = np.linalg.eigh(Covariance)
			eigen_values[eigen_values < tolerance] = tolerance
			Covariance = V@np.diag(eigen_values)@V.T

			# C from the equation C = S.U
			C = im2col(img=y, k=M, stride=M)
			s_square = np.linalg.inv(Covariance)@C
			s_square = np.sum(s_square * C, 0)/(M*M)
			s_square = s_square.reshape((int(y_size[0]/M), int(y_size[1]/M)))
			
			S_squared_all.append(s_square)
			EigenValues_all.append(eigen_values)

		return S_squared_all, EigenValues_all


	def Channel_Estimation(self, pyr_ref, pyr_dist, subband_keys):
		M = self.M
		tolerance = 1e-15
		G_all = []
		Sigma_v_squared_all = []

		for i, subband_key in enumerate(subband_keys):
			y_ref = pyr_ref[subband_key]
			y_dist = pyr_dist[subband_key]

			level = int(np.ceil((i+1)/2))
			B = 2**level + 1

			y_size = (int(y_ref.shape[0]/M)*M, int(y_ref.shape[1]/M)*M)
			y_ref = y_ref[:y_size[0], :y_size[1]]
			y_dist = y_dist[:y_size[0], :y_size[1]]

			# Covariance and Cross-Covariance
			_, _, var_x, var_y, cov_xy = moments(y_ref, y_dist, B, M, "reflect")

			# Estimating coefficients
			g = cov_xy / (var_x + tolerance)
			sigma_v_squared = var_y - g*cov_xy

			# Corrections
			g[var_x < tolerance] = 0
			sigma_v_squared[var_x < tolerance] = var_y[var_x < tolerance]
			var_x[var_x < tolerance] = 0
			g[var_y < tolerance] = 0
			sigma_v_squared[var_y < tolerance] = 0
			sigma_v_squared[g < 0] = var_y[g < 0]
			g[g < 0] = 0
			sigma_v_squared[sigma_v_squared < tolerance] = tolerance

			G_all.append(g)
			Sigma_v_squared_all.append(sigma_v_squared)

		return G_all, Sigma_v_squared_all
	

	def Decomposation(self, img):
		if self.wavelet == 'steerable':
			# Steerable pyramidal decomposation
			pyr = SPyr(image=img, height=self.num_levels, order=5, edge_type='reflect1').pyr_coeffs
			subband_keys = []
			for key in list(pyr.keys())[1:-2:3]:
				subband_keys.append(key)
		else:
			# Wavelet decomposation
			ret = wavedec2(data=img, wavelet=self.wavelet, mode='reflect', level=self.num_levels)
			pyr = {}
			subband_keys = []
			for i in range(self.num_levels):
				pyr[(self.num_levels-1-i, 0)] = ret[i+1][0]
				pyr[(self.num_levels-1-i, 1)] = ret[i+1][1]
				subband_keys.append((self.num_levels-1-i, 0))
				subband_keys.append((self.num_levels-1-i, 1))
			pyr[self.num_levels] = ret[0]

		subband_keys.sort(reverse=True)
		return pyr, subband_keys
	

	def Subband_Eigen_Information_Matrix(self, subband_keys, G_all, Sigma_v_squared_all, S_squared_all, EigenValues_all):
		# No.of subbands
		n_subbands = len(subband_keys)
		num_eigen_values = len(EigenValues_all[0])
		
		# Information in each sub-band along each basis vector i.e per eigen value
		Info_Distored = np.zeros((n_subbands,num_eigen_values))
		Info_Reference = np.zeros((n_subbands,num_eigen_values))

		for i in range(n_subbands):
			g = G_all[i]
			sigma_v_squared = Sigma_v_squared_all[i]
			s_squared = S_squared_all[i]
			eigen_values = EigenValues_all[i]

			num_eigen_values = len(eigen_values)
			level = int(np.ceil((i+1)/2))
			winsize = 2**level + 1
			offset = (winsize - 1)/2
			offset = int(np.ceil(offset/self.M))

			g = g[offset:-offset, offset:-offset]
			sigma_v_squared = sigma_v_squared[offset:-offset, offset:-offset]
			s_squared = s_squared[offset:-offset, offset:-offset]
			if s_squared.shape != g.shape:
				s_squared = s_squared[:g.shape[0], :g.shape[1]]

			
			for j in range(num_eigen_values):
				Info_Distored[i][j] = np.mean(np.log(1 + g*g*s_squared*eigen_values[j]/(sigma_v_squared + self.sigma_n_squared)))
				Info_Reference[i][j] = np.mean(np.log(1 + s_squared*eigen_values[j]/self.sigma_n_squared))

		return Info_Reference, Info_Distored
	

	def Reference_Subband_Eigen_Information_Matrix(self, subband_keys, S_squared_all, EigenValues_all):
		# No.of subbands
		n_subbands = len(subband_keys)
		num_eigen_values = len(EigenValues_all[0])
		
		# Information in each sub-band along each basis vector i.e per eigen value
		Info_Reference = np.zeros((n_subbands,num_eigen_values))

		for i in range(n_subbands):
			s_squared = S_squared_all[i]
			eigen_values = EigenValues_all[i]

			num_eigen_values = len(eigen_values)
			level = int(np.ceil((i+1)/2))
			winsize = 2**level + 1
			offset = (winsize - 1)/2
			offset = int(np.ceil(offset/self.M))
			s_squared = s_squared[offset:-offset, offset:-offset]

			for j in range(num_eigen_values):
				Info_Reference[i][j] = np.mean(np.log(1 + s_squared*eigen_values[j]/self.sigma_n_squared))

		return Info_Reference

	
	def VIF_Wavelet(self, img_ref, img_dist): 	
		# Decomposation of reference and distored images
		pyr_ref, subband_keys = self.Decomposation(img=img_ref)
		pyr_dist, subband_keys = self.Decomposation(img=img_dist)
		subband_keys.sort(reverse=True)

		# Channel Estimation
		[G_all, Sigma_v_squared_all] = self.Channel_Estimation(pyr_ref, pyr_dist, subband_keys)

		# GSM Model
		[S_squared_all, EigenValues_all] = self.GSM_Model(pyr_ref, subband_keys)

		# Information in each subband along each eigen value
		Info_Reference, Info_Distored = self.Subband_Eigen_Information_Matrix(
			subband_keys, G_all, Sigma_v_squared_all, S_squared_all, EigenValues_all
		)

		# Mutual-Information along each subband
		Mutual_Information_Distored = np.sum(Info_Distored, axis=1)
		Mutual_Information_Reference = np.sum(Info_Reference, axis=1)

		return np.mean(Mutual_Information_Distored)/np.mean(Mutual_Information_Reference), Mutual_Information_Distored, Mutual_Information_Reference