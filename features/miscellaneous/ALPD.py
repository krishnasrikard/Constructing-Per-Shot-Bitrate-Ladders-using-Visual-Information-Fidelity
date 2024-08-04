import numpy as np
import cv2
from scipy.signal import find_peaks
import pywt

from tqdm import tqdm

class ALPD_Features():
	def __init__(self,
		rgb:bool=True,
		level=3,
		wavelet_name="sym4"
	):
		"""
		ALPD: Average Local Peak Distance
		- Finding Average Local Peak Distance features like mean and standard-deviation.
		- ALPD is computed in the third level of Discrete Wavelet Transform.

		References:
		- https://github.com/AngKats/StudyVideoTextureRD/blob/35d27c520eba6aee099e6238f1d631b58380c4ce/FeatureExtraction/computeALPD.m
		- https://pywavelets.readthedocs.io/en/latest/ref/swt-stationary-wavelet-transform.html
		"""
		self.rgb = rgb
		self.level = level
		self.wavelet_name = wavelet_name


	def get_luma_component(self,
		img:np.array=None
	):
		assert (img.dtype == np.uint8) and (np.min(img) >= 0 and np.max(img) <= 255), "Input Image/Videos should of type uint8 and should have range [0,255]."
		
		# Converting to int32 to avoid overflow during operations.
		if self.rgb:
			return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)[:,:,0].astype(np.int32)
		else:
			return img.astype(np.int32)


	def compute_swt2d(self,
		img:np.array=None,
	):
		coeffs = pywt.swt2(data=img, wavelet=self.wavelet_name, level=self.level, start_level=0)[0]
		return coeffs


	def compute_ALPD(self,
		img:np.array=None,
	):
		img = self.get_luma_component(img)

		coeffs = self.compute_swt2d(img)
		horizontal_coeffs = np.array(coeffs[1][0])
		vertical_coeffs = np.array(coeffs[1][1])
		
		horizontal_local_peak_distances = []
		for i in range(img.shape[0]):
			hc = horizontal_coeffs[i,:]
			peak_indices,_ = find_peaks(x=hc)

			mask = np.where(np.abs(hc[peak_indices])>10)[0]
			peak_indices = peak_indices[mask]
			peak_indices = peak_indices.tolist()

			L1 = np.array(peak_indices + [img.shape[1]-1])
			L2 = np.array([0] + peak_indices)
			
			horizontal_local_peak_distances.append(np.mean(L1 - L2))
			
		vertical_local_peak_distances = []
		for j in range(img.shape[1]):
			vc = vertical_coeffs[:,j]
			peak_indices,_ = find_peaks(x=vc)

			mask = np.where(np.abs(vc[peak_indices])>10)[0]
			peak_indices = peak_indices[mask]
			peak_indices = peak_indices.tolist()

			L1 = np.array(peak_indices + [img.shape[0]-1])
			L2 = np.array([0] + peak_indices)

			vertical_local_peak_distances.append(np.mean(L1 - L2))
			
		return np.mean(horizontal_local_peak_distances) + np.mean(vertical_local_peak_distances)


	def compute_image_features(self,
		img:np.array=None,
	):
		return self.compute_ALPD(img)
	
		
	def compute_video_features(self,
		video:np.array=None
	):
		num_frames = video.shape[0]
		Features = []

		for i in range(0,num_frames):
			Features.append(self.compute_ALPD(video[i]))

		Features = np.array([np.mean(Features), np.std(Features)])
		return Features