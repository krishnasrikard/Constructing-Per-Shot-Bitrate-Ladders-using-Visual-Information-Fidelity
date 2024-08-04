import numpy as np
import cv2
from scipy.fftpack import dct


class Texture_DCT_Features():
	def __init__(self,
	    block_size:tuple=(-1,-1),
	    rgb:bool=True
	):
		"""
		E: Average Spatial Energy
		h: Average Temporal Energy
		L: Luma Brightness

		References:
		- https://vca.itec.aau.at/
		"""
		assert block_size[0] != -1 and block_size[1] != -1, "Invalid block size."
		self.block_size = block_size
		self.rgb = rgb

		self.weights = np.zeros(self.block_size)
		for i in range(self.block_size[0]):
			for j in range(self.block_size[1]):
				self.weights[i][j] = np.exp(np.abs(np.square(i*j/np.prod(self.block_size)) - 1))


	def get_component(self,
		img:np.array=None,
		component:str=None
	):
		assert (img.dtype == np.uint8) and (np.min(img) >= 0 and np.max(img) <= 255), "Input Image/Videos should of type uint8 and should have range [0,255]."
		component2index = {"Y":0, "U":1, "V":2}
		
		# Converting to int32 to avoid overflow during operations.
		if self.rgb:
			return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)[:,:,component2index[component]].astype(np.int32)
		else:
			return img.astype(np.int32)


	def DCT_2D(self,
		img:np.array=None
	):
		return dct(dct(img.T, type=2, norm="ortho").T, type=2, norm="ortho")


	def compute_image_features(self,
		frame:np.array=None,
		component:str=None,
		H_prev:list=None
	):
		frame = self.get_component(frame, component)

		H = []
		L = []

		for i in range(0,frame.shape[0],self.block_size[0]):
			for j in range(0,frame.shape[1],self.block_size[1]):
				# Block
				block = frame[i:i+self.block_size[0], j:j+self.block_size[1]]
				
				if (block.shape == self.block_size):
					# DCT of Block
					DCT_Coeffs = self.DCT_2D(block)

					# Luminance of Block
					L.append(np.sqrt(DCT_Coeffs[0][0]))

					# Reference Paper: A New Energy Function for Segmentation and Compression
					DCT_Coeffs[0][0] = 0

					# Energy of Block
					H.append(np.sum(np.multiply(self.weights, np.abs(DCT_Coeffs))))
				
		# Spatial-Texture Energy
		E = np.mean(H)/np.prod(self.block_size)

		# Temporal-Texture Energy
		if H_prev is None:
			h = 0
		else:
			h = np.mean(np.abs(np.asarray(H)-np.asarray(H_prev)))/np.prod(self.block_size)

		# Luminanace
		L = np.mean(L)/np.prod(self.block_size)
		return H,[E,h,L]

	
	def compute_video_features(self,
		video:np.array=None,
		component:str=None
	):
		"""
		Args:
			video (np.array): Video
			component (str): "Y" or "U" or "V" component
		"""
		num_frames = video.shape[0]

		Features = []
		for n in range(0,num_frames):
			if n==0:
				H_prev, F = self.compute_image_features(video[n], component=component, H_prev=None)
				Features.append(F)
			else:
				H_prev, F = self.compute_image_features(video[n], component=component, H_prev=H_prev)
				Features.append(F)

		per_frame_features = np.asarray(Features)
		features = np.mean(Features, axis=0)
		features[1] = features[1] * (per_frame_features.shape[0]/(per_frame_features.shape[0] - 1))

		return per_frame_features, features