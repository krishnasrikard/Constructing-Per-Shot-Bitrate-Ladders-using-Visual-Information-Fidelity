import numpy as np
from skimage.feature import graycomatrix, graycoprops
import cv2
import scipy

import sys
sys.path.append("/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity")


class GLCM_Features():
	def __init__(self,
	    descriptors:list=["contrast","correlation","energy","homogeneity"],
		angles:list=[0],
		distance:int=1,
		levels:int=256,
		block_size:tuple=(64,64),
		rgb:bool=True
	):
		"""
		GLCM (Gray Level Co-occurrence Matrix): Function to find calculate features on 
		of various GLCM descriptors like "contrast","correlation","energy" and "homogeneity". The features are calculated on the image or across all frames for a video by pooling spatially and temporally using statistical operations.

		References:
		- https://github.com/AngKats/StudyVideoTextureRD/blob/35d27c520eba6aee099e6238f1d631b58380c4ce/FeatureExtraction/ExtractSTFeat.m
		- https://stackoverflow.com/questions/69796388/extracting-glcm-features
		"""
		self.descriptors = descriptors
		self.angles = angles
		self.distance = distance
		self.levels=levels
		self.block_size=block_size
		self.rgb = rgb


	def get_luma_component(self,
		img:np.array=None
	):
		assert (img.dtype == np.uint8) and (np.min(img) >= 0 and np.max(img) <= 255), "Input Image/Videos should of type uint8 and should have range [0,255]."
		
		# Converting to int32 to avoid overflow during operations.
		if self.rgb:
			return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)[:,:,0].astype(np.int32)
		else:
			return img.astype(np.int32)
		
	
	def compute_statistics(self,
		data:np.array=None,
		stat:str=None
	):
		data = data.flatten()
		if stat == "mean":
			return np.mean(data)
		elif stat == "std":
			return np.std(data)
		elif stat == "max":
			return np.max(data)
		elif stat == "min":
			return np.min(data)
		elif stat == "skew" or stat == "skewness":
			return scipy.stats.skew(data)
		elif stat == "kurt" or stat == "kurtosis":
			return scipy.stats.kurtosis(data, axis=0, fisher=False)
		else:
			assert False, "{} is an invalid statistic property.".format(stat)
		

	def split_image(self,
		img:np.array=None
	):
		"""
		Splitting image in to non-overlapping patches
		"""
		if self.block_size == None:
			self.block_size = img.shape
		patches = np.lib.stride_tricks.sliding_window_view(img, window_shape=self.block_size)[::self.block_size[0], ::self.block_size[1]].reshape(-1, self.block_size[0], self.block_size[1])
		return patches
	

	def compute_GLCM(self,
		img:np.array=None
	):
		"""
		Calculating GLCM matrix on the image
		"""
		glcm = graycomatrix(image=img, distances=[self.distance], angles=self.angles, levels=self.levels, normed=True, symmetric=False)

		# Summing GLCM values for all angles and all distances
		glcm = np.sum(glcm, axis=(2,3), keepdims=True)
		return glcm
	

	def compute_descriptors(self,
		img:np.array=None
	):
		"""
		Calculating GLCM Descriptors
		"""
		glcm = self.compute_GLCM(img)
		Property_Values = []			

		for i,descriptor in enumerate(self.descriptors):
			Property_Values.append(graycoprops(glcm, descriptor)[0][0])

		return Property_Values
	

	def compute_image_glcm_features(self,
		frame:np.array=None,
		stats:list=None
	):
		"""
		Args:
			frame (np.array): Image or frame of video.
			stats (list): List of statistics to be computed. Example: ["mean", "std"]. Options for stats: ["mean", "std", "max", "min", "skewness", "kurtosis"]
		Returns:
			(list): Statistics computed GLCM descriptors pooled spatially for each patch/block of image.
		"""
		# Extracting luma component
		frame = self.get_luma_component(frame)
	
		# Splitting Image to Patches
		patches = self.split_image(frame)

		# GLCM Descriptors
		GLCM_Descriptors_per_block = []
		for i in range(patches.shape[0]):
			GLCM_Descriptors_per_block.append(self.compute_descriptors(patches[i]))
		GLCM_Descriptors_per_block = np.asarray(GLCM_Descriptors_per_block)

		# Spatial Pooling of GLCM Descriptors
		GLCM_Descriptors_Information = np.empty((len(self.descriptors), len(stats)))
		for i in range(len(self.descriptors)):
			for j in range(len(stats)):
				GLCM_Descriptors_Information[i][j] = self.compute_statistics(data=GLCM_Descriptors_per_block[:,i], stat=stats[j])

		return GLCM_Descriptors_Information


	def compute_video_glcm_features(self,
		video:np.array=None,
		stats:list=None
	):
		"""
		Args:
			video (np.array): Video
			stats (list): List of lists of statistics to be computed. Second element of tuple calculates feature along spatially and first element of tuple pools the calculated feature temporally.  Example: [["mean", "std"], ["mean", "mean"]]. Options for spatial/temporal stats: ["mean", "std", "max", "min", "skewness", "kurtosis"]
		Returns:
			(list): Statistics computed on GLCM descriptors pooled spatially and temporally for each patch/block of image.
		"""
		# No.of frames of video
		num_frames = video.shape[0]

		# Statistics to calculate on each frame
		stats = np.asarray(stats)
		spatial_stats = np.unique(stats[:,1])
		spatial_stats2index = {k:v for v,k in enumerate(spatial_stats)}

		GLCM_Descriptors_Information_per_frame = []
		for i in range(0,num_frames):
			GLCM_Descriptors_Information_per_frame.append(self.compute_image_glcm_features(frame=video[i], stats=spatial_stats))
		GLCM_Descriptors_Information_per_frame = np.asarray(GLCM_Descriptors_Information_per_frame)

		GLCM_Descriptors_Information = np.empty((len(self.descriptors), len(stats)))
		for i in range(len(self.descriptors)):
			for j in range(len(stats)):
				pool_stat = stats[j,0]
				spatial_stat_index = spatial_stats2index[stats[j,1]]
				GLCM_Descriptors_Information[i][j] = self.compute_statistics(data=GLCM_Descriptors_Information_per_frame[:,i,spatial_stat_index], stat=pool_stat)
		GLCM_Descriptors_Information = np.asarray(GLCM_Descriptors_Information)

		return np.round(GLCM_Descriptors_Information_per_frame.reshape(num_frames, -1), decimals=6), np.round(GLCM_Descriptors_Information.flatten(), decimals=6)