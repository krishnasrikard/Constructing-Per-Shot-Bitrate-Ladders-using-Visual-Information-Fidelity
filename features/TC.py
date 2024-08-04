import numpy as np
import cv2
from scipy import signal
import scipy


class TC_Features():
	def __init__(self,
		rgb:bool=True
	):
		"""
		TC (Temporal Coherance): Function to calculate of various TC features spatially and temporally. 

		References:
		- https://github.com/AngKats/StudyVideoTextureRD/blob/35d27c520eba6aee099e6238f1d631b58380c4ce/FeatureExtraction/ExtractSTFeat.m
		- https://stackoverflow.com/questions/69796388/extracting-glcm-features
		"""
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
		

	def compute_TC_features(self,
		reference:np.array=None,
		target:np.array=None
	):
		assert reference.shape == target.shape, "reference and target don't have the same shape."
		reference = self.get_luma_component(reference)
		target = self.get_luma_component(target)
		
		N = reference.flatten().shape[0]
		L = int(N//4.5)
		nfft = max(256, 2**np.ceil(np.log2(L)))
		f, Cxy = signal.coherence(reference.flatten(), target.flatten(), fs=1, nfft=nfft, window=signal.get_window('hamming', L, fftbins=False), noverlap=int(L//2), detrend=None)

		return Cxy


	def compute_image_tc_features(self,
		reference:np.array=None,
		target:np.array=None,
		stats:np.array=None
	):
		"""
		Args:
			reference (np.array): Reference image or frame of video.
			target (np.array): Target image or next frame of video. 
			stats (list): List of statistics to be computed. Example: ["mean", "std"]. Options for stats: ["mean", "std", "max", "min", "skewness", "kurtosis"]
		Returns:
			(list): Statistics computed TC features pooled spatially.
		"""
		# Temporal Coherence
		Cxy = self.compute_TC_features(reference=reference, target=target)

		# Temporal-Coherence features
		TC_Features = []
		for i in range(len(stats)):
			TC_Features.append(self.compute_statistics(data=Cxy, stat=stats[i]))

		return np.asarray(TC_Features)
	
	
	def compute_video_tc_features(self,
		video:np.array=None,
		stats:list=None
	):
		"""
		Args:
			video (np.array): Video
			stats (list): List of lists of statistics to be computed. Second element of tuple calculates feature along spatially and first element of tuple pools the calculated feature temporally.  Example: [["mean", "std"], ["mean", "mean"]]. Options for spatial/temporal stats: ["mean", "std", "max", "min", "skewness", "kurtosis"]
		"""
		# No.of frames of video
		num_frames = video.shape[0]

		# Statistics to calculate on each frame
		stats = np.asarray(stats)
		spatial_stats = np.unique(stats[:,1])
		spatial_stats2index = {k:v for v,k in enumerate(spatial_stats)}

		Temporal_Coherence_Information_per_frame = []
		for i in range(1,num_frames):
			Temporal_Coherence_Information_per_frame.append(self.compute_image_tc_features(reference=video[i-1], target=video[i], stats=spatial_stats))
		Temporal_Coherence_Information_per_frame = np.asarray(Temporal_Coherence_Information_per_frame)

		Temporal_Coherence_Information = []
		for i in range(len(stats)):
			pool_stat = stats[i,0]
			spatial_stat_index = spatial_stats2index[stats[i,1]]
			Temporal_Coherence_Information.append(self.compute_statistics(data=Temporal_Coherence_Information_per_frame[:,spatial_stat_index], stat=pool_stat))
		Temporal_Coherence_Information = np.asarray(Temporal_Coherence_Information)

		return np.round(Temporal_Coherence_Information_per_frame, decimals=6), np.round(Temporal_Coherence_Information, decimals=6)