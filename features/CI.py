import numpy as np
import scipy
import cv2


class CI_Features():
	def __init__(self,
		rgb:bool=True,
		WR:float=5
	):
		"""
		Chroma-Information: Function is modified to calculate features on Chroma components.
		
		References:
		- Towards Perceptually-Optimized Compression Of User Generated Content (UGC) - Prediction Of UGC Rate-Distortion Category
		"""
		self.rgb = rgb
		self.WR = WR


	def get_chroma_component(self,
		img:np.array=None,
		component:str=None
	):
		assert (img.dtype == np.uint8) and (np.min(img) >= 0 and np.max(img) <= 255), "Input Image/Videos should of type uint8 and should have range [0,255]."
		component2index = {"U":1, "V":2}
		
		# Converting to int32 to avoid overflow during operations.
		if self.rgb:
			return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)[:,:,component2index[component]].astype(np.int32)
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


	def compute_image_chroma_information(self,
		frame:np.array=None,
		component:str=None,
		stats:list=None
	):
		"""
		Args:
			frame (np.array): Image or frame of video.
			stats (list): List of statistics to be computed. Example: ["mean", "std"]. Options for stats: ["mean", "std", "max", "min", "skewness", "kurtosis"]
		Returns:
			(list): Statistics computed on chroma components
		"""
		# Chroma Components
		I = self.get_chroma_component(frame, component=component)

		# Chroma Information Features/Statistics
		Chroma_Information = []
		for i in range(len(stats)):
			Chroma_Information.append(self.compute_statistics(data=I, stat=stats[i]))
		
		return np.asarray(Chroma_Information)


	def compute_video_chroma_information(self,
		video:np.array=None,
		component:str=None,
		stats:list=None
	):
		"""
		Args:
			video (np.array): Video
			component (str): "U" or "V" component
			stats (list): List of lists of statistics to be computed. Second element of tuple calculates feature along spatially and first element of tuple pools the calculated feature temporally.  Example: [["mean", "std"], ["mean", "mean"]]. Options for spatial/temporal stats: ["mean", "std", "max", "min", "skewness", "kurtosis"]
		Returns:
			(list): Statistics computed on chroma components
		"""
		# No.of frames of video
		num_frames = video.shape[0]

		# Statistics to calculate on each frame
		stats = np.asarray(stats)
		spatial_stats = np.unique(stats[:,1])
		spatial_stats2index = {k:v for v,k in enumerate(spatial_stats)}

		# Chroma information
		Chroma_Information_per_frame = []
		for i in range(0,num_frames):
			Chroma_Information_per_frame.append(self.compute_image_chroma_information(frame=video[i], component=component, stats=spatial_stats))
		Chroma_Information_per_frame = np.asarray(Chroma_Information_per_frame)

		# Multiplying WR to Chroma Information of "V" component
		if component == "V":
			Chroma_Information_per_frame = self.WR * Chroma_Information_per_frame

		Chroma_Information = []
		for i in range(len(stats)):
			pool_stat = stats[i,0]
			spatial_stat_index = spatial_stats2index[stats[i,1]]
			Chroma_Information.append(self.compute_statistics(data=Chroma_Information_per_frame[:,spatial_stat_index], stat=pool_stat))
		Chroma_Information = np.asarray(Chroma_Information)

		return np.round(Chroma_Information_per_frame, decimals=6), np.round(Chroma_Information, decimals=6)