import numpy as np
import scipy
import cv2
from scipy.ndimage import sobel


class SI_Features():
	def __init__(self,
	    rgb:bool=True
	):
		"""
		Spatial-Information: Function is modified to calculate features when Sobel filter is applied. Spatial-Information of an image/frame is generally calculated as std(Sobel(frame)).
		
		References:
		- https://github.com/slhck/siti/blob/master/siti/__main__.py
		- Towards Perceptually-Optimized Compression Of User Generated Content (UGC) - Prediction Of UGC Rate-Distortion Category
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


	def compute_sobel_filter_response(self,
		frame:np.array=None
	):
		frame = self.get_luma_component(frame)

		# Calculate horizontal/vertical operators
		sob_x = sobel(frame, axis=0)
		sob_y = sobel(frame, axis=1)

		# Crop output to valid window, calculate gradient magnitude
		SI = np.hypot(sob_x, sob_y)[1:-1, 1:-1]
		return SI


	def compute_image_spatial_information(self,
		frame:np.array=None,
		stats:list=None
	):
		"""
		Args:
			frame (np.array): Image or frame of video.
			stats (list): List of statistics to be computed. Example: ["mean", "std"]. Options for stats: ["mean", "std", "max", "min", "skewness", "kurtosis"]
		Returns:
			(list): Statistics computed on Sobel(Frame)

		Notes:
			Spatial-Information is generally calculated as std(Sobel(frame)).
		"""
		# Spatial Information
		SI = self.compute_sobel_filter_response(frame)

		# Spatial Information Features/Statistics
		Spatial_Information = []
		for i in range(len(stats)):
			Spatial_Information.append(self.compute_statistics(data=SI, stat=stats[i]))

		return np.asarray(Spatial_Information)


	def compute_video_spatial_information(self,
		video:np.array=None,
		stats:list=None
	):
		"""
		Args:
			video (np.array): Video
			stats (list): List of lists of statistics to be computed. Second element of tuple calculates feature along spatially and first element of tuple pools the calculated feature temporally.  Example: [["mean", "std"], ["mean", "mean"]]. Options for spatial/temporal stats: ["mean", "std", "max", "min", "skewness", "kurtosis"]
		Returns:
			(list): Statistics computed on Sobel(Frame)
		"""
		# No.of frames of video
		num_frames = video.shape[0]

		# Statistics to calculate on each frame
		stats = np.asarray(stats)
		spatial_stats = np.unique(stats[:,1])
		spatial_stats2index = {k:v for v,k in enumerate(spatial_stats)}

		Spatial_Information_per_frame = []
		for i in range(0,num_frames):
			Spatial_Information_per_frame.append(self.compute_image_spatial_information(frame=video[i], stats=spatial_stats))
		Spatial_Information_per_frame = np.asarray(Spatial_Information_per_frame)

		Spatial_Information = []
		for i in range(len(stats)):
			pool_stat = stats[i,0]
			spatial_stat_index = spatial_stats2index[stats[i,1]]
			Spatial_Information.append(self.compute_statistics(data=Spatial_Information_per_frame[:,spatial_stat_index], stat=pool_stat))
		Spatial_Information = np.asarray(Spatial_Information)

		return np.round(Spatial_Information_per_frame, decimals=6), np.round(Spatial_Information, decimals=6)