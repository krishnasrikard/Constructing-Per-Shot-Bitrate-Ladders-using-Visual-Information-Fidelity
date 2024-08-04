import numpy as np
import scipy
import cv2
from scipy.ndimage import sobel

from tqdm import tqdm

class CF_Features():
	def __init__(self,
	):
		"""
		ColorFulness: Function to calculate ColorFulness.
		
		References:
		- https://stackoverflow.com/questions/73478461/computing-colorfulness-of-an-image-in-python-fast
		- Towards Perceptually-Optimized Compression Of User Generated Content (UGC) - Prediction Of UGC Rate-Distortion Category
		"""
		None
	
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


	def compute_image_colorfulness(self,
		frame:np.array=None,
	):
		"""
		Args:
			frame (np.array): Image or frame of video.
			stats (list): List of statistics to be computed. Example: ["mean", "std"]. Options for stats: ["mean", "std", "max", "min", "skewness", "kurtosis"]
		Returns:
			(list): Statistics computed on colorfulness
		"""
		assert (frame.dtype == np.uint8) and (np.min(frame) >= 0 and np.max(frame) <= 255), "Input Image/Videos should of type uint8 and should have range [0,255]."

		# Splitting Image
		# CF value only difference by a factor 255 depending on our the range of our pixel space i.e either [0.0,1.0] or [0,255].
		R = frame[:,:,0]/255.0
		G = frame[:,:,1]/255.0
		B = frame[:,:,2]/255.0

		# Compute rg = R - G
		rg = np.absolute(R - G)

		# Compute yb = 0.5 * (R + G) - B
		yb = np.absolute(0.5 * (R + G) - B)

		# Compute the mean and standard deviation of both `rg` and `yb`
		(rgMean, rgStd) = (np.mean(rg), np.std(rg))
		(ybMean, ybStd) = (np.mean(yb), np.std(yb))

		# Combine the mean and standard deviations
		stdRoot = np.sqrt((rgStd ** 2) + (ybStd ** 2))
		meanRoot = np.sqrt((rgMean ** 2) + (ybMean ** 2))

		# "Colorfulness" metric
		return stdRoot + (0.3 * meanRoot)


	def compute_video_colorfulness(self,
		video:np.array=None,
		stats:list=None
	):
		"""
		Args:
			video (np.array): Video
			stats (list): List of statistics to be computed to pool temporally. Example: ["mean", "std"]. Options for stats: ["mean", "std", "max", "min", "skewness", "kurtosis"]
		Returns:
			(list): Statistics computed on colorfulness
		"""
		# No.of frames of video
		num_frames = video.shape[0]

		# Statistics to calculate on each frame
		stats = np.asarray(stats)

		ColorFulness_per_frame = []
		for i in range(0,num_frames):
			ColorFulness_per_frame.append(self.compute_image_colorfulness(frame=video[i]))
		ColorFulness_per_frame = np.asarray(ColorFulness_per_frame)

		ColorFulness = []
		for i in range(len(stats)):
			ColorFulness.append(self.compute_statistics(data=ColorFulness_per_frame, stat=stats[i]))
		ColorFulness = np.asarray(ColorFulness)

		return np.round(ColorFulness_per_frame, decimals=6), np.round(ColorFulness, decimals=6)