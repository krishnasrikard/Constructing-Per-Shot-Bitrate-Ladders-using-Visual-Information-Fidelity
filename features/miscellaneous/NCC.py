import numpy as np
import scipy
from skimage.feature import match_template
import cv2

from tqdm import tqdm

class NCC_Features():
	def __init__(self,
		rgb:bool=True,
		confine_search=True,
		window_size:tuple=(-1,-1)
	):
		"""
		NCC: Normalized Cross-Correlation
		- Finding Normalized Cross-Correlation features like mean, standard-deviation, skewness and kurtosis between successive frames of a video.
		- The maximum correlation coefficient is considered when a template is matched with an image.

		References:
		- https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.match_descriptors
		- https://github.com/AngKats/StudyVideoTextureRD/blob/35d27c520eba6aee099e6238f1d631b58380c4ce/FeatureExtraction/ComputeNCC.m
		"""
		assert window_size[0] != -1 and window_size[1] != -1, "Invalid window_size"
		self.rgb = rgb
		self.confine_search = confine_search
		self.window_size = window_size


	def get_luma_component(self,
		img:np.array=None
	):
		assert (img.dtype == np.uint8) and (np.min(img) >= 0 and np.max(img) <= 255), "Input Image/Videos should of type uint8 and should have range [0,255]."
		
		# Converting to int32 to avoid overflow during operations.
		if self.rgb:
			return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)[:,:,0].astype(np.int32)
		else:
			return img.astype(np.int32)


	def template_matching(self,
		template:np.array=None,
		image:np.array=None
	):
		return np.max(match_template(image,template,pad_input=True))


	def compute_NCC_properties(self,
		reference:np.array=None,
		target:np.array=None
	):
		assert reference.shape == target.shape, "reference and target don't have the same shape."
		reference = self.get_luma_component(reference)
		target = self.get_luma_component(target)

		NCC = []
		for i in range(0, reference.shape[0], self.window_size[0]):
			for j in range(0, reference.shape[1], self.window_size[1]):
				template = reference[i:i+self.window_size[0], j:j+self.window_size[1]]
				if self.confine_search:
					target_ = target[max(0,i-int(0.5*self.window_size[0])):min(i+int(1.5*self.window_size[0]),target.shape[0]), max(0,j-int(0.5*self.window_size[1])):min(j+int(1.5*self.window_size[1]),target.shape[1])]
				else:
					target_ = target
				correlation_coefficients = self.template_matching(template,target_)
				NCC.append(correlation_coefficients)

		Properties = [np.mean(NCC), np.std(NCC), scipy.stats.skew(NCC), scipy.stats.kurtosis(NCC, axis=0, fisher=False)]
		return np.array(Properties)


	def compute_image_features(self,
		reference:np.array=None,
		target:np.array=None
	):
		return self.compute_NCC_properties(reference,target)

		
	def compute_video_features(self,
		video:np.array=None,
		step=2
	):
		num_frames = video.shape[0]
		Features = []

		for i in range(0,num_frames,step):
			Features.append(self.compute_NCC_properties(video[i],video[i+1]))

		Features = np.mean(Features, axis=0)
		return Features