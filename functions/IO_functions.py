import numpy as np
import cv2

import os
import json

def read_create_jsonfile(path, create:bool=None):
	"""
	Reading an existing json file or creating a new one.
	Args:
		path (str): Path to json file.
		create (bool): Whether to create a file or not. None as input creates the file. (Default: None)
	Return:
		data (dict): Data in json file.
	"""

	if os.path.exists(path) == False:
		if create is None or create == True:
			json_object = json.dumps({}, indent=4)
			with open(path, "w") as f:
				f.write(json_object)
		else:
			assert False, "File in the path {} does not exist.".format(path)

	# Reading File
	f = open(path)
	data = json.load(f)
	f.close()

	return data


def save_jsonfile(path,data):
	"""
	Reading an existing json file or creating a new one.
	Args:
		path (str): Path to json file.
	Return:
		data (dict): Data in json file.
	"""

	json_object = json.dumps(data, indent=4)
	with open(path, "w") as f:
		f.write(json_object)



class YUV_Reader(object):
	"""
		- Source: https://github.com/Netflix/vmaf/blob/5ee0051cd7b1337e033558910c30525d73edfd76/python/vmaf/tools/reader.py
		- The class is a modifed version of the source.
	"""
	
	# Supported YUV types
	SUPPORTED_YUV_8BIT_TYPES = ['yuv420p',
								'yuv422p',
								'yuv444p',
								'gray',
								]

	SUPPORTED_YUV_10BIT_LE_TYPES = ['yuv420p10le',
									'yuv422p10le',
									'yuv444p10le',
									'gray10le',
									]

	SUPPORTED_YUV_12BIT_LE_TYPES = ['yuv420p12le',
									'yuv422p12le',
									'yuv444p12le',
									'gray12le',
									]

	SUPPORTED_YUV_16BIT_LE_TYPES = ['yuv420p16le',
									'yuv422p16le',
									'yuv444p16le',
									'gray16le',
									]

	# Multiplies for the width and height of U/V wrt Y. (Example: For yuv420p, the width and height of U/V is 0.5x, 0.5x of Y)
	UV_WIDTH_HEIGHT_MULTIPLIERS_DICT = {'yuv420p': (0.5, 0.5),
										'yuv422p': (0.5, 1.0),
										'yuv444p': (1.0, 1.0),
										'gray': (0.0, 0.0),
										'yuv420p10le': (0.5, 0.5),
										'yuv422p10le': (0.5, 1.0),
										'yuv444p10le': (1.0, 1.0),
										'gray10le': (0.0, 0.0),
										'yuv420p12le': (0.5, 0.5),
										'yuv422p12le': (0.5, 1.0),
										'yuv444p12le': (1.0, 1.0),
										'gray12le': (0.0, 0.0),
										'yuv420p16le': (0.5, 0.5),
										'yuv422p16le': (0.5, 1.0),
										'yuv444p16le': (1.0, 1.0),
										'gray16le': (0.0, 0.0),
										}


	def __init__(self, filepath, width, height, yuv_type):
		self.filepath = filepath
		self.width = width
		self.height = height
		self.yuv_type = yuv_type
		self._asserts()
		self.file = open(self.filepath, 'rb')


	def __enter__(self):
		"""
			To make YUV_Reader withable.
			Example: 
			```
			with YUV_Reader(...) as yuv_reader: 
				...
			```
		"""
		return self

	def __iter__(self):
		"""
			To make YUV_Reader iterable.
			Example: 
			```
			for y, u, v in yuv_reader: 
				...
			```
		"""
		return self

	def __next__(self):
		"""
			next() is for python2 only, in python3 all you need to define is __next__(self)
		"""
		return self.next()
	
	def close(self):
		self.file.close()

	def __exit__(self, exc_type, exc_value, traceback):
		self.close()
	

	def _asserts(self):
		# Assert file exists
		self._assert_file_exist()

		# Assert YUV type
		self._assert_yuv_type()

		# Assert file size: If the no.of frames is an integer
		assert isinstance(self.num_frames, int)


	def _assert_file_exist(self):
		# Assert file existance
		assert os.path.exists(self.filepath), \
			"File does not exist: {}".format(self.filepath)
		
	def _assert_yuv_type(self):
		# Assert YUV type
		assert (self.yuv_type in self.SUPPORTED_YUV_8BIT_TYPES
				or self.yuv_type in self.SUPPORTED_YUV_10BIT_LE_TYPES
				or self.yuv_type in self.SUPPORTED_YUV_12BIT_LE_TYPES
				or self.yuv_type in self.SUPPORTED_YUV_16BIT_LE_TYPES), \
			'Unsupported YUV type: {}'.format(self.yuv_type)

	def _is_8bit(self):
		# Assert YUV type is 8-bit
		return self.yuv_type in self.SUPPORTED_YUV_8BIT_TYPES

	def _is_10bitle(self):
		# Assert YUV type is 10-bit
		return self.yuv_type in self.SUPPORTED_YUV_10BIT_LE_TYPES

	def _is_12bitle(self):
		# Assert YUV type is 12-bit
		return self.yuv_type in self.SUPPORTED_YUV_12BIT_LE_TYPES

	def _is_16bitle(self):
		# Assert YUV type is 16-bit
		return self.yuv_type in self.SUPPORTED_YUV_16BIT_LE_TYPES
	

	def _get_uv_width_height_multiplier(self):
		# Width and Height multipliers of U/V components
		self._assert_yuv_type()
		return self.UV_WIDTH_HEIGHT_MULTIPLIERS_DICT[self.yuv_type]
	

	@property
	def num_bytes(self):
		# File size in bytes
		self._assert_file_exist()
		return os.path.getsize(self.filepath)


	@property
	def num_frames(self):
		# No.of frames of video in YUV file

		w_multiplier, h_multiplier = self._get_uv_width_height_multiplier()

		if self._is_10bitle() or self._is_12bitle() or self._is_16bitle():
			num_frames = float(self.num_bytes) / self.width / self.height / (1.0 + w_multiplier * h_multiplier * 2) / 2
		elif self._is_8bit():
			num_frames = float(self.num_bytes) / self.width / self.height / (1.0 + w_multiplier * h_multiplier * 2)
		else:
			assert False

		assert num_frames.is_integer(), 'Number of frames is not integer: {}'.format(num_frames)

		return int(num_frames)


	def next(self, format='uint'):
		# Returns Y,U and V components of each frame. n-th call of next() returns Y,U and V components of n-th frame.

		assert format == 'uint' or format == 'float'

		y_width = self.width
		y_height = self.height
		uv_w_multiplier, uv_h_multiplier = self._get_uv_width_height_multiplier()
		uv_width = int(y_width * uv_w_multiplier)
		uv_height = int(y_height * uv_h_multiplier)

		if self._is_8bit():
			pix_type = np.uint8
			word = 1
		elif self._is_10bitle() or self._is_12bitle() or self._is_16bitle():
			pix_type = np.uint16
			word = 2
		else:
			assert False

		# Y component
		y = np.frombuffer(self.file.read(y_width * y_height * word), pix_type)
		if y.size == 0:
			raise StopIteration

		# U and V compoenets
		if uv_width == 0 and uv_height == 0:
			u = None
			v = None
		elif uv_width > 0 and uv_height > 0:
			u = np.frombuffer(self.file.read(uv_width * uv_height * word), pix_type)
			if u.size == 0:
				raise StopIteration
			v = np.frombuffer(self.file.read(uv_width * uv_height * word), pix_type)
			if v.size == 0:
				raise StopIteration
		else:
			assert False, f'Unsupported uv_width and uv_height: {uv_width}, {uv_height}'


		# Reshaping Y,U and V components
		y = y.reshape(y_height, y_width)
		u = u.reshape(uv_height, uv_width) if u is not None else None
		v = v.reshape(uv_height, uv_width) if v is not None else None


		# Returning Y,U and V components
		if format == 'uint':
			return y, u, v
		elif format == 'float':
			if self._is_8bit():
				y = y.astype(np.double) / (2.0**8 - 1.0)
				u = u.astype(np.double) / (2.0**8 - 1.0) if u is not None else None
				v = v.astype(np.double) / (2.0**8 - 1.0) if v is not None else None
				return y, u, v
			elif self._is_10bitle():
				y = y.astype(np.double) / (2.0**10 - 1.0)
				u = u.astype(np.double) / (2.0**10 - 1.0) if u is not None else None
				v = v.astype(np.double) / (2.0**10 - 1.0) if v is not None else None
				return y, u, v
			elif self._is_12bitle():
				y = y.astype(np.double) / (2.0**12 - 1.0)
				u = u.astype(np.double) / (2.0**12 - 1.0) if u is not None else None
				v = v.astype(np.double) / (2.0**12 - 1.0) if v is not None else None
				return y, u, v
			elif self._is_16bitle():
				y = y.astype(np.double) / (2.0**16 - 1.0)
				u = u.astype(np.double) / (2.0**16 - 1.0) if u is not None else None
				v = v.astype(np.double) / (2.0**16 - 1.0) if v is not None else None
				return y, u, v
			else:
				assert False
		else:
			assert False

	def next_rgb(self):
		# Returns next RGB frame
		y,u,v = self.next("float")

		u = cv2.resize(u, (y.shape[1], y.shape[0]), interpolation = cv2.INTER_NEAREST)
		v = cv2.resize(v, (y.shape[1], y.shape[0]), interpolation = cv2.INTER_NEAREST)

		yuv_frame = np.concatenate((np.expand_dims(y, axis=-1), np.expand_dims(u, axis=-1), np.expand_dims(v, axis=-1)), axis=-1)
		yuv_frame = np.uint8(255.0 * yuv_frame)

		rgb_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB)

		return rgb_frame


	def get_YUV_video(self, format='uint'):
		# Returns video in YUV format.
		yuv_video = []

		for i in range(self.num_frames):
			y,u,v = self.next(format)
			u = cv2.resize(u, (y.shape[1], y.shape[0]), interpolation = cv2.INTER_NEAREST)
			v = cv2.resize(v, (y.shape[1], y.shape[0]), interpolation = cv2.INTER_NEAREST)

			yuv_video.append(np.concatenate((np.expand_dims(y, axis=-1), np.expand_dims(u, axis=-1), np.expand_dims(v, axis=-1)), axis=-1))

		return np.array(yuv_video)
	
	def get_RGB_video(self):
		"""
		Returns video in RGB format stored in a numpy array with datatype np.uint8.
		"""
		rgb_video = []

		for i in range(self.num_frames):
			y,u,v = self.next("float")
			u = cv2.resize(u, (y.shape[1], y.shape[0]), 0, 0, interpolation = cv2.INTER_NEAREST)
			v = cv2.resize(v, (y.shape[1], y.shape[0]), 0, 0, interpolation = cv2.INTER_NEAREST)

			yuv_frame = np.concatenate((np.expand_dims(y, axis=-1), np.expand_dims(u, axis=-1), np.expand_dims(v, axis=-1)), axis=-1)
			yuv_frame = np.clip(np.round(255.0 * yuv_frame), 0, 255).astype(np.uint8)

			rgb_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB)
			rgb_video.append(rgb_frame)

		rgb_video = np.array(rgb_video)
		return rgb_video


def save_rgb_video(frames,fps,video_path):
	"""
	Args:
		frames (np.array): Numpy array of frames.
		fps (float): No.of frames per second.
		video_path (str): Video path.
	"""
	assert frames.dtype == np.uint8, "Invalid data type of frames"
	
	size = frames[0].shape[:2]
	frames = list(frames)

	video = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]))
	for frame in frames:
		video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
	video.release()


def read_rgb_video(video_path):
	"""
	Args:
		video_path (str): Video path.
	Returns:
		frames (np.array): Numpy array of frames.
	"""
	video = cv2.VideoCapture(video_path)
	success,image = video.read()

	frames = []
	while success:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		frames.append(image)

		success,image = video.read()

	video.release()
	return np.array(frames, dtype=np.uint8)