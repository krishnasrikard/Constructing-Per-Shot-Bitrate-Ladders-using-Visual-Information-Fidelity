import numpy as np
import math
from scipy.interpolate import CubicHermiteSpline

import torch

import os
import json

def dydx(x, y, return_size="same"):
	"""
	Returns slopes by calculating dy/dx i.e (y2-y1)/(x2-x1)
	Args:
		x (list/np.array): Values of x
		y (list/np.array): Values of y
		return_size (str): Options: ["same", "valid"], return size of slopes. If set to "valid", returns slopes of length len(x)-1 slope values. Else returns slopes of size len(x) by appending the last value again.
	"""
	den = np.diff(x)
	num = np.diff(y)

	slope = list(np.divide(num,den+1e-8))
	if return_size == "same":
		slope.append(slope[-1])
	else:
		None

	# Handling Nans in slope
	for i in range(len(slope)):
		if math.isnan(slope[i]):
			slope[i] = slope[i-1]
	
	slope = np.asarray(slope)
	
	# Assertion
	assert (return_size == "same") and (slope.shape[0] == y.shape[0]), "dy/dx and y do not have same shape."

	return slope


def Find_Intersection(
	f:np.array=None,
	g:np.array=None,
	x:np.array=None,
	use_interpolation:bool=True
):
	"""
	Finding intersection between two functions f and g for given values of x using Cubic-Hermite Interpolation.
	Args:
		f (np.array): 2D numpy array with f[:,0] containing x values and f[:,1] containing output of the function. (Default: None)
		g (np.array): 2D numpy array with f[:,0] containing x values and f[:,1] containing output of the function. (Default: None)
		x (np.array): Values of x for which the intersection point needs to be found. (Default: None)
		use_interpolation (bool): Whether or not to use interpolation. (Default: True)
	Returns:
		f_new (np.array): Interpolated values of function "f" for values of x.
		g_new (np.array): Interpolated values of function "g" for values of x.
		(bool): True if f > g after last intersection point or if f > g when there is no intersection point. False if f < g for the cases mentioned.
	"""
	if use_interpolation == True:
		# Checking if Inputs are in increasing sequence
		if np.all(np.diff(f[:,0]) >= 0) and np.all(np.diff(f[:,1]) >= 0) and np.all(np.diff(g[:,0]) >= 0) and np.all(np.diff(g[:,1]) >= 0):
			None
		else:
			assert False, "Non-Increasing Sequence Found!!!"

		f_Function = CubicHermiteSpline(f[:,0], f[:,1], dydx=dydx(f[:,0],f[:,1]))
		g_Function = CubicHermiteSpline(g[:,0], g[:,1], dydx=dydx(g[:,0],g[:,1]))

		f_new = np.round(f_Function(x), decimals=3)
		g_new = np.round(g_Function(x), decimals=3)
	else:
		assert np.array_equal(f[:,0], g[:,0]), "Interpolation not used. The 'x' values of functions 'f' and 'g' do not match."
		f_new = f[:,1]
		g_new = g[:,1]
	
	sign = np.sign(f_new - g_new)
	diff = np.diff(sign)
	idx = np.argwhere(diff).flatten()

	if idx.shape[0] == 0:
		if np.all(sign[-5:,] >= 0):
			# No-intersection and f > g for given values of x
			return (np.asarray(f_new),np.asarray(g_new), True, [])
		else:
			# No-intersection and f < g for given values of x
			return (np.asarray(f_new),np.asarray(g_new), False, [])
	else:
		if diff[idx[-1]] > 0:
			# Intersection and after final intesection point idx[-1], f > g for given values of x.
			return (np.asarray(f_new),np.asarray(g_new), True, [idx[-1]])
		else:
			# Intersection and after final intesection point idx[-1], f < g for given values of x.
			return (np.asarray(f_new),np.asarray(g_new), False, [idx[-1]])


def upscaling_command(
	input_video_path:str=None,
	output_video_path:str=None,
	vmaf_resolution:tuple=(1920,1080),
	codec:str=None,
	preset:str=None,
	ffmpeg_path:str="ffmpeg/ffmpeg-6.0-amd64-static"
):
	"""
	Function to generate ffmpeg command for bitrate estimation of a video.
	Args:
		input_video_path (str): The path to the input file. (Default: None)
		output_video_path (str): The path to the output file.
		vmaf_resolution (tuple): The resolution at which VMAF is calculated. (Default: (1920,1080))
		codec (str): The encoder/decoder video codec. (Default: None)
		preset (str): A preset of the encoder. (Default: None)
		ffmpeg_path (str): The path to ffmpeg. (Default: "ffmpeg/ffmpeg-6.0-amd64-static")
	Returns:
		cmd (str): The ffmpeg command to estimate bitrate of the given video.
	"""
	
	# The input video
	assert os.path.exists(input_video_path), "Provide a valid input_video_path."

	cmd = "{}/ffmpeg -i {} -vf scale={}x{}:flags=lanczos -codec:v {} -preset {} -qp 0 -y {}".format(ffmpeg_path,input_video_path,vmaf_resolution[0],vmaf_resolution[1],codec,preset,output_video_path)

	return cmd