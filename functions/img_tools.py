"""
Source: https://github.com/utlive/live_python_qa/blob/main/utils.py
"""
import numpy as np

def integral_image(x):
	M, N = x.shape
	int_x = np.zeros((M+1, N+1))
	int_x[1:, 1:] = np.cumsum(np.cumsum(x, 0), 1)
	return int_x

def im2col(img, k, stride=1):
	# Parameters
	m, n = img.shape
	s0, s1 = img.strides
	nrows = m - k + 1
	ncols = n - k + 1
	shape = (k, k, nrows, ncols)
	arr_stride = (s0, s1, s0, s1)

	ret = np.lib.stride_tricks.as_strided(img, shape=shape, strides=arr_stride)
	return ret[:, :, ::stride, ::stride].reshape(k*k, -1)

def moments(x, y, k, stride, padding=None):
	kh = kw = k

	k_norm = k**2

	if padding is None:
		x_pad = x
		y_pad = y
	else:
		x_pad = np.pad(x, int((kh - stride)/2), mode=padding)
		y_pad = np.pad(y, int((kh - stride)/2), mode=padding)

	int_1_x = integral_image(x_pad)
	int_1_y = integral_image(y_pad)

	mu_x = (int_1_x[:-kh:stride, :-kw:stride] - int_1_x[:-kh:stride, kw::stride] - int_1_x[kh::stride, :-kw:stride] + int_1_x[kh::stride, kw::stride]) / k_norm
	mu_y = (int_1_y[:-kh:stride, :-kw:stride] - int_1_y[:-kh:stride, kw::stride] - int_1_y[kh::stride, :-kw:stride] + int_1_y[kh::stride, kw::stride]) / k_norm

	int_2_x = integral_image(x_pad**2)
	int_2_y = integral_image(y_pad**2)

	int_xy = integral_image(x_pad*y_pad)

	var_x = (int_2_x[:-kh:stride, :-kw:stride] - int_2_x[:-kh:stride, kw::stride] - int_2_x[kh::stride, :-kw:stride] + int_2_x[kh::stride, kw::stride]) / k_norm - mu_x**2
	var_y = (int_2_y[:-kh:stride, :-kw:stride] - int_2_y[:-kh:stride, kw::stride] - int_2_y[kh::stride, :-kw:stride] + int_2_y[kh::stride, kw::stride]) / k_norm - mu_y**2

	cov_xy = (int_xy[:-kh:stride, :-kw:stride] - int_xy[:-kh:stride, kw::stride] - int_xy[kh::stride, :-kw:stride] + int_xy[kh::stride, kw::stride]) / k_norm - mu_x*mu_y

	# Correcting negative values of variance.
	mask_x = (var_x < 0)
	mask_y = (var_y < 0)

	var_x[mask_x] = 0
	var_y[mask_y] = 0

	# If either variance was negative, it has been set to zero.
	# So, the correponding covariance should also be zero.
	cov_xy[mask_x | mask_y] = 0

	return (mu_x, mu_y, var_x, var_y, cov_xy)