import numpy as np
from scipy.interpolate import CubicHermiteSpline

import os, sys, warnings
import pickle
from tqdm import tqdm
warnings.filterwarnings("ignore")
sys.path.append("/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity")
import functions.utils as utils

def Pareto_Front_Points(
	RQ_pairs:dict=None,
	Resolutions:list=None,
	use_interpolated_points=False
):
	"""
	Extracting points from RQ_pairs to generate Pareto-Front. Cubic-Hermite Interpolation functions for various resolutions are used to compare mutiple resolutions at different bitrates.
	Args:
		RQ_pairs (dict): Dictionary with resolution as keys containing (bitrate (in kbps), quality) points extracted from selected rate_control json file. (Default: None)
		Resolutions (list): Resolutions that needs to be considered while plotting RQ points. (Default: None)
		use_interpolated_points (bool): Whether to use interpolated points or not. If True, it doesn't return CRF values of each RQ-point in ParetoFront_pairs and ParetoFront_info. (Default: False)
	Returns:
		ParetoFront_pairs (dict): Dictionary with resolution as keys containing (bitrate (in kbps), quality, rate_control_setting_value) points on pareto-front.
	"""
	# All the bitrates obtained from encoding with various resolution and rc pairs of a particular video are considered.
	# Cubic-Hermite Interpolation functions for various resolutions.

	all_bitrates = []
	RQ_pairs_Interpolation_Functions = {}
	for res in Resolutions:
		# Need atleast 2 RQ points for a RQ curve of a resolution
		if len(RQ_pairs[res]) > 1:
			bitrate = RQ_pairs[res][:,0]
			quality = RQ_pairs[res][:,1]

			RQ_pairs_Interpolation_Functions[res] = CubicHermiteSpline(bitrate, quality, dydx=utils.dydx(bitrate, quality))

		if len(RQ_pairs[res]) > 0:
			all_bitrates.append(np.asarray(RQ_pairs[res])[:,0])

	# All Bitrates
	all_bitrates = np.concatenate(all_bitrates, axis=0)
	all_bitrates = np.sort(all_bitrates)

	# Extracting points on Pareto-Front
	ParetoFront_pairs = {res:[] for res in Resolutions}

	for _,bitrate in enumerate(all_bitrates):
		optimal_rq_pair = []
		optimal_res = (-1,-1)

		for _,res in enumerate(Resolutions):
			if len(RQ_pairs[res]) > 1:
				if bitrate <= np.max(RQ_pairs[res][:,0]) and bitrate >= np.min(RQ_pairs[res][:,0]):
					# Interpolated Quality
					quality = RQ_pairs_Interpolation_Functions[res](bitrate)

					# Optimal (Rate,Quality) pair
					if len(optimal_rq_pair) == 0:
						optimal_rq_pair = [bitrate, quality]
						optimal_res = res
					else:
						if optimal_rq_pair[1] < quality:
							# Always the best quality is considered
							optimal_rq_pair = [bitrate, quality]
							optimal_res = res
						elif np.round(optimal_rq_pair[1], decimals=2) == np.round(quality, decimals=2):
							# If at the same bitrates, two resolutions have same quality, the highest resolution is considered
							if optimal_res[0] < res[0]:
								optimal_rq_pair = [bitrate, quality]
								optimal_res = res
						else:
							None

		# Selecting Points on Pareto-Front
		if use_interpolated_points == False:
			RQ_data = RQ_pairs[optimal_res][:,0:2]
			Optimal_data = np.asarray(optimal_rq_pair)

			if any(np.array_equal(np.round(Optimal_data, decimals=2), np.round(row, decimals=2)) for row in RQ_data):
				mask = np.all(np.round(Optimal_data, decimals=2) == np.round(RQ_data, decimals=2), axis=1)
				idx = np.argmax(mask)
				
				ParetoFront_pairs[optimal_res].append(list(RQ_pairs[optimal_res][idx]))
		else:
			ParetoFront_pairs[optimal_res].append(optimal_rq_pair)


	# Sorting (R,Q,rc) pairs by bitrate
	for res in Resolutions:
		ParetoFront_pairs[res].sort()
		ParetoFront_pairs[res] = np.asarray(ParetoFront_pairs[res])

	return ParetoFront_pairs


def reject_outliers(data, m=2):
	d = data - np.median(data)
	mdev = np.median(np.abs(d))
	s = d / (mdev if mdev else 1.0)
	return np.logical_and(s <= m, s >= -m)


def Correct_Pareto_Front(
	RQ_pairs:dict,
):
	"""
	Correcting the pareto-front so that
	- Resolutions are in ascending order
	- There is no overlap between resolutions. Let R2 be high resolution and R1 be lower resolution. 
		- RQ-Points from R2 start after highest bitrate of R1.
		- If RQ-points of R2 exist below highest bitrate of R1, they are removed from RQ_pairs.
		- This is considered for smooth continuity of pareto-front.
	Args:
		RQ_pairs (dict): Dictionary with resolution as keys containing (bitrate (in kbps), quality, rate_control_setting_value) points extracted from selected rate_control json file.
	Returns:
		Updated_RQ_pairs (dict): Updated RQ_pairs after correcting the Pareto-Front.
	"""
	# Resolutions
	Resolutions = list(RQ_pairs.keys())
	Resolutions.sort(reverse=True)

	# Points on Pareto-Front
	Updated_RQ_pairs = {res:[] for res in Resolutions}

	# Starting Bitrate
	next_max_bitrate = np.inf

	for _,res in enumerate(Resolutions):
		# Removing Outliers
		if len(RQ_pairs[res]) > 0:
			mask = reject_outliers(data=np.copy(RQ_pairs[res][:,0]), m=3)
			Data = np.copy(RQ_pairs[res])[mask]
		else:
			Data = np.copy(RQ_pairs[res])
		
		# Selecting Points on Pareto-Front
		for _,data in enumerate(Data):
			if data[0] > next_max_bitrate:
				None
			else:
				Updated_RQ_pairs[res].append(data.tolist())

		if len(Updated_RQ_pairs[res]) > 0:
			Updated_RQ_pairs[res].sort()
			Updated_RQ_pairs[res] = np.asarray(Updated_RQ_pairs[res])
			next_max_bitrate = np.min(Updated_RQ_pairs[res][:,0])
		else:
			Updated_RQ_pairs[res] = np.array([])

	return Updated_RQ_pairs