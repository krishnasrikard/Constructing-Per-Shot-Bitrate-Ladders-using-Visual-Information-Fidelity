import numpy as np

import os, sys, warnings
import pickle
from tqdm import tqdm
warnings.filterwarnings("ignore")
sys.path.append("/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity")
import functions.utils as utils

def CrossOver_Quality(
	RQ_pairs:dict=None,
	Resolutions:list=None
):
	"""
	Estimating cross-over quality points.
	Args:
		RQ_pairs (dict): Dictionary with resolution as keys containing (bitrate (in kbps), quality, rate_control_setting_value) points extracted from selected rate_control json file. (Default: None)
		Resolutions (list): Resolutions that needs to be considered while plotting RQ points. (Default: None)
	Returns:
		crossover_qualites (np.array): The cross-over quality values between adjacent high resolutions to low-resolutions.
	"""
	Resolutions.sort(reverse=True)
	crossover_qualites = np.inf*np.ones((len(Resolutions)-1,))

	for i in range(len(Resolutions)-1):
		f = RQ_pairs[Resolutions[i]]
		g = RQ_pairs[Resolutions[i+1]]

		assert Resolutions[i][0] >= Resolutions[i+1][0] and Resolutions[i][0] >= Resolutions[i+1][1], "Wrong order of Resolutions"

		if f.shape[0] <= 1 or g.shape[0] <= 1:
			continue

		if np.min(f[:,0]) >= np.max(g[:,0]):
			# Doesn't have a common bitrate range, then the lowest quality of higher resolution is considered as cross-over quality
			crossover_qualites[i] = np.min(f[:,1])
		else:			
			# start and end shows bitrate range of intersection between two resolutions
			b_start = max(np.min(f[:,0]),np.min(g[:,0]))
			b_end = min(np.max(f[:,0]),np.max(g[:,0]))
			num_points = 50*np.ceil(b_end-b_start).astype(int)
			x = np.round(np.linspace(start=b_start, stop=b_end, num=num_points, endpoint=True), decimals=4)

			results = utils.Find_Intersection(f,g,x)
			# print (results[2], len(results[3]), x[results[3]])
			if results[2] == True and len(results[3]) > 0:
				# If after the final intersection, higher resolution has higher quality than lower resolutions, considering intersection bitrate as cross-over bitrate.
				crossover_bitrate = x[results[3]]
			elif results[2] == False and len(results[3]) > 0:
				# If after the final intersection, lower resolution has higher quality than higher resolutions, considering highest bitrate where overlap ends is considered as cross-over bitrate.
				crossover_bitrate = b_end
			elif results[2] == False and len(results[3]) == 0:
				# If no intersection is found between lower and higher resolutions and lowest resolution dominates, considering highest bitrate where overlap ends as cross-over bitrate.
				crossover_bitrate = b_end
			else:
				# If no intersection is found between lower and higher resolutions and highest resolution dominates, considering lowest bitrate where overlap starts as cross-over bitrate.
				crossover_bitrate = b_start

			assert crossover_bitrate >= b_start and crossover_bitrate <= b_end, "The values of cross-over bitrate did not lie in the intersection bitrate region."

			# Selecting cross-over quality based on quality of high resolution RQ curve at cross-over bitrate
			idx_crossover_bitrate = np.argmin(np.abs(x - crossover_bitrate))
			crossover_qualites[i] = results[0][idx_crossover_bitrate]

	# Rounding Qualities
	crossover_qualites = np.round(crossover_qualites, decimals=4)

	# Imposing Monotonicity on estimated cross-over qualites
	for i in range(1,len(crossover_qualites)):
		crossover_qualites[i] = min(crossover_qualites[i], crossover_qualites[i-1])

	return crossover_qualites


def CrossOver_Quality_Pareto_Front_Points(
	RQ_pairs:dict=None,
	RQ_info:list=None,
):
	"""
	Constructing Pareto-Front from cross-over quality points.
	Args:
		RQ_pairs (dict): Dictionary with resolution as keys containing (bitrate (in kbps), quality) points extracted from selected rate_control json file. (Default: None)
	Returns:
		CO_Quality_RQ_pairs (dict): Dictionary with resolution as keys containing (bitrate (in kbps), quality, rate_control_setting_value) points on pareto-front.
		CO_Quality_RQ_info (list): List of lists containing (bitrate (in kbps), quality, resolution, rate_control_setting_value) points points on pareto-front.
	"""
	Resolutions = list(RQ_pairs.keys())
	Resolutions.sort(reverse=True)

	co_quality = CrossOver_Quality(
		RQ_pairs=RQ_pairs,
		Resolutions=Resolutions
	)

	CO_Quality_RQ_pairs = {res:[] for res in Resolutions}
	CO_Quality_RQ_info = RQ_info.copy()

	for i,res in enumerate(Resolutions):
		for j,data in enumerate(RQ_pairs[res]):
			# When quality >= cross-over-quality, higher-resolution to corresponding cross-over-quality is selected.
			if i==0:
				if data[1] >= co_quality[i]:
					CO_Quality_RQ_pairs[res].append(data.tolist())
				else:
					CO_Quality_RQ_info.remove([data[0],data[1],res,data[2]])
			elif i<=len(Resolutions)-2:
				if data[1] >= co_quality[i] and data[1] < co_quality[i-1]:
					CO_Quality_RQ_pairs[res].append(data.tolist())
				else:
					CO_Quality_RQ_info.remove([data[0],data[1],res,data[2]])
			else:
				if data[1] < co_quality[i-1]:
					CO_Quality_RQ_pairs[res].append(data.tolist())
				else:
					CO_Quality_RQ_info.remove([data[0],data[1],res,data[2]])				

		if len(CO_Quality_RQ_pairs[res]) > 0:
			CO_Quality_RQ_pairs[res].sort()
			CO_Quality_RQ_pairs[res] = np.array(CO_Quality_RQ_pairs[res])
		else:
			CO_Quality_RQ_pairs[res] = np.array([])
			
	return CO_Quality_RQ_pairs,CO_Quality_RQ_info