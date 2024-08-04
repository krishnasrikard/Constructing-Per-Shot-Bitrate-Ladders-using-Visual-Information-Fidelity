# Importing Libraries
import numpy as np

import os, sys, warnings
import pickle
from tqdm import tqdm
warnings.filterwarnings("ignore")
sys.path.append("/home/krishna/Efficient-Dynamic-Optimizer-using-Visual-Information-Fidelity-Working")
import functions.IO_functions as IO_functions
import functions.extract_functions as extract_functions
import functions.pareto_front_points as pareto_front_points
import defaults


# Reference Bitrate Ladder
def Reference_Bitrate_Ladder(
	video_file:str,
	codec:str,
	preset:str,
	bitrates:list
):
	"""
	Returns Bitrate-Ladder for corresponding bitrates using Reference Pareto Front constructed
	Args:
		video_file (str): The video file name.
		codec (str): Codec used to generate RQ points that need to be extracted. Options: ["libx265", "libx264"]
		preset (str): Preset used to generate RQ points that need to be extracted. Options: ["slow", "medium", "fast", "veryfast", "ultrafast"]
		bitrates (list): List of bitrates present to be in the bitrate ladder.
	"""
	# Resolutions
	Resolutions = defaults.resolutions
	Resolutions.sort(reverse=True)
	bitrates.sort(reverse=True)

	# Reading RQ-Info of video file
	video_rq_points_info = IO_functions.read_create_jsonfile(os.path.join(defaults.rq_points_dataset_path, codec, preset, video_file, "crfs.json"))

	# Rate-Quality Dataset
	RQ_pairs = extract_functions.Extract_RQ_Information(
		video_rq_points_info=video_rq_points_info,
		quality_metric="vmaf",
		resolutions=defaults.resolutions,
		CRFs=defaults.CRFs,
		bitrates=None,
		QPs=None,
		min_quality=defaults.min_quality,
		max_quality=defaults.max_quality,
		min_bitrate=defaults.min_bitrate,
		max_bitrate=defaults.max_bitrate,
		set_bitrate_log_base=2
	)

	# Constructing Pareto-Front
	ParetoFront_pairs = pareto_front_points.Pareto_Front_Points(
		RQ_pairs=RQ_pairs,
		Resolutions=defaults.resolutions,
		use_interpolated_points=False
	)
	ParetoFront_pairs = pareto_front_points.Correct_Pareto_Front(ParetoFront_pairs)

	# Cross-Over Bitrates
	CrossOver_Bitrates = []
	for i in range(len(Resolutions)-1):
		if ParetoFront_pairs[Resolutions[i]].shape[0] > 0:
			# Switching happens to higher resolution from cross-over bitrate
			CrossOver_Bitrates.append(np.min(ParetoFront_pairs[Resolutions[i]][:,0]))
		else:
			if i==0:
				# If Pareto-Front doesn't contain highest resolution, we assuming highest resolution dominates after infinity.
				# Generally, higher resolution can dominate lower resolution after for some smaller CRF value. But consider our quality constraints and CRF values, that CRF value doesn't lie in our experiments.
				CrossOver_Bitrates.append(np.inf)
			else:
				# If Pareto-Front doesn't contain a resolution, cross-over bitrate of previous highest resolution is cross-over bitrate of current resolution.
				CrossOver_Bitrates.append(CrossOver_Bitrates[-1])

	# Code to check the validity of Cross-Over bitrates created using Reference Bitrate Ladder.
	"""
	Info = []
	for k,v in ParetoFront_pairs.items():
		if len(v) > 0:
			Info.append(k)
		else:
			Info.append((0,0))
	for i in range(len(Info)-1):
		if Info[i][0] == 0:
			if i==0:
				None
			elif (CrossOver_Bitrates[i] == CrossOver_Bitrates[i-1]) == False:
				assert False, "Error"
	"""

	# Imposing Monotonicity on estimated cross-over bitrates
	for i in range(1,len(CrossOver_Bitrates)):
		CrossOver_Bitrates[i] = min(CrossOver_Bitrates[i], CrossOver_Bitrates[i-1])

	# Calculating Bitrate-Ladder
	Bitrate_Ladder = {}
	for i in range(len(bitrates)):
		# Switching happends to higher resolution when bitrate >= cross-over-bitrate of corresponding higher resolution.
		b = bitrates[i]
		Bitrate_Ladder[b] = Resolutions[0]

		Bitrate_Ladder[b] = None

		for j in range(1+len(CrossOver_Bitrates)):
			if (j==0) and (b >= CrossOver_Bitrates[j]):
				Bitrate_Ladder[b] = Resolutions[0]
			elif (j <= len(CrossOver_Bitrates)-1) and (CrossOver_Bitrates[j] <= b < CrossOver_Bitrates[j-1]):
				Bitrate_Ladder[b] = Resolutions[j]
			elif (j==len(CrossOver_Bitrates)) and (b < CrossOver_Bitrates[j-1]):
				Bitrate_Ladder[b] = Resolutions[-1]
			else:
				None

		if Bitrate_Ladder[b] is None:
			assert False, "Something is Wrong"
	
	return Bitrate_Ladder


def Construct_Reference_Bitrate_Ladder(
	video_files:list,
	codec:str,
	preset:str
):
	# Reference Bitrate Ladder i.e Exhaustive Encoding
	print ("-"*50)
	print ("Reference Bitrate Ladder i.e Exhaustive Encoding")
	print ()

	# Creating Bitrate-Ladders for each video-file
	Bitrate_Ladders = {}
	for video_file in video_files:
		BL = Reference_Bitrate_Ladder(
			video_file=video_file,
			codec=codec,
			preset=preset,
			bitrates=defaults.evaluation_bitrates
		)
		Bitrate_Ladders[video_file] = BL

	# Saving Bitrate-Ladder
	np.save("/home/krishna/Efficient-Dynamic-Optimizer-using-Visual-Information-Fidelity-Working/bitrate_ladder_construction/bitrate_ladders/standard/{}_reference_bitrate_ladder.npy".format(preset), Bitrate_Ladders)


# Fixed Bitrate Ladder
def Apple_Fixed_Bitrate_Ladder(
	bitrates:list
):
	"""
	Returns Bitrate-Ladder for corresponding bitrates using Apple's Bitrate Ladder
	Args:
		bitrates (list): List of bitrates present to be in the bitrate ladder.
	"""
	# Resolutions
	Resolutions = defaults.resolutions
	bitrates.sort(reverse=True)

	# Apple's Bitrate Ladder's Cross-Over Bitrate
	CrossOver_Bitrates = np.asarray([11600, 8100, 4500, 2400, 600, 300, 145])
	CrossOver_Bitrates = np.round(np.log2(1000*CrossOver_Bitrates), decimals=4)

	# Calculating Bitrate-Ladder
	Bitrate_Ladder = {}
	for i in range(len(bitrates)):
		# Switching happends to higher resolution when bitrate >= cross-over-bitrate of corresponding higher resolution.
		b = bitrates[i]

		Bitrate_Ladder[b] = None

		for j in range(1+len(CrossOver_Bitrates)):
			if (j==0) and (b >= CrossOver_Bitrates[j]):
				Bitrate_Ladder[b] = Resolutions[0]
			elif (j <= len(CrossOver_Bitrates)-1) and (CrossOver_Bitrates[j] <= b < CrossOver_Bitrates[j-1]):
				Bitrate_Ladder[b] = Resolutions[j]
			elif (j==len(CrossOver_Bitrates)) and (b < CrossOver_Bitrates[j-1]):
				Bitrate_Ladder[b] = Resolutions[-1]
			else:
				None

		if Bitrate_Ladder[b] is None:
			assert False, "Something is Wrong"

	return Bitrate_Ladder


def Construct_Fixed_Bitrate_Ladder():
	# Apple's Fixed Bitrate Ladder
	print ("-"*50)
	print ("Apple's Fixed Bitrate Ladder")
	print ()

	# Fixed Bitrate Ladder
	BL = Apple_Fixed_Bitrate_Ladder(
		bitrates=defaults.evaluation_bitrates
	)
	print (BL)

	# Saving Bitrate-Ladder
	np.save("/home/krishna/Efficient-Dynamic-Optimizer-using-Visual-Information-Fidelity-Working/bitrate_ladder_construction/bitrate_ladders/standard/fixed_bitrate_ladder.npy", BL)
