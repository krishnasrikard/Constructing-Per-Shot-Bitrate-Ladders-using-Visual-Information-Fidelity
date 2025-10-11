"""
Calculating BD-Metrics from Bitrate and Quality Ladders
"""
# Importing Libraries
import numpy as np
np.set_printoptions(suppress=True)

import os, sys, warnings
sys.path.append("/home/kd28684/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity-Working")
import functions.IO_functions as IO_functions
import functions.extract_functions as extract_functions
import functions.bjontegaard_metric as bd_metrics
import functions.correction_algorithms as correction_algorithms
import defaults


# Create Rate-Quality Curve using Bitrate Ladder
def Rate_Quality_Curve_from_Bitrate_Ladder(
	RQ_pairs:dict,
	Bitrate_Ladder:dict
):
	"""
	Create Pareto-Front using Bitrate Ladder
	Args:
		RQ_pairs (dict): The rate-quality information of a video.
		Bitrate_Ladder (dict): The bitrate-ladder that should be used for construction of pareto-front.
	Returns:
		(dict): Pareto-Front with reoslutions as keys and (rate, quality) as values for each resolution.
		(list): Bitrate-Quality points on the pareto-front without resolution information.
	"""
	# Resolutions
	Resolutions = defaults.resolutions

	# Pareto-Front
	Pareto_Front = {}
	for res in defaults.resolutions:
		Pareto_Front[res] = []

	# Sorting Bitrate Ladder
	Bitrate_Ladder = dict(sorted(Bitrate_Ladder.items(), reverse=True))
	# print (Bitrate_Ladder)
	# print ()

	# Thresholds
	min_q = defaults.min_quality
	max_q = defaults.max_quality
	min_b = defaults.min_bitrate
	max_b = defaults.max_bitrate

	# Adding Points to Pareto-Front
	for b_step in Bitrate_Ladder.keys():
		# Updating Thresholds
		min_b = b_step

		# Resolution the video should be encoded according to bitrate ladder
		res = Bitrate_Ladder[b_step]

		for rq_point in RQ_pairs[res]:
			if (rq_point[0] >= min_b and rq_point[0] <= max_b) and (rq_point[1] >= min_q and rq_point[1] <= max_q):

				if any(np.array_equal(np.round([rq_point[0], rq_point[1]], decimals=2), np.round(row, decimals=2)) for row in Pareto_Front[res]):
					None
				else:
					Pareto_Front[res].append([rq_point[0], rq_point[1]])
			else:
				None

		# Updating Thresholds
		data = np.asarray(Pareto_Front[res])
		if len(data) > 0:
			max_q = np.min(data[:,1])
			max_b = b_step

	Points = []
	for res in Resolutions:
		Pareto_Front[res].sort()
		Points += Pareto_Front[res]
		Pareto_Front[res] = np.asarray(Pareto_Front[res])
		# print (res)
		# print (Pareto_Front[res])

	Points.sort()
	Points = np.asarray(Points)
	# print ()
	# print (Points)
	# print ("-"*25)
	# print ()

	return Pareto_Front, Points


# Create Rate-Quality Curve using Quality Ladder
def Rate_Quality_Curve_from_Quality_Ladder(
	RQ_pairs:dict,
	Quality_Ladder:dict
):
	"""
	Create Pareto-Front using Quality Ladder
	Args:
		RQ_pairs (dict): The rate-quality information of a video.
		Quality_Ladder (dict): The quality-ladder that should be used for construction of pareto-front.
	Returns:
		(dict): Pareto-Front with reoslutions as keys and (rate, quality) as values for each resolution.
		(list): Bitrate-Quality points on the pareto-front without resolution information.
	"""
	# Resolutions
	Resolutions = defaults.resolutions

	# Pareto-Front
	Pareto_Front = {}
	for res in defaults.resolutions:
		Pareto_Front[res] = []

	# Sorting Quality Ladder
	Quality_Ladder = dict(sorted(Quality_Ladder.items(), reverse=True))
	# print (Quality_Ladder)
	# print ()

	# Thresholds
	min_q = defaults.min_quality
	max_q = defaults.max_quality
	min_b = defaults.min_bitrate
	max_b = defaults.max_bitrate

	# Adding Points to Pareto-Front
	for q_step in Quality_Ladder.keys():
		# Updating Thresholds
		min_q = q_step*100

		# Resolution the video should be encoded according to quality ladder
		res = Quality_Ladder[q_step]

		for rq_point in RQ_pairs[res]:
			if (rq_point[0] >= min_b and rq_point[0] <= max_b) and (rq_point[1] >= min_q and rq_point[1] <= max_q):

				if any(np.array_equal(np.round([rq_point[0], rq_point[1]], decimals=2), np.round(row, decimals=2)) for row in Pareto_Front[res]):
					None
				else:
					Pareto_Front[res].append([rq_point[0], rq_point[1]])
			else:
				None

		# Updating Thresholds
		data = np.asarray(Pareto_Front[res])
		if len(data) > 0:
			max_q = q_step*100
			max_b = np.min(data[:,0])

	Points = []
	for res in Resolutions:
		Pareto_Front[res].sort()
		Points += Pareto_Front[res]
		Pareto_Front[res] = np.asarray(Pareto_Front[res])
		# print (res)
		# print (Pareto_Front[res])

	Points.sort()
	Points = np.asarray(Points)
	# print ()
	# print (Points)
	# print ("-"*25)
	# print ()

	return Pareto_Front, Points


# Calculate BD-metrics for Bitrate Ladders
def Calculate_BD_metrics_for_Bitrate_Ladders(
	video_file:str,
	codec:str,
	preset:str,
	bitrate_ladder_path:str,
	fixed_bitrate_ladder_path:str,
	reference_bitrate_ladder_path:str
):
	"""
	Args:
		video_file (str): The video file name.
		codec (str): Codec used to generate RQ points that need to be extracted. Options: ["libx265", "libx264"]
		preset (str): Preset used to generate RQ points that need to be extracted. Options: ["slow", "medium", "fast", "veryfast", "ultrafast"]
		bitrate_ladder_path (str): The path to Bitrate Ladder that needs to be considered.
		fixed_bitrate_ladder_path (str): The path to Fixed Bitrate Ladder that needs to be considered.
		reference_bitrate_ladder_path (str): The path to Reference Bitrate Ladder that needs to be considered.
	Returns
		(float): BD-rate in percentage wrt Fixed Bitrate-Ladder
		(float): BD-quality wrt Fixed Bitrate-Ladder
		(float): BD-rate in percentage wrt Reference Bitrate-Ladder
		(float): BD-quality wrt Reference Bitrate-Ladder
	"""
	# Rate-Quality points
	RQ_pairs = extract_functions.Extract_RQ_Information(
		video_rq_points_info=IO_functions.read_create_jsonfile(os.path.join(defaults.rq_points_dataset_path, codec, preset, video_file, "crfs.json")),
		quality_metric="vmaf",
		resolutions=defaults.resolutions,
		CRFs=defaults.CRFs,
		QPs=None,
		min_quality=defaults.min_quality,
		max_quality=defaults.max_quality,
		min_bitrate=defaults.min_bitrate,
		max_bitrate=defaults.max_bitrate,
		set_bitrate_log_base=2
	)

	# Fixed Bitrate-Ladder
	AL = np.load(fixed_bitrate_ladder_path, allow_pickle=True)[()]
	AL = correction_algorithms.Top_Bottom(AL)

	# Reference Bitrate Ladder
	RL = np.load(reference_bitrate_ladder_path, allow_pickle=True)[()][video_file]
	RL = correction_algorithms.Top_Bottom(RL)

	# Predicted Bitrate Ladder
	BL = np.load(bitrate_ladder_path, allow_pickle=True)[()][video_file]
	BL = correction_algorithms.Top_Bottom(BL)

	# Constructing Pareto-Fronts and Converting Bitrate to normal scale from log-scale
	_, Fixed_RQ_points = Rate_Quality_Curve_from_Bitrate_Ladder(
		RQ_pairs=RQ_pairs,
		Bitrate_Ladder=AL
	)
	Fixed_RQ_points[:,0] = np.round(np.power(2, Fixed_RQ_points[:,0]), decimals=3)

	_, Reference_RQ_points = Rate_Quality_Curve_from_Bitrate_Ladder(
		RQ_pairs=RQ_pairs,
		Bitrate_Ladder=RL
	)
	Reference_RQ_points[:,0] = np.round(np.power(2, Reference_RQ_points[:,0]), decimals=3)

	_, RQ_points = Rate_Quality_Curve_from_Bitrate_Ladder(
		RQ_pairs=RQ_pairs,
		Bitrate_Ladder=BL
	)
	RQ_points[:,0] = np.round(np.power(2, RQ_points[:,0]), decimals=3)


	# Assertions
	assert (np.all(Fixed_RQ_points[:,0] <= defaults.max_bitrate) and np.all(Fixed_RQ_points[:,0] >= defaults.min_bitrate)) and (np.all(Fixed_RQ_points[:,1] <= defaults.max_quality) and np.all(Fixed_RQ_points[:,1] >= defaults.min_quality)), "Fixed Bitrate Ladder Pareto-Front points are in the wrong range."

	assert (np.all(Reference_RQ_points[:,0] <= defaults.max_bitrate) and np.all(Reference_RQ_points[:,0] >= defaults.min_bitrate)) and (np.all(Reference_RQ_points[:,1] <= defaults.max_quality) and np.all(Reference_RQ_points[:,1] >= defaults.min_quality)), "Reference Bitrate Ladder Pareto-Front points are in the wrong range."

	assert (np.all(RQ_points[:,0] <= defaults.max_bitrate) and np.all(RQ_points[:,0] >= defaults.min_bitrate)) and (np.all(RQ_points[:,1] <= defaults.max_quality) and np.all(Reference_RQ_points[:,1] >= defaults.min_quality)), "Bitrate Ladder Pareto-Front points are in the wrong range."


	# BD-Metrics wrt Apple Fixed Bitrate Ladder
	f_bd_rate = bd_metrics.BD_Rate(
		R1=Fixed_RQ_points[:,0],
		Q1=Fixed_RQ_points[:,1],
		R2=RQ_points[:,0],
		Q2=RQ_points[:,1],
		piecewise=True
	)
	f_bd_quality = bd_metrics.BD_Quality(
		R1=Fixed_RQ_points[:,0],
		Q1=Fixed_RQ_points[:,1],
		R2=RQ_points[:,0],
		Q2=RQ_points[:,1],
		piecewise=True
	)

	# BD-Metrics wrt Reference Bitrate Ladder
	r_bd_rate = bd_metrics.BD_Rate(
		R1=Reference_RQ_points[:,0],
		Q1=Reference_RQ_points[:,1],
		R2=RQ_points[:,0],
		Q2=RQ_points[:,1],
		piecewise=True
	)
	r_bd_quality = bd_metrics.BD_Quality(
		R1=Reference_RQ_points[:,0],
		Q1=Reference_RQ_points[:,1],
		R2=RQ_points[:,0],
		Q2=RQ_points[:,1],
		piecewise=True
	)

	return np.round(f_bd_rate, decimals=4), np.round(f_bd_quality, decimals=4), np.round(r_bd_rate, decimals=4), np.round(r_bd_quality, decimals=4)


# Calculate BD-metrics for Quality Ladders
def Calculate_BD_metrics_for_Quality_Ladders(
	video_file:str,
	codec:str,
	preset:str,
	quality_ladder_path:str,
	fixed_bitrate_ladder_path:str,
	reference_bitrate_ladder_path:str
):
	"""
	Args:
		video_file (str): The video file name.
		codec (str): Codec used to generate RQ points that need to be extracted. Options: ["libx265", "libx264"]
		preset (str): Preset used to generate RQ points that need to be extracted. Options: ["slow", "medium", "fast", "veryfast", "ultrafast"]
		quality_ladder_path (str): The path to Quality Ladder that needs to be considered.
		fixed_bitrate_ladder_path (str): The path to Fixed Bitrate Ladder that needs to be considered.
		reference_bitrate_ladder_path (str): The path to Reference Bitrate Ladder that needs to be considered.
	Returns
		(float): BD-rate in percentage wrt Fixed Bitrate-Ladder
		(float): BD-quality wrt Fixed Bitrate-Ladder
		(float): BD-rate in percentage wrt Reference Bitrate-Ladder
		(float): BD-quality wrt Reference Bitrate-Ladder
	"""
	# Rate-Quality points
	RQ_pairs = extract_functions.Extract_RQ_Information(
		video_rq_points_info=IO_functions.read_create_jsonfile(os.path.join(defaults.rq_points_dataset_path, codec, preset, video_file, "crfs.json")),
		quality_metric="vmaf",
		resolutions=defaults.resolutions,
		CRFs=defaults.CRFs,
		QPs=None,
		min_quality=defaults.min_quality,
		max_quality=defaults.max_quality,
		min_bitrate=defaults.min_bitrate,
		max_bitrate=defaults.max_bitrate,
		set_bitrate_log_base=2
	)

	# Fixed Bitrate-Ladder
	AL = np.load(fixed_bitrate_ladder_path, allow_pickle=True)[()]
	AL = correction_algorithms.Top_Bottom(AL)

	# Reference Bitrate Ladder
	RL = np.load(reference_bitrate_ladder_path, allow_pickle=True)[()][video_file]
	RL = correction_algorithms.Top_Bottom(RL)

	# Predicted Quality Ladder
	QL = np.load(quality_ladder_path, allow_pickle=True)[()][video_file]
	QL = correction_algorithms.Bottom_Top(QL)

	# Constructing Pareto-Fronts and Converting Bitrate to normal scale from log-scale
	_, Fixed_RQ_points = Rate_Quality_Curve_from_Bitrate_Ladder(
		RQ_pairs=RQ_pairs,
		Bitrate_Ladder=AL
	)
	Fixed_RQ_points[:,0] = np.round(np.power(2, Fixed_RQ_points[:,0]), decimals=3)

	_, Reference_RQ_points = Rate_Quality_Curve_from_Bitrate_Ladder(
		RQ_pairs=RQ_pairs,
		Bitrate_Ladder=RL
	)
	Reference_RQ_points[:,0] = np.round(np.power(2, Reference_RQ_points[:,0]), decimals=3)

	_, RQ_points = Rate_Quality_Curve_from_Quality_Ladder(
		RQ_pairs=RQ_pairs,
		Quality_Ladder=QL
	)
	RQ_points[:,0] = np.round(np.power(2, RQ_points[:,0]), decimals=3)


	# Assertions
	assert (np.all(Fixed_RQ_points[:,0] <= defaults.max_bitrate) and np.all(Fixed_RQ_points[:,0] >= defaults.min_bitrate)) and (np.all(Fixed_RQ_points[:,1] <= defaults.max_quality) and np.all(Fixed_RQ_points[:,1] >= defaults.min_quality)), "Fixed Bitrate Ladder Pareto-Front points are in the wrong range."

	assert (np.all(Reference_RQ_points[:,0] <= defaults.max_bitrate) and np.all(Reference_RQ_points[:,0] >= defaults.min_bitrate)) and (np.all(Reference_RQ_points[:,1] <= defaults.max_quality) and np.all(Reference_RQ_points[:,1] >= defaults.min_quality)), "Reference Bitrate Ladder Pareto-Front points are in the wrong range."

	assert (np.all(RQ_points[:,0] <= defaults.max_bitrate) and np.all(RQ_points[:,0] >= defaults.min_bitrate)) and (np.all(RQ_points[:,1] <= defaults.max_quality) and np.all(Reference_RQ_points[:,1] >= defaults.min_quality)), "Bitrate Ladder Pareto-Front points are in the wrong range."


	# BD-Metrics wrt Apple Fixed Bitrate Ladder
	f_bd_rate = bd_metrics.BD_Rate(
		R1=Fixed_RQ_points[:,0],
		Q1=Fixed_RQ_points[:,1],
		R2=RQ_points[:,0],
		Q2=RQ_points[:,1],
		piecewise=True
	)
	f_bd_quality = bd_metrics.BD_Quality(
		R1=Fixed_RQ_points[:,0],
		Q1=Fixed_RQ_points[:,1],
		R2=RQ_points[:,0],
		Q2=RQ_points[:,1],
		piecewise=True
	)

	# BD-Metrics wrt Reference Bitrate Ladder
	r_bd_rate = bd_metrics.BD_Rate(
		R1=Reference_RQ_points[:,0],
		Q1=Reference_RQ_points[:,1],
		R2=RQ_points[:,0],
		Q2=RQ_points[:,1],
		piecewise=True
	)
	r_bd_quality = bd_metrics.BD_Quality(
		R1=Reference_RQ_points[:,0],
		Q1=Reference_RQ_points[:,1],
		R2=RQ_points[:,0],
		Q2=RQ_points[:,1],
		piecewise=True
	)

	return np.round(f_bd_rate, decimals=4), np.round(f_bd_quality, decimals=4), np.round(r_bd_rate, decimals=4), np.round(r_bd_quality, decimals=4)


# Calculating Closeness to Performance of Reference Bitrate Ladder
def Calculate_Closeness(
	Predicted_Metrics:any,
	Reference_Metrics:any,
):
	"""
	Args:
		Predicted_Metrics (any): BD-Metrics of Predicted Bitrate Ladder against Fixed Bitrate Ladder.
		Reference_Metrics (any): BD-Metrics of Reference Bitrate Ladder against Fixed Bitrate Ladder.
	Returns:
		(int): Returns 1, if f_{25} is True
		(int): Returns 1, if f_{50} is True
		(int): Returns 1, if f_{75} is True
		(int): Returns 1, if f_{85} is True 
	"""
	# Closeness before Correction
	# Calculating fraction of samples close to reference bitrate ladder performance
	if Predicted_Metrics[0] < 0.25*Reference_Metrics[0] and Predicted_Metrics[1] > 0.25*Reference_Metrics[1]:
		f_25 = 1
	else:
		f_25 = 0

	if Predicted_Metrics[0] < 0.50*Reference_Metrics[0] and Predicted_Metrics[1] > 0.50*Reference_Metrics[1]:
		f_50 = 1
	else:
		f_50 = 0

	if Predicted_Metrics[0] < 0.75*Reference_Metrics[0] and Predicted_Metrics[1] > 0.75*Reference_Metrics[1]:
		f_75 = 1
	else:
		f_75 = 0

	if Predicted_Metrics[0] < 0.85*Reference_Metrics[0] and Predicted_Metrics[1] > 0.85*Reference_Metrics[1]:
		f_85 = 1
	else:
		f_85 = 0

	return f_25, f_50, f_75, f_85