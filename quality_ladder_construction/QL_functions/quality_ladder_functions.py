import numpy as np
import matplotlib.pyplot as plt

import os, sys, warnings
import pickle
from tqdm import tqdm
sys.path.append("/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity")
import functions.bjontegaard_metric as bd_metrics
import functions.IO_functions as IO_functions
import functions.extract_functions as extract_functions
import functions.extract_features as extract_features
import functions.correction_algorithms as correction_algorithms
import functions.utils as utils
from bitrate_ladder_construction.BL_functions.bitrate_ladder_functions import Pareto_Front_from_Bitrate_Ladder
import quality_ladder_construction.QL_functions.dataset_evaluation_functions as dataset_evaluation_functions
import defaults


# Bitrate: Select Models and Features
def Bitrate_Select_Models_Features(
	video_file:str,
	models_path:str,
	design_type:str,
	features_names:list,
	temporal_low_level_features:bool,
	codec:str,
	preset:str,
	evaluation_qualities:list
):
	"""
	Function to select model and dataset

	Args:
		video_file (str): The video file name
		models_path (str): Path to models
		design_type: Type of features to use. Options:["ml_metadata", "ml_llf", "ml_vif_approach1", "dl_vif_approach1", "ml_ensemble_llfvif_approach1", "ml_ensemble_llfvif_approach1"]
		features_names (list): List of features to be considered when low-level features are used.
		codec (str): Codec used to generate RQ points that need to be extracted. Options: ["libx265", "libx264"]
		preset (str): Preset used to generate RQ points that need to be extracted. Options: ["slow", "medium", "fast", "veryfast", "ultrafast"]
		evaluation_qualities (list):  List of evaluation_qualities present to be in the quality ladder.
	returns:
		(list): List of Models used for prediction
		(list): Features after feature-selection
	"""
	# Design Type
	design_type_list = design_type.split("_")

	# No.of Rate-Control Parameters
	num_rate_control_parameters = len(evaluation_qualities)

	# Meta_Data and Meta_Information
	Meta_Data = []
	Meta_Information = {}

	# By Quality i.e (All Rs for Q1, All Rs for Q2, ....)
	num_samples = num_rate_control_parameters * len(defaults.resolutions)
	Meta_Information[video_file] = np.zeros((num_samples,5))

	for q in evaluation_qualities:
		for res in defaults.resolutions:
			Meta_Data.append([q,res[0]/3840,res[1]/3840])

	Meta_Data = np.asarray(Meta_Data)
	Meta_Information[video_file][:,[1,3,4]] = Meta_Data

	# Amount of Meta_Data used to predict bitrate
	num_samples = Meta_Data.shape[0]
	len_meta_data = Meta_Data.shape[1]


	## Mapping approach to function inputs
	Approach2Info = {
		"1": [["vif_info"], "per_scale"],
		"2": [["vif_info"], "per_subband"],
		"3": [["vif_info"], "per_eigen_value"],
		"4": [["vif_info", "mean_abs_frame_diff"], "per_scale"],
		"5": [["vif_info", "mean_abs_frame_diff"], "per_subband"],
		"6": [["vif_info", "mean_abs_frame_diff"], "per_eigen_value"],
		"7": [["vif_info", "mean_abs_frame_diff", "diff_vif_info"], "per_scale"],
		"8": [["vif_info", "mean_abs_frame_diff", "diff_vif_info"], "per_subband"],
		"9": [["vif_info", "mean_abs_frame_diff", "diff_vif_info"], "per_eigen_value"],
	}


	## Using Metadata to predict Quality of compressed video
	if design_type == "ml_metadata":
		# Creating Input
		X = Meta_Data
		
		# Loading Model
		path = os.path.join(models_path, "metadata/model_metadata.pkl")
		Model = pickle.load(open(path, "rb"))
		print ("Loading model from", path)

		# Type-Casting
		X = X.astype(np.float32)

		# Rounding
		X = np.round(X, decimals=4)

		# Returning
		return X, Model
	

	## Using Low-Level features to predict Quality of compressed video
	elif design_type == "ml_llf":
		# Names of Custom-Features
		if temporal_low_level_features:
			F1 = features_names
			F2 = list(defaults.per_frame_quality_texture_features.keys())
			custom_features_names = list(sorted(set(F1) & set(F2), key = F1.index))
		else:
			F1 = features_names
			F2 = list(defaults.quality_texture_features.keys())
			custom_features_names = list(sorted(set(F1) & set(F2), key = F1.index))
		
		features_names_without_custom = [x for x in features_names if x not in custom_features_names]

		# Extracting Low-Level Features
		features = extract_features.Extract_Low_Level_Features(
			features_names=features_names_without_custom,
			video_filenames=[video_file],
			temporal_low_level_features=temporal_low_level_features
		)

		# Extracting Custom Features
		custom_features = extract_features.Compute_Quality_Custom_Features(
			custom_features_names=custom_features_names,
			quality_metric="vmaf",
			Meta_Information=Meta_Information,
			temporal_low_level_features=temporal_low_level_features
		)

		# Creating Input
		LLF_Data = np.repeat(np.expand_dims(features[video_file], axis=0), num_samples, axis=0)
		Custom_Data = custom_features[video_file]

		if temporal_low_level_features:
			min_temporal_length = min(LLF_Data.shape[1], Custom_Data.shape[1])
			Meta_Data = np.repeat(np.expand_dims(Meta_Data, axis=1), min_temporal_length, axis=1)
			LLF_Data = LLF_Data[:,-min_temporal_length:,:]
			Custom_Data = Custom_Data[:,-min_temporal_length:,:]

		if len(custom_features_names) == 0:
			X = np.concatenate([LLF_Data,Meta_Data], axis=-1)
		else:
			X = np.concatenate([LLF_Data,Custom_Data,Meta_Data], axis=-1)

		# Type-Casting
		X = X.astype(np.float32)

		# Rounding
		X = np.round(X, decimals=4)
		
		# Loading Model
		path = os.path.join(models_path, "low_level_features/model_llf.pkl")
		Model = pickle.load(open(path, "rb"))
		print ("Loading model from", path)

		# Returning
		return X, Model
	

	## Using VIF features to predict quality
	elif "vif" in design_type_list:
		# Approach Number
		approach_number = design_type_list[-1][-1]

		# Features that should be used
		features_name = "vif_features"
		
		# Loading Model and Feature Indices
		path = os.path.join(models_path, features_name, "feature_indices_approach_{}.npy".format(approach_number))
		feature_indices = np.load(path, allow_pickle=True)[()]

		path = os.path.join(models_path, features_name, "model_approach_{}.pkl".format(approach_number))
		Model = pickle.load(open(path, "rb"))
		print ("Loading model from", path)

		# Settings when loading data
		per_frame = False
		per_frame_features_flatten = False

		# Creating Input
		X,_ = dataset_evaluation_functions.VIFFeatures_Bitrate_Dataset(
			codec=codec,
			preset=preset,
			quality_metric="vmaf",
			video_filenames=[video_file],
			Resolutions_Considered=defaults.resolutions,
			CRFs_Considered=[defaults.CRFs[0]],
			bitrates_Considered=None,
			QPs_Considered=None,
			vif_setting=Approach2Info[approach_number][1],
			vif_features_list=Approach2Info[approach_number][0],
			per_frame=per_frame,
			per_frame_features_flatten=per_frame_features_flatten,
			min_quality=-np.inf,
			max_quality=np.inf
		)
		X = np.repeat(X, num_rate_control_parameters, axis=0)

		if len(X.shape) == 2:
			X[...,-1*len_meta_data:] = Meta_Data
		else:
			X[...,-1*len_meta_data:] = np.repeat(np.expand_dims(Meta_Data, axis=1), X.shape[1], axis=1)
		
		# Type Casting
		X = X.astype(np.float32)

		# Rounding
		X = np.round(X, decimals=4)

		# Feature Selection
		if feature_indices is not None:
			X = X[...,feature_indices]

		# Returning
		return X, Model
		

	## Using Ensemble of Low-level features and VIF features to predict quality
	elif design_type_list[1] == "ensemble":
		# Approach Number
		approach_number = design_type_list[-1][-1]

		# Features that should be used
		features_name = "ensemble_low_level_features_vif_features"
		
		# Loading Model and Feature Indices
		path = os.path.join(models_path, features_name, "feature_indices_approach_{}.npy".format(approach_number))
		feature_indices = np.load(path, allow_pickle=True)[()]

		path = os.path.join(models_path, features_name, "model_approach_{}.pkl".format(approach_number))
		Model = pickle.load(open(path, "rb"))
		print ("Loading model from", path)

		# Settings when loading data
		if design_type_list[0] == "ml":
			per_frame = False
			per_frame_features_flatten = False
		else:
			per_frame = True
			per_frame_features_flatten = False


		# Names of Custom-Features
		if temporal_low_level_features:
			F1 = features_names
			F2 = list(defaults.per_frame_quality_texture_features.keys())
			custom_features_names = list(sorted(set(F1) & set(F2), key = F1.index))
		else:
			F1 = features_names
			F2 = list(defaults.quality_texture_features.keys())
			custom_features_names = list(sorted(set(F1) & set(F2), key = F1.index))
		
		features_names_without_custom = [x for x in features_names if x not in custom_features_names]

		# Extracting Low-Level Features
		features = extract_features.Extract_Low_Level_Features(
			features_names=features_names_without_custom,
			video_filenames=[video_file],
			temporal_low_level_features=temporal_low_level_features
		)

		# Extracting Custom Features
		custom_features = extract_features.Compute_Quality_Custom_Features(
			custom_features_names=custom_features_names,
			quality_metric="vmaf",
			Meta_Information=Meta_Information,
			temporal_low_level_features=temporal_low_level_features
		)

		# Creating Input
		LLF_Data = np.repeat(np.expand_dims(features[video_file], axis=0), num_samples, axis=0)
		Custom_Data = custom_features[video_file]

		if temporal_low_level_features:
			min_temporal_length = min(LLF_Data.shape[1], Custom_Data.shape[1])
			LLF_Data = LLF_Data[:,-min_temporal_length:,:]
			Custom_Data = Custom_Data[:,-min_temporal_length:,:]

		if len(custom_features_names) == 0:
			X1 = LLF_Data
		else:
			X1 = np.concatenate([LLF_Data, Custom_Data], axis=-1)

		X2,_ = dataset_evaluation_functions.VIFFeatures_Bitrate_Dataset(
			codec=codec,
			preset=preset,
			quality_metric="vmaf",
			video_filenames=[video_file],
			Resolutions_Considered=defaults.resolutions,
			CRFs_Considered=[defaults.CRFs[0]],
			bitrates_Considered=None,
			QPs_Considered=None,
			vif_setting=Approach2Info[approach_number][1],
			vif_features_list=Approach2Info[approach_number][0],
			per_frame=per_frame,
			per_frame_features_flatten=per_frame_features_flatten,
			min_quality=-np.inf,
			max_quality=np.inf
		)
		X2 = np.repeat(X2, num_rate_control_parameters, axis=0)
		if len(X2.shape) == 2:
			X2[...,-1*len_meta_data:] = Meta_Data
		else:
			X2[...,-1*len_meta_data:] = np.repeat(np.expand_dims(Meta_Data, axis=1), X2.shape[1], axis=1)


		if (per_frame==False and temporal_low_level_features==False) or (per_frame==True and per_frame_features_flatten==True and temporal_low_level_features==False):
			# Concatenating
			X = np.concatenate([X1, X2], axis=-1)

			# Type Casting
			X = X.astype(np.float32)

			# Rounding
			X = np.round(X, decimals=4)
		
			# Feature Selection
			if feature_indices is not None:
				X = X[...,feature_indices]

			# Returning
			return X, Model

		elif per_frame==True and per_frame_features_flatten==False and temporal_low_level_features==True:
			# Concatenating
			min_temporal_length = min(X1.shape[1], X2.shape[1])
			X = np.concatenate([X1[:,-min_temporal_length:,:], X2[:,-min_temporal_length:,:]], axis=-1)

			# Type Casting
			X = X.astype(np.float32)

			# Rounding
			X = np.round(X, decimals=4)
			
			# Feature Selection
			if feature_indices is not None:
				X = X[...,feature_indices]

			# Returning
			return X, Model

		else:
			# Type Casting
			X1 = X1.astype(np.float32)
			X2 = X2.astype(np.float32)

			# Rounding
			X1 = np.round(X1, decimals=4)
			X2 = np.round(X2, decimals=4)

			# Returning
			return X1,X2,Model
			
	else:
		assert False, "Invalid Features"


# CrossOver-Quality: Select Models and Features
def CrossOver_Quality_Select_Models_Features(
	video_file:str,
	models_path:str,
	design_type:str,
	features_names:list,
	temporal_low_level_features:bool,
	codec:str,
	preset:str
):
	"""
	Function to select model and dataset

	Args:
		video_file (str): The video file name
		models_path (str): Path to models
		design_type: Type of features to use. Options:["ml_llf"]
		features_names (list): List of features to be considered when low-level features are used.
		codec (str): Codec used to generate RQ points that need to be extracted. Options: ["libx265", "libx264"]
		preset (str): Preset used to generate RQ points that need to be extracted. Options: ["slow", "medium", "fast", "veryfast", "ultrafast"]
	Returns:
		(list): List of Models used for prediction
		(list): Features after feature-selection
	"""
	
	## Using Low-Level features to predict CrossOver Qualities
	if design_type == "ml_llf":
		# Creating Input
		X,_ = dataset_evaluation_functions.LowLevelFeatures_CrossOverQualities_Dataset(
			codec=codec,
			preset=preset,
			quality_metric="vmaf",
			features_names=features_names,
			video_filenames=[video_file],
			temporal_low_level_features=temporal_low_level_features,
			Resolutions_Considered=defaults.resolutions,
			CRFs_Considered=defaults.CRFs,
			bitrates_Considered=None,
			QPs_Considered=None,
			high_res=defaults.resolutions[0],
			low_res=defaults.resolutions[1]
		)

		# Type-Casting
		X = X.astype(np.float32)
		
		# Rounding
		X = np.round(X, decimals=4)
		
		# Loading Models and Feature-Indices
		Models = []
		feature_indices = []
		for i in range(len(defaults.resolutions)-1):
			path = os.path.join(models_path, "low_level_features/feature_indices_coq_{}.npy".format(i+1))
			feature_indices.append(np.load(path, allow_pickle=True)[()])

			path = os.path.join(models_path, "low_level_features/model_coq_{}.pkl".format(i+1))
			Model = pickle.load(open(path, "rb"))
			print ("Loading model from", path)

			Models.append(Model)

		# Returning
		return X, feature_indices, Models	


# Evaluating Monotonicity of Bitrate Prediction Models
def Bitrate_Monotonicity(
	Model:any,
	X:np.array,
	evaluation_qualities:list
):
	"""
	Function to evaluate monotonicity of Quality Prediction models.
	Args:
		Model (any): Model used to predict.
		X (np.array): List of features for each cross-over bitrate.
		evaluation_qualities (list): List of evaluation_qualities present to be in the quality ladder. 
	Returns:
		(bool): Whether the model's predictions follow monotonicity or not for the given input.
	"""
	# Resolutions
	Resolutions = defaults.resolutions

	# Adjusting Inoputs
	X = X.reshape(len(evaluation_qualities), len(Resolutions), X.shape[-1])
	X = np.transpose(X, (1,0,2))
	X = X.reshape(len(Resolutions) * len(evaluation_qualities), X.shape[-1])

	for i in range(len(Resolutions)):
		x = X[i*len(evaluation_qualities):(i+1)*len(evaluation_qualities)]

		if ("sklearn" in str(type(Model))) or ("lineartree" in str(type(Model))):
			# Sklearn Model
			y_pred = Model.predict(x).flatten()
		else:
			assert False, "Invalid Model"

		y_pred = np.round(y_pred, decimals=6)
		if np.all(np.diff(y_pred) >= -1e-4):
			None
		else:
			return False

	return True


# Bitrate Quality Ladder
def Bitrate_Quality_Ladder(
	Model:any,
	X:np.array,
	evaluation_qualities:list
):
	"""
	Function to return Quality-Ladder for corresponding evaluation_qualities using Bitrate Prediction models.
	Args:
		Model (any): Model used to predict.
		X (np.array): List of features for each cross-over quality.
		evaluation_qualities (list): List of evaluation_qualities present to be in the quality ladder.
	Returns:
		(dict): The quality-ladder i.e a dictionary {quality: resolution} containing the quality as key and the resolution it should be encoded as value for the provided evaluation_qualities.
	"""
	# Resolutions
	Resolutions = defaults.resolutions

	# Quality Ladder
	Quality_Ladder = {}

	for i in range(len(evaluation_qualities)):
		q = evaluation_qualities[i]
		x = X[i*len(defaults.resolutions):(i+1)*len(defaults.resolutions)]

		if ("sklearn" in str(type(Model))) or ("lineartree" in str(type(Model))):
			# Sklearn Model
			y_pred = Model.predict(x).flatten()
		else:
			assert False, "Invalid Model"

		y_pred = np.round(y_pred, decimals=6)
		Quality_Ladder[q] = Resolutions[np.argmin(y_pred)]

	return Quality_Ladder


# Cross-Over Quality Ladder
def CrossOverQualities_Quality_Ladder(
	Models:list,
	X:np.array,
	feature_indices:list,
	evaluation_qualities:list
):
	"""
	Function to return Quality-Ladder for corresponding evaluation_qualities using Quality Prediction models.
	Args:
		Models (list): List of sklearn models trained to predict each cross-over evaluation_qualities.
		X (np.array): List of features for each cross-over quality.
		feature_indices (list): List of feature-indices to consider after feature-selection.
		evaluation_qualities (list): List of evaluation_qualities present to be in the quality ladder.
	Returns:
		(dict): The quality-ladder i.e a dictionary {quality: resolution} containing the quality as key and the resolution it should be encoded as value for the provided evaluation_qualities.
	"""
	# Assertions
	assert len(Models) == len(feature_indices) == len(defaults.resolutions)-1, "The length of list of models and X should be no.of resolutions - 1."

	# Resolutions
	Resolutions = defaults.resolutions

	# Predicting Cross-Over Qualities
	CrossOver_Qualities = []

	for i in range(len(defaults.resolutions)-1):
		x = np.concatenate([X, np.asarray(CrossOver_Qualities).reshape(1,-1)], axis=-1)
		
		# Rounding
		x = np.round(x, decimals=4)
		
		x = x[...,feature_indices[i]]
		x = x.reshape(1, -1)
		model = Models[i]
		y_pred = model.predict(x)
		CrossOver_Qualities.append(y_pred[0])

	# Calculating Quality-Ladder
	Quality_Ladder = {}
	for i in range(len(evaluation_qualities)):
		# Switching happens to higher resolution when quality >= crossover_quality of corresponding higher resolution.
		q = evaluation_qualities[i]
		Quality_Ladder[q] = None

		for j in range(1+len(CrossOver_Qualities)):
			if (j==0) and (q >= CrossOver_Qualities[j]):
				Quality_Ladder[q] = Resolutions[0]
			elif (j <= len(CrossOver_Qualities)-1) and (CrossOver_Qualities[j] <= q < CrossOver_Qualities[j-1]):
				Quality_Ladder[q] = Resolutions[j]
			elif (j==len(CrossOver_Qualities)) and (q < CrossOver_Qualities[j-1]):
				Quality_Ladder[q] = Resolutions[-1]
			else:
				None

		if Quality_Ladder[q] is None:
			assert False, "Something is Wrong"

	return Quality_Ladder


# Create Pareto-Front using Quality Ladder
def Pareto_Front_from_Quality_Ladder(
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


# Calculate BD-metrics
def Calculate_BD_metrics(
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
		bitrates=None,
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
	_, Fixed_Pareto_Front_Points = Pareto_Front_from_Bitrate_Ladder(
		RQ_pairs=RQ_pairs,
		Bitrate_Ladder=AL
	)
	Fixed_Pareto_Front_Points[:,0] = np.round(np.power(2, Fixed_Pareto_Front_Points[:,0]), decimals=3)

	_, Reference_Pareto_Front_Points = Pareto_Front_from_Bitrate_Ladder(
		RQ_pairs=RQ_pairs,
		Bitrate_Ladder=RL
	)
	Reference_Pareto_Front_Points[:,0] = np.round(np.power(2, Reference_Pareto_Front_Points[:,0]), decimals=3)

	_, Pareto_Front_Points = Pareto_Front_from_Quality_Ladder(
		RQ_pairs=RQ_pairs,
		Quality_Ladder=QL
	)
	Pareto_Front_Points[:,0] = np.round(np.power(2, Pareto_Front_Points[:,0]), decimals=3)


	# Assertions
	assert (np.all(Fixed_Pareto_Front_Points[:,0] <= defaults.max_bitrate) and np.all(Fixed_Pareto_Front_Points[:,0] >= defaults.min_bitrate)) and (np.all(Fixed_Pareto_Front_Points[:,1] <= defaults.max_quality) and np.all(Fixed_Pareto_Front_Points[:,1] >= defaults.min_quality)), "Fixed Bitrate Ladder Pareto-Front points are in the wrong range."

	assert (np.all(Reference_Pareto_Front_Points[:,0] <= defaults.max_bitrate) and np.all(Reference_Pareto_Front_Points[:,0] >= defaults.min_bitrate)) and (np.all(Reference_Pareto_Front_Points[:,1] <= defaults.max_quality) and np.all(Reference_Pareto_Front_Points[:,1] >= defaults.min_quality)), "Reference Bitrate Ladder Pareto-Front points are in the wrong range."

	assert (np.all(Pareto_Front_Points[:,0] <= defaults.max_bitrate) and np.all(Pareto_Front_Points[:,0] >= defaults.min_bitrate)) and (np.all(Pareto_Front_Points[:,1] <= defaults.max_quality) and np.all(Reference_Pareto_Front_Points[:,1] >= defaults.min_quality)), "Bitrate Ladder Pareto-Front points are in the wrong range."


	# BD-Metrics wrt Apple Fixed Bitrate Ladder
	f_bd_rate = bd_metrics.BD_Rate(
		R1=Fixed_Pareto_Front_Points[:,0],
		Q1=Fixed_Pareto_Front_Points[:,1],
		R2=Pareto_Front_Points[:,0],
		Q2=Pareto_Front_Points[:,1],
		piecewise=True
	)
	f_bd_quality = bd_metrics.BD_Quality(
		R1=Fixed_Pareto_Front_Points[:,0],
		Q1=Fixed_Pareto_Front_Points[:,1],
		R2=Pareto_Front_Points[:,0],
		Q2=Pareto_Front_Points[:,1],
		piecewise=True
	)

	# BD-Metrics wrt Reference Bitrate Ladder
	r_bd_rate = bd_metrics.BD_Rate(
		R1=Reference_Pareto_Front_Points[:,0],
		Q1=Reference_Pareto_Front_Points[:,1],
		R2=Pareto_Front_Points[:,0],
		Q2=Pareto_Front_Points[:,1],
		piecewise=True
	)
	r_bd_quality = bd_metrics.BD_Quality(
		R1=Reference_Pareto_Front_Points[:,0],
		Q1=Reference_Pareto_Front_Points[:,1],
		R2=Pareto_Front_Points[:,0],
		Q2=Pareto_Front_Points[:,1],
		piecewise=True
	)

	return np.round(f_bd_rate, decimals=4), np.round(f_bd_quality, decimals=4), np.round(r_bd_rate, decimals=4), np.round(r_bd_quality, decimals=4)