"""
Construct Bitrate Ladders
"""
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt

import os, sys, warnings
sys.path.append("/home/kd28684/Efficient-Dynamic-Optimizer-using-Visual-Information-Fidelity-Working")
import pickle
import modules.bitrate_ladder_construction_functions as bitrate_ladder_construction_functions
import defaults


def construct_crossoverbitrates_bitrate_ladder(
	# Files
	video_files:list,

	# Method Arguments
	features_names:list,
	temporal_low_level_features:bool,

	# Path
	models_path:str,
	save_path:str,

	# Arguments
	codec:str,
	preset:str,
	quality_metric:str,
	Resolutions_Considered:list,
	evaluation_bitrates:list,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate,
):
	# Models and Feature-Indices
	Models = []
	Feature_Indices = []
	for i in range(len(Resolutions_Considered)-1):
		Models.append(pickle.load(open(os.path.join(models_path, "low_level_features_cob_{}.pkl".format(i)), "rb")))
		Feature_Indices.append(np.load(os.path.join(models_path, "low_level_features_indices_{}.npy".format(i))))

	# Inputs
	Inputs = bitrate_ladder_construction_functions.Bitrate_Ladder_Construction_with_CrossOver_Bitrates(
		# Files
		video_filenames=video_files,

		# Method Arguments
		features_names=features_names,
		temporal_low_level_features=temporal_low_level_features,

		# Arguments
		codec=codec,
		preset=preset,
		quality_metric=quality_metric,
		Resolutions_Considered=Resolutions_Considered,
		evaluation_bitrates=evaluation_bitrates,
		min_quality=min_quality,
		max_quality=max_quality,
		min_bitrate=min_bitrate,
		max_bitrate=max_bitrate
	)

	# Creating Bitrate-Ladders for each video-file
	Bitrate_Ladders = {}
	for video_file in video_files:
		X = Inputs[video_file]

		# Constructing Bitrate-Ladder
		BL = bitrate_ladder_construction_functions.Predict_CrossOver_Bitrates_Bitrate_Ladder(
			Models=Models,
			Feature_Indices=Feature_Indices,
			X = X,
			evaluation_bitrates=defaults.evaluation_bitrates
		)
	
		Bitrate_Ladders[video_file] = BL

	# Saving Bitrate-Ladders
	np.save(save_path, Bitrate_Ladders)



def construct_metadata_bitrate_ladder(
	# Files
	video_files:list,

	# Path
	models_path:str,
	save_path:str,

	# Arguments
	codec:str,
	preset:str,
	quality_metric:str,
	Resolutions_Considered:list,
	evaluation_bitrates:list,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate,
):
	# Load Model
	Model =  pickle.load(open(models_path, "rb"))

	# Inputs
	Inputs, _, _ = bitrate_ladder_construction_functions.Bitrate_Ladder_Construction_with_Metadata(
		# Files
		video_filenames=video_files,

		# Arguments
		codec=codec,
		preset=preset,
		quality_metric=quality_metric,
		Resolutions_Considered=Resolutions_Considered,
		evaluation_bitrates=evaluation_bitrates,
		min_quality=min_quality,
		max_quality=max_quality,
		min_bitrate=min_bitrate,
		max_bitrate=max_bitrate
	)

	# Creating Bitrate-Ladders for each video-file
	Bitrate_Ladders = {}
	for video_file in video_files:
		X = Inputs[video_file]

		# Constructing Bitrate-Ladder
		BL = bitrate_ladder_construction_functions.Predict_Bitrate_Ladder(
			Model=Model,
			X = X,
			evaluation_bitrates=defaults.evaluation_bitrates
		)
	
		Bitrate_Ladders[video_file] = BL

	# Saving Bitrate-Ladders
	np.save(save_path, Bitrate_Ladders)


def construct_low_level_features_bitrate_ladder(
	# Files
	video_files:list,

	# Method Arguments
	features_names:list,
	temporal_low_level_features:bool,

	# Path
	models_path:str,
	save_path:str,

	# Arguments
	codec:str,
	preset:str,
	quality_metric:str,
	Resolutions_Considered:list,
	evaluation_bitrates:list,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate,
):
	# Load Model
	Model =  pickle.load(open(models_path, "rb"))

	# Inputs
	Inputs, _, _ = bitrate_ladder_construction_functions.Bitrate_Ladder_Construction_with_LowLevelFeatures(
		# Files
		video_filenames=video_files,

		# Method Arguments
		features_names=features_names,
		temporal_low_level_features=temporal_low_level_features,

		# Arguments
		codec=codec,
		preset=preset,
		quality_metric=quality_metric,
		Resolutions_Considered=Resolutions_Considered,
		evaluation_bitrates=evaluation_bitrates,
		min_quality=min_quality,
		max_quality=max_quality,
		min_bitrate=min_bitrate,
		max_bitrate=max_bitrate
	)

	# Creating Bitrate-Ladders for each video-file
	Bitrate_Ladders = {}
	for video_file in video_files:
		X = Inputs[video_file]

		# Constructing Bitrate-Ladder
		BL = bitrate_ladder_construction_functions.Predict_Bitrate_Ladder(
			Model=Model,
			X = X,
			evaluation_bitrates=defaults.evaluation_bitrates
		)
	
		Bitrate_Ladders[video_file] = BL

	# Saving Bitrate-Ladders
	np.save(save_path, Bitrate_Ladders)


def construct_vif_features_bitrate_ladder(
	# Files
	video_files:list,

	# Method Arguments
	per_frame:bool,
	per_frame_features_flatten:bool,
	vif_setting:str,
	vif_features_list:list,

	# Path
	models_path:str,
	save_path:str,

	# Arguments
	codec:str,
	preset:str,
	quality_metric:str,
	Resolutions_Considered:list,
	evaluation_bitrates:list,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate,
):
	# Load Model
	Model =  pickle.load(open(models_path, "rb"))

	# Inputs
	Inputs, _, _ = bitrate_ladder_construction_functions.Bitrate_Ladder_Construction_with_VIFFeatures(
		# Files
		video_filenames=video_files,

		# Method Arguments
		per_frame=per_frame,
		per_frame_features_flatten=per_frame_features_flatten,
		vif_setting=vif_setting,
		vif_features_list=vif_features_list,

		# Arguments
		codec=codec,
		preset=preset,
		quality_metric=quality_metric,
		Resolutions_Considered=Resolutions_Considered,
		evaluation_bitrates=evaluation_bitrates,
		min_quality=min_quality,
		max_quality=max_quality,
		min_bitrate=min_bitrate,
		max_bitrate=max_bitrate
	)

	# Creating Bitrate-Ladders for each video-file
	Bitrate_Ladders = {}
	for video_file in video_files:
		X = Inputs[video_file]

		# Constructing Bitrate-Ladder
		BL = bitrate_ladder_construction_functions.Predict_Bitrate_Ladder(
			Model=Model,
			X = X,
			evaluation_bitrates=defaults.evaluation_bitrates
		)
	
		Bitrate_Ladders[video_file] = BL

	# Saving Bitrate-Ladders
	np.save(save_path, Bitrate_Ladders)


def construct_low_level_features_vif_features_bitrate_ladder(
	# Files
	video_files:list,

	# Method Arguments
	features_names:list,
	temporal_low_level_features:bool,
	per_frame:bool,
	per_frame_features_flatten:bool,
	vif_setting:str,
	vif_features_list:list,

	# Path
	models_path:str,
	save_path:str,

	# Arguments
	codec:str,
	preset:str,
	quality_metric:str,
	Resolutions_Considered:list,
	evaluation_bitrates:list,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate,
):
	# Load Model
	Model =  pickle.load(open(models_path, "rb"))

	# Inputs
	Inputs, _, _ = bitrate_ladder_construction_functions.Bitrate_Ladder_Construction_with_LowLevelFeatures_VIFFeatures(
		# Files
		video_filenames=video_files,

		# Method Arguments
		features_names=features_names,
		temporal_low_level_features=temporal_low_level_features,
		per_frame=per_frame,
		per_frame_features_flatten=per_frame_features_flatten,
		vif_setting=vif_setting,
		vif_features_list=vif_features_list,

		# Arguments
		codec=codec,
		preset=preset,
		quality_metric=quality_metric,
		Resolutions_Considered=Resolutions_Considered,
		evaluation_bitrates=evaluation_bitrates,
		min_quality=min_quality,
		max_quality=max_quality,
		min_bitrate=min_bitrate,
		max_bitrate=max_bitrate
	)

	# Creating Bitrate-Ladders for each video-file
	Bitrate_Ladders = {}
	for video_file in video_files:
		X = Inputs[video_file]

		# Constructing Bitrate-Ladder
		BL = bitrate_ladder_construction_functions.Predict_Bitrate_Ladder(
			Model=Model,
			X = X,
			evaluation_bitrates=defaults.evaluation_bitrates
		)
	
		Bitrate_Ladders[video_file] = BL

	# Saving Bitrate-Ladders
	np.save(save_path, Bitrate_Ladders)




def main(
	# Files
	Test_Video_Files:list,

	# Path
	results_dir,

	# Method Arguments
	features_names:list,
	vif_approach_numbers:list
):
	# Features Settings
	temporal_low_level_features=False
	per_frame = False
	per_frame_features_flatten = False
	VIF_Approach_Map = {
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


	# Arguments
	arguments = {
		# Encoder-Settings
		"codec": "libx265",
		"preset": "medium",
		"quality_metric": "vmaf",

		# RQ-Points
		"Resolutions_Considered": defaults.resolutions,
		"evaluation_bitrates": defaults.evaluation_bitrates,
		"min_quality": defaults.min_quality,
		"max_quality": defaults.max_quality,
		"min_bitrate": defaults.min_bitrate,
		"max_bitrate": defaults.max_bitrate,
	}


	## Constructing Bitrate Ladders using Cross-Over Bitrates
	print ()
	print ("-"*10, "Cross-Over Bitrates Bitrate Ladder", "-"*10)
	print ()

	# Creating Directories
	os.makedirs(os.path.join(results_dir, "CrossOver_Bitrates", "bitrate_ladders",), exist_ok=True)

	construct_crossoverbitrates_bitrate_ladder(
		# Files
		video_files=Test_Video_Files,

		# Method Arguments
		features_names=features_names[:-3],
		temporal_low_level_features=temporal_low_level_features,

		# Path
		models_path=os.path.join(results_dir, "CrossOver_Bitrates", "models"),
		save_path=os.path.join(results_dir, "CrossOver_Bitrates", "bitrate_ladders", "low_level_features.npy"),

		# Arguments
		codec=arguments["codec"],
		preset=arguments["preset"],
		quality_metric=arguments["quality_metric"],
		Resolutions_Considered=arguments["Resolutions_Considered"],
		evaluation_bitrates=arguments["evaluation_bitrates"],
		min_quality=arguments['min_quality'],
		max_quality=arguments["max_quality"],
		min_bitrate=arguments["min_bitrate"],
		max_bitrate=arguments["max_bitrate"],
	)


	# Creating Directories
	os.makedirs(os.path.join(results_dir, "Bitrate_Ladder_Prediction", "bitrate_ladders",), exist_ok=True)


	# Constructing Bitrate Ladders using Metadata
	print ()
	print ("-"*10, "Metadata Bitrate Ladder", "-"*10)
	print ()

	construct_metadata_bitrate_ladder(
		# Files
		video_files=Test_Video_Files,

		# Path
		models_path=os.path.join(results_dir, "Bitrate_Ladder_Prediction", "models", "metadata.pkl"),
		save_path=os.path.join(results_dir, "Bitrate_Ladder_Prediction", "bitrate_ladders", "metadata.npy"),

		# Arguments
		**arguments
	)


	# Constructing Bitrate Ladders using Low-Level Features
	print ()
	print ("-"*10, "Low-Level Features Bitrate Ladder", "-"*10)
	print ()

	construct_low_level_features_bitrate_ladder(
		# Files
		video_files=Test_Video_Files,

		# Method Arguments
		features_names=features_names,
		temporal_low_level_features=temporal_low_level_features,

		# Path
		models_path=os.path.join(results_dir, "Bitrate_Ladder_Prediction", "models", "low_level_features.pkl"),
		save_path=os.path.join(results_dir, "Bitrate_Ladder_Prediction", "bitrate_ladders", "low_level_features.npy"),

		# Arguments
		**arguments
	)


	# Constructing Bitrate Ladders using VIF Features
	print ()
	print ("-"*10, "VIF Features Bitrate Ladder", "-"*10)
	print ()

	for vif_approach_number in vif_approach_numbers:
		construct_vif_features_bitrate_ladder(
			# Files
			video_files=Test_Video_Files,

			# Method Arguments
			per_frame=per_frame,
			per_frame_features_flatten=per_frame_features_flatten,
			vif_setting=VIF_Approach_Map[vif_approach_number][1],
			vif_features_list=VIF_Approach_Map[vif_approach_number][0],

			# Path
			models_path=os.path.join(results_dir, "Bitrate_Ladder_Prediction", "models", "vif_features_{}.pkl".format(vif_approach_number)),
			save_path=os.path.join(results_dir, "Bitrate_Ladder_Prediction", "bitrate_ladders", "vif_features_{}.npy".format(vif_approach_number)),

			# Arguments
			**arguments
		)


	# Constructing Bitrate Ladders using Low-Level Features and VIF Features
	print ()
	print ("-"*10, "Low-Level Features and VIF Features Bitrate Ladder", "-"*10)
	print ()

	for vif_approach_number in vif_approach_numbers:
		construct_low_level_features_vif_features_bitrate_ladder(
			# Files
			video_files=Test_Video_Files,

			# Method Arguments
			features_names=features_names,
			temporal_low_level_features=temporal_low_level_features,
			per_frame=per_frame,
			per_frame_features_flatten=per_frame_features_flatten,
			vif_setting=VIF_Approach_Map[vif_approach_number][1],
			vif_features_list=VIF_Approach_Map[vif_approach_number][0],

			# Path
			models_path=os.path.join(results_dir, "Bitrate_Ladder_Prediction", "models", "low_level_features_vif_features_{}.pkl".format(vif_approach_number)),
			save_path=os.path.join(results_dir, "Bitrate_Ladder_Prediction", "bitrate_ladders", "low_level_features_vif_features_{}.npy".format(vif_approach_number)),

			# Arguments
			**arguments
		)
