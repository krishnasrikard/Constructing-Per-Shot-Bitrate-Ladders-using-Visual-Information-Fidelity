"""
Functions to Construct Bitrate Ladders
"""
# Importing Libraries
import numpy as np

import os, sys, warnings
sys.path.append("/home/kd28684/Efficient-Dynamic-Optimizer-using-Visual-Information-Fidelity-Working")
import functions.pareto_front_points as pareto_front_points
import functions.extract_functions as extract_functions
import functions.extract_features as extract_features
import functions.IO_functions as IO_functions
import modules.bitrate_ladder_prediction_dataset_functions as dataset_functions
import defaults


def Bitrate_Ladder_Construction_with_CrossOver_Bitrates(
	codec:str,
	preset:str,
	quality_metric:str,
	features_names:list,
	video_filenames:list,
	temporal_low_level_features:bool,
	Resolutions_Considered:list,
	evaluation_bitrates:list,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate
):
	"""
	Args:
		codec (str): Codec used to generate RQ points that need to be extracted.
		preset (str): Preset used to generate RQ points that need to be extracted.
		quality_metric (str): Quality Metric to consider.
		compression_statistics (bool): If True, during testing i.e predicting bitrate video is assumed to be compressed but quality estimation is not performed. If False, during testing, neither compression not quality estimation is performed.
		features_names (list): List of features to be considered. 
		video_filenames (list): List of video filenames to be considered for feature-extraction.
		temporal_low_level_features (bool): If True, everything is extracted per frame instead of pooling using various statistics.
		Resolutions_Considered (list): Resolutions to be considered.
		evaluation_bitrates (list): List of bitrates for evaluation.
		min_quality (float): Minimum quality to be considered for in output pairs/info.
		max_quality (float): Maximum quality to be considered for in output pairs/info.
		min_bitrate (float): Minimum bitrate (in kbps) to be considered for in output pairs/info.
		max_bitrate (float): Maximum bitrate (in kbps) to be considered for in output pairs/info.
	Returns:
		(np.array): Input Data
		(np.array): Target Data
	"""
	# Creating Input
	X = {}
	for video_file in video_filenames:
		I, _ = dataset_functions.LowLevelFeatures_CrossOverBitrates_Dataset(
			codec=codec,
			preset=preset,
			quality_metric=quality_metric,
			features_names=features_names,
			video_filenames=[video_file],
			temporal_low_level_features=temporal_low_level_features,
			Resolutions_Considered=Resolutions_Considered,
			CRFs_Considered=defaults.CRFs,
			QPs_Considered=None,
			high_res=Resolutions_Considered[0],
			low_res=Resolutions_Considered[1],
			min_quality=min_quality,
			max_quality=max_quality,
			min_bitrate=min_bitrate,
			max_bitrate=max_bitrate
		)
		X[video_file] = I

		# Type-Casting
		X[video_file] = X[video_file].astype(np.float32)
		
		# Rounding
		X[video_file] = np.round(X[video_file], decimals=4)

	return X


def Bitrate_Ladder_Construction_with_Metadata(
	codec:str,
	preset:str,
	quality_metric:str,
	video_filenames:list,
	Resolutions_Considered:list,
	evaluation_bitrates:list,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate,
):
	"""
	Args:
		codec (str): Codec used to generate RQ points that need to be extracted.
		preset (str): Preset used to generate RQ points that need to be extracted.
		quality_metric (str): Quality Metric to consider.
		video_filenames (list): List of video filenames to be considered for feature-extraction.
		Resolutions_Considered (list): Resolutions to be considered.
		evaluation_bitrates (list): List of bitrates for evaluation.
		min_quality (float): Minimum quality to be considered for in output pairs/info.
		max_quality (float): Maximum quality to be considered for in output pairs/info.
		min_bitrate (float): Minimum bitrate (in kbps) to be considered for in output pairs/info.
		max_bitrate (float): Maximum bitrate (in kbps) to be considered for in output pairs/info.
	Returns:
		(np.array): Input Data
		(np.array): Target Data
	"""
	# Extracting RQ Information
	Meta_Information = extract_features.Extract_RQ_Features(
		codec=codec,
		preset=preset,
		quality_metric=quality_metric,
		video_filenames=video_filenames,
		Resolutions_Considered=Resolutions_Considered,
		CRFs_Considered=defaults.CRFs,
		QPs_Considered=None,
		min_quality=min_quality,
		max_quality=max_quality,
		min_bitrate=min_bitrate,
		max_bitrate=max_bitrate
	)
	for video_file in video_filenames:
		# Order: By Bitrate i.e (All Rs for B1, All Rs for B2, ....)
		num_samples = len(evaluation_bitrates) * len(defaults.resolutions)
		Meta_Information[video_file] = np.zeros((num_samples,11))

		New_Meta_Data = []
		for b in evaluation_bitrates:
			for res in defaults.resolutions:
				New_Meta_Data.append([b,res[0]/3840,res[1]/3840])

		New_Meta_Data = np.asarray(New_Meta_Data)
		Meta_Information[video_file][:,[0,3,4]] = New_Meta_Data
	
	X = {}
	y = {}
	CRFs_after_constraints = {}

	for video_file in video_filenames:
		# No.of RQ-points obtained by compressing the uncompressed video under different settings
		num_samples = Meta_Information[video_file].shape[0]

		# Finding CRFs after after applying quality constraints.
		CRFs_after_constraints[video_file] = {}

		for _,res in enumerate(defaults.resolutions):
			CRFs_after_constraints[video_file][res] = []
			scaled_h = np.round(res[1]/3840, decimals=4)

			for i in range(Meta_Information[video_file].shape[0]):
				h = Meta_Information[video_file][i,-1]

				if (np.isclose(np.round(h, decimals=4), scaled_h)):
					CRFs_after_constraints[video_file][res].append(
						Meta_Information[video_file][i,-3]
					)

		# Target: Quality
		Target = np.expand_dims(Meta_Information[video_file][:,1], axis=-1)

		# Repeating Meta_Data along temporal-axis
		# Meta_Data containing [bitrate, width, height]
		Meta_Data = Meta_Information[video_file][:,[0,3,4]]

		X[video_file] = Meta_Data
		y[video_file] = Target

		# Scaling
		if quality_metric == "vmaf":
			y[video_file] = y[video_file]/100.0

		# Type-Casting
		X[video_file] = X[video_file].astype(np.float32)
		y[video_file] = y[video_file].astype(np.float32)

		# Rounding
		X[video_file] = np.round(X[video_file], decimals=4)
		y[video_file] = np.round(y[video_file], decimals=4)

	return X,y,CRFs_after_constraints
	

def Bitrate_Ladder_Construction_with_LowLevelFeatures(
	codec:str,
	preset:str,
	quality_metric:str,
	features_names:list,
	video_filenames:list,
	temporal_low_level_features:bool,
	Resolutions_Considered:list,
	evaluation_bitrates:list,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate
):
	"""
	Args:
		codec (str): Codec used to generate RQ points that need to be extracted.
		preset (str): Preset used to generate RQ points that need to be extracted.
		quality_metric (str): Quality Metric to consider.
		features_names (list): List of features to be considered. 
		video_filenames (list): List of video filenames to be considered for feature-extraction.
		temporal_low_level_features (bool): If True, everything is extracted per frame instead of pooling using various statistics.
		Resolutions_Considered (list): Resolutions to be considered.
		evaluation_bitrates (list): List of bitrates for evaluation.
		min_quality (float): Minimum quality to be considered for in output pairs/info.
		max_quality (float): Maximum quality to be considered for in output pairs/info.
		min_bitrate (float): Minimum bitrate (in kbps) to be considered for in output pairs/info.
		max_bitrate (float): Maximum bitrate (in kbps) to be considered for in output pairs/info.
	Returns:
		(np.array): Input Data
		(np.array): Target Data
	"""
	# Names of Custom-Features
	# Custom-Features are returned in the same order as features_names so that there won't we any trouble while accessing them using "X".
	if temporal_low_level_features:
		F1 = features_names
		F2 = list(defaults.per_frame_bitrate_texture_features.keys())
		custom_features_names = list(sorted(set(F1) & set(F2), key = F1.index))
	else:
		F1 = features_names
		F2 = list(defaults.bitrate_texture_features.keys())
		custom_features_names = list(sorted(set(F1) & set(F2), key = F1.index))
	
	features_names_without_custom = [x for x in features_names if x not in custom_features_names]

	# Extracting RQ Information
	Meta_Information = extract_features.Extract_RQ_Features(
		codec=codec,
		preset=preset,
		quality_metric=quality_metric,
		video_filenames=video_filenames,
		Resolutions_Considered=Resolutions_Considered,
		CRFs_Considered=defaults.CRFs,
		QPs_Considered=None,
		min_quality=min_quality,
		max_quality=max_quality,
		min_bitrate=min_bitrate,
		max_bitrate=max_bitrate
	)
	for video_file in video_filenames:
		# Order: By Bitrate i.e (All Rs for B1, All Rs for B2, ....)
		num_samples = len(evaluation_bitrates) * len(defaults.resolutions)
		Meta_Information[video_file] = np.zeros((num_samples,11))

		New_Meta_Data = []
		for b in evaluation_bitrates:
			for res in defaults.resolutions:
				New_Meta_Data.append([b,res[0]/3840,res[1]/3840])

		New_Meta_Data = np.asarray(New_Meta_Data)
		Meta_Information[video_file][:,[0,3,4]] = New_Meta_Data

	# Extracting Low-Level Features
	features = extract_features.Extract_Low_Level_Features(
		features_names=features_names_without_custom,
		video_filenames=video_filenames,
		temporal_low_level_features=temporal_low_level_features
	)

	# Extracting Custom Features
	custom_features = extract_features.Compute_Bitrate_Custom_Features(
		custom_features_names=custom_features_names,
		Meta_Information=Meta_Information,
		temporal_low_level_features=temporal_low_level_features
	)
	
	X = {}
	y = {}
	CRFs_after_constraints = {}

	for video_file in video_filenames:
		# No.of RQ-points obtained by compressing the uncompressed video under different settings
		num_samples = Meta_Information[video_file].shape[0]

		# Finding CRFs after after applying quality constraints.
		CRFs_after_constraints[video_file] = {}

		for _,res in enumerate(defaults.resolutions):
			CRFs_after_constraints[video_file][res] = []
			scaled_h = np.round(res[1]/3840, decimals=4)

			for i in range(Meta_Information[video_file].shape[0]):
				h = Meta_Information[video_file][i,-1]

				if (np.isclose(np.round(h, decimals=4), scaled_h)):
					CRFs_after_constraints[video_file][res].append(
						Meta_Information[video_file][i,-3]
					)

		# Target: Quality
		Target = np.expand_dims(Meta_Information[video_file][:,1], axis=-1)
		
		# LLF_Data
		LLF_Data = np.repeat(np.expand_dims(features[video_file], axis=0), num_samples, axis=0)

		# Custom_Data
		Custom_Data = custom_features[video_file]

		# Repeating Meta_Data along temporal-axis
		# Meta_Data containing [bitrate, width, height]
		Meta_Data = Meta_Information[video_file][:,[0,3,4]]

		# Final Features
		# Matching temporal-length of LLF_Data, Custom_Data and Meta_Data
		if temporal_low_level_features:
			# Temporal-Length
			min_temporal_length = min(LLF_Data.shape[1], Custom_Data.shape[1])

			# Repeating Metadata along temporal-axis
			Meta_Data = np.repeat(np.expand_dims(Meta_Data, axis=1), min_temporal_length, axis=1)

			LLF_Data = LLF_Data[:,-min_temporal_length:,:]
			Custom_Data = Custom_Data[:,-min_temporal_length:,:]

		if len(custom_features_names) == 0:
			Final_Features = np.concatenate([LLF_Data,Meta_Data], axis=-1)
		else:
			Final_Features = np.concatenate([LLF_Data,Custom_Data,Meta_Data], axis=-1)

		X[video_file] = Final_Features
		y[video_file] = Target

		# Scaling
		if quality_metric == "vmaf":
			y[video_file] = y[video_file]/100.0

		# Type-Casting
		X[video_file] = X[video_file].astype(np.float32)
		y[video_file] = y[video_file].astype(np.float32)

		# Rounding
		X[video_file] = np.round(X[video_file], decimals=4)
		y[video_file] = np.round(y[video_file], decimals=4)
	
	return X,y,CRFs_after_constraints


def Bitrate_Ladder_Construction_with_VIFFeatures(
	codec:str,
	preset:str,
	quality_metric:str,
	video_filenames:list,
	Resolutions_Considered:list,
	evaluation_bitrates:list,
	vif_setting:str,
	vif_features_list:list,
	per_frame:bool,
	per_frame_features_flatten:bool,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate
):
	"""
	Args:
		codec (str): Codec used to generate RQ points that need to be extracted.
		preset (str): Preset used to generate RQ points that need to be extracted.
		quality_metric (str): Quality Metric to consider.
		video_filenames (list): List of video filenames to be considered for feature-extraction.
		Resolutions_Considered (list): Resolutions to be considered.
		evaluation_bitrates (list): List of bitrates for evaluation.
		vif_setting (str): Select one VIF setting i.e how VIF information extracted from compressed videos should be used. Options: ["per_scale", "per_subband", "per_eigen_value"]
		vif_features_list (list): List of VIF features to be considered as input features for the dataset. Options: ["vif_info", "mean_abs_frame_diff", "diff_vif_info"]
		per_frame (bool): Whether features should be given per frame or average along temporal-axis.
		per_frame_features_flatten (bool): Whether to flatten features per each frames to a vector of shape (frames*features).
		min_quality (float): Minimum quality to be considered for in output pairs/info.
		max_quality (float): Maximum quality to be considered for in output pairs/info.
		min_bitrate (float): Minimum bitrate (in kbps) to be considered for in output pairs/info.
		max_bitrate (float): Maximum bitrate (in kbps) to be considered for in output pairs/info.
	Returns:
		(np.array): Input Data
		(np.array): Target Data
	"""
	# Extracting RQ Information
	Meta_Information = extract_features.Extract_RQ_Features(
		codec=codec,
		preset=preset,
		quality_metric=quality_metric,
		video_filenames=video_filenames,
		Resolutions_Considered=Resolutions_Considered,
		CRFs_Considered=defaults.CRFs,
		QPs_Considered=None,
		min_quality=min_quality,
		max_quality=max_quality,
		min_bitrate=min_bitrate,
		max_bitrate=max_bitrate
	)
	for video_file in video_filenames:
		# Order: By Bitrate i.e (All Rs for B1, All Rs for B2, ....)
		num_samples = len(evaluation_bitrates) * len(defaults.resolutions)
		Meta_Information[video_file] = np.zeros((num_samples,11))

		New_Meta_Data = []
		for b in evaluation_bitrates:
			for res in defaults.resolutions:
				New_Meta_Data.append([b,res[0]/3840,res[1]/3840])

		New_Meta_Data = np.asarray(New_Meta_Data)
		Meta_Information[video_file][:,[0,3,4]] = New_Meta_Data

	# Extracting VIF Features
	VIF_Features = extract_features.Extract_VIF_Features(
		video_filenames=video_filenames,
		vif_setting=vif_setting,
		vif_features_list=vif_features_list,
		per_frame=per_frame,
		per_frame_features_flatten=per_frame_features_flatten
	)

	X = {}
	y = {}
	CRFs_after_constraints = {}

	for video_file in video_filenames:
		# No.of RQ-points obtained by compressing the uncompressed video under different settings
		num_samples = Meta_Information[video_file].shape[0]

		# Finding CRFs after after applying quality constraints.
		CRFs_after_constraints[video_file] = {}

		for _,res in enumerate(defaults.resolutions):
			CRFs_after_constraints[video_file][res] = []
			scaled_h = np.round(res[1]/3840, decimals=4)

			for i in range(Meta_Information[video_file].shape[0]):
				h = Meta_Information[video_file][i,-1]

				if (np.isclose(np.round(h, decimals=4), scaled_h)):
					CRFs_after_constraints[video_file][res].append(
						Meta_Information[video_file][i,-3]
					)

		# Target: Quality
		Target = np.expand_dims(Meta_Information[video_file][:,1], axis=-1)

		# VIF_Data
		VIF_Data = np.repeat(np.expand_dims(VIF_Features[video_file], axis=0), num_samples, axis=0)
		temporal_length = VIF_Data.shape[1]

		# Repeating Meta_Data along temporal-axis
		# Meta_Data containing [bitrate, width, height]
		Meta_Data = Meta_Information[video_file][:,[0,3,4]]
		Meta_Data = np.repeat(np.expand_dims(Meta_Data, axis=1), temporal_length, axis=1)
			
		# Final Features
		Final_Features = np.concatenate([VIF_Data, Meta_Data], axis=-1)

		if (per_frame == False) or (per_frame == True and per_frame_features_flatten == True):
			Final_Features = Final_Features[:,0,:]

		X[video_file] = Final_Features
		y[video_file] = Target

		# Scaling
		if quality_metric == "vmaf":
			y[video_file] = y[video_file]/100.0

		# Type-Casting
		X[video_file] = X[video_file].astype(np.float32)
		y[video_file] = y[video_file].astype(np.float32)

		# Rounding
		X[video_file] = np.round(X[video_file], decimals=4)
		y[video_file] = np.round(y[video_file], decimals=4)

	return X,y,CRFs_after_constraints


def Bitrate_Ladder_Construction_with_LowLevelFeatures_VIFFeatures(
	codec:str,
	preset:str,
	quality_metric:str,
	features_names:list,
	video_filenames:list,
	temporal_low_level_features:bool,
	Resolutions_Considered:list,
	evaluation_bitrates:list,
	vif_setting:str,
	vif_features_list:list,
	per_frame:bool,
	per_frame_features_flatten:bool,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate
):
	"""
	Args:
		codec (str): Codec used to generate RQ points that need to be extracted.
		preset (str): Preset used to generate RQ points that need to be extracted.
		quality_metric (str): Quality Metric to consider.
		features_names (list): List of features to be considered. 
		video_filenames (list): List of video filenames to be considered for feature-extraction.
		temporal_low_level_features (bool): If True, everything is extracted per frame instead of pooling using various statistics.
		Resolutions_Considered (list): Resolutions to be considered.
		evaluation_crfs (list): CRFs to be considered for evaluation.
		evaluation_qps (list): QPs to be considered for evaluation.
		evaluation_bitrates (list): List of bitrates for evaluation.
		vif_setting (str): Select one VIF setting i.e how VIF information extracted from compressed videos should be used. Options: ["per_scale", "per_subband", "per_eigen_value"]
		vif_features_list (list): List of VIF features to be considered as input features for the dataset. Options: ["vif_info", "mean_abs_frame_diff", "diff_vif_info"]
		per_frame (bool): Whether features should be given per frame or average along temporal-axis.
		per_frame_features_flatten (bool): Whether to flatten features per each frames to a vector of shape (frames*features).
		min_quality (float): Minimum quality to be considered for in output pairs/info.
		max_quality (float): Maximum quality to be considered for in output pairs/info.
		min_bitrate (float): Minimum bitrate (in kbps) to be considered for in output pairs/info.
		max_bitrate (float): Maximum bitrate (in kbps) to be considered for in output pairs/info.
	Returns:
		(np.array): Input Data
		(np.array): Target Data
	"""
	# Names of Custom-Features
	# Custom-Features are returned in the same order as features_names so that there won't we any trouble while accessing them using "X".
	if temporal_low_level_features:
		F1 = features_names
		F2 = list(defaults.per_frame_bitrate_texture_features.keys())
		custom_features_names = list(sorted(set(F1) & set(F2), key = F1.index))
	else:
		F1 = features_names
		F2 = list(defaults.bitrate_texture_features.keys())
		custom_features_names = list(sorted(set(F1) & set(F2), key = F1.index))
	
	features_names_without_custom = [x for x in features_names if x not in custom_features_names]

	# Extracting RQ Information
	Meta_Information = extract_features.Extract_RQ_Features(
		codec=codec,
		preset=preset,
		quality_metric=quality_metric,
		video_filenames=video_filenames,
		Resolutions_Considered=Resolutions_Considered,
		CRFs_Considered=defaults.CRFs,
		QPs_Considered=None,
		min_quality=min_quality,
		max_quality=max_quality,
		min_bitrate=min_bitrate,
		max_bitrate=max_bitrate
	)
	for video_file in video_filenames:
		# Order: By Bitrate i.e (All Rs for B1, All Rs for B2, ....)
		num_samples = len(evaluation_bitrates) * len(defaults.resolutions)
		Meta_Information[video_file] = np.zeros((num_samples,11))

		New_Meta_Data = []
		for b in evaluation_bitrates:
			for res in defaults.resolutions:
				New_Meta_Data.append([b,res[0]/3840,res[1]/3840])

		New_Meta_Data = np.asarray(New_Meta_Data)
		Meta_Information[video_file][:,[0,3,4]] = New_Meta_Data
	
	# Extracting Low-Level Features
	features = extract_features.Extract_Low_Level_Features(
		features_names=features_names_without_custom,
		video_filenames=video_filenames,
		temporal_low_level_features=temporal_low_level_features
	)

	# Extracting Custom Features
	custom_features = extract_features.Compute_Bitrate_Custom_Features(
		custom_features_names=custom_features_names,
		Meta_Information=Meta_Information,
		temporal_low_level_features=temporal_low_level_features
	)

	# Extracting VIF Features
	VIF_Features = extract_features.Extract_VIF_Features(
		video_filenames=video_filenames,
		vif_setting=vif_setting,
		vif_features_list=vif_features_list,
		per_frame=per_frame,
		per_frame_features_flatten=per_frame_features_flatten
	)

	X1 = {}
	X2 = {}
	y = {}
	CRFs_after_constraints = {}

	for video_file in video_filenames:
		#  No.of RQ-points obtained by compressing the uncompressed video under different settings
		num_samples = Meta_Information[video_file].shape[0]

		# Finding CRFs after after applying quality constraints.
		CRFs_after_constraints[video_file] = {}

		for _,res in enumerate(defaults.resolutions):
			CRFs_after_constraints[video_file][res] = []
			scaled_h = np.round(res[1]/3840, decimals=4)

			for i in range(Meta_Information[video_file].shape[0]):
				h = Meta_Information[video_file][i,-1]

				if (np.isclose(np.round(h, decimals=4), scaled_h)):
					CRFs_after_constraints[video_file][res].append(
						Meta_Information[video_file][i,-3]
					)

		# Target: Quality
		Target = np.expand_dims(Meta_Information[video_file][:,1], axis=-1)

		# LLF_Data
		LLF_Data = np.repeat(np.expand_dims(features[video_file], axis=0), num_samples, axis=0)

		# Custom_Data
		Custom_Data = custom_features[video_file]

		# Matching temporal-length of LLF_Data and Custom_Data
		if temporal_low_level_features:
			min_temporal_length = min(LLF_Data.shape[1], Custom_Data.shape[1])
			LLF_Data = LLF_Data[:,-min_temporal_length:,:]
			Custom_Data = Custom_Data[:,-min_temporal_length:,:]

		# VIF_Data
		VIF_Data = np.repeat(np.expand_dims(VIF_Features[video_file], axis=0), num_samples, axis=0)
		temporal_length = VIF_Data.shape[1]

		# Repeating Meta_Data along temporal-axis
		# Meta_Data containing [bitrate, width, height]
		Meta_Data = Meta_Information[video_file][:,[0,3,4]]
		Meta_Data = np.repeat(np.expand_dims(Meta_Data, axis=1), temporal_length, axis=1)

		# Final Features
		if len(custom_features_names) == 0:
			X1[video_file] = LLF_Data
		else:
			X1[video_file] = np.concatenate([LLF_Data, Custom_Data], axis=-1)

		Final_Features = np.concatenate([VIF_Data, Meta_Data], axis=-1)
		if (per_frame == False) or (per_frame == True and per_frame_features_flatten == True):
			Final_Features = Final_Features[:,0,:]

		X2[video_file] = Final_Features
		y[video_file] = Target

		# Scaling
		if quality_metric == "vmaf":
			y[video_file] = y[video_file]/100.0

	if (per_frame==False and temporal_low_level_features==False) or (per_frame==True and per_frame_features_flatten==True and temporal_low_level_features==False):
		X = {}
		for video_file in video_filenames:
			# Concatenating
			X[video_file] = np.concatenate([X1[video_file], X2[video_file]], axis=-1)

			# Type Casting
			X[video_file] = X[video_file].astype(np.float32)
			y[video_file] = y[video_file].astype(np.float32)

			# Rounding
			X[video_file] = np.round(X[video_file], decimals=4)
			y[video_file] = np.round(y[video_file], decimals=4)

		return X,y,CRFs_after_constraints
	
	elif per_frame==True and per_frame_features_flatten==False and temporal_low_level_features==True:
		X = {}
		for video_file in video_filenames:
			# Concatenating
			min_temporal_length = min(X1[video_file].shape[1], X2[video_file].shape[1])
			X[video_file] = np.concatenate([X1[video_file][:,-min_temporal_length:,:], X2[video_file][:,-min_temporal_length:,:]], axis=-1)

			# Type Casting
			X[video_file] = X[video_file].astype(np.float32)
			y[video_file] = y[video_file].astype(np.float32)

			# Rounding
			X[video_file] = np.round(X[video_file], decimals=4)
			y[video_file] = np.round(y[video_file], decimals=4)

		return X,y,CRFs_after_constraints
	
	else:
		for video_file in video_filenames:
			# Type Casting
			X1[video_file] = X1[video_file].astype(np.float32)
			X2[video_file] = X2[video_file].astype(np.float32)
			y[video_file] = y[video_file].astype(np.float32)

			# Rounding
			X1[video_file] = np.round(X1[video_file], decimals=4)
			X2[video_file] = np.round(X2[video_file], decimals=4)
			y[video_file] = np.round(y[video_file], decimals=4)

		return X1,X2,y,CRFs_after_constraints


# Predict Bitrate Ladder using CrossOver Bitrates
def Predict_CrossOver_Bitrates_Bitrate_Ladder(
	Models:list,
	Feature_Indices:list,
	X:np.array,
	evaluation_bitrates:list
):
	"""
	Function to return Bitrate-Ladder for corresponding evaluation_bitrates using Quality Prediction models.
	Args:
		Models (list): Model used to predict.
		Feature_Indices (list): Feature Indices.
		X (np.array): List of features for each cross-over bitrate.
		evaluation_bitrates (list): List of evaluation_bitrates present to be in the bitrate ladder.
	Returns:
		(dict): The bitrate-ladder i.e a dictionary {bitrate: resolution} containing the bitrate as key and the resolution it should be encoded as value for the provided evaluation_bitrates.
	"""
	# Assertions
	assert len(Models) == len(Feature_Indices) == len(defaults.resolutions)-1, "The length of list of models and X should be no.of resolutions - 1."

	# Resolutions
	Resolutions = defaults.resolutions

	# Predicting Cross-Over Bitrates
	CrossOver_Bitrates = []

	for i in range(len(defaults.resolutions)-1):
		x = np.concatenate([X, np.asarray(CrossOver_Bitrates).reshape(1,-1)], axis=-1)
		
		# Rounding
		x = np.round(x, decimals=4)
		
		x = x[...,Feature_Indices[i]]
		x = x.reshape(1, -1)
		model = Models[i]
		y_pred = model.predict(x)
		CrossOver_Bitrates.append(y_pred[0])

	# Calculating Bitrate-Ladder
	Bitrate_Ladder = {}
	for i in range(len(evaluation_bitrates)):
		# Switching happens to higher resolution when bitrate >= crossover_bitrate of corresponding higher resolution.
		b = evaluation_bitrates[i]
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


# Predicting Bitrate Ladder
def Predict_Bitrate_Ladder(
	Model:any,
	X:np.array,
	evaluation_bitrates:list
):
	"""
	Function to return Bitrate-Ladder for corresponding evaluation_bitrates using Quality Prediction models.
	Args:
		Model (any): Model used to predict.
		X (np.array): List of features for each cross-over bitrate.
		evaluation_bitrates (list): List of evaluation_bitrates present to be in the bitrate ladder.
	Returns:
		(dict): The bitrate-ladder i.e a dictionary {bitrate: resolution} containing the bitrate as key and the resolution it should be encoded as value for the provided evaluation_bitrates.
	"""
	# Resolutions
	Resolutions = defaults.resolutions

	# Bitrate Ladder
	Bitrate_Ladder = {}

	for i in range(len(evaluation_bitrates)):
		b = evaluation_bitrates[i]
		x = X[i*len(Resolutions):(i+1)*len(Resolutions)]

		if ("sklearn" in str(type(Model))):
			# Sklearn Model
			y_pred = Model.predict(x).flatten()
		else:
			assert False, "Invalid Model"

		y_pred = np.clip(y_pred, 0, 1)
		Bitrate_Ladder[b] = Resolutions[np.argmax(y_pred)]

	return Bitrate_Ladder