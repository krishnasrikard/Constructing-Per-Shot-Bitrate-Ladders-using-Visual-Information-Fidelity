import numpy as np
import matplotlib.pyplot as plt
import scipy

import os, sys, warnings
import pickle
from tqdm import tqdm
sys.path.append("/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity")
import functions.extract_functions as extract_functions
import functions.extract_features as extract_features
import functions.IO_functions as IO_functions
import bitrate_ladder_construction.BL_functions.cross_over_rate_control as cross_over_rate_control
import defaults


# Extract Cross-Over Bitrates
def Extract_CrossOver_Bitrates(
	codec:str,
	preset:str,
	quality_metric:str,
	video_filenames:list,
	Resolutions_Considered:list,
	CRFs_Considered:list,
	bitrates_Considered:list,
	QPs_Considered:list,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate
):
	"""
	Function to extract rate-quality features of uncompressed videos when compressed using different settings

	Args:
		codec (str): Codec used to generate RQ points that need to be extracted. Options: ["libx265", "libx264"]
		preset (str): Preset used to generate RQ points that need to be extracted. Options: ["slow", "medium", "fast", "veryfast", "ultrafast"]
		quality_metric (str): Quality Metric to consider.
		video_filenames (list): List of video filenames to be considered for feature-extraction.
		Resolutions_Considered (list): Resolutions to be considered.
		CRFs_Considered (list): CRFs to be considered.
		bitrates_Considered (list): Bitrates to be considered.
		QPs_Considered (list): QPs to be considered.
		min_quality (float): Minimum quality to be considered for in output pairs/info.
		max_quality (float): Maximum quality to be considered for in output pairs/info.
		min_bitrate (float): Minimum bitrate (in kbps) to be considered for in output pairs/info.
		max_bitrate (float): Maximum bitrate (in kbps) to be considered for in output pairs/info.
	Returns:
		Meta_Information (dict): Dictionary containing rate-quality information of uncompressed videos compressed with different encoding settings.
	"""
	CrossOver_Bitrates = {}

	for video_file in video_filenames:
		# Cross-Over Bitrate Information
		if CRFs_Considered is not None:
			video_rq_points_info = IO_functions.read_create_jsonfile(os.path.join(defaults.rq_points_dataset_path, codec, preset, video_file, "crfs.json"))
		elif bitrates_Considered is not None:
			video_rq_points_info = IO_functions.read_create_jsonfile(os.path.join(defaults.rq_points_dataset_path, codec, preset, video_file, "bitrates.json"))
		elif QPs_Considered is not None:
			video_rq_points_info = IO_functions.read_create_jsonfile(os.path.join(defaults.rq_points_dataset_path, codec, preset, video_file, "qps.json"))
		else:
			assert False, "Only one of CRFs/bitrates/QPs should be given."

		RQ_pairs = extract_functions.Extract_RQ_Information(
			video_rq_points_info=video_rq_points_info,
			quality_metric=quality_metric,
			resolutions=Resolutions_Considered,
			CRFs=CRFs_Considered,
			bitrates=bitrates_Considered,
			QPs=QPs_Considered,
			min_quality=min_quality,
			max_quality=max_quality,
			min_bitrate=min_bitrate,
			max_bitrate=max_bitrate,
			set_bitrate_log_base=2
		)

		CrossOver_Bitrates[video_file] = cross_over_rate_control.CrossOver_Bitrates(
			RQ_pairs=RQ_pairs,
			Resolutions=Resolutions_Considered
		)

	return CrossOver_Bitrates


# Creating Dataset for Cross-Over Bitrate Prediction
def LowLevelFeatures_CrossOverBitrates_Dataset(
	codec:str,
	preset:str,
	quality_metric:str,
	features_names:list,
	video_filenames:list,
	temporal_low_level_features:bool,
	Resolutions_Considered:list,
	CRFs_Considered:list,
	bitrates_Considered:list,
	QPs_Considered:list,
	high_res:tuple,
	low_res:tuple,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate
):
	"""
	Args:
		codec (str): Codec used to generate RQ points that need to be extracted. Options: ["libx265", "libx264"]
		preset (str): Preset used to generate RQ points that need to be extracted. Options: ["slow", "medium", "fast", "veryfast", "ultrafast"]
		quality_metric (str): Quality Metric to consider.
		features_names (list): List of features to be considered. 
		video_filenames (list): List of video filenames to be considered for feature-extraction.
		temporal_low_level_features (bool): If True, everything is extracted per frame instead of pooling using various statistics.
		Resolutions_Considered (list): Resolutions to be considered.
		CRFs_Considered (list): CRFs to be considered.
		bitrates_Considered (list): Bitrates to be considered.
		QPs_Considered (list): QPs to be considered.
		high_res (tuple): The value of high-resolution of a cross-over bitrate point.
		low_res (tuple): The value of low-resolution of a cross-over bitrate point.
		min_quality (float): Minimum quality to be considered for in output pairs/info.
		max_quality (float): Maximum quality to be considered for in output pairs/info.
		min_bitrate (float): Minimum bitrate (in kbps) to be considered for in output pairs/info.
		max_bitrate (float): Maximum bitrate (in kbps) to be considered for in output pairs/info.
	Returns:
		(np.array): Input Data
		(np.array): Target Data
	"""
	Resolutions_Considered = sorted(Resolutions_Considered, reverse=True)
	CrossOver_Bitrates = Extract_CrossOver_Bitrates(
		codec=codec,
		preset=preset,
		quality_metric=quality_metric,
		video_filenames=video_filenames,
		Resolutions_Considered=Resolutions_Considered,
		CRFs_Considered=CRFs_Considered,
		bitrates_Considered=bitrates_Considered,
		QPs_Considered=QPs_Considered,
		min_quality=min_quality,
		max_quality=max_quality,
		min_bitrate=min_bitrate,
		max_bitrate=max_bitrate
	)
	features = extract_features.Extract_Low_Level_Features(
		features_names=features_names,
		video_filenames=video_filenames,
		temporal_low_level_features=temporal_low_level_features,
	)

	X = []
	Targets = []

	high_res_index = Resolutions_Considered.index(high_res)
	low_res_index = Resolutions_Considered.index(low_res)
	assert high_res_index == low_res_index - 1, "Resolutions should be consecutive"

	for video_file in video_filenames:
		X.append(features[video_file])
		Targets.append(CrossOver_Bitrates[video_file])

	X = np.asarray(X)
	y = np.asarray(Targets)[:, high_res_index:high_res_index+1]

	# Type-Casting
	X = X.astype(np.float32)
	y = y.astype(np.float32)

	# Rounding
	X = np.round(X, decimals=4)
	y = np.round(y, decimals=4)

	return X,y


# Creating Datasets for Quality Prediction
def Metadata_Quality_Dataset(
	codec:str,
	preset:str,
	quality_metric:str,
	video_filenames:list,
	Resolutions_Considered:list,
	CRFs_Considered:list,
	bitrates_Considered:list,
	QPs_Considered:list,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate,
):
	"""
	Args:
		codec (str): Codec used to generate RQ points that need to be extracted. Options: ["libx265", "libx264"]
		preset (str): Preset used to generate RQ points that need to be extracted. Options: ["slow", "medium", "fast", "veryfast", "ultrafast"]
		quality_metric (str): Quality Metric to consider.
		video_filenames (list): List of video filenames to be considered for feature-extraction.
		Resolutions_Considered (list): Resolutions to be considered.
		CRFs_Considered (list): CRFs to be considered.
		bitrates_Considered (list): Bitrates to be considered.
		QPs_Considered (list): QPs to be considered.
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
		CRFs_Considered=CRFs_Considered,
		bitrates_Considered=bitrates_Considered,
		QPs_Considered=QPs_Considered,
		min_quality=min_quality,
		max_quality=max_quality,
		min_bitrate=min_bitrate,
		max_bitrate=max_bitrate
	)
	
	X = []
	y = []

	for video_file in video_filenames:
		# No.of RQ-points obtained by compressing the uncompressed video under different settings
		num_samples = Meta_Information[video_file].shape[0]

		# Target: Quality
		Target = np.expand_dims(Meta_Information[video_file][:,1], axis=-1)

		# Repeating Meta_Data along temporal-axis
		# Meta_Data containing [bitrate, width, height]
		Meta_Data = Meta_Information[video_file][:,[0,3,4]]

		X.append(Meta_Data)
		y.append(Target)

	# Reshaping
	X = np.concatenate(X, axis=0)
	y = np.concatenate(y, axis=0)

	# Scaling
	if quality_metric == "vmaf":
		y = y/100.0

	# Type-Casting
	X = X.astype(np.float32)
	y = y.astype(np.float32)

	# Rounding
	X = np.round(X, decimals=4)
	y = np.round(y, decimals=4)

	return X,y


def LowLevelFeatures_Quality_Dataset(
	codec:str,
	preset:str,
	quality_metric:str,
	features_names:list,
	video_filenames:list,
	temporal_low_level_features:bool,
	Resolutions_Considered:list,
	CRFs_Considered:list,
	bitrates_Considered:list,
	QPs_Considered:list,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate
):
	"""
	Args:
		codec (str): Codec used to generate RQ points that need to be extracted. Options: ["libx265", "libx264"]
		preset (str): Preset used to generate RQ points that need to be extracted. Options: ["slow", "medium", "fast", "veryfast", "ultrafast"]
		quality_metric (str): Quality Metric to consider.
		features_names (list): List of features to be considered. 
		video_filenames (list): List of video filenames to be considered for feature-extraction.
		temporal_low_level_features (bool): If True, everything is extracted per frame instead of pooling using various statistics.
		Resolutions_Considered (list): Resolutions to be considered.
		CRFs_Considered (list): CRFs to be considered.
		bitrates_Considered (list): Bitrates to be considered.
		QPs_Considered (list): QPs to be considered.
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
		CRFs_Considered=CRFs_Considered,
		bitrates_Considered=bitrates_Considered,
		QPs_Considered=QPs_Considered,
		min_quality=min_quality,
		max_quality=max_quality,
		min_bitrate=min_bitrate,
		max_bitrate=max_bitrate
	)
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
	
	X = []
	y = []

	for video_file in video_filenames:
		# No.of RQ-points obtained by compressing the uncompressed video under different settings
		num_samples = Meta_Information[video_file].shape[0]

		# Target: Quality
		Target = np.expand_dims(Meta_Information[video_file][:,1], axis=-1)
		
		# LLF_Data
		LLF_Data = np.repeat(np.expand_dims(features[video_file], axis=0), num_samples, axis=0)

		# Custom_Data
		Custom_Data = custom_features[video_file]

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

		X.append(Final_Features)
		y.append(Target)

	# Reshaping
	X = np.concatenate(X, axis=0)
	y = np.concatenate(y, axis=0)

	# Scaling
	if quality_metric == "vmaf":
		y = y/100.0

	# Type-Casting
	X = X.astype(np.float32)
	y = y.astype(np.float32)

	# Rounding
	X = np.round(X, decimals=4)
	y = np.round(y, decimals=4)
	
	return X,y


def VIFFeatures_Quality_Dataset(
	codec:str,
	preset:str,
	quality_metric:str,
	video_filenames:list,
	Resolutions_Considered:list,
	CRFs_Considered:list,
	bitrates_Considered:list,
	QPs_Considered:list,
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
		codec (str): Codec used to generate RQ points that need to be extracted. Options: ["libx265", "libx264"]
		preset (str): Preset used to generate RQ points that need to be extracted. Options: ["slow", "medium", "fast", "veryfast", "ultrafast"]
		quality_metric (str): Quality Metric to consider.
		video_filenames (list): List of video filenames to be considered for feature-extraction.
		Resolutions_Considered (list): Resolutions to be considered.
		CRFs_Considered (list): CRFs to be considered.
		bitrates_Considered (list): Bitrates to be considered.
		QPs_Considered (list): QPs to be considered.
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
	# Extracting all VIF Information and RQ Information
	Meta_Information = extract_features.Extract_RQ_Features(
		codec=codec,
		preset=preset,
		quality_metric=quality_metric,
		video_filenames=video_filenames,
		Resolutions_Considered=Resolutions_Considered,
		CRFs_Considered=CRFs_Considered,
		bitrates_Considered=bitrates_Considered,
		QPs_Considered=QPs_Considered,
		min_quality=min_quality,
		max_quality=max_quality,
		min_bitrate=min_bitrate,
		max_bitrate=max_bitrate
	)
	VIF_Features = extract_features.Extract_VIF_Features(
		video_filenames=video_filenames,
		vif_setting=vif_setting,
		vif_features_list=vif_features_list,
		per_frame=per_frame,
		per_frame_features_flatten=per_frame_features_flatten
	)

	X = []
	y = []

	for video_file in video_filenames:
		# No.of RQ-points obtained by compressing the uncompressed video under different settings
		num_samples = Meta_Information[video_file].shape[0]

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

		X.append(Final_Features)
		y.append(Target)

	# Reshaping
	X = np.concatenate(X, axis=0)
	y = np.concatenate(y, axis=0)

	# Scaling
	if quality_metric == "vmaf":
		y = y/100.0

	# Type-Casting
	X = X.astype(np.float32)
	y = y.astype(np.float32)

	# Rounding
	X = np.round(X, decimals=4)
	y = np.round(y, decimals=4)

	return X,y


def LowLevelFeatures_VIFFeatures_Quality_Dataset(
	codec:str,
	preset:str,
	quality_metric:str,
	features_names:list,
	video_filenames:list,
	temporal_low_level_features:bool,
	Resolutions_Considered:list,
	CRFs_Considered:list,
	bitrates_Considered:list,
	QPs_Considered:list,
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
		codec (str): Codec used to generate RQ points that need to be extracted. Options: ["libx265", "libx264"]
		preset (str): Preset used to generate RQ points that need to be extracted. Options: ["slow", "medium", "fast", "veryfast", "ultrafast"]
		quality_metric (str): Quality Metric to consider.
		features_names (list): List of features to be considered. 
		video_filenames (list): List of video filenames to be considered for feature-extraction.
		temporal_low_level_features (bool): If True, everything is extracted per frame instead of pooling using various statistics.
		Resolutions_Considered (list): Resolutions to be considered.
		CRFs_Considered (list): CRFs to be considered.
		bitrates_Considered (list): Bitrates to be considered.
		QPs_Considered (list): QPs to be considered.
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
		CRFs_Considered=CRFs_Considered,
		bitrates_Considered=bitrates_Considered,
		QPs_Considered=QPs_Considered,
		min_quality=min_quality,
		max_quality=max_quality,
		min_bitrate=min_bitrate,
		max_bitrate=max_bitrate
	)
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
	# Extracting VIF Information
	VIF_Features = extract_features.Extract_VIF_Features(
		video_filenames=video_filenames,
		vif_setting=vif_setting,
		vif_features_list=vif_features_list,
		per_frame=per_frame,
		per_frame_features_flatten=per_frame_features_flatten
	)

	X1 = []
	X2 = []
	y = []

	for video_file in video_filenames:
		#  No.of RQ-points obtained by compressing the uncompressed video under different settings
		num_samples = Meta_Information[video_file].shape[0]

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
			X1.append(LLF_Data)
		else:
			X1.append(np.concatenate([LLF_Data, Custom_Data], axis=-1))

		Final_Features = np.concatenate([VIF_Data, Meta_Data], axis=-1)
		if (per_frame == False) or (per_frame == True and per_frame_features_flatten == True):
			Final_Features = Final_Features[:,0,:]

		X2.append(Final_Features)
		y.append(Target)
		
	# Concatenating
	X1 = np.concatenate(X1, axis=0)
	X2 = np.concatenate(X2, axis=0)
	y = np.concatenate(y, axis=0)

	# Scaling
	if quality_metric == "vmaf":
		y = y/100.0

	if (per_frame==False and temporal_low_level_features==False) or (per_frame==True and per_frame_features_flatten==True and temporal_low_level_features==False):
		# Concatenating
		X = np.concatenate([X1, X2], axis=-1)

		# Type Casting
		X = X.astype(np.float32)
		y = y.astype(np.float32)

		# Rounding
		X = np.round(X, decimals=4)
		y = np.round(y, decimals=4)

		return X,y
	
	elif per_frame==True and per_frame_features_flatten==False and temporal_low_level_features==True:
		# Concatenating
		min_temporal_length = min(X1.shape[1], X2.shape[1])
		X = np.concatenate([X1[:,-min_temporal_length:,:], X2[:,-min_temporal_length:,:]], axis=-1)

		# Type Casting
		X = X.astype(np.float32)
		y = y.astype(np.float32)

		# Rounding
		X = np.round(X, decimals=4)
		y = np.round(y, decimals=4)

		return X, y
	
	else:
		# Type Casting
		X1 = X1.astype(np.float32)
		X2 = X2.astype(np.float32)
		y = y.astype(np.float32)

		# Rounding
		X1 = np.round(X1, decimals=4)
		X2 = np.round(X2, decimals=4)
		y = np.round(y, decimals=4)

		return X1,X2,y



# Quality Prediction Functions
def Predict_Metadata_Quality(
	Model:any,
	codec:str,
	preset:str,
	quality_metric:str,
	video_filenames:list,
	Resolutions_Considered:list, 
	CRFs_Considered:list,
	bitrates_Considered:list,
	QPs_Considered:list,
	feature_indices:list,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate
):
	"""
	Predicting Quality Estimations of each of the video files per each resolution and crf value. The order of resolutions values matter and along crfs axis, the values are considered from highest to lowest crf or in ascending order of bitrate.
	Args:
		Model (any): sklearn model used for evaluation.
		codec (str): Codec used to generate RQ points that need to be extracted. Options: ["libx265", "libx264"]
		preset (str): Preset used to generate RQ points that need to be extracted. Options: ["slow", "medium", "fast", "veryfast", "ultrafast"]
		quality_metric (str): Quality Metric to consider.
		video_filenames (list): List of video filenames to be considered for feature-extraction.
		Resolutions_Considered (list): Resolutions to be considered.
		CRFs_Considered (list): CRFs to be considered.
		bitrates_Considered (list): Bitrates to be considered.
		QPs_Considered (list): QPs to be considered.
		feature_indices (list): List of indices to considered after combined feature elimination of Low-Level features.
		min_quality (float): Minimum quality to be considered for in output pairs/info.
		max_quality (float): Maximum quality to be considered for in output pairs/info.
		min_bitrate (float): Minimum bitrate (in kbps) to be considered for in output pairs/info.
		max_bitrate (float): Maximum bitrate (in kbps) to be considered for in output pairs/info.
	Returns:
		y_pred_Results (np.array): Predicted Quality Estimations
		y_Results (np.array): True Quality Estimations
		bitrates_Results (np.array): Corresponding bitrate values
	"""
	# Results
	y_pred_Results = -np.inf*np.ones((len(video_filenames), len(Resolutions_Considered), len(CRFs_Considered)))
	y_Results = -np.inf*np.ones((len(video_filenames), len(Resolutions_Considered), len(CRFs_Considered)))
	bitrate_Results = -np.inf*np.ones((len(video_filenames), len(Resolutions_Considered), len(CRFs_Considered)))

	for i,video_file in enumerate(video_filenames):
		for j,res in enumerate(Resolutions_Considered):
			# Meta Information for each resolution of each video
			Meta_Information = extract_features.Extract_RQ_Features(
				codec=codec,
				preset=preset,
				quality_metric=quality_metric,
				video_filenames=[video_file],
				Resolutions_Considered=[res],
				CRFs_Considered=CRFs_Considered,
				bitrates_Considered=bitrates_Considered,
				QPs_Considered=QPs_Considered,
				min_quality=min_quality,
				max_quality=max_quality,
				min_bitrate=min_bitrate,
				max_bitrate=max_bitrate
			)

			# Dataset with video_file and res
			X,y = Metadata_Quality_Dataset(
				codec=codec,
				preset=preset,
				quality_metric=quality_metric,
				video_filenames=[video_file],
				Resolutions_Considered=[res],
				CRFs_Considered=CRFs_Considered,
				bitrates_Considered=bitrates_Considered,
				QPs_Considered=QPs_Considered,
				min_quality=min_quality,
				max_quality=max_quality,
				min_bitrate=min_bitrate,
				max_bitrate=max_bitrate
			)

			# Feature Selection
			if feature_indices is not None:
				X = X[...,feature_indices]

			# Predictions
			if ("sklearn" in str(type(Model))) or ("lineartree" in str(type(Model))):
				# Sklearn Model
				y_pred = Model.predict(X).flatten()
			else:
				assert False, "Invalid Model"

			y = y.flatten()
			assert y_pred.shape == y.shape

			# Set Results
			y_pred_Results[i,j,:y_pred.shape[0]] = y_pred
			y_Results[i,j,:y_pred.shape[0]] = y
			bitrate_Results[i,j,:y_pred.shape[0]] = Meta_Information[video_file][:,0]

	return y_pred_Results, y_Results, bitrate_Results


def Predict_LowLevelFeatures_Quality(
	Model:any,
	codec:str,
	preset:str,
	quality_metric:str,
	features_names:list,
	video_filenames:list,
	temporal_low_level_features:bool,
	Resolutions_Considered:list, 
	CRFs_Considered:list,
	bitrates_Considered:list,
	QPs_Considered:list,
	feature_indices:list,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate
):
	"""
	Predicting Quality Estimations of each of the video files per each resolution and crf value. The order of resolutions values matter and along crfs axis, the values are considered from highest to lowest crf or in ascending order of bitrate.
	Args:
		Model (any): sklearn model used for evaluation.
		codec (str): Codec used to generate RQ points that need to be extracted. Options: ["libx265", "libx264"]
		preset (str): Preset used to generate RQ points that need to be extracted. Options: ["slow", "medium", "fast", "veryfast", "ultrafast"]
		quality_metric (str): Quality Metric to consider.
		features_names (list): List of features to be considered. 
		feature_indices (list): List of indices to considered after feature elimination i.e indices along last axis/dim of features_names to consider.
		video_filenames (list): List of video filenames to be considered for feature-extraction.
		temporal_low_level_features (bool): If True, everything is extracted per frame instead of pooling using various statistics.
		Resolutions_Considered (list): Resolutions to be considered.
		CRFs_Considered (list): CRFs to be considered.
		bitrates_Considered (list): Bitrates to be considered.
		QPs_Considered (list): QPs to be considered.
		feature_indices (list): List of indices to considered after combined feature elimination of Low-Level features.
		min_quality (float): Minimum quality to be considered for in output pairs/info.
		max_quality (float): Maximum quality to be considered for in output pairs/info.
		min_bitrate (float): Minimum bitrate (in kbps) to be considered for in output pairs/info.
		max_bitrate (float): Maximum bitrate (in kbps) to be considered for in output pairs/info.
	Returns:
		y_pred_Results (np.array): Predicted Quality Estimations
		y_Results (np.array): True Quality Estimations
		bitrates_Results (np.array): Corresponding bitrate values
	"""
	# Results
	y_pred_Results = -np.inf*np.ones((len(video_filenames), len(Resolutions_Considered), len(CRFs_Considered)))
	y_Results = -np.inf*np.ones((len(video_filenames), len(Resolutions_Considered), len(CRFs_Considered)))
	bitrate_Results = -np.inf*np.ones((len(video_filenames), len(Resolutions_Considered), len(CRFs_Considered)))

	for i,video_file in enumerate(video_filenames):
		for j,res in enumerate(Resolutions_Considered):
			# Meta Information for each resolution of each video
			Meta_Information = extract_features.Extract_RQ_Features(
				codec=codec,
				preset=preset,
				quality_metric=quality_metric,
				video_filenames=[video_file],
				Resolutions_Considered=[res],
				CRFs_Considered=CRFs_Considered,
				bitrates_Considered=bitrates_Considered,
				QPs_Considered=QPs_Considered,
				min_quality=min_quality,
				max_quality=max_quality,
				min_bitrate=min_bitrate,
				max_bitrate=max_bitrate
			)

			# Dataset with video_file and res
			X,y = LowLevelFeatures_Quality_Dataset(
				codec=codec,
				preset=preset,
				quality_metric=quality_metric,
				features_names=features_names,
				video_filenames=[video_file],
				temporal_low_level_features=temporal_low_level_features,
				Resolutions_Considered=[res],
				CRFs_Considered=CRFs_Considered,
				bitrates_Considered=bitrates_Considered,
				QPs_Considered=QPs_Considered,
				min_quality=min_quality,
				max_quality=max_quality,
				min_bitrate=min_bitrate,
				max_bitrate=max_bitrate
			)

			# Feature Selection
			if feature_indices is not None:
				X = X[...,feature_indices]

			# Predictions
			if ("sklearn" in str(type(Model))) or ("lineartree" in str(type(Model))):
				# Sklearn Model
				y_pred = Model.predict(X).flatten()
			else:
				assert False, "Invalid Model"

			y = y.flatten()
			assert y_pred.shape == y.shape

			# Set Results
			y_pred_Results[i,j,:y_pred.shape[0]] = y_pred
			y_Results[i,j,:y_pred.shape[0]] = y
			bitrate_Results[i,j,:y_pred.shape[0]] = Meta_Information[video_file][:,0]

	return y_pred_Results, y_Results, bitrate_Results


def Predict_VIFFeatures_Quality(
	Model:any,
	codec:str,
	preset:str,
	quality_metric:str,
	video_filenames:list,
	Resolutions_Considered:list,
	CRFs_Considered:list,
	bitrates_Considered:list,
	QPs_Considered:list,
	vif_setting:str,
	vif_features_list:list,
	per_frame:bool,
	per_frame_features_flatten:bool,
	feature_indices:list,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate
):
	"""
	Predicting Quality Estimations of each of the video files per each resolution and crf value. The order of resolutions values matter and along crfs axis, the values are considered from highest to lowest crf or in ascending order of bitrate.
	Args:
		Model (any): sklearn model used for evaluation.
		codec (str): Codec used to generate RQ points that need to be extracted. Options: ["libx265", "libx264"]
		preset (str): Preset used to generate RQ points that need to be extracted. Options: ["slow", "medium", "fast", "veryfast", "ultrafast"]
		quality_metric (str): Quality Metric to consider.
		video_filenames (list): List of video filenames to be considered for feature-extraction.
		Resolutions_Considered (list): Resolutions to be considered.
		CRFs_Considered (list): CRFs to be considered.
		bitrates_Considered (list): Bitrates to be considered.
		QPs_Considered (list): QPs to be considered.
		vif_setting (str): Select one VIF setting i.e how VIF information extracted from compressed videos should be used. Options: ["per_scale", "per_subband", "per_eigen_value"]
		vif_features_list (list): List of VIF features to be considered as input features for the dataset. Options: ["vif_info", "mean_abs_frame_diff", "diff_vif_info"]
		per_frame (bool): Whether features should be given per frame or average along temporal-axis.
		per_frame_features_flatten (bool): Whether to flatten features per each frames to a vector of shape(frames*features)
		features_indices (list): List of indices of VIF features to consider obtained from feature-selection.
		min_quality (float): Minimum quality to be considered for in output pairs/info.
		max_quality (float): Maximum quality to be considered for in output pairs/info.
		min_bitrate (float): Minimum bitrate (in kbps) to be considered for in output pairs/info.
		max_bitrate (float): Maximum bitrate (in kbps) to be considered for in output pairs/info.
	Returns:
		y_pred_Results (np.array): Predicted Quality Estimations
		y_Results (np.array): True Quality Estimations
		bitrates_Results (np.array): Corresponding bitrate values
	"""
	# Results
	y_pred_Results = -np.inf*np.ones((len(video_filenames), len(Resolutions_Considered), len(CRFs_Considered)))
	y_Results = -np.inf*np.ones((len(video_filenames), len(Resolutions_Considered), len(CRFs_Considered)))
	bitrate_Results = -np.inf*np.ones((len(video_filenames), len(Resolutions_Considered), len(CRFs_Considered)))

	for i,video_file in enumerate(video_filenames):
		for j,res in enumerate(Resolutions_Considered):
			# Meta Information for each resolution of each video
			Meta_Information = extract_features.Extract_RQ_Features(
				codec=codec,
				preset=preset,
				quality_metric=quality_metric,
				video_filenames=[video_file],
				Resolutions_Considered=[res],
				CRFs_Considered=CRFs_Considered,
				bitrates_Considered=bitrates_Considered,
				QPs_Considered=QPs_Considered,
				min_quality=min_quality,
				max_quality=max_quality,
				min_bitrate=min_bitrate,
				max_bitrate=max_bitrate
			)

			# Dataset with video_file and res
			X,y = VIFFeatures_Quality_Dataset(
				codec=codec,
				preset=preset,
				quality_metric=quality_metric,
				video_filenames=[video_file],
				Resolutions_Considered=[res],
				CRFs_Considered=CRFs_Considered,
				bitrates_Considered=bitrates_Considered,
				QPs_Considered=QPs_Considered,
				vif_setting=vif_setting,
				vif_features_list=vif_features_list,
				per_frame=per_frame,
				per_frame_features_flatten=per_frame_features_flatten,
				min_quality=min_quality,
				max_quality=max_quality,
				min_bitrate=min_bitrate,
				max_bitrate=max_bitrate
			)

			# Feature Selection
			if feature_indices is not None:
				X = X[...,feature_indices]

			# Predictions
			if ("sklearn" in str(type(Model))) or ("lineartree" in str(type(Model))):
				# Sklearn Model
				y_pred = Model.predict(X).flatten()
			else:
				assert False, "Invalid Model"

			y = y.flatten()
			assert y_pred.shape == y.shape

			# Set Results
			y_pred_Results[i,j,:y_pred.shape[0]] = y_pred
			y_Results[i,j,:y_pred.shape[0]] = y
			bitrate_Results[i,j,:y_pred.shape[0]] = Meta_Information[video_file][:,0]

	return y_pred_Results, y_Results, bitrate_Results


def Predict_LowLevelFeatures_VIFFeatures_Quality(
	Model:any,
	codec:str,
	preset:str,
	quality_metric:str,
	features_names:list,
	video_filenames:list,
	temporal_low_level_features:bool,
	Resolutions_Considered:list,
	CRFs_Considered:list,
	bitrates_Considered:list,
	QPs_Considered:list,
	vif_setting:str,
	vif_features_list:list,
	per_frame:bool,
	per_frame_features_flatten:bool,
	feature_indices:list,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate
):
	"""
	Predicting Quality Estimations of each of the video files per each resolution and crf value. The order of resolutions values matter and along crfs axis, the values are considered from highest to lowest crf or in ascending order of bitrate.
	Args:
		codec (str): Codec used to generate RQ points that need to be extracted. Options: ["libx265", "libx264"]
		preset (str): Preset used to generate RQ points that need to be extracted. Options: ["slow", "medium", "fast", "veryfast", "ultrafast"]
		quality_metric (str): Quality Metric to consider.
		features_names (list): List of features to be considered.
		video_filenames (list): List of video filenames to be considered for feature-extraction.
		temporal_low_level_features (bool): If True, everything is extracted per frame instead of pooling using various statistics.
		Resolutions_Considered (list): Resolutions to be considered.
		CRFs_Considered (list): CRFs to be considered.
		bitrates_Considered (list): Bitrates to be considered.
		QPs_Considered (list): QPs to be considered.
		vif_setting (str): Select one VIF setting i.e how VIF information extracted from compressed videos should be used. Options: ["per_scale", "per_subband", "per_eigen_value"]
		vif_features_list (list): List of VIF features to be considered as input features for the dataset. Options: ["vif_info", "mean_abs_frame_diff", "diff_vif_info"]
		per_frame (bool): Whether features should be given per frame or average along temporal-axis.
		per_frame_features_flatten (bool): Whether to flatten features per each frames to a vector of shape (frames*features).
		feature_indices (list): List of indices to considered after combined feature elimination of VIF and Low-Level features.
		min_quality (float): Minimum quality to be considered for in output pairs/info.
		max_quality (float): Maximum quality to be considered for in output pairs/info.
		min_bitrate (float): Minimum bitrate (in kbps) to be considered for in output pairs/info.
		max_bitrate (float): Maximum bitrate (in kbps) to be considered for in output pairs/info.
	Returns:
		(np.array): Input Data
		(np.array): Target Data
	"""
	# Results
	y_pred_Results = -np.inf*np.ones((len(video_filenames), len(Resolutions_Considered), len(CRFs_Considered)))
	y_Results = -np.inf*np.ones((len(video_filenames), len(Resolutions_Considered), len(CRFs_Considered)))
	bitrate_Results = -np.inf*np.ones((len(video_filenames), len(Resolutions_Considered), len(CRFs_Considered)))

	for i,video_file in enumerate(video_filenames):
		for j,res in enumerate(Resolutions_Considered):
			# Meta Information for each resolution of each video
			Meta_Information = extract_features.Extract_RQ_Features(
				codec=codec,
				preset=preset,
				quality_metric=quality_metric,
				video_filenames=[video_file],
				Resolutions_Considered=[res],
				CRFs_Considered=CRFs_Considered,
				bitrates_Considered=bitrates_Considered,
				QPs_Considered=QPs_Considered,
				min_quality=min_quality,
				max_quality=max_quality,
				min_bitrate=min_bitrate,
				max_bitrate=max_bitrate
			)

			# Dataset with video_file and res
			F = LowLevelFeatures_VIFFeatures_Quality_Dataset(
				codec=codec,
				preset=preset,
				quality_metric=quality_metric,
				features_names=features_names,
				video_filenames=[video_file],
				temporal_low_level_features=temporal_low_level_features,
				Resolutions_Considered=[res],
				CRFs_Considered=CRFs_Considered,
				bitrates_Considered=bitrates_Considered,
				QPs_Considered=QPs_Considered,
				vif_setting=vif_setting,
				vif_features_list=vif_features_list,
				per_frame=per_frame,
				per_frame_features_flatten=per_frame_features_flatten,
				min_quality=min_quality,
				max_quality=max_quality,
				min_bitrate=min_bitrate,
				max_bitrate=max_bitrate
			)
			if len(F) == 2:
				X, y = F[0], F[1]
			else:
				X1, X2, y = F[0], F[1], F[2]
			
			
			if (per_frame==False and temporal_low_level_features==False) or (per_frame==True and per_frame_features_flatten==True and temporal_low_level_features==False):
				# Feature Selection
				if feature_indices is not None:
					X = X[...,feature_indices]

				# Predictions
				if ("sklearn" in str(type(Model))) or ("lineartree" in str(type(Model))):
					# Sklearn Model
					y_pred = Model.predict(X).flatten()
				else:
					assert False, "Invalid Model"

			else:
				assert False, "Right now, doesn't support process input with settings provided. The functions only supports the following settings: (per_frame==False and temporal_low_level_features==False) or (per_frame==True and per_frame_features_flatten==True and temporal_low_level_features==False)"


			y = y.flatten()
			assert y_pred.shape == y.shape

			# Set Results
			y_pred_Results[i,j,:y_pred.shape[0]] = y_pred
			y_Results[i,j,:y_pred.shape[0]] = y
			bitrate_Results[i,j,:y_pred.shape[0]] = Meta_Information[video_file][:,0]

	return y_pred_Results, y_Results, bitrate_Results