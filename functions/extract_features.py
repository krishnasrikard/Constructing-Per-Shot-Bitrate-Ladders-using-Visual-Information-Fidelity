import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch

import os, sys, warnings
import pickle
from tqdm import tqdm
warnings.filterwarnings("ignore")
sys.path.append("/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity")
import functions.extract_functions as extract_functions
import functions.IO_functions as IO_functions
import defaults


# Extracting Features
def Extract_RQ_Features(
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
		min_quality (float): Minimum quality to be considered for in output pairs/info. (Default: -np.inf)
		max_quality (float): Maximum quality to be considered for in output pairs/info. (Default: np.inf)
		min_bitrate (float): Minimum bitrate (in kbps) to be considered for in output pairs/info. (Default: -np.inf)
		max_bitrate (float): Maximum bitrate (in kbps) to be considered for in output pairs/info. (Default: np.inf)
	Returns:
		Meta_Information (dict): Dictionary with video_filenames as keys containing (bitrate (in kbps), quality, rate_control_setting_value, w/3840, h/3840) rate-quality features or meta-information of compressed videos obtained from uncompressed videos compressed with different encoding settings. The resolutions compressed videos we consider have width/3840 = height/2160. So, to reduce redundancy of a features, we consider w/3840, h/3840 as features i.e instead of using large values of resolutions as features, we will use scaled resolutions.
	"""
	Meta_Information = {}

	for video_file in video_filenames:
		# Rate-Quality/Meta Information
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

		Meta_Information[video_file] = []
		for res in Resolutions_Considered:
			# The resolutions compressed videos we consider have width/3840 = height/2160. So, to reduce redundancy of a features, we consider w/3840, h/3840 as features i.e instead of using large values of resolutions as features, we will use scaled resolutions.
			if len(RQ_pairs[res]) > 0:
				Res_Info = np.repeat([[res[0]/3840, res[1]/3840]], len(RQ_pairs[res]), axis=0)
				Meta_Information[video_file].append(np.concatenate([RQ_pairs[res], Res_Info], axis=-1))

		Meta_Information[video_file] = np.concatenate(Meta_Information[video_file], axis=0).reshape(-1,5)

	return Meta_Information


def Extract_Low_Level_Features(
	features_names:list,
	video_filenames:list,
	temporal_low_level_features:bool
):
	"""
	Extracting low-level features from uncompressed videos.
	
	Args:
		features_names (list): List of features to be considered.  if temporal=False, or else i.e if temporal=True ["GLCM_contrast", "GLCM_correlation", "GLCM_energy", "GLCM_homogeneity", "TC_mean", "TC_std", "TC_skewness", "TC_kurtosis", "SI", "TI", "E", "h", "L"]
		video_filenames (list): List of video filenames to be considered for feature-extraction.
		temporal_low_level_features (bool): If True, everything is extracted per frame instead of pooling using various statistics.
	Returns:
		features (dict): Dictionary of features_names for uncompressed video_filenames.
	"""
	if temporal_low_level_features == False:
		# Loading Low-Level Features
		df = pd.read_csv(os.path.join(defaults.llf_features_path,"uncompressed_low_level_features_info.csv"))
		filenames = df.loc[:, "filename"].to_numpy()
		Results = df.loc[:, features_names].to_numpy()
		Results = np.round(Results, decimals=6)

		features = {}
		for i,filename in enumerate(filenames):
			if filename in video_filenames:
				features[filename] = np.round(Results[i], decimals=6)
				
		return features
	else:
		# Loading Low-Level Features
		features = {}

		for video_file in video_filenames:
			file_path = os.path.join(defaults.llf_features_path, video_file+".npy")
			extracted_data = np.load(file_path, allow_pickle=True)[()]
			
			min_temporal_length = np.inf
			for f in extracted_data.keys():
				min_temporal_length = min(min_temporal_length, len(extracted_data[f]))

			features[video_file] = np.asarray([extracted_data[i][-min_temporal_length:] for i in features_names]).T
			features[video_file] = np.round(features[video_file], decimals=6)
			
	return features



def Extract_VIF_Features(
	video_filenames:list,
	vif_setting:str,
	vif_features_list:list,
	per_frame:bool,
	per_frame_features_flatten:bool
):
	"""
	Function to extract VIF features of uncompressed videos.

	Args:
		video_filenames (list): List of video filenames to be considered for feature-extraction.
		vif_setting (str): Select one VIF setting i.e how VIF information extracted from compressed videos should be used. Options: ["per_scale", "per_subband", "per_eigen_value"]
		vif_features_list (list): List of VIF features to be considered as input features for the dataset. Options: ["vif_info", "mean_abs_frame_diff", "diff_vif_info"]
		per_frame (bool): Whether features should be given per frame or average along temporal-axis.
		per_frame_features_flatten (bool): Whether to flatten features per each frames to a vector of shape (frames*features).
	Returns:
		(dict): Dictionary containing VIF-Features for each uncompressed video. The video filenames are used as keys and VIF-Features are returned as values.
	"""
	Frame_VIF_Information = {}
	Diff_Frame_Information = {}
	Diff_Frame_VIF_Information = {}
	
	for video_file in video_filenames:
		# Video Spatial and Temporal VIF Info
		video_vif_info = np.load(os.path.join(defaults.vif_information_path, video_file+".npy"), allow_pickle=True)

		# VIF Information extracted for each frame/along temporal-axis
		Frame_VIF_Information[video_file] = []

		# Mean Absolute Frame difference for consecutive frames
		Diff_Frame_Information[video_file] = []

		# Diff-frame VIF Information extracted for each frame/along temporal-axis
		Diff_Frame_VIF_Information[video_file] = []
		
		for i in range(len(video_vif_info)):
			Frame_VIF_Information[video_file].append(video_vif_info[i]["vif_info"])
			Diff_Frame_Information[video_file].append(video_vif_info[i]["mean_abs_frame_diff"])
			Diff_Frame_VIF_Information[video_file].append(video_vif_info[i]["diff_vif_info"])


		Frame_VIF_Information[video_file] = np.asarray(Frame_VIF_Information[video_file])
		Diff_Frame_Information[video_file] = np.asarray(Diff_Frame_Information[video_file])
		Diff_Frame_VIF_Information[video_file] = np.asarray(Diff_Frame_VIF_Information[video_file])
		

	# No.of frames in the uncompressed video
	num_frames = Frame_VIF_Information[video_file].shape[0]
	

	# Creating VIF-Features considering VIF settings
	VIF_Features = {}
	for video_file in video_filenames:
		# Final Features that needs to be considered 
		Final_Features = []

		# Frame VIF Info of Uncompressed Video
		if "vif_info" in vif_features_list:
			F_VIF = Frame_VIF_Information[video_file]

			if vif_setting == "per_scale":
				# Mutual Information per scale as features
				F_VIF = np.mean(np.sum(F_VIF, axis=-1).reshape(num_frames,4,2), axis=-1)
			elif vif_setting == "per_subband":
				# Mutual Information per subband as features
				F_VIF = np.sum(F_VIF, axis=-1)
			elif vif_setting == "per_eigen_value":
				# Entropy Difference along each eigen value as features
				F_VIF = F_VIF
			else:
				assert False, "Invalid VIF-Setting"

			F_VIF = F_VIF.reshape(num_frames, -1)
			Final_Features.append(F_VIF)


		# MAD Frame-Difference VIF Info of Uncompressed Video
		if "mean_abs_frame_diff" in vif_features_list:
			D = Diff_Frame_Information[video_file]

			# Temporal-Difference gives us (num_frames-1) length
			D = D[1:]

			D = D.reshape(num_frames-1, -1)
			Final_Features.append(D)


		# Frame-Difference VIF Info of Uncompressed Video
		if "diff_vif_info" in vif_features_list:
			D_VIF = Diff_Frame_VIF_Information[video_file]

			# Temporal-Difference gives us (num_frames-1) length
			D_VIF = D_VIF[1:,:,:]

			if vif_setting == "per_scale":
				# Mutual Information per scale as features
				D_VIF = np.mean(np.sum(D_VIF, axis=-1).reshape(num_frames-1,4,2), axis=-1)
			elif vif_setting == "per_subband":
				# Mutual Information per subband as features
				D_VIF = np.sum(D_VIF, axis=-1)
			elif vif_setting == "per_eigen_value":
				# Entropy Difference along each eigen value as features
				D_VIF = D_VIF
			else:
				assert False, "Invalid VIF-Setting"

			D_VIF = D_VIF.reshape(num_frames-1, -1)
			Final_Features.append(D_VIF)
		

		# Matching temporal-length of all VIF_Infos
		min_temporal_length = np.inf
		for i in range(len(Final_Features)):
			min_temporal_length = min(min_temporal_length, Final_Features[i].shape[0])

		for i in range(len(Final_Features)):
			Final_Features[i] = Final_Features[i][-min_temporal_length:]


		# Concatenating all VIF Info
		Final_Features = np.concatenate(Final_Features, axis=-1)
		if per_frame == False:
			# If per-frame features are not needed, the average value along temporal-axis is considered
			Final_Features = np.mean(Final_Features, axis=0, keepdims=True)
		else:
			if per_frame_features_flatten:
				Final_Features = np.expand_dims(Final_Features.flatten(), axis=0)


		# Rounding and Assigning Features
		VIF_Features[video_file] = np.round(Final_Features, decimals=4)

	return VIF_Features



def Compute_Bitrate_Custom_Features(
	custom_features_names:list,
	Meta_Information:dict,
	temporal_low_level_features:bool
):
	"""
	Computes Custom Features
	Bitrate Customized Features include Bitrate-Texture-Features
	Args:
		custom_features_names (list): List of custom features to be computed.
		features (dict): Low-Level Features corresponding to features_names for each video-file.
		Meta_Information (dict): Meta-Information for each video-file.
		temporal_low_level_features (bool): If True, everything is extracted per frame instead of pooling using various statistics.
	"""
	# Video Filenames
	video_filenames = list(Meta_Information.keys())

	# Low-Level Features needed for Custom-Features
	features_names = []
	if temporal_low_level_features:
		for cf in custom_features_names:
			features_names += defaults.per_frame_bitrate_texture_features[cf]
	else:
		for cf in custom_features_names:
			features_names += defaults.bitrate_texture_features[cf]
	

	# Extracting Low-Level Features needed for Custom-Features
	features = Extract_Low_Level_Features(
		features_names=features_names,
		video_filenames=video_filenames,
		temporal_low_level_features=temporal_low_level_features
	)

	# Custom_Features
	Custom_Features = {}

	for video_file in video_filenames:
		#  No.of RQ-points obtained by compressing the uncompressed video under different settings
		num_samples = Meta_Information[video_file].shape[0]

		# Low-Level Features
		LLF_Data = np.repeat(np.expand_dims(features[video_file], axis=0), num_samples, axis=0)

		# Meta_Data containing [bitrate]. Repeating Meta_Data along temporal-axis
		Meta_Data = np.copy(Meta_Information[video_file][:,0:1])

		if temporal_low_level_features:
			temporal_length = LLF_Data.shape[1]
			Meta_Data = np.repeat(np.expand_dims(Meta_Data, axis=1), temporal_length, axis=1)

		# Custom_Data
		Custom_Data = []

		if temporal_low_level_features:
			for cf in custom_features_names:
				if cf == "log2(sqrt(h_Y/E_Y)) + 2b":
					# Indices
					i = features_names.index("h_Y")
					j = features_names.index("E_Y")

					# Compute
					h = LLF_Data[...,i]
					E = LLF_Data[...,j]
					b = Meta_Data[...,0]
					Custom_Data.append(np.log2(np.sqrt(np.divide(h, E))) + 2*b)

				elif cf == "log2(sqrt(h_U/E_U)) + 2b":
					# Indices
					i = features_names.index("h_U")
					j = features_names.index("E_U")

					# Compute
					h = LLF_Data[...,i]
					E = LLF_Data[...,j]
					b = Meta_Data[...,0]
					Custom_Data.append(np.log2(np.sqrt(np.divide(h, E))) + 2*b)

				elif cf == "log2(sqrt(h_V/E_V)) + 2b":
					# Indices
					i = features_names.index("h_V")
					j = features_names.index("E_V")

					# Compute
					h = LLF_Data[...,i]
					E = LLF_Data[...,j]
					b = Meta_Data[...,0]
					Custom_Data.append(np.log2(np.sqrt(np.divide(h, E))) + 2*b)

				else:
					assert False, "Invalid feature {}".format(cf)
		else:
			for cf in custom_features_names:
				if cf == "log2(sqrt(mean_h_Y/mean_E_Y)) + 2b":
					# Indices
					i = features_names.index("mean_h_Y")
					j = features_names.index("mean_E_Y")

					# Compute
					h = LLF_Data[...,i]
					E = LLF_Data[...,j]
					b = Meta_Data[...,0]
					Custom_Data.append(np.log2(np.sqrt(np.divide(h, E))) + 2*b)

				elif cf == "log2(sqrt(mean_h_U/mean_E_U)) + 2b":
					# Indices
					i = features_names.index("mean_h_U")
					j = features_names.index("mean_E_U")

					# Compute
					h = LLF_Data[...,i]
					E = LLF_Data[...,j]
					b = Meta_Data[...,0]
					Custom_Data.append(np.log2(np.sqrt(np.divide(h, E))) + 2*b)

				elif cf == "log2(sqrt(mean_h_V/mean_E_V)) + 2b":
					# Indices
					i = features_names.index("mean_h_V")
					j = features_names.index("mean_E_V")

					# Compute
					h = LLF_Data[...,i]
					E = LLF_Data[...,j]
					b = Meta_Data[...,0] 
					Custom_Data.append(np.log2(np.sqrt(np.divide(h, E))) + 2*b)

				else:
					assert False, "Invalid feature {}".format(cf)

		Custom_Data = np.asarray(Custom_Data)
		if Custom_Data.shape[0] == 0:
			if temporal_low_level_features:
				Custom_Data = np.zeros((0,Meta_Data.shape[0],temporal_length))
			else:
				Custom_Data = np.zeros((0,Meta_Data.shape[0]))
				
		if temporal_low_level_features:
			Custom_Features[video_file] = Custom_Data.transpose(1,2,0)
		else:
			if len(Custom_Data.shape) == 1:
				# assert False, "Reached a line in Custom Bitrate Features function which is written for an unknown reason at the moment."
				Custom_Features[video_file] = Custom_Data.T
			else:
				Custom_Features[video_file] = Custom_Data.transpose(1,0)

	return Custom_Features



def Compute_Quality_Custom_Features(
	custom_features_names:list,
	quality_metric:str,
	Meta_Information:dict,
	temporal_low_level_features:bool
):
	"""
	Computes Custom Features
	Quality Customized Features include Quality-Texture-Features
	Args:
		custom_features_names (list): List of custom features to be computed.
		features (dict): Low-Level Features corresponding to features_names for each video-file.
		quality_metric (str): Quality Metric to consider.
		Meta_Information (dict): Meta-Information for each video-file.
		temporal_low_level_features (bool): If True, everything is extracted per frame instead of pooling using various statistics.
	"""
	# Video Filenames
	video_filenames = list(Meta_Information.keys())

	# Low-Level Features needed for Custom-Features
	features_names = []
	if temporal_low_level_features:
		for cf in custom_features_names:
			features_names += defaults.per_frame_quality_texture_features[cf]
	else:
		for cf in custom_features_names:
			features_names += defaults.quality_texture_features[cf]
	

	# Extracting Low-Level Features needed for Custom-Features
	features = Extract_Low_Level_Features(
		features_names=features_names,
		video_filenames=video_filenames,
		temporal_low_level_features=temporal_low_level_features
	)

	# Custom_Features
	Custom_Features = {}

	for video_file in video_filenames:
		#  No.of RQ-points obtained by compressing the uncompressed video under different settings
		num_samples = Meta_Information[video_file].shape[0]

		# Low-Level Features
		LLF_Data = np.repeat(np.expand_dims(features[video_file], axis=0), num_samples, axis=0)

		# Meta_Data containing [quality]. Repeating Meta_Data along temporal-axis
		Meta_Data = np.copy(Meta_Information[video_file][:,1:2])

		# Scaling
		if quality_metric == "vmaf":
			Meta_Data[:,0] = Meta_Data[:,0]/100.0
			
		if temporal_low_level_features:
			temporal_length = LLF_Data.shape[1]
			Meta_Data = np.repeat(np.expand_dims(Meta_Data, axis=1), temporal_length, axis=1)

		# Custom_Data
		Custom_Data = []

		if temporal_low_level_features:
			for cf in custom_features_names:
				if cf == "0.5*(q - log2(sqrt(h_Y/E_Y)))":
					# Indices
					i = features_names.index("h_Y")
					j = features_names.index("E_Y")

					# Compute
					h = LLF_Data[...,i]
					E = LLF_Data[...,j]
					q = Meta_Data[...,0]
					Custom_Data.append(0.5*(q - np.log2(np.sqrt(np.divide(h, E)))))

				elif cf == "0.5*(q - log2(sqrt(h_U/E_U)))":
					# Indices
					i = features_names.index("h_U")
					j = features_names.index("E_U")

					# Compute
					h = LLF_Data[...,i]
					E = LLF_Data[...,j]
					q = Meta_Data[...,0]
					Custom_Data.append(0.5*(q - np.log2(np.sqrt(np.divide(h, E)))))

				elif cf == "0.5*(q - log2(sqrt(h_V/E_V)))":
					# Indices
					i = features_names.index("h_V")
					j = features_names.index("E_V")

					# Compute
					h = LLF_Data[...,i]
					E = LLF_Data[...,j]
					q = Meta_Data[...,0]
					Custom_Data.append(0.5*(q - np.log2(np.sqrt(np.divide(h, E)))))

				else:
					assert False, "Invalid feature {}".format(cf)
		else:
			for cf in custom_features_names:
				if cf == "0.5*(q - log2(sqrt(mean_h_Y/mean_E_Y)))":
					# Indices
					i = features_names.index("mean_h_Y")
					j = features_names.index("mean_E_Y")

					# Compute
					h = LLF_Data[...,i]
					E = LLF_Data[...,j]
					q = Meta_Data[...,0]
					Custom_Data.append(0.5*(q - np.log2(np.sqrt(np.divide(h, E)))))

				elif cf == "0.5*(q - log2(sqrt(mean_h_U/mean_E_U)))":
					# Indices
					i = features_names.index("mean_h_U")
					j = features_names.index("mean_E_U")

					# Compute
					h = LLF_Data[...,i]
					E = LLF_Data[...,j]
					q = Meta_Data[...,0]
					Custom_Data.append(0.5*(q - np.log2(np.sqrt(np.divide(h, E)))))

				elif cf == "0.5*(q - log2(sqrt(mean_h_V/mean_E_V)))":
					# Indices
					i = features_names.index("mean_h_V")
					j = features_names.index("mean_E_V")

					# Compute
					h = LLF_Data[...,i]
					E = LLF_Data[...,j]
					q = Meta_Data[...,0] 
					Custom_Data.append(0.5*(q - np.log2(np.sqrt(np.divide(h, E)))))

				else:
					assert False, "Invalid feature {}".format(cf)

		Custom_Data = np.asarray(Custom_Data)
		if Custom_Data.shape[0] == 0:
			if temporal_low_level_features:
				Custom_Data = np.zeros((0,Meta_Data.shape[0],temporal_length))
			else:
				Custom_Data = np.zeros((0,Meta_Data.shape[0]))
				
		if temporal_low_level_features:
			Custom_Features[video_file] = Custom_Data.transpose(1,2,0)
		else:
			if len(Custom_Data.shape) == 1:
				# assert False, "Reached a line in Custom Bitrate Features function which is written for an unknown reason at the moment."
				Custom_Features[video_file] = Custom_Data.T
			else:
				Custom_Features[video_file] = Custom_Data.transpose(1,0)

	return Custom_Features