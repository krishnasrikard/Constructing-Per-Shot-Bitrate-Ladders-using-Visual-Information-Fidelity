# Time-Complexity

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

import sys, os, time
sys.path.append("/home/kd28684/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity-Working")
import pickle, argparse
from tqdm import tqdm
import features.VIF as VIF
import features.GLCM as GLCM
import features.TC as TC
import features.SI as SI
import features.TI as TI
import features.CF as CF
import features.CI as CI
import features.CTI as CTI
import features.Texture_DCT as Texture_DCT
import functions.IO_functions as IO_functions
import modules.bitrate_ladder_construction_functions as bitrate_ladder_construction_functions
import defaults


# Fexture Extraction Modules
cf = CF.CF_Features()
ci = CI.CI_Features(rgb=True, WR=5)
cti = CTI.CTI_Features(rgb=True)
glcm = GLCM.GLCM_Features(descriptors=["contrast","correlation","energy","homogeneity"],angles=[0],distance=1,levels=256,block_size=(64,64),rgb=True)
tc = TC.TC_Features(rgb=True)
si = SI.SI_Features(rgb=True)
ti = TI.TI_Features(rgb=True)
texture_dct_features = Texture_DCT.Texture_DCT_Features(block_size=(32,32),rgb=True)


# System Time for Compression and VMAF
def Compression_VMAF_system_time():
	# Compression and VMAF Time
	Compression_time = []
	VMAF_time = []

	# Iterating for all videos in the dataset
	for video_filename in tqdm(defaults.Video_Titles):
		path = os.path.join(defaults.rq_points_dataset_path, "libx265", "medium", video_filename, "crfs.json")
		data = IO_functions.read_create_jsonfile(path)
		
		for resolution in defaults.resolutions:
			for crf in defaults.CRFs:
				info = data["{}x{}".format(resolution[0], resolution[1])][str(crf)]
				Compression_time.append(info['downscaling_compression_time'])
				VMAF_time.append(info["upscaling_quality_estimation_time"])

	# Printing Statistics
	print (np.max(Compression_time), np.mean(Compression_time), np.median(Compression_time), np.sum(Compression_time))
	print (np.max(VMAF_time), np.mean(VMAF_time), np.median(VMAF_time), np.sum(VMAF_time))


#  VIF Feature Extraction on a video from calculate_vif_features.py
def extract_vif_features(video, compute_features_list=["vif_info", "diff_vif_info", "mean_abs_frame_diff"]):
	# Initializing VIF
	VIF_Function = VIF.Compute_VIF()

	# Computing Reference Video Features
	Reference_Video_Features = []

	# Iterating for each frame
	for i in range(video.shape[0]):
		# Calculating VIF features
		frame = np.copy(video[i])

		# Luma Component of current frame
		# Converting to int32 to avoid overflow during operations.
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)[:,:,0]
		frame = frame.astype(np.int32)

		# Assertion
		assert (frame.dtype == np.int32) and (np.min(frame) >= 0 and np.max(frame) <= 255), "Before calculation frame should of type uint8 and should have range [0,255]."

		# Calculating VIF Features for reference frame
		if "vif_info" in compute_features_list:
			# Decomposation
			vif_pyr_ref, vif_subband_keys = VIF_Function.Decomposation(frame)
			vif_subband_keys.sort(reverse=True)

			# GSM Model
			[vif_S_squared_all, vif_EigenValues_all] = VIF_Function.GSM_Model(vif_pyr_ref, vif_subband_keys)

			# Information in each subband along each eigen value
			vif_features_reference = VIF_Function.Reference_Subband_Eigen_Information_Matrix(
				subband_keys=vif_subband_keys, S_squared_all=vif_S_squared_all, EigenValues_all=vif_EigenValues_all
			)
		else:
			vif_features_reference = None


		# Calculating Diff-VIF (T-VIF) Features
		if i == 0:
			current_frame = np.zeros(video[i].shape, dtype=np.uint8)
			previous_frame = np.zeros(video[i].shape, dtype=np.uint8)
		else:
			current_frame = np.copy(video[i])
			previous_frame = np.copy(video[i-1])

		# Luma Component of current frame
		# Converting to int32 to avoid overflow during operations.
		current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2YUV)[:,:,0]
		current_frame = current_frame.astype(np.int32)
			
		# Luma Component of previous frame
		# Converting to int32 to avoid overflow during operations.
		previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_RGB2YUV)[:,:,0]
		previous_frame = previous_frame.astype(np.int32)

		# Assertions
		assert (current_frame.dtype == np.int32) and (np.min(current_frame) >= 0 and np.max(current_frame) <= 255), "Before calculation frame should of type int32 and should have range [0,255]."
		assert (previous_frame.dtype == np.int32) and (np.min(previous_frame) >= 0 and np.max(previous_frame) <= 255), "Before calculation frame should of type int32 and should have range [0,255]."

		# Frame Difference
		if "mean_abs_frame_diff" in compute_features_list:
			diff_frame = np.copy(current_frame - previous_frame)
		else:
			diff_frame = np.zeros_like(current_frame)

		# Calculating VIF Features for diff frame
		if "diff_vif_info" in compute_features_list:
			# Decomposation
			diff_vif_pyr_ref, diff_vif_subband_keys = VIF_Function.Decomposation(diff_frame)
			diff_vif_subband_keys.sort(reverse=True)

			# GSM Model
			[diff_vif_S_squared_all, diff_vif_EigenValues_all] = VIF_Function.GSM_Model(diff_vif_pyr_ref, diff_vif_subband_keys)

			# Information in each subband along each eigen value
			diff_vif_features_reference = VIF_Function.Reference_Subband_Eigen_Information_Matrix(
				subband_keys=diff_vif_subband_keys, S_squared_all=diff_vif_S_squared_all, EigenValues_all=diff_vif_EigenValues_all
			)
		else:
			diff_vif_features_reference = None

		# Appending reference video features and all other parameters
		Reference_Video_Features.append({
			"vif_info":vif_features_reference, 
			"diff_vif_info":diff_vif_features_reference, 
			"mean_abs_frame_diff":np.mean(np.abs(diff_frame))
		})



# Low-Level Feature calculation from calculate_low_level_features.py
class generate_low_level_features():
	def __init__(self,
		video:np.array,
		features_to_compute:list = ["glcm", "tc", "si", "ti", "cti", "cf", "ci", "texture_dct"]
	):
		# Video
		self.video = video

		# Features
		self.features = {}
		self.per_frame_features = {}

		# Generate
		if "glcm" in features_to_compute:
			self.generate_glcm_features()
		if "tc" in features_to_compute:
			self.generate_tc_features()
		if "si" in features_to_compute:
			self.generate_si_features()
		if "ti" in features_to_compute:
			self.generate_ti_features()
		if "cti" in features_to_compute:
			self.generate_cti_features()
		if "cf" in features_to_compute:
			self.generate_cf_features()
		if "ci" in features_to_compute:
			self.generate_ci_features()
		if "texture_dct" in features_to_compute:
			self.generate_texture_dct_features()


	def generate_per_frame_low_level_features(self):
		return self.per_frame_features
	

	def generate_low_level_features(self):
		return self.features


	def generate_glcm_features(self):
		# GLCM Features
		print ("GLCM Features:\n",flush=True)

		# Stats
		spatial_stats = np.sort(["mean", "std"])
		temporal_stats = np.sort(["mean", "std", "skew", "kurt"])
		stats = []
		for t in temporal_stats:
			for s in spatial_stats:
				stats.append([t,s])

		# Feature Names
		per_frame_features_names = []
		for f in ["GLCM_contrast", "GLCM_correlation", "GLCM_energy", "GLCM_homogeneity"]:
			for s in spatial_stats:
				per_frame_features_names.append(f+"_"+s)

		features_names = []
		for f in ["GLCM_contrast", "GLCM_correlation", "GLCM_energy", "GLCM_homogeneity"]:
			for t in temporal_stats:
				for s in spatial_stats:
					features_names.append(t+"_"+f+"_"+s)
		
		# Compute
		time_instant = time.time()
		per_frame_features, features = glcm.compute_video_glcm_features(video=np.copy(self.video), stats=stats)
		compute_time = time.time() - time_instant

		# Assertions
		assert len(per_frame_features_names) == per_frame_features.shape[1], "Dimensions of per-frame features do not match."
		assert len(features_names) == features.shape[0], "Dimensions of per-frame features do not match."

		for i,name in enumerate(per_frame_features_names):
			self.per_frame_features[name] = per_frame_features[:,i]

		for i,name in enumerate(features_names):
			self.features[name] = features[i]
		self.features["GLCM_compute_time"] = np.round(compute_time, decimals=6)


	def generate_tc_features(self):
		# TC Features
		print ("TC Features:\n",flush=True)

		# Stats
		spatial_stats = np.sort(["mean", "std", "skew", "kurt"])
		temporal_stats = np.sort(["mean", "std"])
		stats = []
		for t in temporal_stats:
			for s in spatial_stats:
				stats.append([t,s])

		# Feature Names
		per_frame_features_names = []
		for f in ["TC"]:
			for s in spatial_stats:
				per_frame_features_names.append(f+"_"+s)

		features_names = []
		for f in ["TC"]:
			for t in temporal_stats:
				for s in spatial_stats:
					features_names.append(t+"_"+f+"_"+s)

		# Compute
		time_instant = time.time()
		per_frame_features, features = tc.compute_video_tc_features(video=np.copy(self.video), stats=stats)
		compute_time = time.time() - time_instant

		# Assertions
		assert len(per_frame_features_names) == per_frame_features.shape[1], "Dimensions of per-frame features do not match."
		assert len(features_names) == features.shape[0], "Dimensions of per-frame features do not match."

		for i,name in enumerate(per_frame_features_names):
			self.per_frame_features[name] = per_frame_features[:,i]

		for i,name in enumerate(features_names):
			self.features[name] = features[i]
		self.features["TC_compute_time"] = np.round(compute_time, decimals=6)

	
	def generate_si_features(self):
		# SI Features
		print ("SI Features:\n",flush=True)

		# Stats
		spatial_stats = np.sort(["mean", "std"])
		temporal_stats = np.sort(["mean", "std", "skew", "kurt"])
		stats = []
		for t in temporal_stats:
			for s in spatial_stats:
				stats.append([t,s])

		# Feature Names
		per_frame_features_names = []
		for f in ["SI"]:
			for s in spatial_stats:
				per_frame_features_names.append(f+"_"+s)

		features_names = []
		for f in ["SI"]:
			for t in temporal_stats:
				for s in spatial_stats:
					features_names.append(t+"_"+f+"_"+s)

		# Compute
		time_instant = time.time()
		per_frame_features, features = si.compute_video_spatial_information(video=np.copy(self.video), stats=stats)
		compute_time = time.time() - time_instant

		# Assertions
		assert len(per_frame_features_names) == per_frame_features.shape[1], "Dimensions of per-frame features do not match."
		assert len(features_names) == features.shape[0], "Dimensions of per-frame features do not match."

		for i,name in enumerate(per_frame_features_names):
			self.per_frame_features[name] = per_frame_features[:,i]

		for i,name in enumerate(features_names):
			self.features[name] = features[i]
		self.features["SI_compute_time"] = np.round(compute_time, decimals=6)


	def generate_ti_features(self):
		# TI Features
		print ("TI Features:\n",flush=True)

		# Stats
		spatial_stats = np.sort(["mean", "std"])
		temporal_stats = np.sort(["mean", "std", "skew", "kurt"])
		stats = []
		for t in temporal_stats:
			for s in spatial_stats:
				stats.append([t,s])

		# Feature Names
		per_frame_features_names = []
		for f in ["TI"]:
			for s in spatial_stats:
				per_frame_features_names.append(f+"_"+s)

		features_names = []
		for f in ["TI"]:
			for t in temporal_stats:
				for s in spatial_stats:
					features_names.append(t+"_"+f+"_"+s)

		# Compute
		time_instant = time.time()
		per_frame_features, features = ti.compute_video_temporal_information(video=np.copy(self.video), stats=stats)
		compute_time = time.time() - time_instant

		# Assertions
		assert len(per_frame_features_names) == per_frame_features.shape[1], "Dimensions of per-frame features do not match."
		assert len(features_names) == features.shape[0], "Dimensions of per-frame features do not match."

		for i,name in enumerate(per_frame_features_names):
			self.per_frame_features[name] = per_frame_features[:,i]

		for i,name in enumerate(features_names):
			self.features[name] = features[i]
		self.features["TI_compute_time"] = np.round(compute_time, decimals=6)


	def generate_cti_features(self):
		# CTI Features
		print ("CTI Features:\n",flush=True)

		# Stats
		spatial_stats = np.sort(["mean", "std"])
		temporal_stats = np.sort(["mean", "std", "skew", "kurt"])
		stats = []
		for t in temporal_stats:
			for s in spatial_stats:
				stats.append([t,s])

		# Feature Names
		per_frame_features_names = []
		for f in ["CTI"]:
			for s in spatial_stats:
				per_frame_features_names.append(f+"_"+s)

		features_names = []
		for f in ["CTI"]:
			for t in temporal_stats:
				for s in spatial_stats:
					features_names.append(t+"_"+f+"_"+s)

		# Compute
		time_instant = time.time()
		per_frame_features, features = cti.compute_video_contrast_information(video=np.copy(self.video), stats=stats)
		compute_time = time.time() - time_instant

		# Assertions
		assert len(per_frame_features_names) == per_frame_features.shape[1], "Dimensions of per-frame features do not match."
		assert len(features_names) == features.shape[0], "Dimensions of per-frame features do not match."

		for i,name in enumerate(per_frame_features_names):
			self.per_frame_features[name] = per_frame_features[:,i]

		for i,name in enumerate(features_names):
			self.features[name] = features[i]
		self.features["CTI_compute_time"] = np.round(compute_time, decimals=6)


	def generate_cf_features(self):
		# CF Features
		print ("CF Features:\n",flush=True)

		# Stats
		temporal_stats = np.sort(["mean", "std", "skew", "kurt"])
		stats = temporal_stats

		# Feature Names
		per_frame_features_names = ["CF"]

		features_names = []
		for f in ["CF"]:
			for t in temporal_stats:
				features_names.append(t+"_"+f)

		# Compute
		time_instant = time.time()
		per_frame_features, features = cf.compute_video_colorfulness(video=np.copy(self.video), stats=stats)
		compute_time = time.time() - time_instant

		# Assertions
		assert len(features_names) == features.shape[0], "Dimensions of per-frame features do not match."

		self.per_frame_features[per_frame_features_names[0]] = per_frame_features

		for i,name in enumerate(features_names):
			self.features[name] = features[i]
		self.features["CF_compute_time"] = np.round(compute_time, decimals=6)


	def generate_ci_features(self):
		# CI Features
		print ("CI Features:\n",flush=True)

		# Stats
		spatial_stats = np.sort(["mean", "std"])
		temporal_stats = np.sort(["mean", "std", "skew", "kurt"])
		stats = []
		for t in temporal_stats:
			for s in spatial_stats:
				stats.append([t,s])

		# Feature Names
		per_frame_features_names = []
		for f in ["CI_U"]:
			for s in spatial_stats:
				per_frame_features_names.append(f+"_"+s)

		features_names = []
		for f in ["CI_U"]:
			for t in temporal_stats:
				for s in spatial_stats:
					features_names.append(t+"_"+f+"_"+s)

		# Compute
		time_instant = time.time()
		per_frame_features, features = ci.compute_video_chroma_information(video=np.copy(self.video), component="U", stats=stats)
		compute_time = time.time() - time_instant

		# Assertions
		assert len(per_frame_features_names) == per_frame_features.shape[1], "Dimensions of per-frame features do not match."
		assert len(features_names) == features.shape[0], "Dimensions of per-frame features do not match."

		for i,name in enumerate(per_frame_features_names):
			self.per_frame_features[name] = per_frame_features[:,i]

		for i,name in enumerate(features_names):
			self.features[name] = features[i]
		self.features["CI_U_compute_time"] = np.round(compute_time, decimals=6)

		# Feature Names
		per_frame_features_names = []
		for f in ["CI_V"]:
			for s in spatial_stats:
				per_frame_features_names.append(f+"_"+s)

		features_names = []
		for f in ["CI_V"]:
			for t in temporal_stats:
				for s in spatial_stats:
					features_names.append(t+"_"+f+"_"+s)

		# Compute
		time_instant = time.time()
		per_frame_features, features = ci.compute_video_chroma_information(video=np.copy(self.video), component="V", stats=stats)
		compute_time = time.time() - time_instant

		# Assertions
		assert len(per_frame_features_names) == per_frame_features.shape[1], "Dimensions of per-frame features do not match."
		assert len(features_names) == features.shape[0], "Dimensions of per-frame features do not match."

		for i,name in enumerate(per_frame_features_names):
			self.per_frame_features[name] = per_frame_features[:,i]

		for i,name in enumerate(features_names):
			self.features[name] = features[i]
		self.features["CI_V_compute_time"] = np.round(compute_time, decimals=6)
		

	def generate_texture_dct_features(self):
		# Texture-DCT Features
		print ("Texture-DCT Features:\n",flush=True)

		# Feature Names
		per_frame_features_names = ["E_Y", "h_Y", "L_Y"]
		features_names = ["mean_E_Y", "mean_h_Y", "mean_L_Y"]

		# Compute
		time_instant = time.time()
		per_frame_features, features = texture_dct_features.compute_video_features(video=np.copy(self.video), component="Y")
		compute_time = time.time() - time_instant

		# Assertions
		assert len(per_frame_features_names) == per_frame_features.shape[1], "Dimensions of per-frame features do not match."
		assert len(features_names) == features.shape[0], "Dimensions of per-frame features do not match."

		for i,name in enumerate(per_frame_features_names):
			self.per_frame_features[name] = per_frame_features[:,i]

		for i,name in enumerate(features_names):
			self.features[name] = features[i]
		self.features["EhL_Y_compute_time"] = np.round(compute_time, decimals=6)


		# Feature Names
		per_frame_features_names = ["E_U", "h_U", "L_U"]
		features_names = ["mean_E_U", "mean_h_U", "mean_L_U"]

		# Compute
		time_instant = time.time()
		per_frame_features, features = texture_dct_features.compute_video_features(np.copy(self.video), component="U")
		compute_time = time.time() - time_instant

		# Assertions
		assert len(per_frame_features_names) == per_frame_features.shape[1], "Dimensions of per-frame features do not match."
		assert len(features_names) == features.shape[0], "Dimensions of per-frame features do not match."

		for i,name in enumerate(per_frame_features_names):
			self.per_frame_features[name] = per_frame_features[:,i]

		for i,name in enumerate(features_names):
			self.features[name] = features[i]
		self.features["EhL_U_compute_time"] = np.round(compute_time, decimals=6)


		# Feature Names
		per_frame_features_names = ["E_V", "h_V", "L_V"]
		features_names = ["mean_E_V", "mean_h_V", "mean_L_V"]

		# Compute
		time_instant = time.time()
		per_frame_features, features = texture_dct_features.compute_video_features(video=np.copy(self.video), component="V")
		compute_time = time.time() - time_instant

		# Assertions
		assert len(per_frame_features_names) == per_frame_features.shape[1], "Dimensions of per-frame features do not match."
		assert len(features_names) == features.shape[0], "Dimensions of per-frame features do not match."

		for i,name in enumerate(per_frame_features_names):
			self.per_frame_features[name] = per_frame_features[:,i]

		for i,name in enumerate(features_names):
			self.features[name] = features[i]
		self.features["EhL_V_compute_time"] = np.round(compute_time, decimals=6)



# Execution Time Functions
def Execution_Time_LLF(video_path):
	# Load Video
	video = np.load(video_path)

	# Calculating Low-Level Features
	F = generate_low_level_features(video, features_to_compute=["glcm", "tc", "si", "ti", "cti", "cf", "ci", "texture_dct"])
	F.generate_low_level_features()


def Execution_Time_GLCM(video_path):
	# Load Video
	video = np.load(video_path)

	# Calculating Low-Level Features
	F = generate_low_level_features(video, features_to_compute=["glcm"])
	F.generate_low_level_features()


def Execution_Time_TC(video_path):
	# Load Video
	video = np.load(video_path)
	
	# Calculating Low-Level Features
	F = generate_low_level_features(video, features_to_compute=["tc"])
	F.generate_low_level_features()


def Execution_Time_SI_TI_CF(video_path):
	# Load Video
	video = np.load(video_path)

	# Calculating Low-Level Features
	F = generate_low_level_features(video, features_to_compute=["si", "ti", "cf"])
	F.generate_low_level_features()


def Execution_Time_CTI_CI(video_path):
	# Load Video
	video = np.load(video_path)

	# Calculating Low-Level Features
	F = generate_low_level_features(video, features_to_compute=["cti", "ci"])
	F.generate_low_level_features()


def Execution_Time_Texture_DCT(video_path):
	# Load Video
	video = np.load(video_path)

	# Calculating Low-Level Features
	F = generate_low_level_features(video, features_to_compute=["texture_dct"])
	F.generate_low_level_features()


def Execution_Time_VIFF9(video_path):
	# Load Video
	video = np.load(video_path)

	# Calculating VIF Features
	extract_vif_features(
		video=video,
		compute_features_list=["vif_info", "diff_vif_info", "mean_abs_frame_diff"]
	)


def Execution_Time_VIFF3(video_path):
	# Load Video
	video = np.load(video_path)

	# Calculating VIF Features
	extract_vif_features(
		video=video,
		compute_features_list=["vif_info"]
	)


def Execution_Time_ExtraTrees(video_file):
	## Features
	# Low-Level Features (Custom-Features always at the end so as to match code in 'dataset_evaluation_functions.py')
	features_names = []
	for features_subset in [defaults.glcm_features, defaults.tc_features, defaults.si_features, defaults.ti_features, defaults.cti_features, defaults.cf_features, defaults.ci_features, defaults.dct_features, list(defaults.bitrate_texture_features.keys())]:
		for f in features_subset:
			features_names.append(f)

	# VIF-Approach Number
	vif_approach_number = "9"

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

	
	Inputs, _, _ = bitrate_ladder_construction_functions.Bitrate_Ladder_Construction_with_LowLevelFeatures_VIFFeatures(
		# Files
		video_filenames=[video_file],

		# Method Arguments
		features_names=features_names,
		temporal_low_level_features=False,
		per_frame=False,
		per_frame_features_flatten=False,
		vif_setting=VIF_Approach_Map[vif_approach_number][1],
		vif_features_list=VIF_Approach_Map[vif_approach_number][0],

		# Arguments
		codec="libx265",
		preset="medium",
		quality_metric="vmaf",
		Resolutions_Considered=defaults.resolutions,
		evaluation_bitrates=defaults.evaluation_bitrates,
		min_quality=defaults.min_quality,
		max_quality=defaults.max_quality,
		min_bitrate=defaults.min_bitrate,
		max_bitrate=defaults.max_bitrate
	)
	
	# Load Model
	Model = Model = pickle.load(open("/home/kd28684/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity-Working/results/main/Bitrate_Ladder_Prediction/models/low_level_features_vif_features_9.pkl", "rb"))

	# Predicting Quality
	Model.predict(Inputs[video_file]).flatten()


# Main Function
if __name__ == "__main__":
	# Get Arguments
	parser = argparse.ArgumentParser(description='')

	parser.add_argument(
		'--func', 
		help='Function to execute.'
	)
	parser.add_argument(
		'--video_file', 
		help='Video-File'
	)
	parser.add_argument(
		'--video_path', 
		help='Width of Source Video'
	)

	# Parse Arguments
	args = parser.parse_args()

	# Get Time
	if args.func == "LLF":
		print (args.func)
		Execution_Time_LLF(
			video_path=args.video_path
		)
	elif args.func == "GLCM":
		print (args.func)
		Execution_Time_GLCM(
			video_path=args.video_path
		)
	elif args.func == "TC":
		print (args.func)
		Execution_Time_TC(
			video_path=args.video_path
		)
	elif args.func == "SI_TI_CF":
		print (args.func)
		Execution_Time_SI_TI_CF(
			video_path=args.video_path
		)
	elif args.func == "CTI_CI":
		print (args.func)
		Execution_Time_CTI_CI(
			video_path=args.video_path
		)
	elif args.func == "Texture_DCT":
		print (args.func)
		Execution_Time_Texture_DCT(
			video_path=args.video_path
		)
	elif args.func == "VIFF3":
		print (args.func)
		Execution_Time_VIFF3(
			video_path=args.video_path
		)
	elif args.func == "VIFF9":
		print (args.func)
		Execution_Time_VIFF9(
			video_path=args.video_path
		)
	elif args.func == "ExtraTrees":
		print (args.func)
		Execution_Time_ExtraTrees(
			video_file=args.video_file
		)
	else:
		assert False, "Unknown Func = {}".format(args.func)