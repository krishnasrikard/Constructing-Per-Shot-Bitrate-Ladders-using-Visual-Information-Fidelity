# Time-Complexity

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt

import sys, os, time
sys.path.append("/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity")
import subprocess
import shlex
from tqdm import tqdm
import functions.IO_functions as IO_functions
import functions.ffmpeg_commands as ffmpeg_commands
import functions.extract_functions as extract_functions
import generate.calculate_low_level_features as calculate_low_level_features
import generate.calculate_vif_information as calculate_vif_information
import bitrate_ladder_construction.BL_functions.bitrate_ladder_functions as bitrate_ladder_functions
import defaults


def LLF_process_time():
	# Reading Video 
	file_path = "/home/krishna/Nebula/krishna/BVT-100_4K/pierseaside-scene1_3840x2160_10bit_420_60fps_frames1-64.yuv"
	yuv_reader = IO_functions.YUV_Reader(
		filepath=file_path,
		width=3840,
		height=2160,
		yuv_type="yuv420p10le"
	)
	video = yuv_reader.get_RGB_video()

	# Calculating Low-Level Features
	F = calculate_low_level_features.generate_low_level_features(video)
	F.generate_low_level_features()


def VIF_process_time():
	# Reading Video 
	file_path = "/home/krishna/Nebula/krishna/BVT-100_4K/pierseaside-scene1_3840x2160_10bit_420_60fps_frames1-64.yuv"

	# Calculating VIF Features
	F = calculate_vif_information.extract_vif_info(uncompressed_video_path=file_path, reference_info_save_path="temp/vif.npy")


def ExtraTrees_process_time():
	# Design
	design = "ml_ensemble_llfvif_approach" + str(9)

	# Features
	features_set = []
	for features_subset in [defaults.glcm_features, defaults.tc_features, defaults.si_features, defaults.ti_features, defaults.cti_features, defaults.cf_features, defaults.ci_features, defaults.dct_features, list(defaults.bitrate_texture_features.keys())]:
		for f in features_subset:
			if "max" in f:
				continue
			features_set.append(f)

	
	X, Model = bitrate_ladder_functions.Quality_Select_Models_Features(
		video_file=defaults.Test_Video_Titles[0],
		models_path=os.path.join("/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/bitrate_ladder_construction", "models/ML", "Quality"),
		design_type=design,
		features_names=features_set,
		temporal_low_level_features=False,
		codec="libx265",
		preset="medium",
		evaluation_bitrates=defaults.evaluation_bitrates
	)
	
	Model.predict(X)


def Compression_VMAF_process_time():
	Compression_time = []
	VMAF_time = []

	for video_filename in tqdm(defaults.Video_Titles):
		path = os.path.join(defaults.rq_points_dataset_path, "libx265", "medium", video_filename, "crfs.json")
		data = IO_functions.read_create_jsonfile(path)
		
		for resolution in defaults.resolutions:
			for crf in defaults.CRFs:

				info = data["{}x{}".format(resolution[0], resolution[1])][str(crf)]
				Compression_time.append(info['downscaling_compression_time'])
				VMAF_time.append(info["upscaling_quality_estimation_time"])

	print (np.max(Compression_time), np.mean(Compression_time), np.median(Compression_time), np.sum(Compression_time))
	print (np.max(VMAF_time), np.mean(VMAF_time), np.median(VMAF_time), np.sum(VMAF_time))



if __name__ == "__main__":
	# -----------------------------------------------------------------
	# Flushing Output
	import functools
	print = functools.partial(print, flush=True)

	# Saving stdout
	sys.stdout = open('logs/{}.log'.format(os.path.basename(__file__)[:-3]), 'w')

	# -----------------------------------------------------------------
	
	Compression_VMAF_process_time()