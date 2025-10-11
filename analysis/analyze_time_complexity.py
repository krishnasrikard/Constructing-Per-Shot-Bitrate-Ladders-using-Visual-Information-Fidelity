"""
Analyzing Time-Complexity
"""
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt

import os, sys, warnings
sys.path.append("/home/kd28684/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity-Working")
from tqdm import tqdm
import functions.IO_functions as IO_functions
import defaults


# Time-Complexity
def calculate_time_complexity(
	codec:str,
	preset:str,
	video_files:list,
):
	"""
	Calculating Time-Complexity

	Args:
		codec (str): Codec
		preset (str): Preset of Codec
		video_files (list): List of video-files to consider.
	"""
	# Mean Compression and VMAF time
	Compression_time = {}
	VMAF_time = {}
	Bitrate = {}
	Quality = {}

	for resolution in defaults.resolutions:
		Compression_time[resolution] = []
		VMAF_time[resolution] = []
		Bitrate[resolution] = []
		Quality[resolution] = []

	# Extract Time
	for video_filename in tqdm(video_files, desc="{}_{}".format(codec, preset)):
		path = os.path.join(defaults.rq_points_dataset_path, codec, preset, video_filename, "crfs.json")
		data = IO_functions.read_create_jsonfile(path)
		
		for resolution in defaults.resolutions:
			for rc in defaults.codec_CRF_ranges[codec]:
				info = data["{}x{}".format(resolution[0], resolution[1])][str(rc)]

				Compression_time[resolution].append(info['downscaling_compression_time'])
				VMAF_time[resolution].append(info["quality_estimation_time"])
				Bitrate[resolution].append(info['bitrate'])
				Quality[resolution].append(info["vmaf"])

	return Compression_time, VMAF_time, Bitrate, Quality



# Get Execution Time
def get_execution_time(
	data_path:str,
	video_files:list,
):
	"""
	Get Execution Time

	Args:
		data_path (str): Data Path
		video_files (list): List of video-files to consider.
	"""

	# Accumulate Data
	Data = {}
	for key in ["LLF_time", "GLCM_time", "TC_time", "SI_TI_CF_time", "CTI_CI_time", "Texture_DCT_time", "VIFF3_time", "VIFF9_time", "ExtraTrees_time"]:
		Data[key] = []
	
	for video_filename in video_files:
		path = os.path.join(data_path, "{}.npy".format(video_filename))
		data = np.load(path, allow_pickle=True)[()]
		
		for key in Data.keys():
			Data[key].append(data[key])

	# Convert to numpy arrays
	for key in Data.keys():
		Data[key] = np.array(Data[key])

	# Logging Average
	print("\nAverage Execution Times (in seconds):")
	for key in Data.keys():
		mean_time = np.mean(Data[key])
		print("{:<20s}: {:.4f} seconds".format(key, mean_time))



# Main Exection
if __name__ == "__main__":
	get_execution_time(
		data_path="/home/kd28684/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity-Working/analysis/logs/misc",
		video_files=defaults.Video_Titles[:100]
	)