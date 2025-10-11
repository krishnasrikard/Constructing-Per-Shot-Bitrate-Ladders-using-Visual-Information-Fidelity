# Importing Libraries
import numpy as np
import pandas as pd
import cv2

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append("/home/kd28684/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity-Working")
import subprocess
import joblib
import functions.IO_functions as IO_functions
import functions.extract_functions as extract_functions
import defaults


# For each video-file
def calculate_execution_time(video_file):
	"""
	Calculates Execution Time for each feature extraction function for a given video file.

	Args:
		video_file (str): Name of the video file (without extension).

	Returns:
		None: Saves the execution times in a .npy file.
	"""
	# Logging
	print ("-"*75 + "\n" + video_file + "\n" + "-"*75, flush=True)

	# ----------------------------------------------------------------------------

	# Reading Video 
	file_path = os.path.join(defaults.source_dataset_path, video_file + ".yuv")
	yuv_reader = IO_functions.YUV_Reader(
		filepath=file_path,
		width=3840,
			height=2160,
			yuv_type="yuv420p10le"
	)
	video = yuv_reader.get_RGB_video()

	# Paths
	video_temp_path = "logs/misc/temp_{}.npy".format(video_file)
	video_time_path = "logs/misc/{}.npy".format(video_file)

	# Saving Video
	np.save(video_temp_path, video)

	# ----------------------------------------------------------------------------

	# Execution Time for Low-Level Features
	print ("Calculating Execution Time for Low-Level Features...", flush=True)

	cmd = "/usr/bin/time python3 get_execution_time_functions.py --func LLF --video_file {} --video_path {}".format(
		video_file,
		video_temp_path
	)

	output = subprocess.getoutput(cmd)
	LLF_time = extract_functions.extract_execution_time(output)

	# ----------------------------------------------------------------------------

	# Execution Time for GLCM Features
	print ("Calculating Execution Time for GLCM Features...", flush=True)

	cmd = "/usr/bin/time python3 get_execution_time_functions.py --func GLCM --video_file {} --video_path {}".format(
		video_file,
		video_temp_path
	)

	output = subprocess.getoutput(cmd)
	GLCM_time = extract_functions.extract_execution_time(output)

	# ----------------------------------------------------------------------------

	# Execution Time for TC Features
	print ("Calculating Execution Time for TC Features...", flush=True)

	cmd = "/usr/bin/time python3 get_execution_time_functions.py --func TC --video_file {} --video_path {}".format(
		video_file,
		video_temp_path
	)

	output = subprocess.getoutput(cmd)
	TC_time = extract_functions.extract_execution_time(output)

	# ----------------------------------------------------------------------------

	# Execution Time for SI_TI_CF Features
	print ("Calculating Execution Time for SI_TI_CF Features...", flush=True)

	cmd = "/usr/bin/time python3 get_execution_time_functions.py --func SI_TI_CF --video_file {} --video_path {}".format(
		video_file,
		video_temp_path
	)

	output = subprocess.getoutput(cmd)
	SI_TI_CF_time = extract_functions.extract_execution_time(output)

	# ----------------------------------------------------------------------------

	# Execution Time for CTI_CI Features
	print ("Calculating Execution Time for CTI_CI Features...", flush=True)

	cmd = "/usr/bin/time python3 get_execution_time_functions.py --func CTI_CI --video_file {} --video_path {}".format(
		video_file,
		video_temp_path
	)

	output = subprocess.getoutput(cmd)
	CTI_CI_time = extract_functions.extract_execution_time(output)

	# ----------------------------------------------------------------------------

	# Execution Time for Texture_DCT Features
	print ("Calculating Execution Time for Texture_DCT Features...", flush=True)

	cmd = "/usr/bin/time python3 get_execution_time_functions.py --func Texture_DCT --video_file {} --video_path {}".format(
		video_file,
		video_temp_path
	)

	output = subprocess.getoutput(cmd)
	Texture_DCT_time = extract_functions.extract_execution_time(output)
	
	# ----------------------------------------------------------------------------

	# Execution Time for VIFF3 Features
	print ("Calculating Execution Time for VIFF3 Features...", flush=True)

	cmd = "/usr/bin/time python3 get_execution_time_functions.py --func VIFF3 --video_file {} --video_path {}".format(
		video_file,
		video_temp_path
	)

	output = subprocess.getoutput(cmd)
	VIFF3_time = extract_functions.extract_execution_time(output)

	# ----------------------------------------------------------------------------

	# Execution Time for VIFF9 Features
	print ("Calculating Execution Time for VIFF9 Features...", flush=True)

	cmd = "/usr/bin/time python3 get_execution_time_functions.py --func VIFF9 --video_file {} --video_path {}".format(
		video_file,
		video_temp_path
	)

	output = subprocess.getoutput(cmd)
	VIFF9_time = extract_functions.extract_execution_time(output)

	# ----------------------------------------------------------------------------

	# Execution Time for Extra-Trees
	print ("Calculating Execution Time for Extra-Trees Features...", flush=True)

	cmd = "/usr/bin/time python3 get_execution_time_functions.py --func ExtraTrees --video_file {} --video_path {}".format(
		video_file,
		video_temp_path
	)

	output = subprocess.getoutput(cmd)
	ExtraTrees_time = extract_functions.extract_execution_time(output)

	# ----------------------------------------------------------------------------

	# Saving Results
	np.save(video_time_path, np.array({
		"LLF_time": LLF_time,
		"GLCM_time": GLCM_time,
		"TC_time": TC_time,
		"SI_TI_CF_time": SI_TI_CF_time,
		"CTI_CI_time": CTI_CI_time,
		"Texture_DCT_time": Texture_DCT_time,
		"VIFF3_time": VIFF3_time,
		"VIFF9_time": VIFF9_time,
		"ExtraTrees_time": ExtraTrees_time,
	}))


	# Delete Temp Video
	os.remove(video_temp_path)


# Main Execution
if __name__ == "__main__":
	# For each video-file
	for video_file in defaults.Video_Titles:
		calculate_execution_time(video_file)