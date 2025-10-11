"""
Calculating BD-metrics
"""
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt

import os, sys, warnings
sys.path.append("/home/kd28684/Efficient-Dynamic-Optimizer-using-Visual-Information-Fidelity-Working")
import pickle
import modules.bitrate_quality_ladder_evaluation_functions as bitrate_quality_ladder_evaluation_functions
import defaults


def calculate_bd_metrics_of_bitrate_ladders(
	# Files
	Test_Video_Files:list,

	# Path
	save_path:str,

	# Arguments:
	codec:str,
	preset:str,
	bitrate_ladder_path:str,
	fixed_bitrate_ladder_path:str,
	reference_bitrate_ladder_path:str
):
	# Calculating BD-Metrics
	BD_Metrics = {}
	skipped_files_count = 0

	for video_file in Test_Video_Files:
		Metrics = bitrate_quality_ladder_evaluation_functions.Calculate_BD_metrics_for_Bitrate_Ladders(
			# Video and Encoder Settings
			video_file=video_file,
			codec=codec,
			preset=preset,

			# Bitrate Ladders
			bitrate_ladder_path=bitrate_ladder_path,
			fixed_bitrate_ladder_path=fixed_bitrate_ladder_path,
			reference_bitrate_ladder_path=reference_bitrate_ladder_path
		)
				
		# Increasing Skipped Files
		if Metrics is None:
			skipped_files_count += 1
		else:
			BD_Metrics[video_file] = Metrics

		# If we skip atleast 10% of the files, throw an error.
		assert skipped_files_count < 0.1*len(Test_Video_Files), "Skipped too many files."

	# Saving BD-Metrics
	np.save(save_path, BD_Metrics)


def calculate_bd_metrics_of_quality_ladders(
	# Files
	Test_Video_Files:list,

	# Path
	save_path:str,

	# Arguments:
	codec:str,
	preset:str,
	quality_ladder_path:str,
	fixed_bitrate_ladder_path:str,
	reference_bitrate_ladder_path:str
):
	# Calculating BD-Metrics
	BD_Metrics = {}
	skipped_files_count = 0

	for video_file in Test_Video_Files:
		Metrics = bitrate_quality_ladder_evaluation_functions.Calculate_BD_metrics_for_Quality_Ladders(
			# Video and Encoder Settings
			video_file=video_file,
			codec=codec,
			preset=preset,

			# Quality and Bitrate Ladders
			quality_ladder_path=quality_ladder_path,
			fixed_bitrate_ladder_path=fixed_bitrate_ladder_path,
			reference_bitrate_ladder_path=reference_bitrate_ladder_path
		)
				
		# Increasing Skipped Files
		if Metrics is None:
			skipped_files_count += 1
		else:
			BD_Metrics[video_file] = Metrics

		# If we skip atleast 10% of the files, throw an error.
		assert skipped_files_count < 0.1*len(Test_Video_Files), "Skipped too many files."

	# Saving BD-Metrics
	np.save(save_path, BD_Metrics)



def main(
	# Files
	Test_Video_Files:list,

	# Path
	results_dir,
):
	# Parameters
	codec = "libx265"
	preset = "medium"

	# Calculating BD Metrics of Cross-Over Bitrate Ladders
	print ()
	print ("-"*10, "Cross-Over Bitrate Ladders", "-"*10)
	print ()

	# Creating Directories
	os.makedirs(os.path.join(results_dir, "CrossOver_Bitrates", "bd_metrics"), exist_ok=True)

	calculate_bd_metrics_of_bitrate_ladders(
		# Files
		Test_Video_Files=Test_Video_Files,

		# Paths
		save_path=os.path.join(
			results_dir, "CrossOver_Bitrates", "bd_metrics", "low_level_features.npy"
		),

		# Arguments
		codec=codec,
		preset=preset,
		bitrate_ladder_path = os.path.join(
			results_dir, "CrossOver_Bitrates", "bitrate_ladders", "low_level_features.npy"
		),
		fixed_bitrate_ladder_path = os.path.join(
			results_dir, "Standard", "bitrate_ladders", "fixed_bitrate_ladder.npy"
		),
		reference_bitrate_ladder_path = os.path.join(
			results_dir, "Standard", "bitrate_ladders", "{}_{}.npy".format(codec, preset)
		)
	)


	# Calculating BD Metrics of Cross-Over Quality Ladders
	print ()
	print ("-"*10, "Cross-Over Quality Ladders", "-"*10)
	print ()

	# Creating Directories
	os.makedirs(os.path.join(results_dir, "CrossOver_Qualities", "bd_metrics"), exist_ok=True)

	calculate_bd_metrics_of_quality_ladders(
		# Files
		Test_Video_Files=Test_Video_Files,

		# Paths
		save_path=os.path.join(
			results_dir, "CrossOver_Qualities", "bd_metrics", "low_level_features.npy"
		),

		# Arguments
		codec=codec,
		preset=preset,
		quality_ladder_path = os.path.join(
			results_dir, "CrossOver_Qualities", "quality_ladders", "low_level_features.npy"
		),
		fixed_bitrate_ladder_path = os.path.join(
			results_dir, "Standard", "bitrate_ladders", "fixed_bitrate_ladder.npy"
		),
		reference_bitrate_ladder_path = os.path.join(
			results_dir, "Standard", "bitrate_ladders", "{}_{}.npy".format(codec, preset)
		)
	)

		
	# Creating Directories
	os.makedirs(os.path.join(results_dir, "Bitrate_Ladder_Prediction", "bd_metrics"), exist_ok=True)
	os.makedirs(os.path.join(results_dir, "Quality_Ladder_Prediction", "bd_metrics"), exist_ok=True)


	# Calculating BD Metrics of Metadata Bitrate Ladders
	print ()
	print ("-"*10, "BD Metrics Metadata Bitrate Ladders", "-"*10)
	print ()

	calculate_bd_metrics_of_bitrate_ladders(
		# Files
		Test_Video_Files=Test_Video_Files,

		# Paths
		save_path=os.path.join(
			results_dir, "Bitrate_Ladder_Prediction", "bd_metrics", "metadata.npy"
		),

		# Arguments
		codec=codec,
		preset=preset,
		bitrate_ladder_path = os.path.join(
			results_dir, "Bitrate_Ladder_Prediction", "bitrate_ladders", "metadata.npy"
		),
		fixed_bitrate_ladder_path = os.path.join(
			results_dir, "Standard", "bitrate_ladders", "fixed_bitrate_ladder.npy"
		),
		reference_bitrate_ladder_path = os.path.join(
			results_dir, "Standard", "bitrate_ladders", "{}_{}.npy".format(codec, preset)
		)
	)


	# Calculating BD Metrics of Metadata Quality Ladders
	print ()
	print ("-"*10, "BD Metrics Metadata Quality Ladders", "-"*10)
	print ()

	calculate_bd_metrics_of_quality_ladders(
		# Files
		Test_Video_Files=Test_Video_Files,

		# Paths
		save_path=os.path.join(
			results_dir, "Quality_Ladder_Prediction", "bd_metrics", "metadata.npy"
		),

		# Arguments
		codec=codec,
		preset=preset,
		quality_ladder_path = os.path.join(
			results_dir, "Quality_Ladder_Prediction", "quality_ladders", "metadata.npy"
		),
		fixed_bitrate_ladder_path = os.path.join(
			results_dir, "Standard", "bitrate_ladders", "fixed_bitrate_ladder.npy"
		),
		reference_bitrate_ladder_path = os.path.join(
			results_dir, "Standard", "bitrate_ladders", "{}_{}.npy".format(codec, preset)
		)
	)


	# Calculating BD Metrics of Low-Level Features Bitrate Ladders
	print ()
	print ("-"*10, "BD Metrics Low-Level Bitrate Ladders", "-"*10)
	print ()

	calculate_bd_metrics_of_bitrate_ladders(
		# Files
		Test_Video_Files=Test_Video_Files,

		# Paths
		save_path=os.path.join(
			results_dir, "Bitrate_Ladder_Prediction", "bd_metrics", "low_level_features.npy"
		),

		# Arguments
		codec=codec,
		preset=preset,
		bitrate_ladder_path = os.path.join(
			results_dir, "Bitrate_Ladder_Prediction", "bitrate_ladders", "low_level_features.npy"
		),
		fixed_bitrate_ladder_path = os.path.join(
			results_dir, "Standard", "bitrate_ladders", "fixed_bitrate_ladder.npy"
		),
		reference_bitrate_ladder_path = os.path.join(
			results_dir, "Standard", "bitrate_ladders", "{}_{}.npy".format(codec, preset)
		)
	)


	# Calculating BD Metrics of Low-Level Features Quality Ladders
	print ()
	print ("-"*10, "BD Metrics Low-Level Quality Ladders", "-"*10)
	print ()

	calculate_bd_metrics_of_quality_ladders(
		# Files
		Test_Video_Files=Test_Video_Files,

		# Paths
		save_path=os.path.join(
			results_dir, "Quality_Ladder_Prediction", "bd_metrics", "low_level_features.npy"
		),

		# Arguments
		codec=codec,
		preset=preset,
		quality_ladder_path = os.path.join(
			results_dir, "Quality_Ladder_Prediction", "quality_ladders", "low_level_features.npy"
		),
		fixed_bitrate_ladder_path = os.path.join(
			results_dir, "Standard", "bitrate_ladders", "fixed_bitrate_ladder.npy"
		),
		reference_bitrate_ladder_path = os.path.join(
			results_dir, "Standard", "bitrate_ladders", "{}_{}.npy".format(codec, preset)
		)
	)


	# Calculating BD Metrics of VIF Features Bitrate Ladders
	print ()
	print ("-"*10, "BD Metrics VIF Bitrate Ladders", "-"*10)
	print ()

	for vif_approach_number in np.arange(1,10):
		calculate_bd_metrics_of_bitrate_ladders(
			# Files
			Test_Video_Files=Test_Video_Files,

			# Paths
			save_path=os.path.join(
				results_dir, "Bitrate_Ladder_Prediction", "bd_metrics", "vif_features_{}.npy".format(str(vif_approach_number))
			),

			# Arguments
			codec=codec,
			preset=preset,
			bitrate_ladder_path = os.path.join(
				results_dir, "Bitrate_Ladder_Prediction", "bitrate_ladders", "vif_features_{}.npy".format(str(vif_approach_number))
			),
			fixed_bitrate_ladder_path = os.path.join(
				results_dir, "Standard", "bitrate_ladders", "fixed_bitrate_ladder.npy"
			),
			reference_bitrate_ladder_path = os.path.join(
				results_dir, "Standard", "bitrate_ladders", "{}_{}.npy".format(codec, preset)
			)
		)


	# Calculating BD Metrics of VIF Features Quality Ladders
	print ()
	print ("-"*10, "BD Metrics VIF Quality Ladders", "-"*10)
	print ()

	for vif_approach_number in np.arange(1,10):
		calculate_bd_metrics_of_quality_ladders(
			# Files
			Test_Video_Files=Test_Video_Files,

			# Paths
			save_path=os.path.join(
				results_dir, "Quality_Ladder_Prediction", "bd_metrics", "vif_features_{}.npy".format(str(vif_approach_number))
			),

			# Arguments
			codec=codec,
			preset=preset,
			quality_ladder_path = os.path.join(
				results_dir, "Quality_Ladder_Prediction", "quality_ladders", "vif_features_{}.npy".format(str(vif_approach_number))
			),
			fixed_bitrate_ladder_path = os.path.join(
				results_dir, "Standard", "bitrate_ladders", "fixed_bitrate_ladder.npy"
			),
			reference_bitrate_ladder_path = os.path.join(
				results_dir, "Standard", "bitrate_ladders", "{}_{}.npy".format(codec, preset)
			)
		)


	# Calculating BD Metrics of Low-Level Features and VIF Features Bitrate Ladders
	print ()
	print ("-"*10, "BD Metrics Low-Level Features and VIF Features Bitrate Ladders", "-"*10)
	print ()

	for vif_approach_number in np.arange(1,10):
		calculate_bd_metrics_of_bitrate_ladders(
			# Files
			Test_Video_Files=Test_Video_Files,

			# Paths
			save_path=os.path.join(
				results_dir, "Bitrate_Ladder_Prediction", "bd_metrics", "low_level_features_vif_features_{}.npy".format(str(vif_approach_number))
			),

			# Arguments
			codec=codec,
			preset=preset,
			bitrate_ladder_path = os.path.join(
				results_dir, "Bitrate_Ladder_Prediction", "bitrate_ladders", "low_level_features_vif_features_{}.npy".format(str(vif_approach_number))
			),
			fixed_bitrate_ladder_path = os.path.join(
				results_dir, "Standard", "bitrate_ladders", "fixed_bitrate_ladder.npy"
			),
			reference_bitrate_ladder_path = os.path.join(
				results_dir, "Standard", "bitrate_ladders", "{}_{}.npy".format(codec, preset)
			)
		)


	# Calculating BD Metrics of Low-Level Features and VIF Features Quality Ladders
	print ()
	print ("-"*10, "BD Metrics Low-Level Features and VIF Features Quality Ladders", "-"*10)
	print ()

	for vif_approach_number in np.arange(1,10):
		calculate_bd_metrics_of_quality_ladders(
			# Files
			Test_Video_Files=Test_Video_Files,

			# Paths
			save_path=os.path.join(
				results_dir, "Quality_Ladder_Prediction", "bd_metrics", "low_level_features_vif_features_{}.npy".format(str(vif_approach_number))
			),

			# Arguments
			codec=codec,
			preset=preset,
			quality_ladder_path = os.path.join(
				results_dir, "Quality_Ladder_Prediction", "quality_ladders", "low_level_features_vif_features_{}.npy".format(str(vif_approach_number))
			),
			fixed_bitrate_ladder_path = os.path.join(
				results_dir, "Standard", "bitrate_ladders", "fixed_bitrate_ladder.npy"
			),
			reference_bitrate_ladder_path = os.path.join(
				results_dir, "Standard", "bitrate_ladders", "{}_{}.npy".format(codec, preset)
			)
		)