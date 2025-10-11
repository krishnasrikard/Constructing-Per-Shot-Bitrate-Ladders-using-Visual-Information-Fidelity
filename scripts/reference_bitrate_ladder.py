"""
Construct Bitrate Ladders for each Convex-Hull
"""
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt

import os, sys, warnings
sys.path.append("/home/kd28684/Efficient-Dynamic-Optimizer-using-Visual-Information-Fidelity-Working")
import modules.reference_bitrate_ladder_functions as reference_bitrate_ladder_functions
import modules.bitrate_quality_ladder_evaluation_functions as bitrate_quality_ladder_evaluation_functions
import defaults


def main(
	# Files
	Test_Video_Files:list,

	# Path
	results_dir,
):
	# Parameters
	codec = "libx265"
	preset = "medium"


	# Creating Directories
	os.makedirs(os.path.join(results_dir, "Standard", "bitrate_ladders"), exist_ok=True)
	os.makedirs(os.path.join(results_dir, "Standard", "bd_metrics", "Reference_Bitrate_Ladder"), exist_ok=True)


	# Constructing Apple's Fixed Bitrate Ladder
	print ()
	print ("-"*10, "Apple's Fixed Bitrate Ladder", "-"*10)
	print ()

	Fixed_Bitrate_Ladder = reference_bitrate_ladder_functions.Construct_Apple_Fixed_Bitrate_Ladder(
		evaluation_bitrates=defaults.evaluation_bitrates,
	)

	np.save(
		os.path.join(
			results_dir, "Standard", "bitrate_ladders", "fixed_bitrate_ladder.npy"
		),
		Fixed_Bitrate_Ladder
	)


	# Constructing Reference Bitrate Ladders
	print ()
	print ("-"*10, "Reference Bitrate Ladder", "-"*10)
	print ()

	Convex_Hull_Bitrate_Ladder = {}

	for video_file in Test_Video_Files:
		Convex_Hull_Bitrate_Ladder[video_file] = reference_bitrate_ladder_functions.Construct_Reference_Bitrate_Ladder(
			# Video and Encoder Settings
			video_file=video_file,
			codec=codec,
			preset=preset,

			# Evaluation Bitrates
			evaluation_bitrates=defaults.evaluation_bitrates
		)

	np.save(
		os.path.join(
			results_dir, "Standard", "bitrate_ladders", "{}_{}.npy".format(codec, preset)
		),
		Convex_Hull_Bitrate_Ladder
	)


	# Calculating BD Metrics of Reference Bitrate Ladders
	print ()
	print ("-"*10, "BD Metrics Reference Bitrate Ladder", "-"*10)
	print ()

	BD_Metrics_Convex_Hull = {}

	for video_file in Test_Video_Files:
		Metrics = bitrate_quality_ladder_evaluation_functions.Calculate_BD_metrics_for_Bitrate_Ladders(
			# Video and Encoder Settings
			video_file=video_file,
			codec=codec,
			preset=preset,

			# Bitrate Ladders
			bitrate_ladder_path = os.path.join(
				results_dir, "Standard", "bitrate_ladders", "{}_{}.npy".format(codec, preset)
			),
			fixed_bitrate_ladder_path = os.path.join(
				results_dir, "Standard", "bitrate_ladders", "fixed_bitrate_ladder.npy"
			),
			reference_bitrate_ladder_path = os.path.join(
				results_dir, "Standard", "bitrate_ladders", "{}_{}.npy".format(codec, preset)
			)
		)

		# Assertion
		if Metrics is None:
			assert False, "BD-Metrics of Convex-Hull is None for video-file: {}".format(video_file)
		else:
			BD_Metrics_Convex_Hull[video_file] = Metrics

	np.save(
		os.path.join(
			results_dir, "Standard", "bd_metrics", "Reference_Bitrate_Ladder", "{}_{}.npy".format(codec, preset)
		),
		BD_Metrics_Convex_Hull
	)