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


def calculate_closeness(
	# Files
	Test_Video_Files:list,

	# Path
	save_path:str,

	# Arguments
	Predicted_BD_Metrics:dict,
	Reference_BD_Metrics:dict
):
	# Assertions
	assert len(Predicted_BD_Metrics.keys()) > 0.1 * len(Test_Video_Files), "No.of video-files in Predicted BD-Metrics is less 90% of no.of Test Video Files"
	assert len(Reference_BD_Metrics.keys()) == len(Test_Video_Files), "No.of video-files in Predicted BD-Metrics is less 100% of no.of Test Video Files"

	# Closeness
	Closeness = {}

	for video_file in Predicted_BD_Metrics.keys():
		Closeness[video_file] = bitrate_quality_ladder_evaluation_functions.Calculate_Closeness(
			Predicted_Metrics=Predicted_BD_Metrics[video_file],
			Reference_Metrics=Reference_BD_Metrics[video_file], 
		)

	# Saving Closeness
	np.save(save_path, Closeness)



def main(
	# Files
	Test_Video_Files:list,

	# Path
	results_dir,
):
	# Parameters
	codec = "libx265"
	preset = "medium"


	# Calculating Closeness of Cross-Over Bitrate Ladders
	print ()
	print ("-"*10, "Closeness Cross-Over Bitrate Ladders", "-"*10)
	print ()

	# Creating Directories
	os.makedirs(os.path.join(results_dir, "CrossOver_Bitrates", "closeness"), exist_ok=True)

	calculate_closeness(
		# Files
		Test_Video_Files=Test_Video_Files,

		# Path
		save_path = os.path.join(
			results_dir, "CrossOver_Bitrates", "closeness", "low_level_features.npy"
		),

		# Arguments
		Predicted_BD_Metrics = np.load(
			os.path.join(
				results_dir, "CrossOver_Bitrates", "bd_metrics", "low_level_features.npy"
			), allow_pickle=True
		)[()],

		Reference_BD_Metrics =  np.load(
			os.path.join(
				results_dir, "Standard", "bd_metrics", "Reference_Bitrate_Ladder", "{}_{}.npy".format(codec, preset)
			), allow_pickle=True
		)[()]
	)


	# Calculating Closeness of Cross-Over Quality Ladders
	print ()
	print ("-"*10, "Closeness Cross-Over Quality Ladders", "-"*10)
	print ()

	# Creating Directories
	os.makedirs(os.path.join(results_dir, "CrossOver_Qualities", "closeness"), exist_ok=True)

	calculate_closeness(
		# Files
		Test_Video_Files=Test_Video_Files,

		# Path
		save_path = os.path.join(
			results_dir, "CrossOver_Qualities", "closeness", "low_level_features.npy"
		),

		# Arguments
		Predicted_BD_Metrics = np.load(
			os.path.join(
				results_dir, "CrossOver_Qualities", "bd_metrics", "low_level_features.npy"
			), allow_pickle=True
		)[()],

		Reference_BD_Metrics =  np.load(
			os.path.join(
				results_dir, "Standard", "bd_metrics", "Reference_Bitrate_Ladder", "{}_{}.npy".format(codec, preset)
			), allow_pickle=True
		)[()]
	)	

		
	# Creating Directories
	os.makedirs(os.path.join(results_dir, "Bitrate_Ladder_Prediction", "closeness"), exist_ok=True)
	os.makedirs(os.path.join(results_dir, "Quality_Ladder_Prediction", "closeness"), exist_ok=True)


	# Calculating Closeness of Metadata Bitrate Ladders
	print ()
	print ("-"*10, "Closeness Metadata Bitrate Ladders", "-"*10)
	print ()

	calculate_closeness(
		# Files
		Test_Video_Files=Test_Video_Files,

		# Path
		save_path = os.path.join(
			results_dir, "Bitrate_Ladder_Prediction", "closeness", "metadata.npy"
		),

		# Arguments
		Predicted_BD_Metrics = np.load(
			os.path.join(
				results_dir, "Bitrate_Ladder_Prediction", "bd_metrics", "metadata.npy"
			), allow_pickle=True
		)[()],

		Reference_BD_Metrics =  np.load(
			os.path.join(
				results_dir, "Standard", "bd_metrics", "Reference_Bitrate_Ladder", "{}_{}.npy".format(codec, preset)
			), allow_pickle=True
		)[()]
	)


	# Calculating Closeness of Metadata Quality Ladders
	print ()
	print ("-"*10, "Closeness Metadata Quality Ladders", "-"*10)
	print ()

	calculate_closeness(
		# Files
		Test_Video_Files=Test_Video_Files,

		# Path
		save_path = os.path.join(
			results_dir, "Quality_Ladder_Prediction", "closeness", "metadata.npy"
		),

		# Arguments
		Predicted_BD_Metrics = np.load(
			os.path.join(
				results_dir, "Quality_Ladder_Prediction", "bd_metrics", "metadata.npy"
			), allow_pickle=True
		)[()],

		Reference_BD_Metrics =  np.load(
			os.path.join(
				results_dir, "Standard", "bd_metrics", "Reference_Bitrate_Ladder", "{}_{}.npy".format(codec, preset)
			), allow_pickle=True
		)[()]
	)


	# Calculating Closeness of Low-Level Features Bitrate Ladders
	print ()
	print ("-"*10, "Closeness Low-Level Bitrate Ladders", "-"*10)
	print ()

	calculate_closeness(
		# Files
		Test_Video_Files=Test_Video_Files,

		# Path
		save_path = os.path.join(
			results_dir, "Bitrate_Ladder_Prediction", "closeness", "low_level_features.npy".format(codec, preset)
		),

		# Arguments
		Predicted_BD_Metrics = np.load(
			os.path.join(
				results_dir, "Bitrate_Ladder_Prediction", "bd_metrics", "low_level_features.npy".format(codec, preset)
			), allow_pickle=True
		)[()],

		Reference_BD_Metrics =  np.load(
			os.path.join(
				results_dir, "Standard", "bd_metrics", "Reference_Bitrate_Ladder", "{}_{}.npy".format(codec, preset)
			), allow_pickle=True
		)[()]
	)


	# Calculating Closeness of Low-Level Features Quality Ladders
	print ()
	print ("-"*10, "Closeness Low-Level Quality Ladders", "-"*10)
	print ()

	calculate_closeness(
		# Files
		Test_Video_Files=Test_Video_Files,

		# Path
		save_path = os.path.join(
			results_dir, "Quality_Ladder_Prediction", "closeness", "low_level_features.npy".format(codec, preset)
		),

		# Arguments
		Predicted_BD_Metrics = np.load(
			os.path.join(
				results_dir, "Quality_Ladder_Prediction", "bd_metrics", "low_level_features.npy".format(codec, preset)
			), allow_pickle=True
		)[()],

		Reference_BD_Metrics =  np.load(
			os.path.join(
				results_dir, "Standard", "bd_metrics", "Reference_Bitrate_Ladder", "{}_{}.npy".format(codec, preset)
			), allow_pickle=True
		)[()]
	)


	# Calculating Closeness of VIF Features Bitrate Ladders
	print ()
	print ("-"*10, "Closeness VIF Bitrate Ladders", "-"*10)
	print ()

	for vif_approach_number in np.arange(1,10):
		calculate_closeness(
			# Files
			Test_Video_Files=Test_Video_Files,

			# Path
			save_path = os.path.join(
				results_dir, "Bitrate_Ladder_Prediction", "closeness", "vif_features_{}.npy".format(str(vif_approach_number))
			),

			# Arguments
			Predicted_BD_Metrics = np.load(
				os.path.join(
					results_dir, "Bitrate_Ladder_Prediction", "bd_metrics", "vif_features_{}.npy".format(str(vif_approach_number))
				), allow_pickle=True
			)[()],

			Reference_BD_Metrics =  np.load(
				os.path.join(
					results_dir, "Standard", "bd_metrics", "Reference_Bitrate_Ladder", "{}_{}.npy".format(codec, preset)
				), allow_pickle=True
			)[()]
		)


	# Calculating Closeness of VIF Features Quality Ladders
	print ()
	print ("-"*10, "Closeness VIF Quality Ladders", "-"*10)
	print ()

	for vif_approach_number in np.arange(1,10):
		calculate_closeness(
			# Files
			Test_Video_Files=Test_Video_Files,

			# Path
			save_path = os.path.join(
				results_dir, "Quality_Ladder_Prediction", "closeness", "vif_features_{}.npy".format(str(vif_approach_number))
			),

			# Arguments
			Predicted_BD_Metrics = np.load(
				os.path.join(
					results_dir, "Quality_Ladder_Prediction", "bd_metrics", "vif_features_{}.npy".format(str(vif_approach_number))
				), allow_pickle=True
			)[()],

			Reference_BD_Metrics =  np.load(
				os.path.join(
					results_dir, "Standard", "bd_metrics", "Reference_Bitrate_Ladder", "{}_{}.npy".format(codec, preset)
				), allow_pickle=True
			)[()]
		)


	# Calculating Closeness of Low-Level Features and VIF Features Bitrate Ladders
	print ()
	print ("-"*10, "Closeness Low-Level Features and VIF Features Bitrate Ladders", "-"*10)
	print ()

	for vif_approach_number in np.arange(1,10):
		calculate_closeness(
			# Files
			Test_Video_Files=Test_Video_Files,

			# Path
			save_path = os.path.join(
				results_dir, "Bitrate_Ladder_Prediction", "closeness", "low_level_features_vif_features_{}.npy".format(str(vif_approach_number))
			),

			# Arguments
			Predicted_BD_Metrics = np.load(
				os.path.join(
					results_dir, "Bitrate_Ladder_Prediction", "bd_metrics", "low_level_features_vif_features_{}.npy".format(str(vif_approach_number))
				), allow_pickle=True
			)[()],

			Reference_BD_Metrics =  np.load(
				os.path.join(
					results_dir, "Standard", "bd_metrics", "Reference_Bitrate_Ladder", "{}_{}.npy".format(codec, preset)
				), allow_pickle=True
			)[()]
		)


	# Calculating Closeness of Low-Level Features and VIF Features Quality Ladders
	print ()
	print ("-"*10, "Closeness Low-Level Features and VIF Features Quality Ladders", "-"*10)
	print ()

	for vif_approach_number in np.arange(1,10):
		calculate_closeness(
			# Files
			Test_Video_Files=Test_Video_Files,

			# Path
			save_path = os.path.join(
				results_dir, "Quality_Ladder_Prediction", "closeness", "low_level_features_vif_features_{}.npy".format(str(vif_approach_number))
			),

			# Arguments
			Predicted_BD_Metrics = np.load(
				os.path.join(
					results_dir, "Quality_Ladder_Prediction", "bd_metrics", "low_level_features_vif_features_{}.npy".format(str(vif_approach_number))
				), allow_pickle=True
			)[()],

			Reference_BD_Metrics =  np.load(
				os.path.join(
					results_dir, "Standard", "bd_metrics", "Reference_Bitrate_Ladder", "{}_{}.npy".format(codec, preset)
				), allow_pickle=True
			)[()]
		)