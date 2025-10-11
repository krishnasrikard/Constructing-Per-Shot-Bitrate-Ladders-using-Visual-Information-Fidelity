# Main Function
# Importing Libraries
import numpy as np

import os, sys, warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import joblib
import scripts.train_bitrate_ladder_prediction_methods as train_bitrate_ladder_prediction_methods
import scripts.train_quality_ladder_prediction_methods as train_quality_ladder_prediction_methods
import scripts.test_bitrate_ladder_prediction_methods as test_bitrate_ladder_prediction_methods
import scripts.test_quality_ladder_prediction_methods as test_quality_ladder_prediction_methods
import scripts.reference_bitrate_ladder as reference_bitrate_ladder
import scripts.bitrate_ladders as bitrate_ladders
import scripts.quality_ladders as quality_ladders
import scripts.bd_metrics as bd_metrics
import scripts.closeness as closeness
import defaults

# Execute Function
def execute_training(
	Train_Video_Files:list,
	Valid_Video_Files:list,
	Test_Video_Files:list,
	save_results_folder:str
):
	"""
	Args:
		Train_Video_Files (list): List of video-files used for training.
		Valid_Video_Files (list): List of video-files used for validation.
		Test_Video_Files (list): List of video-files used for testing.
		save_results_folder (list): Folder name to in results folder to save results.
	"""
	## Assertions
	# """
	print ()
	print ("-"*25, "Assertions", "-"*25)
	print ()

	for video_file in tqdm(Train_Video_Files + Valid_Video_Files + Test_Video_Files):
		# Check for YUV file
		assert os.path.exists(
			os.path.join(defaults.source_dataset_path, video_file + ".yuv")
		), "For video-file: {}, YUV does not exist."

		# Check for RQ Points
		assert os.path.exists(
			os.path.join(defaults.rq_points_dataset_path, "libx265", "medium", video_file, "crfs.json")
		)
	# """


	## Features
	# Low-Level Features (Custom-Features always at the end so as to match code in 'dataset_evaluation_functions.py')
	bitrate_ladder_prediction_features_names = []
	for features_subset in [defaults.glcm_features, defaults.tc_features, defaults.si_features, defaults.ti_features, defaults.cti_features, defaults.cf_features, defaults.ci_features, defaults.dct_features, list(defaults.bitrate_texture_features.keys())]:
		for f in features_subset:
			bitrate_ladder_prediction_features_names.append(f)

	quality_ladder_prediction_features_names = []
	for features_subset in [defaults.glcm_features, defaults.tc_features, defaults.si_features, defaults.ti_features, defaults.cti_features, defaults.cf_features, defaults.ci_features, defaults.dct_features, list(defaults.quality_texture_features.keys())]:
		for f in features_subset:
			quality_ladder_prediction_features_names.append(f)

	# VIF-Approach Number
	vif_approach_numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]


	## Training Regressors
	# """
	print ()
	print ("-"*25, "Training", "-"*25)
	print ()

	train_bitrate_ladder_prediction_methods.main(
		Train_Video_Files=Train_Video_Files,
		Valid_Video_Files=Valid_Video_Files,
		Test_Video_Files=Test_Video_Files,
		results_dir=save_results_folder,
		features_names=bitrate_ladder_prediction_features_names,
		vif_approach_numbers=vif_approach_numbers
	)

	train_quality_ladder_prediction_methods.main(
		Train_Video_Files=Train_Video_Files,
		Valid_Video_Files=Valid_Video_Files,
		Test_Video_Files=Test_Video_Files,
		results_dir=save_results_folder,
		features_names=quality_ladder_prediction_features_names,
		vif_approach_numbers=vif_approach_numbers
	)
	# """
	


# Execute Function
def execute(
	Train_Video_Files:list,
	Valid_Video_Files:list,
	Test_Video_Files:list,
	save_results_folder:str
):
	"""
	Args:
		Train_Video_Files (list): List of video-files used for training.
		Valid_Video_Files (list): List of video-files used for validation.
		Test_Video_Files (list): List of video-files used for testing.
		save_results_folder (list): Folder name to in results folder to save results.
	"""
	## Features
	# Low-Level Features (Custom-Features always at the end so as to match code in 'dataset_evaluation_functions.py')
	bitrate_ladder_prediction_features_names = []
	for features_subset in [defaults.glcm_features, defaults.tc_features, defaults.si_features, defaults.ti_features, defaults.cti_features, defaults.cf_features, defaults.ci_features, defaults.dct_features, list(defaults.bitrate_texture_features.keys())]:
		for f in features_subset:
			bitrate_ladder_prediction_features_names.append(f)

	quality_ladder_prediction_features_names = []
	for features_subset in [defaults.glcm_features, defaults.tc_features, defaults.si_features, defaults.ti_features, defaults.cti_features, defaults.cf_features, defaults.ci_features, defaults.dct_features, list(defaults.quality_texture_features.keys())]:
		for f in features_subset:
			quality_ladder_prediction_features_names.append(f)

	# VIF-Approach Number
	vif_approach_numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]


	## Convex-Hulls
	# """
	print ()
	print ("-"*25, "Convex-Hulls", "-"*25)
	print ()

	reference_bitrate_ladder.main(
		Test_Video_Files=Test_Video_Files,
		results_dir=save_results_folder,
	)
	# """


	## Predict Bitrate/Quality Ladders
	# """
	print ()
	print ("-"*25, "Bitrate and Quality Ladders", "-"*25)
	print ()

	bitrate_ladders.main(
		Test_Video_Files=Test_Video_Files,
		results_dir=save_results_folder,
		features_names=bitrate_ladder_prediction_features_names,
		vif_approach_numbers=vif_approach_numbers
	)

	quality_ladders.main(
		Test_Video_Files=Test_Video_Files,
		results_dir=save_results_folder,
		features_names=quality_ladder_prediction_features_names,
		vif_approach_numbers=vif_approach_numbers
	)
	# """


	## Calculate BD-Metrics
	# """
	print ()
	print ("-"*25, "BD-Metrics", "-"*25)
	print ()

	bd_metrics.main(
		Test_Video_Files=Test_Video_Files,
		results_dir=save_results_folder,
	)
	# """


	## Calculate Closeness
	# """
	print ()
	print ("-"*25, "Closeness", "-"*25)
	print ()

	closeness.main(
		Test_Video_Files=Test_Video_Files,
		results_dir=save_results_folder,
	)
	# """



# Execute Function
def execute_testing(
	Train_Video_Files:list,
	Valid_Video_Files:list,
	Test_Video_Files:list,
	save_results_folder:str
):
	"""
	Args:
		Train_Video_Files (list): List of video-files used for training.
		Valid_Video_Files (list): List of video-files used for validation.
		Test_Video_Files (list): List of video-files used for testing.
		save_results_folder (list): Folder name to in results folder to save results.
	"""
	## Assertions
	# """
	print ()
	print ("-"*25, "Assertions", "-"*25)
	print ()

	for video_file in tqdm(Train_Video_Files + Valid_Video_Files + Test_Video_Files):
		# Check for YUV file
		assert os.path.exists(
			os.path.join(defaults.source_dataset_path, video_file + ".yuv")
		), "For video-file: {}, YUV does not exist."

		# Check for RQ Points
		assert os.path.exists(
			os.path.join(defaults.rq_points_dataset_path, "libx265", "medium", video_file, "crfs.json")
		)
	# """


	## Features
	# Low-Level Features (Custom-Features always at the end so as to match code in 'dataset_evaluation_functions.py')
	bitrate_ladder_prediction_features_names = []
	for features_subset in [defaults.glcm_features, defaults.tc_features, defaults.si_features, defaults.ti_features, defaults.cti_features, defaults.cf_features, defaults.ci_features, defaults.dct_features, list(defaults.bitrate_texture_features.keys())]:
		for f in features_subset:
			bitrate_ladder_prediction_features_names.append(f)

	quality_ladder_prediction_features_names = []
	for features_subset in [defaults.glcm_features, defaults.tc_features, defaults.si_features, defaults.ti_features, defaults.cti_features, defaults.cf_features, defaults.ci_features, defaults.dct_features, list(defaults.quality_texture_features.keys())]:
		for f in features_subset:
			quality_ladder_prediction_features_names.append(f)

	# VIF-Approach Number
	vif_approach_numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]


	## Testing Regressors
	# """
	print ()
	print ("-"*25, "Testing", "-"*25)
	print ()

	test_bitrate_ladder_prediction_methods.main(
		Train_Video_Files=Train_Video_Files,
		Valid_Video_Files=Valid_Video_Files,
		Test_Video_Files=Test_Video_Files,
		results_dir=save_results_folder,
		features_names=bitrate_ladder_prediction_features_names,
		vif_approach_numbers=vif_approach_numbers
	)

	test_quality_ladder_prediction_methods.main(
		Train_Video_Files=Train_Video_Files,
		Valid_Video_Files=Valid_Video_Files,
		Test_Video_Files=Test_Video_Files,
		results_dir=save_results_folder,
		features_names=quality_ladder_prediction_features_names,
		vif_approach_numbers=vif_approach_numbers
	)
	# """



if __name__ == "__main__":
	"""
	print ("-"*100)
	print ("-"*100)
	execute_training(
		Train_Video_Files=defaults.Train_Video_Titles,
		Valid_Video_Files=defaults.Valid_Video_Titles,
		Test_Video_Files=defaults.Test_Video_Titles,
		save_results_folder="results/main"
	)
	"""

	"""
	execute(
		Train_Video_Files=defaults.Train_Video_Titles,
		Valid_Video_Files=defaults.Valid_Video_Titles,
		Test_Video_Files=defaults.Test_Video_Titles,
		save_results_folder="results/main"
	)
	"""

	# """
	execute_testing(
		Train_Video_Files=defaults.Train_Video_Titles,
		Valid_Video_Files=defaults.Valid_Video_Titles,
		Test_Video_Files=defaults.Test_Video_Titles,
		save_results_folder="results/main"
	)
	# """