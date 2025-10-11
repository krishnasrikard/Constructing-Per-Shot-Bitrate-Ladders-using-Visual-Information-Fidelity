# Testing Regressors with various input features
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
from sklearn.feature_selection import RFE
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor

import os, sys, warnings
sys.path.append("/home/kd28684/Efficient-Dynamic-Optimizer-using-Visual-Information-Fidelity-Working")
import pickle
import functions.plot_functions as plot_functions
import modules.quality_ladder_prediction_dataset_functions as quality_ladder_prediction_dataset_functions
import defaults


# Evaluate Monotonicity
def evaluate_monotonicity(y_pred_Results, Resolutions_Considered):
	Checks = []
	for j in range(len(defaults.Test_Video_Titles)):
		Failed_for_Current_Video = False

		for i in range(len(Resolutions_Considered)-1):	
			# Assertions
			assert y_pred_Results.shape[0] == len(defaults.Test_Video_Titles), "Test Video Titles and Predicted Results do not match"

			# Mask
			mask = np.nonzero(np.where((y_pred_Results[j,i:i+1,:] != -np.inf), 1, 0))
			y_pred = y_pred_Results[j,i:i+1,:][mask]

			# Check
			y_pred = np.round(y_pred, decimals=4).flatten()
			if np.all(np.diff(y_pred) >= -1e-4):
				None
			else:
				Failed_for_Current_Video = True
				break
		
		if Failed_for_Current_Video:
			Checks.append(False)
		else:
			Checks.append(True)
	
	return np.mean(np.array(Checks).astype(int))



# Testing to Low-Level features to predict Cross-Over bitrates
def test_crossoverqualities(
	# Files
	Train_Video_Files:list,
	Valid_Video_Files:list,
	Test_Video_Files:list,

	# Method Arguments
	features_names:list,
	temporal_low_level_features:bool,

	# Paths
	results_dir:str,

	# Arguments
	codec:str,
	preset:str,
	quality_metric:str,
	Resolutions_Considered:list,
	CRFs_Considered:list,
	QPs_Considered:list,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate,
):
	# Predicted CrossOver_Qualities
	valid_Predicted_CrossOver_Qualities = []

	# Resolutions
	Resolutions = sorted(Resolutions_Considered, reverse=True)

	for i in range(len(Resolutions)-1):
		## Test Datasets
		X_test, y_test = quality_ladder_prediction_dataset_functions.LowLevelFeatures_CrossOverQualities_Dataset(
			# Files
			video_filenames=Test_Video_Files,

			# Method Arguments
			features_names=features_names,
			temporal_low_level_features=temporal_low_level_features,

			# Arguments
			codec=codec,
			preset=preset,
			quality_metric=quality_metric,
			Resolutions_Considered=Resolutions_Considered,
			CRFs_Considered=CRFs_Considered,
			QPs_Considered=QPs_Considered,
			high_res=Resolutions[i],
			low_res=Resolutions[i+1],
			min_quality=min_quality,
			max_quality=max_quality,
			min_bitrate=min_bitrate,
			max_bitrate=max_bitrate
		)

		# Appending previously predicted Cross-Over bitrates
		X_test = np.concatenate([X_test, *valid_Predicted_CrossOver_Qualities], axis=1)


		# Load Model and
		test_model = pickle.load(
			open(os.path.join(
				results_dir, "models", "low_level_features_cob_{}.pkl".format(i)
			), "rb")
		)
		indices = np.load(
			open(os.path.join(
				results_dir, "models", "low_level_features_indices_{}.npy".format(i)
			), "rb")
		)

		# Feature-Elimination on Inputs
		X_test = X_test[...,indices]
		

		## Testing best model on best features
		# Predictions
		y_test_pred = np.round(test_model.predict(X_test).reshape(-1,1), decimals=4)

		# Performance
		print ("Test:\n")
		print ("MSE =", np.round(sklearn.metrics.mean_squared_error(np.squeeze(y_test_pred), np.squeeze(y_test)), decimals=3))
		print ("PLCC =", np.round(scipy.stats.pearsonr(np.squeeze(y_test_pred), np.squeeze(y_test))[0], decimals=3))
		print ("SRCC =", np.round(scipy.stats.spearmanr(np.squeeze(y_test_pred), np.squeeze(y_test))[0], decimals=3))
		print ()

		# Predicting for CrossOver Bitrates for next set of resolutions
		valid_Predicted_CrossOver_Qualities.append(y_test_pred)



# Testing using Metadata
def test_metadata(
	# Files
	Train_Video_Files:list,
	Valid_Video_Files:list,
	Test_Video_Files:list,

	# Paths
	results_dir:str,

	# Arguments
	codec:str,
	preset:str,
	quality_metric:str,
	Resolutions_Considered:list,
	CRFs_Considered:list,
	QPs_Considered:list,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate,
):
	## Test Dataset
	X_test, y_test = quality_ladder_prediction_dataset_functions.Metadata_Bitrate_Dataset(
		# Files
		video_filenames=Test_Video_Files,
		
		# Arguments
		codec=codec,
		preset=preset,
		quality_metric=quality_metric,
		Resolutions_Considered=Resolutions_Considered,
		CRFs_Considered=CRFs_Considered,
		QPs_Considered=QPs_Considered,
		min_quality=min_quality,
		max_quality=max_quality,
		min_bitrate=min_bitrate,
		max_bitrate=max_bitrate
	)

	print ("Dimensions of test dataset = ", X_test.shape, y_test.shape)
	print ()


	## Load Model
	test_model = pickle.load(
		open(os.path.join(
			results_dir, "models", "metadata.pkl"
		), "rb")
	)


	## Testing best model on best features
	# Performance on Test Set
	print("Performance on Test Set")
	y_pred_Results, y_Results, _ = quality_ladder_prediction_dataset_functions.Predict_Metadata_Bitrate(
		# Model
		Model=test_model,

		# Files
		video_filenames=Test_Video_Files,
		
		# Arguments
		codec=codec,
		preset=preset,
		quality_metric=quality_metric,
		Resolutions_Considered=Resolutions_Considered,
		CRFs_Considered=CRFs_Considered,
		QPs_Considered=QPs_Considered,
		min_quality=min_quality,
		max_quality=max_quality,
		min_bitrate=min_bitrate,
		max_bitrate=max_bitrate
	)

	# Plotting Performance
	plot_functions.Plot_Predictions(
		y_pred_Results=y_pred_Results,
		y_Results=y_Results,
		Resolutions=Resolutions_Considered,
		plot_save_path=None,
		show=False,
		save_results=None,
	)

	# Evaluate Monotonicity
	print ("Monotonicity Check Results:", evaluate_monotonicity(y_pred_Results, Resolutions_Considered))



# Testing using Low-Level features
def test_low_level_features(
	# Files
	Train_Video_Files:list,
	Valid_Video_Files:list,
	Test_Video_Files:list,

	# Method Arguments
	features_names:list,
	temporal_low_level_features:bool,

	# Paths
	results_dir:str,

	# Arguments
	codec:str,
	preset:str,
	quality_metric:str,
	Resolutions_Considered:list,
	CRFs_Considered:list,
	QPs_Considered:list,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate,
):
	## Test Dataset
	X_test, y_test = quality_ladder_prediction_dataset_functions.LowLevelFeatures_Bitrate_Dataset(
		# Files
		video_filenames=Test_Video_Files,

		# Method Arguments
		features_names=features_names,
		temporal_low_level_features=temporal_low_level_features,
		
		# Arguments
		codec=codec,
		preset=preset,
		quality_metric=quality_metric,
		Resolutions_Considered=Resolutions_Considered,
		CRFs_Considered=CRFs_Considered,
		QPs_Considered=QPs_Considered,
		min_quality=min_quality,
		max_quality=max_quality,
		min_bitrate=min_bitrate,
		max_bitrate=max_bitrate
	)

	print ("Dimensions of test dataset = ", X_test.shape, y_test.shape)
	print ()


	## Load Model
	test_model = pickle.load(
		open(os.path.join(
			results_dir, "models", "low_level_features.pkl"
		), "rb")
	)


	## Testing best model on best features
	# Performance on Test Set
	print("Performance on Test Set")
	y_pred_Results, y_Results, _ = quality_ladder_prediction_dataset_functions.Predict_LowLevelFeatures_Bitrate(
		# Model
		Model=test_model,

		# Files
		video_filenames=Test_Video_Files,

		# Method Arguments
		features_names=features_names,
		temporal_low_level_features=temporal_low_level_features,
		
		# Arguments
		codec=codec,
		preset=preset,
		quality_metric=quality_metric,
		Resolutions_Considered=Resolutions_Considered,
		CRFs_Considered=CRFs_Considered,
		QPs_Considered=QPs_Considered,
		min_quality=min_quality,
		max_quality=max_quality,
		min_bitrate=min_bitrate,
		max_bitrate=max_bitrate
	)

	# Plotting Performance
	plot_functions.Plot_Predictions(
		y_pred_Results=y_pred_Results,
		y_Results=y_Results,
		Resolutions=Resolutions_Considered,
		plot_save_path=None,
		show=False,
		save_results=None,
	)

	# Evaluate Monotonicity
	print ("Monotonicity Check Results:", evaluate_monotonicity(y_pred_Results, Resolutions_Considered))



# Testing using VIF features
def test_vif_features(
	# Files
	Train_Video_Files:list,
	Valid_Video_Files:list,
	Test_Video_Files:list,

	# Method Arguments
	per_frame:bool,
	per_frame_features_flatten:bool,
	vif_setting:str,
	vif_features_list:list,
	vif_approach_number:str,

	# Paths
	results_dir:str,

	# Arguments
	codec:str,
	preset:str,
	quality_metric:str,
	Resolutions_Considered:list,
	CRFs_Considered:list,
	QPs_Considered:list,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate,
):
	## Test Dataset
	X_test, y_test = quality_ladder_prediction_dataset_functions.VIFFeatures_Bitrate_Dataset(
		# Files
		video_filenames=Test_Video_Files,

		# Method Arguments
		per_frame=per_frame,
		per_frame_features_flatten=per_frame_features_flatten,
		vif_setting=vif_setting,
		vif_features_list=vif_features_list,
		
		# Arguments
		codec=codec,
		preset=preset,
		quality_metric=quality_metric,
		Resolutions_Considered=Resolutions_Considered,
		CRFs_Considered=CRFs_Considered,
		QPs_Considered=QPs_Considered,
		min_quality=min_quality,
		max_quality=max_quality,
		min_bitrate=min_bitrate,
		max_bitrate=max_bitrate
	)

	print ("Dimensions of test dataset = ", X_test.shape, y_test.shape)
	print ()


	## Load Model
	test_model = pickle.load(
		open(os.path.join(
			results_dir, "models", "vif_features_{}.pkl".format(vif_approach_number)
		), "rb")
	)


	## Testing best model on best features
	# Performance on Test Set
	print("Performance on Test Set")
	y_pred_Results, y_Results, _ = quality_ladder_prediction_dataset_functions.Predict_VIFFeatures_Bitrate(
		# Model
		Model=test_model,

		# Files
		video_filenames=Test_Video_Files,

		# Method Arguments
		per_frame=per_frame,
		per_frame_features_flatten=per_frame_features_flatten,
		vif_setting=vif_setting,
		vif_features_list=vif_features_list,
		
		# Arguments
		codec=codec,
		preset=preset,
		quality_metric=quality_metric,
		Resolutions_Considered=Resolutions_Considered,
		CRFs_Considered=CRFs_Considered,
		QPs_Considered=QPs_Considered,
		min_quality=min_quality,
		max_quality=max_quality,
		min_bitrate=min_bitrate,
		max_bitrate=max_bitrate
	)

	# Plotting Performance
	plot_functions.Plot_Predictions(
		y_pred_Results=y_pred_Results,
		y_Results=y_Results,
		Resolutions=Resolutions_Considered,
		plot_save_path=None,
		show=False,
		save_results=None,
	)

	# Evaluate Monotonicity
	print ("Monotonicity Check Results:", evaluate_monotonicity(y_pred_Results, Resolutions_Considered))



# Testing using Low-Level Features and VIF features
def test_low_level_features_vif_features(
	# Files
	Train_Video_Files:list,
	Valid_Video_Files:list,
	Test_Video_Files:list,

	# Method Arguments
	features_names:list,
	temporal_low_level_features:bool,
	per_frame:bool,
	per_frame_features_flatten:bool,
	vif_setting:str,
	vif_features_list:list,
	vif_approach_number:str,

	# Paths
	results_dir:str,

	# Arguments
	codec:str,
	preset:str,
	quality_metric:str,
	Resolutions_Considered:list,
	CRFs_Considered:list,
	QPs_Considered:list,
	min_quality=defaults.min_quality,
	max_quality=defaults.max_quality,
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate,
):
	## Test Dataset
	X_test, y_test = quality_ladder_prediction_dataset_functions.LowLevelFeatures_VIFFeatures_Bitrate_Dataset(
		# Files
		video_filenames=Test_Video_Files,

		# Method Arguments
		features_names=features_names,
		temporal_low_level_features=temporal_low_level_features,
		per_frame=per_frame,
		per_frame_features_flatten=per_frame_features_flatten,
		vif_setting=vif_setting,
		vif_features_list=vif_features_list,
		
		# Arguments
		codec=codec,
		preset=preset,
		quality_metric=quality_metric,
		Resolutions_Considered=Resolutions_Considered,
		CRFs_Considered=CRFs_Considered,
		QPs_Considered=QPs_Considered,
		min_quality=min_quality,
		max_quality=max_quality,
		min_bitrate=min_bitrate,
		max_bitrate=max_bitrate
	)

	print ("Dimensions of test dataset = ", X_test.shape, y_test.shape)
	print ()


	## Testing Models
	test_model = pickle.load(
		open(os.path.join(
			results_dir, "models", "low_level_features_vif_features_{}.pkl".format(vif_approach_number)
		), "rb")
	)


	## Testing best model on best features
	# Performance on Test Set
	print("Performance on Test Set")
	y_pred_Results, y_Results, _ = quality_ladder_prediction_dataset_functions.Predict_LowLevelFeatures_VIFFeatures_Bitrate(
		# Model
		Model=test_model,

		# Files
		video_filenames=Test_Video_Files,

		# Method Arguments
		codec=codec,
		preset=preset,
		features_names=features_names,
		temporal_low_level_features=temporal_low_level_features,
		per_frame=per_frame,
		per_frame_features_flatten=per_frame_features_flatten,
		vif_setting=vif_setting,
		vif_features_list=vif_features_list,
		
		# Arguments
		quality_metric=quality_metric,
		Resolutions_Considered=Resolutions_Considered,
		CRFs_Considered=CRFs_Considered,
		QPs_Considered=QPs_Considered,
		min_quality=min_quality,
		max_quality=max_quality,
		min_bitrate=min_bitrate,
		max_bitrate=max_bitrate
	)

	# Plotting Performance
	plot_functions.Plot_Predictions(
		y_pred_Results=y_pred_Results,
		y_Results=y_Results,
		Resolutions=Resolutions_Considered,
		plot_save_path=None,
		show=False,
		save_results=None,
	)

	# Evaluate Monotonicity
	print ("Monotonicity Check Results:", evaluate_monotonicity(y_pred_Results, Resolutions_Considered))



def main(
	# Files
	Train_Video_Files:list,
	Valid_Video_Files:list,
	Test_Video_Files:list,

	# Path
	results_dir,

	# Method Arguments
	features_names:list,
	vif_approach_numbers:list
):
	# Features Settings
	temporal_low_level_features=False
	per_frame = False
	per_frame_features_flatten = False
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


	# Arguments
	arguments = {
		# Encoder-Settings
		"codec": "libx265",
		"preset": "medium",
		"quality_metric": "vmaf",

		# RQ-Points
		"Resolutions_Considered": defaults.resolutions,
		"CRFs_Considered": None,
		"QPs_Considered": None,
		"min_quality": defaults.min_quality,
		"max_quality": defaults.max_quality,
		"min_bitrate": defaults.min_bitrate,
		"max_bitrate": defaults.max_bitrate,
	}


	# Updating Arguments
	arguments["CRFs_Considered"] = defaults.CRFs


	## Testing Cross-Over Bitrates prediction model
	# Creating Directories
	os.makedirs(os.path.join(results_dir, "CrossOver_Qualities", "models"), exist_ok=True)

	test_crossoverqualities(
		# Files
		Train_Video_Files=Train_Video_Files,
		Valid_Video_Files=Valid_Video_Files,
		Test_Video_Files=Test_Video_Files,

		# Method Arguments
		features_names=features_names[:-3],
		temporal_low_level_features=temporal_low_level_features,

		# Path
		results_dir=os.path.join(results_dir, "CrossOver_Qualities"),

		# Arguments
		**arguments
	)



	# Creating Directories
	os.makedirs(os.path.join(results_dir, "Quality_Ladder_Prediction", "models"), exist_ok=True)
	os.makedirs(os.path.join(results_dir, "Quality_Ladder_Prediction", "bitrate_prediction_results"), exist_ok=True)
	os.makedirs(os.path.join(results_dir, "Quality_Ladder_Prediction", "bitrate_prediction_plots"), exist_ok=True)


	# Metadata
	print ()
	print ("-"*10, "Testing using Metadata", "-"*10)
	print ()

	test_metadata(
		# Files
		Train_Video_Files=Train_Video_Files,
		Valid_Video_Files=Valid_Video_Files,
		Test_Video_Files=Test_Video_Files,

		# Path
		results_dir=os.path.join(results_dir, "Quality_Ladder_Prediction"),

		# Arguments
		**arguments
	)


	# Low-Level Features
	print ()
	print ("-"*10, "Testing using Low-Level Features", "-"*10)
	print ()

	test_low_level_features(
		# Files
		Train_Video_Files=Train_Video_Files,
		Valid_Video_Files=Valid_Video_Files,
		Test_Video_Files=Test_Video_Files,

		# Method Arguments
		features_names=features_names,
		temporal_low_level_features=temporal_low_level_features,

		# Path
		results_dir=os.path.join(results_dir, "Quality_Ladder_Prediction"),

		# Arguments
		**arguments
	)


	# VIF Features
	print ()
	print ("-"*10, "Testing using VIF Features", "-"*10)
	print ()

	for vif_approach_number in vif_approach_numbers:
		test_vif_features(
			# Files
			Train_Video_Files=Train_Video_Files,
			Valid_Video_Files=Valid_Video_Files,
			Test_Video_Files=Test_Video_Files,

			# Method Arguments
			per_frame=per_frame,
			per_frame_features_flatten=per_frame_features_flatten,
			vif_setting=VIF_Approach_Map[vif_approach_number][1],
			vif_features_list=VIF_Approach_Map[vif_approach_number][0],
			vif_approach_number=vif_approach_number,

			# Path
			results_dir=os.path.join(results_dir, "Quality_Ladder_Prediction"),

			# Arguments
			**arguments
		)


	# Low-Level Features and VIF Features
	print ()
	print ("-"*10, "Testing using Low-Level Features and VIF Features", "-"*10)
	print ()

	for vif_approach_number in vif_approach_numbers:
		test_low_level_features_vif_features(
			# Files
			Train_Video_Files=Train_Video_Files,
			Valid_Video_Files=Valid_Video_Files,
			Test_Video_Files=Test_Video_Files,

			# Method Arguments
			features_names=features_names,
			temporal_low_level_features=temporal_low_level_features,
			per_frame=per_frame,
			per_frame_features_flatten=per_frame_features_flatten,
			vif_setting=VIF_Approach_Map[vif_approach_number][1],
			vif_features_list=VIF_Approach_Map[vif_approach_number][0],
			vif_approach_number=vif_approach_number,

			# Path
			results_dir=os.path.join(results_dir, "Quality_Ladder_Prediction"),

			# Arguments
			**arguments
		)