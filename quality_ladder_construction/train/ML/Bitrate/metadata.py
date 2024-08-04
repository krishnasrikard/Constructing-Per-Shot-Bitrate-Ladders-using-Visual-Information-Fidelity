## Predicting Bitrate using Meta Data Features using Machine Learning Models

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import sklearn
from sklearn.feature_selection import RFECV, RFECV, SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor

import os, sys, warnings
import pickle
from tqdm import tqdm
warnings.filterwarnings("ignore")
sys.path.append("/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity")
import quality_ladder_construction.QL_functions.dataset_evaluation_functions as dataset_evaluation_functions
import functions.plot_functions as plot_functions
import defaults


def train():
	"""
	Args:
		None
	"""

	## Training, Validation and Test Datasets
	X_train, y_train = dataset_evaluation_functions.Metadata_Bitrate_Dataset(
		video_filenames=input_arguments["train_video_filenames"],
		**arguments,
	)

	X_valid, y_valid = dataset_evaluation_functions.Metadata_Bitrate_Dataset(
		video_filenames=input_arguments["valid_video_filenames"],
		**arguments
	)

	print ("Dimensions of training dataset = ", X_train.shape, y_train.shape)
	print ("Dimensions of validation dataset = ", X_valid.shape, y_valid.shape)
	print ()


	## Training Models
	"""
	# Extra-Trees Regressor
	print ("Extra-Trees Regressor:")
	ETR_model = ExtraTreesRegressor(n_estimators=1250, random_state=2, max_depth=3)
	ETR_model.fit(X_train, y_train)

	plot_functions.Calculate_Prediction_Performance_Metrics(
		y_pred=ETR_model.predict(X_valid),
		y_true=y_valid,
		resolution_data=X_valid[:,-2:]
	)


	# XG-Boost
	print ("XG-Boost Regressor:")
	XGB_model = XGBRegressor(n_estimators=1250, learning_rate=0.0075, device="cuda", random_state=2, max_depth=3)
	XGB_model.fit(X_train,y_train)

	plot_functions.Calculate_Prediction_Performance_Metrics(
		y_pred=XGB_model.predict(X_valid),
		y_true=y_valid,
		resolution_data=X_valid[:,-2:]
	)


	# Random Forest
	print ("Random-Forest Regressor:")
	RF_model = RandomForestRegressor(n_estimators=1250, random_state=2, max_depth=3)
	RF_model.fit(X_train, y_train)

	plot_functions.Calculate_Prediction_Performance_Metrics(
		y_pred=RF_model.predict(X_valid),
		y_true=y_valid,
		resolution_data=X_valid[:,-2:]
	)
	"""


	## Re-Training best model on best features
	test_model = ExtraTreesRegressor(n_estimators=1250, random_state=2, max_depth=3)
	test_model.fit(X_train, y_train)

	# Saving the weights
	pickle.dump(test_model, open(os.path.join(models_path, "model_metadata.pkl"), "wb"))
	np.save(os.path.join(models_path,"feature_indices_metadata.npy"), np.array(None))


	## Performance
	# Performance on Validation Set
	print("Performance on Validation Set")
	y_pred_Results, y_Results, quality_Results = dataset_evaluation_functions.Predict_Metadata_Bitrate(
		Model=test_model,
		video_filenames=input_arguments["valid_video_filenames"],
		feature_indices=None,
		**arguments,
	)
	plot_functions.Plot_Predictions(y_pred_Results=y_pred_Results, y_Results=y_Results, Resolutions=arguments["Resolutions_Considered"], plot_save_path="plots/metadata_valid.png", show=False, save_results="results/metadata_valid.npy")

	# Performance on Test Set
	print("Performance on Test Set")
	y_pred_Results, y_Results, quality_Results = dataset_evaluation_functions.Predict_Metadata_Bitrate(
		Model=test_model,
		video_filenames=input_arguments["test_video_filenames"],
		feature_indices=None,
		**arguments,
	)
	plot_functions.Plot_Predictions(y_pred_Results=y_pred_Results, y_Results=y_Results, Resolutions=arguments["Resolutions_Considered"], plot_save_path="plots/metadata_test.png", show=False, save_results="results/metadata_test.npy")



if __name__ == "__main__":
	# -----------------------------------------------------------------
	# Flushing Output
	import functools
	print = functools.partial(print, flush=True)

	# Saving stdout
	sys.stdout = open('stdouts/{}.log'.format(os.path.basename(__file__)[:-3]), 'w')

	# -----------------------------------------------------------------

	# Arguments
	arguments = {
		# RQ-Points
		"Resolutions_Considered": defaults.resolutions,
		"CRFs_Considered": defaults.CRFs,
		"bitrates_Considered": None,
		"QPs_Considered": None,
		"min_quality": defaults.min_quality,
		"max_quality": defaults.max_quality,
		"min_bitrate": defaults.min_bitrate,
		"max_bitrate": defaults.max_bitrate,

		# Task Arguments

		# Encoder-Settings
		"codec": "libx265",
		"preset": "medium",
		"quality_metric": "vmaf"
	}

	# Input Argumnets
	input_arguments = {
		# Datasets
		"train_video_filenames": defaults.Train_Video_Titles,
		"valid_video_filenames": defaults.Valid_Video_Titles,
		"test_video_filenames": defaults.Test_Video_Titles
	}

	# Paths
	home_path = "/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/quality_ladder_construction"
	models_path = os.path.join(home_path, "models/ML/Bitrate", "metadata")
	if os.path.exists(models_path) == False:
		os.makedirs(models_path)

	# Execute
	train()