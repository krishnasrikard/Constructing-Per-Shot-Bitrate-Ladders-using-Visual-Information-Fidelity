# Training Regressors with various input features
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


# Training to Low-Level features to predict Cross-Over bitrates
def train_crossoverqualities(
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
	train_Predicted_CrossOver_Qualities = []
	valid_Predicted_CrossOver_Qualities = []

	# Resolutions
	Resolutions = sorted(Resolutions_Considered, reverse=True)

	for i in range(len(Resolutions)-1):
		## Training, Validation and Test Datasets
		X_train, y_train = quality_ladder_prediction_dataset_functions.LowLevelFeatures_CrossOverQualities_Dataset(
			# Files
			video_filenames=Train_Video_Files,

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
		X_valid, y_valid = quality_ladder_prediction_dataset_functions.LowLevelFeatures_CrossOverQualities_Dataset(
			# Files
			video_filenames=Valid_Video_Files,

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
		X_train = np.concatenate([X_train, *train_Predicted_CrossOver_Qualities], axis=1)
		X_valid = np.concatenate([X_valid, *valid_Predicted_CrossOver_Qualities], axis=1)


		# Recursive Feature Elimination
		rfe = RFE(
			estimator=RandomForestRegressor(
				n_estimators=1250, 
				criterion="squared_error", random_state=2, 
				max_depth=12, max_features="log2", n_jobs=-1
			), 
			n_features_to_select=9
		)
		rfe.fit(X_train, y_train)
		indices = [i for i,v in enumerate(rfe.support_) if v]

		# Feature-Elimination on Inputs
		X_train = X_train[...,indices]
		X_valid = X_valid[...,indices]
		

		## Training best model on best features
		test_model = ExtraTreesRegressor(n_estimators=1250, random_state=2, max_depth=12, n_jobs=-1, max_features="log2")
		test_model.fit(X_train, y_train)

		# Saving the weights
		pickle.dump(
			test_model, 
			open(os.path.join(
				results_dir, "models", "low_level_features_cob_{}.pkl".format(i)
			), "wb")
		)

		np.save(
			open(os.path.join(
				results_dir, "models", "low_level_features_indices_{}.npy".format(i)
			), "wb"), 
			indices
		)

		# Predictions
		y_train_pred = np.round(test_model.predict(X_train).reshape(-1,1), decimals=4)
		y_valid_pred = np.round(test_model.predict(X_valid).reshape(-1,1), decimals=4)

		# Performance
		print ("Validation:\n")
		print ("MSE =", np.round(sklearn.metrics.mean_squared_error(np.squeeze(y_valid_pred), np.squeeze(y_valid)), decimals=3))
		print ("PLCC =", np.round(scipy.stats.pearsonr(np.squeeze(y_valid_pred), np.squeeze(y_valid))[0], decimals=3))
		print ("SRCC =", np.round(scipy.stats.spearmanr(np.squeeze(y_valid_pred), np.squeeze(y_valid))[0], decimals=3))
		print ()

		# Predicting for CrossOver Bitrates for next set of resolutions
		train_Predicted_CrossOver_Qualities.append(y_train_pred)
		valid_Predicted_CrossOver_Qualities.append(y_valid_pred)



# Training using Metadata
def train_metadata(
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
	## Training, Validation and Test Datasets
	X_train, y_train = quality_ladder_prediction_dataset_functions.Metadata_Bitrate_Dataset(
		# Files
		video_filenames=Train_Video_Files,

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

	X_valid, y_valid = quality_ladder_prediction_dataset_functions.Metadata_Bitrate_Dataset(
		# Files
		video_filenames=Valid_Video_Files,
		
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


	## Training best model on best features
	test_model = ExtraTreesRegressor(n_estimators=1250, random_state=2, max_depth=3, n_jobs=-1)
	test_model.fit(X_train, y_train)

	# Saving the weights
	pickle.dump(
		test_model, 
		open(os.path.join(
			results_dir, "models", "metadata.pkl"
		), "wb")
	)


	## Performance
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
		plot_save_path=os.path.join(results_dir, "bitrate_prediction_plots", "metadata.png"),
		show=False,
		save_results=os.path.join(results_dir, "bitrate_prediction_results", "metadata.npy"),
	)



# Training using Low-Level features
def train_low_level_features(
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
	## Training, Validation and Test Datasets
	X_train, y_train = quality_ladder_prediction_dataset_functions.LowLevelFeatures_Bitrate_Dataset(
		# Files
		video_filenames=Train_Video_Files,

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

	X_valid, y_valid = quality_ladder_prediction_dataset_functions.LowLevelFeatures_Bitrate_Dataset(
		# Files
		video_filenames=Valid_Video_Files,

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

	print ("Dimensions of training dataset = ", X_train.shape, y_train.shape)
	print ("Dimensions of validation dataset = ", X_valid.shape, y_valid.shape)
	print ()


	## Training Models
	"""
	# Extra-Trees Regressor
	print ("Extra-Trees Regressor:")
	ETR_model = ExtraTreesRegressor(n_estimators=1250, random_state=2, max_depth=11)
	ETR_model.fit(X_train, y_train)

	plot_functions.Calculate_Prediction_Performance_Metrics(
		y_pred=ETR_model.predict(X_valid),
		y_true=y_valid,
		resolution_data=X_valid[:,-2:]
	)


	# XG-Boost
	print ("XG-Boost Regressor:")
	XGB_model = XGBRegressor(n_estimators=1250, learning_rate=0.0075, device="cuda", random_state=2, max_depth=11)
	XGB_model.fit(X_train,y_train)

	plot_functions.Calculate_Prediction_Performance_Metrics(
		y_pred=XGB_model.predict(X_valid),
		y_true=y_valid,
		resolution_data=X_valid[:,-2:]
	)


	# Random Forest
	print ("Random-Forest Regressor:")
	RF_model = RandomForestRegressor(n_estimators=1250, random_state=2, max_depth=11)
	RF_model.fit(X_train, y_train)

	plot_functions.Calculate_Prediction_Performance_Metrics(
		y_pred=RF_model.predict(X_valid),
		y_true=y_valid,
		resolution_data=X_valid[:,-2:]
	)
	"""


	## Training best model on best features
	test_model = ExtraTreesRegressor(n_estimators=1250, random_state=2, max_depth=11, n_jobs=-1)
	test_model.fit(X_train, y_train)

	# Saving the weights
	pickle.dump(
		test_model, 
		open(os.path.join(
			results_dir, "models", "low_level_features.pkl"
		), "wb")
	)


	## Performance
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
		plot_save_path=os.path.join(results_dir, "bitrate_prediction_plots", "low_level_features.png"),
		show=False,
		save_results=os.path.join(results_dir, "bitrate_prediction_results", "low_level_features.npy"),
	)



# Training using VIF features
def train_vif_features(
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
	## Training, Validation and Test Datasets
	X_train, y_train = quality_ladder_prediction_dataset_functions.VIFFeatures_Bitrate_Dataset(
		# Files
		video_filenames=Train_Video_Files,

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

	X_valid, y_valid = quality_ladder_prediction_dataset_functions.VIFFeatures_Bitrate_Dataset(
		# Files
		video_filenames=Valid_Video_Files,

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

	print ("Dimensions of training dataset = ", X_train.shape, y_train.shape)
	print ("Dimensions of validation dataset = ", X_valid.shape, y_valid.shape)
	print ()


	## Training Models
	"""
	# Extra-Trees Regressor
	print ("Extra-Trees Regressor:")
	ETR_model = ExtraTreesRegressor(n_estimators=1250, random_state=2, max_depth=8)
	ETR_model.fit(X_train, y_train)

	plot_functions.Calculate_Prediction_Performance_Metrics(
		y_pred=ETR_model.predict(X_valid),
		y_true=y_valid,
		resolution_data=X_valid[:,-2:]
	)


	# XG-Boost
	print ("XG-Boost Regressor:")
	XGB_model = XGBRegressor(n_estimators=1250, learning_rate=0.0075, device="cuda", random_state=2, max_depth=8)
	XGB_model.fit(X_train,y_train)

	plot_functions.Calculate_Prediction_Performance_Metrics(
		y_pred=XGB_model.predict(X_valid),
		y_true=y_valid,
		resolution_data=X_valid[:,-2:]
	)


	# Random Forest
	print ("Random-Forest Regressor:")
	RF_model = RandomForestRegressor(n_estimators=1250, random_state=2, max_depth=8)
	RF_model.fit(X_train, y_train)

	plot_functions.Calculate_Prediction_Performance_Metrics(
		y_pred=RF_model.predict(X_valid),
		y_true=y_valid,
		resolution_data=X_valid[:,-2:]
	)
	"""


	## Training best model on best features
	test_model = ExtraTreesRegressor(n_estimators=1250, random_state=2, max_depth=8, n_jobs=-1)
	test_model.fit(X_train, y_train)

	# Saving the weights
	pickle.dump(
		test_model, 
		open(os.path.join(
			results_dir, "models", "vif_features_{}.pkl".format(vif_approach_number)
		), "wb")
	)


	## Performance
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
		plot_save_path=os.path.join(results_dir, "bitrate_prediction_plots", "vif_features_{}.png".format(vif_approach_number)),
		show=False,
		save_results=os.path.join(results_dir, "bitrate_prediction_results", "vif_features_{}.npy".format(vif_approach_number)),
	)



# Training using Low-Level Features and VIF features
def train_low_level_features_vif_features(
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
	## Training, Validation and Test Datasets
	X_train, y_train = quality_ladder_prediction_dataset_functions.LowLevelFeatures_VIFFeatures_Bitrate_Dataset(
		# Files
		video_filenames=Train_Video_Files,

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

	X_valid, y_valid = quality_ladder_prediction_dataset_functions.LowLevelFeatures_VIFFeatures_Bitrate_Dataset(
		# Files
		video_filenames=Valid_Video_Files,

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

	print ("Dimensions of training dataset = ", X_train.shape, y_train.shape)
	print ("Dimensions of validation dataset = ", X_valid.shape, y_valid.shape)
	print ()


	## Training Models
	"""
	# Extra-Trees Regressor
	print ("Extra-Trees Regressor:")
	ETR_model = ExtraTreesRegressor(n_estimators=1250, random_state=2, max_depth=12)
	ETR_model.fit(X_train, y_train)

	plot_functions.Calculate_Prediction_Performance_Metrics(
		y_pred=ETR_model.predict(X_valid),
		y_true=y_valid,
		resolution_data=X_valid[:,-2:]
	)


	# XG-Boost
	print ("XG-Boost Regressor:")
	XGB_model = XGBRegressor(n_estimators=1250, learning_rate=0.0075, device="cuda", random_state=2, max_depth=12)
	XGB_model.fit(X_train,y_train)

	plot_functions.Calculate_Prediction_Performance_Metrics(
		y_pred=XGB_model.predict(X_valid),
		y_true=y_valid,
		resolution_data=X_valid[:,-2:]
	)


	# Random Forest
	print ("Random-Forest Regressor:")
	RF_model = RandomForestRegressor(n_estimators=1250, random_state=2, max_depth=12)
	RF_model.fit(X_train, y_train)

	plot_functions.Calculate_Prediction_Performance_Metrics(
		y_pred=RF_model.predict(X_valid),
		y_true=y_valid,
		resolution_data=X_valid[:,-2:]
	)
	"""


	## Training best model on best features
	test_model = ExtraTreesRegressor(n_estimators=1250, random_state=2, max_depth=12, n_jobs=-1)
	test_model.fit(X_train, y_train)

	# Saving the weights
	pickle.dump(
		test_model, 
		open(os.path.join(
			results_dir, "models", "low_level_features_vif_features_{}.pkl".format(vif_approach_number)
		), "wb")
	)


	## Performance
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
		plot_save_path=os.path.join(results_dir, "bitrate_prediction_plots", "low_level_features_vif_features_{}.png".format(vif_approach_number)),
		show=False,
		save_results=os.path.join(results_dir, "bitrate_prediction_results", "low_level_features_vif_features_{}.npy".format(vif_approach_number)),
	)



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


	## Training Cross-Over Qualities prediction model
	# Creating Directories
	os.makedirs(os.path.join(results_dir, "CrossOver_Qualities", "models"), exist_ok=True)

	train_crossoverqualities(
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
	print ("-"*10, "Training using Metadata", "-"*10)
	print ()

	train_metadata(
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
	print ("-"*10, "Training using Low-Level Features", "-"*10)
	print ()

	train_low_level_features(
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
	print ("-"*10, "Training using VIF Features", "-"*10)
	print ()

	for vif_approach_number in vif_approach_numbers:
		train_vif_features(
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
	print ("-"*10, "Training using Low-Level Features and VIF Features", "-"*10)
	print ()

	for vif_approach_number in vif_approach_numbers:
		train_low_level_features_vif_features(
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