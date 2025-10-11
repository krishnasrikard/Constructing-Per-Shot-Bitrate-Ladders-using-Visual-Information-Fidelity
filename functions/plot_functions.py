# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import seaborn as sns

import os, sys, warnings
import pickle, operator
from tqdm import tqdm
warnings.filterwarnings("ignore")
sys.path.append("/home/kd28684/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity-Working")
import functions.extract_functions as extract_functions
import functions.IO_functions as IO_functions
import functions.correction_algorithms as correction_algorithms
import modules.bitrate_quality_ladder_evaluation_functions as bitrate_quality_ladder_evaluation_functions
import defaults


# Evaluation Function
def Calculate_Prediction_Performance_Metrics(
	y_pred:np.array,
	y_true:np.array,
	resolution_data:np.array
):
	"""
	Returns mean of mae, std, plcc and srocc across each resolution.
	Args:
		y_pred (np.array): Predicted quality values per video-file per resolution per crf.
		y_true (np.array): True quality values per video-file per resolution per crf.
		resolution_data (np.array): Resolution Information from last two columns of "X"
	Returns:
		mean (float): Mean of error between true and predicted qualities.
		std (float): Standard deviation of error between true and predicted qualities.
		plcc (np.array): Pearson Correlation Coefficient
		srcc (np.array): Spearman Rank Correlation Coefficient
	"""
	# Transformation
	y_true = y_true.flatten()
	y_pred = y_pred.flatten()

	# Assertions
	assert y_pred.shape == y_true.shape, "Shape of true and predicted quality arrays is not the same"

	# Resolution-Data
	scaled_heights = resolution_data[:,-1]

	# Correlation Coeffients
	mae = np.zeros((len(defaults.resolutions)))
	std = np.zeros((len(defaults.resolutions)))
	plcc = np.zeros((len(defaults.resolutions)))
	srcc = np.zeros((len(defaults.resolutions)))


	# Iterating over each resolution
	for i,resolution in enumerate(defaults.resolutions):
		# Scaled Height
		scaled_h = np.round(resolution[1]/3840, decimals=4)

		# Mask
		mask = [j for j,h in enumerate(scaled_heights) if np.isclose(np.round(h, decimals=4), scaled_h)]

		# Pred and True corresponding to resolution
		pred = y_pred[mask]
		true = y_true[mask]

		# Mean and Standard Deviation
		mae[i] = np.mean(np.abs(true - pred))
		std[i] = np.std(true - pred)

		# Pearson Correlation Coefficient
		r = scipy.stats.pearsonr(true, pred)[0]
		try:
			plcc[i] = r
		except:
			plcc[i] = r[0]

		# Spearman Rank Correlation Coefficient
		r = scipy.stats.spearmanr(true, pred)[0]
		try:
			srcc[i] = r
		except:
			srcc[i] = r[0]


	print ("MAE =", np.round(mae, decimals=2))
	print ("PLCC =", np.round(plcc, decimals=2))
	print ("SRCC =", np.round(srcc, decimals=2))
	print ("\n")



# Plotting Predictions per Resolutions Function
def Plot_Predictions(
	y_pred_Results:np.array,
	y_Results:np.array,
	Resolutions:np.array,
	plot_save_path:str,
	show:bool=False,
	save_results:str=None
):
	if save_results is not None:
		R = []

	plt.figure(figsize=(16,12))
	for i in range(len(Resolutions)):
		mask = np.nonzero(np.where((y_pred_Results[:,i:i+1,:].flatten() != -np.inf), 1, 0))
		y_pred = y_pred_Results[:,i:i+1,:].flatten()[mask]

		mask = np.nonzero(np.where((y_Results[:,i:i+1,:].flatten() != -np.inf), 1, 0))
		y = y_Results[:,i:i+1,:].flatten()[mask]

		mae = np.round(np.mean(np.abs(y_pred - y)), decimals=3)
		std = np.round(np.std(np.abs(y_pred - y)), decimals=3)
		plcc = np.round(scipy.stats.pearsonr(y_pred, y)[0], decimals=3)
		srcc = np.round(scipy.stats.spearmanr(y_pred, y)[0], decimals=3)

		if save_results is not None:
			R.append([mae, std, plcc, srcc])

		plt.subplot(2,3,i+1)
		plt.grid()
		plt.title("{}, PLCC = {}, SRCC = {}\nMAE = {}, STD = {}".format(Resolutions[i], plcc, srcc, mae, std))
		plt.xlabel("y_true")
		plt.ylabel("y_pred")
		plt.scatter(y, y_pred)
		plt.plot([np.min(y), np.max(y)], [np.min(y), np.max(y)])

	if plot_save_path is None:
		print ("Plot save path is None. Hence, not saving the plot")
	else:
		plt.savefig(plot_save_path, dpi=400, bbox_inches='tight')

	if show:
		plt.show()
	else:
		plt.close()


	if save_results is not None:
		np.save(save_results, np.asarray(R))



#  Plotting BD-Histograms
def Plot_BD_Metrics(
	bd_metrics_paths:list,
	save_path:str,
):
	# Plotting
	plt.figure(figsize=(12,8))

	# Histogram plot of BD-metrics
	bitrate_bins = [-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60]
	quality_bins = [-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7]

	# BD-Metrics
	BD_Metrics = []

	for path in bd_metrics_paths:
		# Calculating BD-metrics
		Metrics = np.load(path, allow_pickle=True)[()]
		Metrics = np.asarray(list(Metrics.values()))
		N = len(Metrics)

		# Replacing NaNs and Infinities
		Metrics = Metrics[np.logical_not(np.all(np.isnan(Metrics), axis=1)), :]

		# Assertions
		assert len(Metrics) > 0.95*N, "More than 10% of BD-Metrics have NaNs"

		# Data
		Metrics = np.clip(Metrics, -60, 60)
		BD_Metrics.append(Metrics)

	
	# Calculating Mean and Standard Deviation
	Mean = []
	Std = []
	for i in range(4):
		single_bd_metric = []
		for j in range(len(bd_metrics_paths)):
			single_bd_metric.append(BD_Metrics[j][:,i])
		
		single_bd_metric = np.concatenate(single_bd_metric, axis=0)

		Mean.append(np.round(np.mean(single_bd_metric), decimals=3))
		Std.append(np.round(np.std(single_bd_metric), decimals=3))
		

	# Plotting
	plt.subplot(2,2,1)
	plt.title(r"BD-Rate wrt AL ($\mu$={}, $\sigma$={})".format(Mean[0], Std[0]))
	plt.grid()
	for i in range(len(bd_metrics_paths)):
		sns.histplot(data=BD_Metrics[i][:,0], bins=bitrate_bins, kde=True, element="step")

	plt.subplot(2,2,2)
	plt.title(r"BD-VMAF wrt AL ($\mu$={}, $\sigma$={})".format(Mean[1], Std[1]))
	plt.grid()
	for i in range(len(bd_metrics_paths)):
		sns.histplot(data=BD_Metrics[i][:,1], bins=quality_bins, kde=True, element="step")

	plt.subplot(2,2,3)
	plt.title(r"BD-Rate wrt RL ($\mu$={}, $\sigma$={})".format(Mean[2], Std[2]))
	plt.grid()
	for i in range(len(bd_metrics_paths)):
		sns.histplot(data=BD_Metrics[i][:,2], bins=bitrate_bins, kde=True, element="step")

	plt.subplot(2,2,4)
	plt.title(r"BD-VMAF wrt RL ($\mu$={}, $\sigma$={})".format(Mean[3], Std[3]))
	plt.grid()
	for i in range(len(bd_metrics_paths)):
		sns.histplot(data=BD_Metrics[i][:,3], bins=quality_bins, kde=True, element="step")

	plt.savefig(save_path, dpi=400, bbox_inches='tight')



# Plotting Convex-Hull of Ladders
def Plot_Predicted_RQ_Curve(
	video_file:str,
	ladder_paths:list,
	ladder_labels:list,
	save_path:str
):
	"""
	Args:
		video_file (str): The video file name.
		ladder_path (list): The path to Ladders that needs to be considered.
		ladder_labels (list): List of labels that describe each ladder to consider.
		save_path (str): Path to save results.
	"""
	# Rate-Quality points
	RQ_pairs = extract_functions.Extract_RQ_Information(
		video_rq_points_info=IO_functions.read_create_jsonfile(os.path.join(defaults.rq_points_dataset_path, "libx265", "medium", video_file, "crfs.json")),
		quality_metric="vmaf",
		resolutions=defaults.resolutions,
		CRFs=defaults.CRFs,
		QPs=None,
		min_quality=defaults.min_quality,
		max_quality=defaults.max_quality,
		min_bitrate=defaults.min_bitrate,
		max_bitrate=defaults.max_bitrate,
		set_bitrate_log_base=2
	)

	# Plot Settings
	resolutions_strings = ["{}x{}".format(res[0],res[1]) for res in defaults.resolutions]
	Resolution_Color_Map = dict(zip(defaults.resolutions, ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']))
	linestyles = ["dotted", "dotted", "dashed", "dashed", "solid", "solid"]

	# Plotting Rate-Quality_Curves
	plt.figure(figsize=(10,8))
	plt.grid()
	plt.xlabel(r"$\log_{2}(Bitrate)$")
	plt.ylabel("VMAF")
	plt.title("Rate-Quality Curves for " + video_file.split("_")[0])


	# Predicted Rate-Quality Curve from Bitrate :adder
	for i,path in enumerate(ladder_paths):
		if "fixed_bitrate_ladder" in path:
			BL = np.load(path, allow_pickle=True)[()]
			BL = correction_algorithms.Top_Bottom(BL)

			# Constructing Rate-Quality_Curve
			RQ_curve_pairs, RQ_Points = bitrate_quality_ladder_evaluation_functions.Rate_Quality_Curve_from_Bitrate_Ladder(
				RQ_pairs=RQ_pairs,
				Bitrate_Ladder=BL
			)
		elif "standard" in path:
			BL = np.load(path, allow_pickle=True)[()][video_file]
			BL = correction_algorithms.Top_Bottom(BL)

			# Constructing Rate-Quality_Curve
			RQ_curve_pairs, RQ_Points = bitrate_quality_ladder_evaluation_functions.Rate_Quality_Curve_from_Bitrate_Ladder(
				RQ_pairs=RQ_pairs,
				Bitrate_Ladder=BL
			)
		elif "bitrate_ladders" in path:
			BL = np.load(path, allow_pickle=True)[()][video_file]
			BL = correction_algorithms.Top_Bottom(BL)

			# Constructing Rate-Quality_Curve
			RQ_curve_pairs, RQ_Points = bitrate_quality_ladder_evaluation_functions.Rate_Quality_Curve_from_Bitrate_Ladder(
				RQ_pairs=RQ_pairs,
				Bitrate_Ladder=BL
			)
		else:
			QL = np.load(path, allow_pickle=True)[()][video_file]
			QL = correction_algorithms.Bottom_Top(QL)

			# Constructing Rate-Quality_Curve
			RQ_curve_pairs, RQ_Points = bitrate_quality_ladder_evaluation_functions.Rate_Quality_Curve_from_Quality_Ladder(
				RQ_pairs=RQ_pairs,
				Quality_Ladder=QL
			)


		# Plotting Rate-Quality_Curve
		plt.plot(RQ_Points[:,0], RQ_Points[:,1], linestyle=linestyles[i] , label=ladder_labels[i], linewidth=3)


		# Scatter Plot for each Resolution
		for res in defaults.resolutions:
			data = RQ_curve_pairs[res]
			res_string = "{}x{}".format(res[0],res[1])
			if data.shape[0] > 0:
				plt.scatter(data[:,0], data[:,1], color=Resolution_Color_Map[res], label=res_string, s=75, marker="o")				
		

	# Legend
	handles, labels = plt.gca().get_legend_handles_labels()
	Handles_Labels_Dict = {}
	
	for res_string in resolutions_strings:
		Handles_Labels_Dict[res_string] = None

	for i,label in enumerate(labels):
		if label in resolutions_strings:
			if isinstance(handles[i], tuple):
				Handles_Labels_Dict[label] = (handles[i][0])
			else:
				Handles_Labels_Dict[label] = handles[i]
		else:
			Handles_Labels_Dict[label] = handles[i]

	Handles_Labels = {k: v for k, v in Handles_Labels_Dict.items() if v is not None}
	labels, handles = tuple(Handles_Labels.keys()), tuple(Handles_Labels.values())
	plt.legend(handles, labels)

	# Save fig
	plt.savefig(save_path, dpi=500, bbox_inches='tight')