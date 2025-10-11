# Importing Libraries
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append("/home/kd28684/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity-Working")
import functions.plot_functions as plot_functions
import functions.IO_functions as IO_functions
import defaults


def print_results(
	performance_path:list
):
	# Filename
	print (os.path.basename(performance_path))
	print ()
	
	# Loading Results
	performance = np.array(list(np.load(performance_path, allow_pickle=True)[()]))

	# Print
	# print (performance)
	print (" & ".join([str(x) for x in performance[:, 2]]))
	print ()
	print ("-"*100)
	print ()
	

def box_plot(
	performance_paths:list,
	plot_title:str,
	metric:str,
	save_path:str
):
	# Index based on metric
	if metric == "MAE":
		index = 0
	elif metric == "STD":
		index = 1
	elif metric == "PLCC":
		index = 2
	elif metric == "SRCC" or metric == "SROCC":
		index = 3
	else:
		raise ValueError("Invalid metric")

	# Loading Results
	performance = []
	for performance_path in performance_paths:
		# Filename
		print (os.path.basename(performance_path))

		performance.append(np.array(list(np.load(performance_path, allow_pickle=True)[()])))
	print ()
	performance = np.array(performance)

	# Plotting
	resolutions_strings = ["{}x{}".format(res[0],res[1]) for res in defaults.resolutions]
	plt.figure(figsize=(16,12))
	plt.title(f"{metric} on Validation and Test Set using {plot_title}")
	sns.heatmap(performance[:,:,index], annot=True, annot_kws={'size':30}, vmin=0.45, vmax=0.85, cmap=sns.color_palette("Blues", as_cmap=True))
	plt.xticks(ticks=0.5+np.arange(len(resolutions_strings)), labels=resolutions_strings, rotation=0, fontsize=17)
	plt.yticks(ticks=0.5+np.arange(9), labels=["VIFF-{}".format(i+1) for i in range(9)], rotation=0, ha='right', fontsize=20)
	plt.savefig(save_path, pad_inches=0.15, dpi=500, bbox_inches='tight')



# Path
results_dir = "../results/main"
i = 0


# Constructing BD-Histograms
for subfolder in ["Bitrate_Ladder_Prediction", "Quality_Ladder_Prediction"]:
	# Logging
	print ()
	print (subfolder)
	print ()
	
	# Performance Resuls Dir
	if subfolder == "Bitrate_Ladder_Prediction":
		subsubfolder = "quality_prediction_results"
	else:
		subsubfolder = "bitrate_prediction_results"

	# Metadata
	performance_path = os.path.join(results_dir, subfolder, subsubfolder, "metadata.npy")
	print_results(performance_path)


	# Low-Level Features
	performance_path = os.path.join(results_dir, subfolder, subsubfolder, "low_level_features.npy")
	print_results(performance_path)


	# VIF Features
	performance_paths = []
	for vif_approach_number in np.arange(1,10):
		performance_paths.append(
			os.path.join(results_dir, subfolder, subsubfolder, "vif_features_{}.npy".format(vif_approach_number))
		)
	box_plot(performance_paths, "VIF Features", "PLCC", os.path.join("plots/regressors_performance", f"{subsubfolder}_vif_features.png"))


	# Low-Level Features
	performance_paths = []
	for vif_approach_number in np.arange(1,10):
		performance_paths.append(
			os.path.join(results_dir, subfolder, subsubfolder, "low_level_features_vif_features_{}.npy".format(vif_approach_number))
		)
	box_plot(performance_paths, "Low-Level & VIF Features", "PLCC", os.path.join("plots/regressors_performance", f"{subsubfolder}_low_level_features_vif_features.png"))