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
	bd_metrics_path:str,
	closeness_path:str
):
	# Filename
	print (os.path.basename(bd_metrics_path))
	print ()
	
	# Loading Results
	BD_metrics = np.array(list(np.load(bd_metrics_path, allow_pickle=True)[()].values()))
	closeness = np.array(list(np.load(closeness_path, allow_pickle=True)[()].values()))[:,0:3]

	# Stats of results
	stats_BD_metrics = np.array([
		np.mean(BD_metrics, axis=0), np.std(BD_metrics, axis=0)
	])
	stats_closeness = np.mean(closeness, axis=0)

	# Rounding
	stats_BD_metrics = np.round(stats_BD_metrics, decimals=3)
	stats_closeness = np.round(stats_closeness, decimals=3)

	# Results Line
	results_line = ""
	for i in range(4):
		results_line += "${}/{}$ & ".format(stats_BD_metrics[0,i], stats_BD_metrics[1,i])
	results_line += "${}$ & ${}$ & ${}$".format(stats_closeness[0], stats_closeness[1], stats_closeness[2])

	# Print
	print (results_line)
	print ()
	print ("-"*100)
	print ()


# Path
results_dir = "../results/main"
i = 0


# Constructing BD-Histograms
for subfolder in ["Bitrate_Ladder_Prediction", "Quality_Ladder_Prediction"]:
	# Logging
	print ()
	print (subfolder)
	print ()

	# Cross-Over Points
	if subfolder == "Bitrate_Ladder_Prediction":
		bd_metrics_path = os.path.join(results_dir, "CrossOver_Bitrates", "bd_metrics", "low_level_features.npy")
		closeness_path = os.path.join(results_dir, "CrossOver_Bitrates", "closeness", "low_level_features.npy")
	else:
		bd_metrics_path = os.path.join(results_dir, "CrossOver_Qualities", "bd_metrics", "low_level_features.npy")
		closeness_path = os.path.join(results_dir, "CrossOver_Qualities", "closeness", "low_level_features.npy")
	print_results(bd_metrics_path, closeness_path)


	# Metadata
	bd_metrics_path = os.path.join(results_dir, subfolder, "bd_metrics", "metadata.npy")
	closeness_path = os.path.join(results_dir, subfolder, "closeness", "metadata.npy")
	print_results(bd_metrics_path, closeness_path)


	# Low-Level Features
	for filename in os.listdir(os.path.join(results_dir, subfolder, "bd_metrics")):
		bd_metrics_path = os.path.join(results_dir, subfolder, "bd_metrics", "low_level_features.npy")
		closeness_path = os.path.join(results_dir, subfolder, "closeness", filename)
		print_results(bd_metrics_path, closeness_path)


	# VIF Features
	for vif_approach_number in np.arange(1,10):
		bd_metrics_path = os.path.join(results_dir, subfolder, "bd_metrics", "vif_features_{}.npy".format(vif_approach_number))
		closeness_path = os.path.join(results_dir, subfolder, "closeness", "vif_features_{}.npy".format(vif_approach_number))
		print_results(bd_metrics_path, closeness_path)


	# Low-Level Features and VIF Features
	for vif_approach_number in np.arange(1,10):
		bd_metrics_path = os.path.join(results_dir, subfolder, "bd_metrics", "low_level_features_vif_features_{}.npy".format(vif_approach_number))
		closeness_path = os.path.join(results_dir, subfolder, "closeness", "low_level_features_vif_features_{}.npy".format(vif_approach_number))
		print_results(bd_metrics_path, closeness_path)