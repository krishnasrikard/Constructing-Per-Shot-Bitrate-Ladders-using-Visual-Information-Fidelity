"""
Plotting BD-Metrics
"""
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

# Path
plots_dir = "plots/bd_histograms"
results_dir = "../results/main"


# Constructing BD-Histograms
for subfolder in ["Bitrate_Ladder_Prediction", "Quality_Ladder_Prediction"]:
	# Creating Directories
	os.makedirs(os.path.join(plots_dir, subfolder), exist_ok=True)
			

	# Metadata
	bd_metrics_paths = []
	for i in range(1):
		bd_metrics_paths.append(
			os.path.join(
				results_dir, subfolder, "bd_metrics", "metadata.npy"
			)
		)

	plot_functions.Plot_BD_Metrics(
		bd_metrics_paths=bd_metrics_paths,
		save_path=os.path.join(plots_dir, subfolder, "metadata.png"),
	)


	# Low-Level Features
	bd_metrics_paths = []
	for i in range(1):
		bd_metrics_paths.append(
			os.path.join(
				results_dir, subfolder, "bd_metrics", "low_level_features.npy"
			)
		)

	plot_functions.Plot_BD_Metrics(
		bd_metrics_paths=bd_metrics_paths,
		save_path=os.path.join(plots_dir, subfolder, "low_level_features.png"),
	)


	# VIF Features
	for vif_approach_number in np.arange(1,10):
		bd_metrics_paths = []
		for i in range(1):
			bd_metrics_paths.append(
				os.path.join(
					results_dir, subfolder, "bd_metrics", "vif_features_{}.npy".format(vif_approach_number)
				)
			)

		plot_functions.Plot_BD_Metrics(
			bd_metrics_paths=bd_metrics_paths,
			save_path=os.path.join(plots_dir, subfolder, "vif_features_{}.png".format(vif_approach_number))
		)


	# Low-Level Features
	for vif_approach_number in np.arange(1,10):
		bd_metrics_paths = []
		for i in range(1):
			bd_metrics_paths.append(
				os.path.join(
					results_dir, subfolder, "bd_metrics", "low_level_features_vif_features_{}.npy".format(vif_approach_number)
				)
			)

		plot_functions.Plot_BD_Metrics(
			bd_metrics_paths=bd_metrics_paths,
			save_path=os.path.join(plots_dir, subfolder, "low_level_features_vif_features_{}.png".format(vif_approach_number))
		)