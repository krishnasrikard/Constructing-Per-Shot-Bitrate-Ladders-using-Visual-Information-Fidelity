# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os, sys, warnings
import pickle, operator, random
random.seed(1)
from tqdm import tqdm
warnings.filterwarnings("ignore")
sys.path.append("/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity")
import functions.plot_functions as plot_functions
import defaults


# Paths
Pairs = [
	("CrossOver_Qualities/llf_coq_bd_metrics.npy", "llf_coq_bd_histogram.png")
] + [
	("Bitrate/llf_bd_metrics.npy", "llf_quality_bd_histogram.png"),
	("Bitrate/metadata_bd_metrics.npy", "metadata_bd_histogram.png")
] + [
	("Bitrate/vif_features/approach_{}_bd_metrics.npy".format(i), "viff_approach_{}_bd_histogram.png".format(i)) for i in range(1,10)
] + [
	("Bitrate/ensemble_low_level_features_vif_features/approach_{}_bd_metrics.npy".format(i), "llfviff_approach_{}_bd_histogram.png".format(i)) for i in range(1,10)
]

bd_metrics_path = "bd_metrics/ML"
results_path = "bd_histograms"


# Plotting BD-Histograms
for path1, path2 in Pairs:
	plot_functions.Plot_BD_Metrics(
		os.path.join(bd_metrics_path, path1), 
		os.path.join(results_path, path2)
	)


# Plotting Convex-Hulls
	
# Paths
fixed_ladder_path = "/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/bitrate_ladder_construction/bitrate_ladders/standard/fixed_bitrate_ladder.npy"

reference_ladder_path = "/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/bitrate_ladder_construction/bitrate_ladders/standard/reference_bitrate_ladder.npy"

bitrate_ladders_path = "/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/bitrate_ladder_construction/bitrate_ladders/ML/"

quality_ladders_path = "/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/quality_ladder_construction/quality_ladders/ML/"


Test_Files = defaults.Test_Video_Titles

# Note: Make sure labels and Ladders Match. (Mainly Approach Numbers)
for _,video_file in enumerate(Test_Files):
	plot_functions.Plot_Pareto_Front(
		video_file=video_file,
		codec="libx265",
		preset="medium",
		ladder_paths=[
			os.path.join(bitrate_ladders_path, "CrossOver_Bitrates/llf_bitrate_ladders.npy"),
			os.path.join(bitrate_ladders_path, "Quality/ensemble_low_level_features_vif_features/approach_7_bitrate_ladders.npy"),
			os.path.join(quality_ladders_path, "CrossOver_Qualities/llf_quality_ladders.npy"),
			os.path.join(quality_ladders_path, "Bitrate/ensemble_low_level_features_vif_features/approach_9_quality_ladders.npy"),
			fixed_ladder_path,
			reference_ladder_path
		],
		ladder_labels=["LLF-1_CoB", "LLF-2_VIFF-7", "LLF-1_CoQ", "LLF-3_VIFF-9", "Fixed", "Reference"],
		results_path="bd_histograms/pareto_fronts"
	)