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
	("CrossOver_Bitrates/llf_cob_bd_metrics.npy", "llf_cob_bd_histogram.png")
] + [
	("Quality/llf_bd_metrics.npy", "llf_quality_bd_histogram.png"),
	("Quality/metadata_bd_metrics.npy", "metadata_bd_histogram.png")
] + [
	("Quality/vif_features/approach_{}_bd_metrics.npy".format(i), "viff_approach_{}_bd_histogram.png".format(i)) for i in range(1,10)
] + [
	("Quality/ensemble_low_level_features_vif_features/approach_{}_bd_metrics.npy".format(i), "llfviff_approach_{}_bd_histogram.png".format(i)) for i in range(1,10)
]

bd_metrics_path = "bd_metrics/ML"
results_path = "bd_histograms"


# Plotting BD-Histograms
for path1, path2 in Pairs:
	plot_functions.Plot_BD_Metrics(
		os.path.join(bd_metrics_path, path1), 
		os.path.join(results_path, path2)
	)