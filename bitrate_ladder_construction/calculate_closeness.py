# Importing Libraries
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import os, sys, warnings
import pickle
from tqdm import tqdm
warnings.filterwarnings("ignore")
sys.path.append("/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity")
import bitrate_ladder_construction.BL_functions.bitrate_ladder_functions as bitrate_ladder_functions
import functions.correction_algorithms as correction_algorithms
import defaults


def Calculate_Closeness(
	bitrate_ladder_path:str
):
	# Video-Files
	Video_Files = defaults.Test_Video_Titles

	# Closeness
	f_25 = []
	f_50 = []
	f_75 = []

	# BD-Metrics
	Reference_BD_Metrics = []
	Predicted_BD_Metrics = []

	for video_file in Video_Files:
		Reference_Metrics = bitrate_ladder_functions.Calculate_BD_metrics(
			video_file=video_file,
			codec="libx265",
			preset="medium",
			bitrate_ladder_path="/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/bitrate_ladder_construction/bitrate_ladders/standard/reference_bitrate_ladder.npy",
			fixed_bitrate_ladder_path="/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/bitrate_ladder_construction/bitrate_ladders/standard/fixed_bitrate_ladder.npy",
			reference_bitrate_ladder_path="/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/bitrate_ladder_construction/bitrate_ladders/standard/reference_bitrate_ladder.npy"
		)

		Predicted_Metrics = bitrate_ladder_functions.Calculate_BD_metrics(
			video_file=video_file,
			codec="libx265",
			preset="medium",
			bitrate_ladder_path=bitrate_ladder_path,
			fixed_bitrate_ladder_path="/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/bitrate_ladder_construction/bitrate_ladders/standard/fixed_bitrate_ladder.npy",
			reference_bitrate_ladder_path="/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/bitrate_ladder_construction/bitrate_ladders/standard/reference_bitrate_ladder.npy"
		)

		# Calculating fraction of samples close to reference bitrate ladder performance
		if Predicted_Metrics[0] < 0.25*Reference_Metrics[0] and Predicted_Metrics[1] > 0.25*Reference_Metrics[1]:
			f_25.append(1)
		else:
			f_25.append(0)

		if Predicted_Metrics[0] < 0.5*Reference_Metrics[0] and Predicted_Metrics[1] > 0.5*Reference_Metrics[1]:
			f_50.append(1)
		else:
			f_50.append(0)

		if Predicted_Metrics[0] < 0.75*Reference_Metrics[0] and Predicted_Metrics[1] > 0.75*Reference_Metrics[1]:
			f_75.append(1)
		else:
			f_75.append(0)

		Reference_BD_Metrics.append(Reference_Metrics)
		Predicted_BD_Metrics.append(Predicted_Metrics)


	Reference_BD_Metrics = np.asarray(Reference_BD_Metrics)
	Predicted_BD_Metrics = np.asarray(Predicted_BD_Metrics)


	# Calculating Mean and Std of BD-metrics
	mean_Rate_Fixed = np.round(np.mean(Predicted_BD_Metrics[:,0]), decimals=3)
	std_Rate_Fixed = np.round(np.std(Predicted_BD_Metrics[:,0]), decimals=3)

	mean_Quality_Fixed = np.round(np.mean(Predicted_BD_Metrics[:,1]), decimals=3)
	std_Quality_Fixed = np.round(np.std(Predicted_BD_Metrics[:,1]), decimals=3)

	mean_Rate_Reference = np.round(np.mean(Predicted_BD_Metrics[:,2]), decimals=3)
	std_Rate_Reference = np.round(np.std(Predicted_BD_Metrics[:,2]), decimals=3)

	mean_Quality_Reference = np.round(np.mean(Predicted_BD_Metrics[:,3]), decimals=3)
	std_Quality_Reference = np.round(np.std(Predicted_BD_Metrics[:,3]), decimals=3)


	return mean_Rate_Fixed, std_Rate_Fixed, mean_Quality_Fixed, std_Quality_Fixed, mean_Rate_Reference, std_Rate_Reference, mean_Quality_Reference, std_Quality_Reference, np.round(np.mean(f_25), decimals=3), np.round(np.mean(f_50), decimals=3), np.round(np.mean(f_75), decimals=3)


Paths = [
	"/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/bitrate_ladder_construction/bitrate_ladders/ML/CrossOver_Bitrates/llf_bitrate_ladders.npy",
] + [
	"/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/bitrate_ladder_construction/bitrate_ladders/ML/Quality/metadata_bitrate_ladders.npy",
	"/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/bitrate_ladder_construction/bitrate_ladders/ML/Quality/llf_bitrate_ladders.npy"
] + [
	"/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/bitrate_ladder_construction/bitrate_ladders/ML/Quality/vif_features/approach_{}_bitrate_ladders.npy".format(i) for i in range(1,10)
] + [
	"/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/bitrate_ladder_construction/bitrate_ladders/ML/Quality/ensemble_low_level_features_vif_features/approach_{}_bitrate_ladders.npy".format(i) for i in range(1,10)
]

for path in Paths:
	print (path)
	print ()
	O = Calculate_Closeness(
		bitrate_ladder_path=path
	)
	print ("${}/{}$ & ${}/{}$ & ${}/{}$ & ${}/{}$ & ${}$ & ${}$ & ${}$".format(O[0],O[1],O[2],O[3],O[4],O[5],O[6],O[7],O[8],O[9],O[10]))
	print ("-"*50)
	print ()
	