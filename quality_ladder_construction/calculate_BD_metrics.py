# Importing Libraries
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt

import os, sys, warnings
import pickle
from tqdm import tqdm
warnings.filterwarnings("ignore")
sys.path.append("/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity")
import quality_ladder_construction.QL_functions.quality_ladder_functions as quality_ladder_functions
import defaults


# Cross-Over Qualities
def Calculate_LLF_COQ_BD_Metrics(video_files, codec, preset):
	# Calculating BD-metrics
	Metrics = []
	for video_file in video_files:
		Metrics.append(quality_ladder_functions.Calculate_BD_metrics(
			video_file=video_file,
			codec=codec,
			preset=preset,
			quality_ladder_path=os.path.join(quality_ladders_path, "CrossOver_Qualities", "llf_quality_ladders.npy"),
			fixed_bitrate_ladder_path=fixed_bitrate_ladder_path,
			reference_bitrate_ladder_path=reference_bitrate_ladder_path
		))
	Metrics = np.asarray(Metrics)
	Metrics = Metrics[np.logical_not(np.all(np.isnan(Metrics), axis=1)), :]

	# Saving Reference and Fixed BD metrics
	np.save(os.path.join(bd_metrics_path, "CrossOver_Qualities", "llf_coq_bd_metrics.npy"), Metrics)


# Quality	
def Calculate_Metadata_BD_Metrics(video_files, codec, preset):
	# Calculating BD-metrics
	Metrics = []
	for video_file in video_files:
		Metrics.append(quality_ladder_functions.Calculate_BD_metrics(
			video_file=video_file,
			codec=codec,
			preset=preset,
			quality_ladder_path=os.path.join(quality_ladders_path, "Bitrate", "metadata_quality_ladders.npy"),
			fixed_bitrate_ladder_path=fixed_bitrate_ladder_path,
			reference_bitrate_ladder_path=reference_bitrate_ladder_path
		))
	Metrics = np.asarray(Metrics)
	Metrics = Metrics[np.logical_not(np.all(np.isnan(Metrics), axis=1)), :]

	# Saving Reference and Fixed BD metrics
	np.save(os.path.join(bd_metrics_path, "Bitrate", "metadata_bd_metrics.npy"), Metrics)


def Calculate_LLF_BD_Metrics(video_files, codec, preset):
	# Calculating BD-metrics
	Metrics = []
	for video_file in video_files:
		Metrics.append(quality_ladder_functions.Calculate_BD_metrics(
			video_file=video_file,
			codec=codec,
			preset=preset,
			quality_ladder_path=os.path.join(quality_ladders_path, "Bitrate", "llf_quality_ladders.npy"),
			fixed_bitrate_ladder_path=fixed_bitrate_ladder_path,
			reference_bitrate_ladder_path=reference_bitrate_ladder_path
		))
	Metrics = np.asarray(Metrics)
	Metrics = Metrics[np.logical_not(np.all(np.isnan(Metrics), axis=1)), :]

	# Saving Reference and Fixed BD metrics
	np.save(os.path.join(bd_metrics_path, "Bitrate", "llf_bd_metrics.npy"), Metrics)


def Calculate_ML_VIF_BD_Metrics(video_files, codec, preset):
	for i in range(1,10):
		# Calculating BD-metrics
		Metrics = []
		for video_file in video_files:
			Metrics.append(quality_ladder_functions.Calculate_BD_metrics(
				video_file=video_file,
				codec=codec,
				preset=preset,
				quality_ladder_path=os.path.join(quality_ladders_path, "Bitrate", "vif_features", "approach_{}_quality_ladders.npy".format(i)),
				fixed_bitrate_ladder_path=fixed_bitrate_ladder_path,
				reference_bitrate_ladder_path=reference_bitrate_ladder_path
			))
		Metrics = np.asarray(Metrics)
		Metrics = Metrics[np.logical_not(np.all(np.isnan(Metrics), axis=1)), :]

		# Saving Reference and Fixed BD metrics
		os.makedirs(os.path.join(bd_metrics_path, "Bitrate", "vif_features"), exist_ok=True)
		np.save(os.path.join(bd_metrics_path, "Bitrate", "vif_features", "approach_{}_bd_metrics.npy".format(i)), Metrics)


def Calculate_Ensemble_ML_LLFVIF_BD_Metrics(video_files, codec, preset):
	for i in range(1,10):
		# Calculating BD-metrics
		Metrics = []
		for video_file in video_files:
			Metrics.append(quality_ladder_functions.Calculate_BD_metrics(
				video_file=video_file,
				codec=codec,
				preset=preset,
				quality_ladder_path=os.path.join(quality_ladders_path, "Bitrate", "ensemble_low_level_features_vif_features", "approach_{}_quality_ladders.npy".format(i)),
				fixed_bitrate_ladder_path=fixed_bitrate_ladder_path,
				reference_bitrate_ladder_path=reference_bitrate_ladder_path
			))
		Metrics = np.asarray(Metrics)
		Metrics = Metrics[np.logical_not(np.all(np.isnan(Metrics), axis=1)), :]

		# Saving Reference and Fixed BD metrics
		os.makedirs(os.path.join(bd_metrics_path, "Bitrate", "ensemble_low_level_features_vif_features"), exist_ok=True)
		np.save(os.path.join(bd_metrics_path, "Bitrate", "ensemble_low_level_features_vif_features", "approach_{}_bd_metrics.npy".format(i)), Metrics)


# Arguments
arguments = {
	# Encoder-Settings
	"codec": "libx265",
	"preset": "medium"
}
input_arguments = {
	"video_files": defaults.Test_Video_Titles,
}


# Paths
home_path = "/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/quality_ladder_construction"

reference_bitrate_ladder_path = os.path.join("/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/bitrate_ladder_construction", "bitrate_ladders/standard", "reference_bitrate_ladder.npy")
fixed_bitrate_ladder_path = os.path.join("/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/bitrate_ladder_construction", "bitrate_ladders/standard", "fixed_bitrate_ladder.npy")

quality_ladders_path = os.path.join(home_path, "quality_ladders/ML")
bd_metrics_path = os.path.join(home_path, "bd_metrics/ML")
os.makedirs(bd_metrics_path, exist_ok=True)


# Execute
Calculate_Metadata_BD_Metrics(video_files=input_arguments["video_files"], **arguments)

Calculate_LLF_BD_Metrics(video_files=input_arguments["video_files"], **arguments)

Calculate_ML_VIF_BD_Metrics(video_files=input_arguments["video_files"], **arguments)

Calculate_Ensemble_ML_LLFVIF_BD_Metrics(video_files=input_arguments["video_files"], **arguments)

Calculate_LLF_COQ_BD_Metrics(video_files=input_arguments["video_files"], **arguments)