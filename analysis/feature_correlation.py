"""
Feature Analysis Functions
"""
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import StandardScaler
from scipy.stats.stats import pearsonr 

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append("/home/kd28684/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity-Working")
import functions.extract_features as extract_features
from tqdm import tqdm
import defaults

def get_low_level_features(
	feature_names:list,
	video_filenames:list,
):
	# Settings
	temporal_low_level_features=False

	# Low-Level features
	F = extract_features.Extract_Low_Level_Features(
		features_names=feature_names,
		video_filenames=video_filenames,
		temporal_low_level_features=temporal_low_level_features
	)
	F =  np.array(list(F.values()))

	return F


# Self-Correlation
def self_correlation(
	X:np.array
):
	"""
	Args:
		X (np.array): (num_samples, n_features)
	Returns:
		correlation_matrix (np.array)
	"""
	correlation_matrix = np.corrcoef(X.T)

	return correlation_matrix


def get_vif_features(
	vif_approach_number:str,
	video_filenames:list
):
	# Settings
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

	# VIF features
	F = extract_features.Extract_VIF_Features(
		video_filenames=video_filenames,
		vif_setting=VIF_Approach_Map[vif_approach_number][1],
		vif_features_list=VIF_Approach_Map[vif_approach_number][0],
		per_frame=per_frame,
		per_frame_features_flatten=per_frame_features_flatten
	)
	F =  np.array(list(F.values()))[:,0,:]

	return F



def get_low_level_features_vif_features(
	feature_names:list,
	vif_approach_number:str,
	video_filenames:list
):
	# Low-Level features
	LLF = get_low_level_features(
		feature_names=feature_names,
		video_filenames=video_filenames
	)
	
	# VIF features
	VIFF = get_vif_features(
		vif_approach_number=vif_approach_number,
		video_filenames=video_filenames
	)

	# Features
	F = np.concatenate([LLF, VIFF], axis=1)

	return F


def study_self_correlation(
	X:np.array,
	save_filename_prefix:str,
	results_dir:str,
):
	"""
	Args:
		X (np.array): Input Features.
		save_filename_prefix (str): Prefix of filename saved.
		results_dir (str): Path to directory to save results.
		y (np.array): Optional Features for comparisons.
		additional_prefix (str): String appended to prefix while cross analysis.
	"""
	# Self-Correlation
	cm = self_correlation(X)
	cm = np.abs(cm)
	img = Image.fromarray(np.uint8(255.0 * cm))
	img.save(os.path.join(
		results_dir, save_filename_prefix + ".png"	
	))

	cm = np.where(cm >= 0.5, cm, 0)
	img = Image.fromarray(np.uint8(255.0 * cm))
	img.save(os.path.join(
		results_dir, save_filename_prefix + "_semi_binary" + ".png"	
	))


def study_cross_correlation(
	X1:np.array,
	X2:np.array,
	save_filename_prefix:str,
	results_dir:str,
):
	"""
	Args:
		X1 (np.array): Input Features.
		X2 (np.array): Input Features.
		save_filename_prefix (str): Prefix of filename saved.
		results_dir (str): Path to directory to save results.
	"""

	# Cross-Correlation
	cross_correlation_values = []
	for i in range(X1.shape[1]):
		cross_correlation_values.append(
			np.round(pearsonr(X1[:,i], X2[:,i])[0], decimals=4)
		)

	plt.figure()
	plt.grid()
	plt.boxplot(cross_correlation_values, patch_artist=True, meanline=True, showmeans=True)
	plt.savefig(os.path.join(
		results_dir, save_filename_prefix + ".png"	
	), dpi=500, bbox_inches='tight')


def main(
	# Files
	Video_Files:list,

	# Path
	results_dir:str
):
	# Features
	LLF_subset_names = {}

	# Low-Level Features
	LLF_subset_names["LLF"] = []
	for features_subset in [defaults.glcm_features, defaults.tc_features, defaults.si_features, defaults.ti_features, defaults.cti_features, defaults.cf_features, defaults.ci_features, defaults.dct_features]:
		for f in features_subset:
			LLF_subset_names["LLF"].append(f)

	LLF_subset_names["GLCM" ] = defaults.glcm_features
	LLF_subset_names["TC"] = defaults.tc_features
	LLF_subset_names["SI"] = defaults.si_features
	LLF_subset_names["TI"] = defaults.ti_features
	LLF_subset_names["CTI"] = defaults.cti_features
	LLF_subset_names["CF"] = defaults.cf_features
	LLF_subset_names["CI"] = defaults.ci_features
	LLF_subset_names["DCT"] = defaults.dct_features


	# Features Directory
	Features = {}
	Features["LLF"] = get_low_level_features(
		feature_names=LLF_subset_names["LLF"],
		video_filenames=Video_Files
	)

	for vif_approach_number in range(1,10):
		Features["VIFF_{}".format(vif_approach_number)] = get_vif_features(
			vif_approach_number=str(vif_approach_number),
			video_filenames=Video_Files
		)

	Features["LLF_VIFF_9"] = get_low_level_features_vif_features(
		feature_names=LLF_subset_names["LLF"],
		vif_approach_number="9",
		video_filenames=Video_Files
	)


	# Self-Correlation
	# """
	for subset_name in ["LLF", "VIFF_9", "LLF_VIFF_9"]:
		study_self_correlation(
			X = Features[subset_name],
			save_filename_prefix=subset_name,
			results_dir=results_dir
		)
	# """


if __name__ == "__main__":
	# Settings
	Video_Files = defaults.Video_Titles
	
	# Path
	results_dir = "plots/correlation"

	# Main
	main(
		Video_Files=Video_Files,
		results_dir=results_dir
	)