# Importing Libraries
import numpy as np
np.set_printoptions(suppress=True)

import os, sys, warnings
import pickle
from tqdm import tqdm
warnings.filterwarnings("ignore")
sys.path.append("/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity")
import quality_ladder_construction.QL_functions.quality_ladder_functions as quality_ladder_functions
import defaults


def Construct_LLF_COQ_Quality_Ladder(video_files, codec, preset):
	print ("-"*100)
	print ("Quality Ladder constructed using Cross-Over Qualities")
	print ()

	# Features
	features_set = []
	for features_subset in [defaults.glcm_features, defaults.tc_features, defaults.si_features, defaults.ti_features, defaults.cti_features, defaults.cf_features, defaults.ci_features, defaults.dct_features]:
		for f in features_subset:
			if "max" in f:
				continue
			features_set.append(f)

	# Creating Quality-Ladders for each video-file
	Quality_Ladders = {}
	for video_file in video_files:
		# Models and Features
		X, feature_indices, Models = quality_ladder_functions.CrossOver_Quality_Select_Models_Features(
			video_file=video_file,
			models_path=os.path.join(models_path,"CrossOver_Qualities"),
			design_type="ml_llf",
			features_names=features_set,
			temporal_low_level_features=False,
			codec=codec,
			preset=preset,
		)

		# Constructing Quality-Ladder
		QL = quality_ladder_functions.CrossOverQualities_Quality_Ladder(
			Models=Models,
			X = X,
			feature_indices=feature_indices,
			evaluation_qualities=defaults.evaluation_qualities
		)
	
		Quality_Ladders[video_file] = QL
		print (QL)
		print ()

	# Saving Quality-Ladders
	np.save(os.path.join(quality_ladders_path, "CrossOver_Qualities", "llf_quality_ladders.npy"), Quality_Ladders)


def Construct_Metadata_ML_Quality_Ladder(video_files, codec, preset):
	print ("-"*100)
	print ("Quality Ladder constructed by predicting quality using only metadata")
	print ()

	# Creating Quality-Ladders for each video-file
	Quality_Ladders = {}
	for video_file in video_files:
		# Models and Features
		X, Model = quality_ladder_functions.Bitrate_Select_Models_Features(
			video_file=video_file,
			models_path=os.path.join(models_path,"Bitrate"),
			design_type="ml_metadata",
			features_names=None,
			temporal_low_level_features=False,
			codec=codec,
			preset=preset,
			evaluation_qualities=defaults.evaluation_qualities
		)

		# Constructing Quality-Ladder
		QL = quality_ladder_functions.Bitrate_Quality_Ladder(
			Model=Model,
			X = X,
			evaluation_qualities=defaults.evaluation_qualities
		)
	
		Quality_Ladders[video_file] = QL
		print (QL)
		print ()

	# Saving Quality-Ladders
	np.save(os.path.join(quality_ladders_path, "Bitrate", "metadata_quality_ladders.npy"), Quality_Ladders)


def Construct_LLF_ML_Quality_Ladder(video_files, codec, preset):
	print ("-"*100)
	print ("Quality Ladder constructed by predicting quality using ML and Low-Level features")
	print ()

	# Features
	features_set = []
	for features_subset in [defaults.glcm_features, defaults.tc_features, defaults.si_features, defaults.ti_features, defaults.cti_features, defaults.cf_features, defaults.ci_features, defaults.dct_features, list(defaults.quality_texture_features.keys())]:
		for f in features_subset:
			if "max" in f:
				continue
			features_set.append(f)

	# Creating Quality-Ladders for each video-file
	Quality_Ladders = {}
	for video_file in video_files:
		# Models and Features
		X, Model = quality_ladder_functions.Bitrate_Select_Models_Features(
			video_file=video_file,
			models_path=os.path.join(models_path,"Bitrate"),
			design_type="ml_llf",
			features_names=features_set,
			temporal_low_level_features=False,
			codec=codec,
			preset=preset,
			evaluation_qualities=defaults.evaluation_qualities
		)

		# Constructing Quality-Ladder
		QL = quality_ladder_functions.Bitrate_Quality_Ladder(
			Model=Model,
			X = X,
			evaluation_qualities=defaults.evaluation_qualities
		)
	
		Quality_Ladders[video_file] = QL
		print (QL)
		print ()

	# Saving Quality-Ladders
	np.save(os.path.join(quality_ladders_path, "Bitrate", "llf_quality_ladders.npy"), Quality_Ladders)


def Construct_ML_VIF_Quality_Ladder(video_files, codec, preset):
	print ("-"*100)
	print ("Quality Ladder constructed by predicting quality using ML and VIF-features")
	print ()

	for i in range(1,10):
		print ("-"*50)
		print ("Approach-{}:".format(i))
		print ()

		# Design
		design = "ml_vif_approach" + str(i)

		# Creating Quality-Ladders for each video-file
		Quality_Ladders = {}
		for video_file in video_files:
			# Models and Features
			X, Model = quality_ladder_functions.Bitrate_Select_Models_Features(
				video_file=video_file,
				models_path=os.path.join(models_path,"Bitrate"),
				design_type=design,
				features_names=None,
				temporal_low_level_features=False,
				codec=codec,
				preset=preset,
				evaluation_qualities=defaults.evaluation_qualities
			)

			# Constructing Quality-Ladder
			QL = quality_ladder_functions.Bitrate_Quality_Ladder(
				Model=Model,
				X = X,
				evaluation_qualities=defaults.evaluation_qualities
			)
		
			Quality_Ladders[video_file] = QL
			print (QL)
			print ()

		# Saving Quality-Ladders
		np.save(os.path.join(quality_ladders_path, "Bitrate", "vif_features", "approach_{}_quality_ladders.npy".format(i)), Quality_Ladders)


def Construct_ML_Ensemble_LLFVIF_Quality_Ladder(video_files, codec, preset):
	print ("-"*100)
	print ("Quality Ladder constructed by predicting quality using ML, Low-Level features and VIF-features")
	print ()

	for i in range(1,10):
		print ("-"*50)
		print ("Approach-{}:".format(i))
		print ()

		# Design
		design = "ml_ensemble_llfvif_approach" + str(i)

		# Features
		features_set = []
		for features_subset in [defaults.glcm_features, defaults.tc_features, defaults.si_features, defaults.ti_features, defaults.cti_features, defaults.cf_features, defaults.ci_features, defaults.dct_features, list(defaults.quality_texture_features.keys())]:
			for f in features_subset:
				if "max" in f:
					continue
				features_set.append(f)

		# Creating Quality-Ladders for each video-file
		Quality_Ladders = {}
		for video_file in video_files:
			# Models and Features
			X, Model = quality_ladder_functions.Bitrate_Select_Models_Features(
				video_file=video_file,
				models_path=os.path.join(models_path,"Bitrate"),
				design_type=design,
				features_names=features_set,
				temporal_low_level_features=False,
				codec=codec,
				preset=preset,
				evaluation_qualities=defaults.evaluation_qualities
			)

			# Constructing Quality-Ladder
			QL = quality_ladder_functions.Bitrate_Quality_Ladder(
				Model=Model,
				X = X,
				evaluation_qualities=defaults.evaluation_qualities
			)
		
			Quality_Ladders[video_file] = QL
			print (QL)
			print ()

		# Saving Quality-Ladders
		np.save(os.path.join(quality_ladders_path, "Bitrate", "ensemble_low_level_features_vif_features", "approach_{}_quality_ladders.npy".format(i)), Quality_Ladders)




# Arguments
arguments = {
	# Encoder-Settings
	"codec": "libx265",
	"preset": "medium"
}
input_arguments = {
	"video_files": defaults.Test_Video_Titles
}


# Paths
home_path = "/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/quality_ladder_construction"
models_path = os.path.join(home_path, "models/ML")
quality_ladders_path = os.path.join(home_path, "quality_ladders/ML")
if os.path.exists(quality_ladders_path) == False:
	os.makedirs(quality_ladders_path)


# Execute
Construct_Metadata_ML_Quality_Ladder(video_files=input_arguments["video_files"], **arguments)

Construct_LLF_ML_Quality_Ladder(video_files=input_arguments["video_files"], **arguments)

Construct_ML_VIF_Quality_Ladder(video_files=input_arguments["video_files"], **arguments)

Construct_ML_Ensemble_LLFVIF_Quality_Ladder(video_files=input_arguments["video_files"], **arguments)

Construct_LLF_COQ_Quality_Ladder(video_files=input_arguments["video_files"], **arguments)