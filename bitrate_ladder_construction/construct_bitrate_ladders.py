# Importing Libraries
import numpy as np
np.set_printoptions(suppress=True)

import os, sys, warnings
import pickle
from tqdm import tqdm
warnings.filterwarnings("ignore")
sys.path.append("/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity")
import bitrate_ladder_construction.BL_functions.bitrate_ladder_functions as bitrate_ladder_functions
import defaults


def Construct_LLF_COB_Bitrate_Ladder(video_files, codec, preset):
	print ("-"*100)
	print ("Bitrate Ladder constructed using Cross-Over Bitrates")
	print ()

	# Features
	features_set = []
	for features_subset in [defaults.glcm_features, defaults.tc_features, defaults.si_features, defaults.ti_features, defaults.cti_features, defaults.cf_features, defaults.ci_features, defaults.dct_features]:
		for f in features_subset:
			if "max" in f:
				continue
			features_set.append(f)

	# Creating Bitrate-Ladders for each video-file
	Bitrate_Ladders = {}
	for video_file in video_files:
		# Models and Features
		X, feature_indices, Models = bitrate_ladder_functions.CrossOver_Bitrate_Select_Models_Features(
			video_file=video_file,
			models_path=os.path.join(models_path,"CrossOver_Bitrates"),
			design_type="ml_llf",
			features_names=features_set,
			temporal_low_level_features=False,
			codec=codec,
			preset=preset,
		)

		# Constructing Bitrate-Ladder
		BL = bitrate_ladder_functions.CrossOverBitrates_Bitrate_Ladder(
			Models=Models,
			X = X,
			feature_indices=feature_indices,
			evaluation_bitrates=defaults.evaluation_bitrates
		)
	
		Bitrate_Ladders[video_file] = BL
		print (BL)
		print ()

	# Saving Bitrate-Ladders
	np.save(os.path.join(bitrate_ladders_path, "CrossOver_Bitrates", "llf_bitrate_ladders.npy"), Bitrate_Ladders)


def Construct_Metadata_ML_Bitrate_Ladder(video_files, codec, preset):
	print ("-"*100)
	print ("Bitrate Ladder constructed by predicting quality using only metadata")
	print ()

	# Creating Bitrate-Ladders for each video-file
	Bitrate_Ladders = {}
	for video_file in video_files:
		# Models and Features
		X, Model = bitrate_ladder_functions.Quality_Select_Models_Features(
			video_file=video_file,
			models_path=os.path.join(models_path,"Quality"),
			design_type="ml_metadata",
			features_names=None,
			temporal_low_level_features=False,
			codec=codec,
			preset=preset,
			evaluation_bitrates=defaults.evaluation_bitrates	
		)

		# Constructing Bitrate-Ladder
		BL = bitrate_ladder_functions.Quality_Bitrate_Ladder(
			Model=Model,
			X = X,
			evaluation_bitrates=defaults.evaluation_bitrates
		)
	
		Bitrate_Ladders[video_file] = BL
		print (BL)
		print ()

	# Saving Bitrate-Ladders
	np.save(os.path.join(bitrate_ladders_path, "Quality", "metadata_bitrate_ladders.npy"), Bitrate_Ladders)


def Construct_LLF_ML_Bitrate_Ladder(video_files, codec, preset):
	print ("-"*100)
	print ("Bitrate Ladder constructed by predicting quality using ML and Low-Level features")
	print ()

	# Features
	features_set = []
	for features_subset in [defaults.glcm_features, defaults.tc_features, defaults.si_features, defaults.ti_features, defaults.cti_features, defaults.cf_features, defaults.ci_features, defaults.dct_features, list(defaults.bitrate_texture_features.keys())]:
		for f in features_subset:
			if "max" in f:
				continue
			features_set.append(f)

	# Creating Bitrate-Ladders for each video-file
	Bitrate_Ladders = {}
	for video_file in video_files:
		# Models and Features
		X, Model = bitrate_ladder_functions.Quality_Select_Models_Features(
			video_file=video_file,
			models_path=os.path.join(models_path,"Quality"),
			design_type="ml_llf",
			features_names=features_set,
			temporal_low_level_features=False,
			codec=codec,
			preset=preset,
			evaluation_bitrates=defaults.evaluation_bitrates
		)

		# Constructing Bitrate-Ladder
		BL = bitrate_ladder_functions.Quality_Bitrate_Ladder(
			Model=Model,
			X = X,
			evaluation_bitrates=defaults.evaluation_bitrates
		)
	
		Bitrate_Ladders[video_file] = BL
		print (BL)
		print ()

	# Saving Bitrate-Ladders
	np.save(os.path.join(bitrate_ladders_path, "Quality", "llf_bitrate_ladders.npy"), Bitrate_Ladders)


def Construct_ML_VIF_Bitrate_Ladder(video_files, codec, preset):
	print ("-"*100)
	print ("Bitrate Ladder constructed by predicting quality using ML and VIF-features")
	print ()

	for i in range(1,10):
		print ("-"*50)
		print ("Approach-{}:".format(i))
		print ()

		# Design
		design = "ml_vif_approach" + str(i)

		# Creating Bitrate-Ladders for each video-file
		Bitrate_Ladders = {}
		for video_file in video_files:
			# Models and Features
			X, Model = bitrate_ladder_functions.Quality_Select_Models_Features(
				video_file=video_file,
				models_path=os.path.join(models_path,"Quality"),
				design_type=design,
				features_names=None,
				temporal_low_level_features=False,
				codec=codec,
				preset=preset,
				evaluation_bitrates=defaults.evaluation_bitrates
			)

			# Constructing Bitrate-Ladder
			BL = bitrate_ladder_functions.Quality_Bitrate_Ladder(
				Model=Model,
				X = X,
				evaluation_bitrates=defaults.evaluation_bitrates
			)
		
			Bitrate_Ladders[video_file] = BL
			print (BL)
			print ()

		# Saving Bitrate-Ladders
		np.save(os.path.join(bitrate_ladders_path, "Quality", "vif_features", "approach_{}_bitrate_ladders.npy".format(i)), Bitrate_Ladders)


def Construct_ML_Ensemble_LLFVIF_Bitrate_Ladder(video_files, codec, preset):
	print ("-"*100)
	print ("Bitrate Ladder constructed by predicting quality using ML, Low-Level features and VIF-features")
	print ()

	for i in range(1,10):
		print ("-"*50)
		print ("Approach-{}:".format(i))
		print ()

		# Design
		design = "ml_ensemble_llfvif_approach" + str(i)

		# Features
		features_set = []
		for features_subset in [defaults.glcm_features, defaults.tc_features, defaults.si_features, defaults.ti_features, defaults.cti_features, defaults.cf_features, defaults.ci_features, defaults.dct_features, list(defaults.bitrate_texture_features.keys())]:
			for f in features_subset:
				if "max" in f:
					continue
				features_set.append(f)

		# Creating Bitrate-Ladders for each video-file
		Bitrate_Ladders = {}
		for video_file in video_files:
			# Models and Features
			X, Model = bitrate_ladder_functions.Quality_Select_Models_Features(
				video_file=video_file,
				models_path=os.path.join(models_path,"Quality"),
				design_type=design,
				features_names=features_set,
				temporal_low_level_features=False,
				codec=codec,
				preset=preset,
				evaluation_bitrates=defaults.evaluation_bitrates
			)

			# Constructing Bitrate-Ladder
			BL = bitrate_ladder_functions.Quality_Bitrate_Ladder(
				Model=Model,
				X = X,
				evaluation_bitrates=defaults.evaluation_bitrates
			)
		
			Bitrate_Ladders[video_file] = BL
			print (BL)
			print ()

		# Saving Bitrate-Ladders
		np.save(os.path.join(bitrate_ladders_path, "Quality", "ensemble_low_level_features_vif_features", "approach_{}_bitrate_ladders.npy".format(i)), Bitrate_Ladders)




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
home_path = "/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/bitrate_ladder_construction"
models_path = os.path.join(home_path, "models/ML")
bitrate_ladders_path = os.path.join(home_path, "bitrate_ladders/ML")
if os.path.exists(bitrate_ladders_path) == False:
	os.makedirs(bitrate_ladders_path)


# Execute
Construct_Metadata_ML_Bitrate_Ladder(video_files=input_arguments["video_files"], **arguments)

Construct_LLF_ML_Bitrate_Ladder(video_files=input_arguments["video_files"], **arguments)

Construct_ML_VIF_Bitrate_Ladder(video_files=input_arguments["video_files"], **arguments)

Construct_ML_Ensemble_LLFVIF_Bitrate_Ladder(video_files=input_arguments["video_files"], **arguments)

Construct_LLF_COB_Bitrate_Ladder(video_files=input_arguments["video_files"], **arguments)