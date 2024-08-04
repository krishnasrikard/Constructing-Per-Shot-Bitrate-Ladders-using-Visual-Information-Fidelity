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


def Evaluate_Metadata_ML_Monotonicity(video_files, codec, preset):
	# Evaluating model used for predicting quality trained using only metadata
	print ("-"*100)
	print ("Evaluating model used for predicting quality trained using only metadata")
	print ()

	# Features
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

		# Evaluate Monotonicity
		state = bitrate_ladder_functions.Quality_Monotonicity(
			Model=Model,
			X = X,
			evaluation_bitrates=defaults.evaluation_bitrates
		)

		if state:
			None
		else:
			print ()
			print ("Fails monotonicity check for", video_file)
			print ()


def Evaluate_LLF_ML_Monotonicity(video_files, codec, preset):
	# Evaluating model used for predicting quality trained using ML and Low-Level features
	print ("-"*100)
	print ("Evaluating model used for predicting quality trained using ML and Low-Level features")
	print ()

	# Features
	features_set = []
	for features_subset in [defaults.glcm_features, defaults.tc_features, defaults.si_features, defaults.ti_features, defaults.cti_features, defaults.cf_features, defaults.ci_features, defaults.dct_features, list(defaults.bitrate_texture_features.keys())]:
		for f in features_subset:
			if "max" in f:
				continue
			features_set.append(f)

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

		# Evaluate Monotonicity
		state = bitrate_ladder_functions.Quality_Monotonicity(
			Model=Model,
			X = X,
			evaluation_bitrates=defaults.evaluation_bitrates
		)

		if state:
			None
		else:
			print ()
			print ("Fails monotonicity check for", video_file)
			print ()


def Evaluate_ML_VIF_Monotonicity(video_files, codec, preset):
	# Evaluating model used for predicting quality trained using ML and VIF-features
	print ("-"*100)
	print ("Evaluating model used for predicting quality trained using ML and VIF-features")
	print ()

	# For different approaches
	for i in range(1,10):
		print ("-"*50)
		print ("Approach-{}:".format(i))
		print ()

		# Design
		design = "ml_vif_approach" + str(i)

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

			# Evaluate Monotonicity
			state = bitrate_ladder_functions.Quality_Monotonicity(
				Model=Model,
				X = X,
				evaluation_bitrates=defaults.evaluation_bitrates
			)

			if state:
				None
			else:
				print ()
				print ("Fails monotonicity check for", video_file)
				print ()


def Evaluate_ML_Ensemble_LLFVIF_Monotonicity(video_files, codec, preset):
	# Evaluating model used for predicting quality trained using ML, Low-Level features and VIF-features
	print ("-"*100)
	print ("Evaluating model used for predicting quality trained using ML, Low-Level features and VIF-features")
	print ()

	# For different approaches
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

			# Evaluate Monotonicity
			state = bitrate_ladder_functions.Quality_Monotonicity(
				Model=Model,
				X = X,
				evaluation_bitrates=defaults.evaluation_bitrates
			)

			if state:
				None
			else:
				print ()
				print ("Fails monotonicity check for", video_file)
				print ()


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

# Execute
Evaluate_Metadata_ML_Monotonicity(video_files=input_arguments["video_files"], **arguments)

Evaluate_LLF_ML_Monotonicity(video_files=input_arguments["video_files"], **arguments)

Evaluate_ML_VIF_Monotonicity(video_files=input_arguments["video_files"], **arguments)

Evaluate_ML_Ensemble_LLFVIF_Monotonicity(video_files=input_arguments["video_files"], **arguments)