"""
Plotting Predicted Rate-Quality Curves
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

# Inputs Settings
results_dir = "../results/main"

# Directory
os.makedirs(os.path.join("plots/predicted_rate_quality_curves"), exist_ok=True)


# Plot
for video_file in defaults.Test_Video_Titles:
    # Logging
    print ()
    print ("Video File: {}".format(video_file))
    print ()

    plot_functions.Plot_Predicted_RQ_Curve(
        video_file=video_file,
        ladder_paths=[
            # Bitrate Ladders
            os.path.join(results_dir, "{}/bitrate_ladders/{}.npy".format("CrossOver_Bitrates", "low_level_features")),
            os.path.join(results_dir, "{}/bitrate_ladders/{}.npy".format("Bitrate_Ladder_Prediction", "low_level_features_vif_features_6")),

            # Quality Ladders
            os.path.join(results_dir, "{}/quality_ladders/{}.npy".format("CrossOver_Qualities", "low_level_features")),
            os.path.join(results_dir, "{}/quality_ladders/{}.npy".format("Quality_Ladder_Prediction", "low_level_features_vif_features_8")),

            # Standard
            os.path.join(results_dir, "Standard/bitrate_ladders/fixed_bitrate_ladder.npy"),
            os.path.join(results_dir, "Standard/bitrate_ladders/{}_{}.npy".format("libx265", "medium")),
        ],
        ladder_labels=[
            # Bitrate Ladders
            "LLF-1_CoB",
            "LLF-2_VIFF-6",
            
            # Quality Ladders
            "LLF-1_CoQ",
            "LLF-3_VIFF-8",

            # Standard
            "Fixed",
            "Reference"
        ],
        save_path="plots/predicted_rate_quality_curves/{}.png".format(video_file)
    )