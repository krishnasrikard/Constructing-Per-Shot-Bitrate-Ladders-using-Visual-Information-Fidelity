import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
from tqdm import tqdm
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.append("/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity")
import defaults

# Parameters
Resolutions = defaults.resolutions
resolutions_strings = ["{}x{}".format(res[0],res[1]) for res in Resolutions]

# -----------------------------------------------------------------------------------------------------------------------------------------------


# Metadata
test = np.load("results/metadata_test.npy")

plt.figure(figsize=(16,12))
plt.title("PLCC on Validation and Test Set using Metadata")
plt.grid()
plt.bar(x=np.arange(len(resolutions_strings)), height=test[:,2], width=0.5)
plt.xticks(ticks=np.arange(len(resolutions_strings)), labels=resolutions_strings,rotation=0)
plt.savefig("plots/bar_plot_plcc_metadata_test.png", pad_inches=0.15, dpi=500, bbox_inches='tight')


# -----------------------------------------------------------------------------------------------------------------------------------------------


# Low-Level Features
test = np.load("results/llf_test.npy")

plt.figure(figsize=(16,12))
plt.title("PLCC on Validation and Test Set using VIF features")
plt.grid()
plt.bar(x=np.arange(len(resolutions_strings)), height=test[:,2], width=0.5)
plt.xticks(ticks=np.arange(len(resolutions_strings)), labels=resolutions_strings,rotation=0)
plt.savefig("plots/bar_plot_plcc_llf_test.png", pad_inches=0.15, dpi=500, bbox_inches='tight')



# -----------------------------------------------------------------------------------------------------------------------------------------------

# VIF features
test = []
for i in range(1,10):
	test.append(np.load("results/viff_approach_{}_test.npy".format(i)))
test = np.asarray(test)

plt.figure(figsize=(16,12))
plt.title("PLCC on Validation and Test Set using VIF features")

sns.heatmap(test[:,:,2], annot=True, annot_kws={'size':30}, vmin=0.45, vmax=0.85, cmap=sns.color_palette("Blues", as_cmap=True))

plt.xticks(ticks=0.5+np.arange(len(resolutions_strings)), labels=resolutions_strings, rotation=0, fontsize=17)
plt.yticks(ticks=0.5+np.arange(9), labels=["VIFF-{}".format(i+1) for i in range(9)], rotation=0, ha='right', fontsize=20)
plt.savefig("plots/heat_map_plcc_viff_test.png", pad_inches=0.15, dpi=500, bbox_inches='tight')

# -----------------------------------------------------------------------------------------------------------------------------------------------

# Low-Level Features + VIF Features
test = []
for i in range(1,10):
	test.append(np.load("results/ensemble_llf_viff_approach_{}_test.npy".format(i)))
test = np.asarray(test)

plt.figure(figsize=(16,12))
plt.title("PLCC on Validation and Test Set using Low-Level & VIF features")

sns.heatmap(test[:,:,2], annot=True, annot_kws={'size':30}, vmin=0.45, vmax=0.85, cmap=sns.color_palette("Blues", as_cmap=True))

plt.xticks(ticks=0.5+np.arange(len(resolutions_strings)), labels=resolutions_strings, rotation=0, fontsize=17)
plt.yticks(ticks=0.5+np.arange(9), labels=["LLF-2_VIFF-{}".format(i+1) for i in range(9)], rotation=0, ha='right', fontsize=20)
plt.savefig("plots/heat_map_plcc_ensemble_llf_viff_test.png", pad_inches=0.15, dpi=500, bbox_inches='tight')
