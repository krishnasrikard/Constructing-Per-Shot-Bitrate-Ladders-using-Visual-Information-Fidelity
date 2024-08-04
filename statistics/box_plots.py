import numpy as np
import matplotlib.pyplot as plt

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append("/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity")
import functions.extract_features as extract_features
import defaults


# Parameters
Resolutions = [(3840,2160),(2560,1440),(1920,1080),(1280,720),(960,540),(768,432),(640,360),(512,288)]
CRFs = list(np.arange(15,46,1))


def Box_Plot(
	Meta_Information:dict,
	x_axis:str,
	y_axis:str,
	x_label:str,
	y_label:str,
	save_path:str
):
	"""
	Box-Plot of compressed videos metadata for different resolutions or CRFs
	Args:
		Meta_Information (dict): Dictionary containing Meta-Information of various
		x_axis (str): Options: ["resolutions", "crfs"].
		y_axis (str): Options: ["bitrate", "quality"].
		save_path (str): Path to save plot.
	"""
	# Assertions
	assert x_axis in ["resolutions", "crfs"], "Invalid x-axis"
	assert y_axis in ["bitrate", "quality"], "Invalid y-axis"

	# Index
	if y_axis == "bitrate":
		index = 0
	else:
		index = 1

	# Box-Plot Data
	Data = []

	# Labels
	if x_axis == "resolutions":
		labels = Resolutions

		for _,res in enumerate(Resolutions):
			scaled_h = np.round(res[1]/3840, decimals=4)
			
			data_per_resolution = []
			for _,video_file in enumerate(Meta_Information.keys()):
				mask = [i for i,h in enumerate(Meta_Information[video_file][:,-1]) if np.isclose(np.round(h, decimals=4), scaled_h)]

				data_per_resolution.append(Meta_Information[video_file][mask][:,index])

			data_per_resolution = np.concatenate(data_per_resolution, axis=0)
			Data.append(data_per_resolution)

	else:
		labels = CRFs

		for _,CRF in enumerate(CRFs):
			data_per_crf = []
			for _,video_file in enumerate(Meta_Information.keys()):
				mask = [i for i,crf in enumerate(Meta_Information[video_file][:,-3]) if crf == CRF]
				data_per_crf.append(Meta_Information[video_file][mask][:,index])

			data_per_crf = np.concatenate(data_per_crf, axis=0)
			Data.append(data_per_crf)


	# Box-Plot
	plt.figure(figsize=(10,8))
	plt.grid()
	plt.title("Box-plot of {} vs {}".format(y_label, x_label))
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.boxplot(Data, labels=labels, patch_artist=True, meanline=True, showmeans=True)
	plt.plot(1+np.arange(len(labels)), np.asarray([np.mean(Data[i]) for i in range(len(Data))]), color="green", linewidth=2, label="Mean")
	plt.plot(1+np.arange(len(labels)), np.asarray([np.median(Data[i]) for i in range(len(Data))]), color="orange", linewidth=2, label="Median")
	plt.legend()
	plt.savefig(save_path, dpi=400, bbox_inches='tight')

	for i in range(len(Data)):
		print (labels[i], np.mean(Data[i]) - np.median(Data[i]))



# Extracting RQ Information for all video files
Meta_Information = extract_features.Extract_RQ_Features(
	codec="libx265",
	preset="medium",
	quality_metric="vmaf",
	video_filenames=defaults.Video_Titles,
	Resolutions_Considered=Resolutions,
	CRFs_Considered=CRFs,
	bitrates_Considered=None,
	QPs_Considered=None,
	min_quality=-np.inf,
	max_quality=np.inf,
	min_bitrate=-np.inf,
	max_bitrate=np.inf
)

for _,x_info in enumerate([("resolutions", "Resolutions"), ("crfs", "CRFs")]):
	for _,y_info in enumerate([("bitrate", "Bitrate"), ("quality", "VMAF")]):
		x_axis,x_label = x_info
		y_axis,y_label = y_info

		Box_Plot(
			Meta_Information=Meta_Information,
			x_axis=x_axis,
			y_axis=y_axis,
			x_label=x_label,
			y_label=y_label,
			save_path="plots/Box_Plot_{}_{}.png".format(x_label,y_label)
		)