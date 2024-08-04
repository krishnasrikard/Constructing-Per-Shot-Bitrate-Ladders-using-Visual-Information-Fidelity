"""
Histogram describing distribution of rate-quality points
"""
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt

import os, sys, warnings
import pickle
from tqdm import tqdm
warnings.filterwarnings("ignore")
sys.path.append("/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity")
import functions.IO_functions as IO_functions
import functions.extract_functions as extract_functions
import functions.extract_features as extract_features
import defaults

# Parameters
Resolutions = [(3840,2160),(2560,1440),(1920,1080),(1280,720),(960,540),(768,432),(640,360),(512,288)]
CRFs = list(np.arange(15,46,1))


# Rate-Quality Points
RQ_Points = extract_features.Extract_RQ_Features(
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
	min_bitrate=defaults.min_bitrate,
	max_bitrate=defaults.max_bitrate,
)

R = []
Q = []
for video_file in defaults.Video_Titles:
	R.append(RQ_Points[video_file][:,0])
	Q.append(RQ_Points[video_file][:,1])

R = np.concatenate(R, axis=0)
Q = np.concatenate(Q, axis=0)

# Plotting RQ-Curve
plt.figure(figsize=(8,6))
plt.grid()
plt.xlabel(r"$\log_{2}(Bitrate)$")
plt.ylabel("VMAF")
plt.title("Heat-Map of RQ points")
R = np.asarray(R).flatten()
Q = np.asarray(Q).flatten()/100
plt.hist2d(x=R,y=Q,bins=[np.arange(15,29,1),np.arange(0,1.05,0.1)],cmap="ocean_r")
plt.colorbar()

Count = np.zeros((len(Resolutions),))
for video_file in defaults.Video_Titles:
	for i,resolution in enumerate(Resolutions):
		mask = (np.isclose(np.round(RQ_Points[video_file][:,3], decimals=4), np.round(resolution[0]/3840, decimals=4))).astype("int")
		Count[i] += np.sum(mask)

plt.savefig('plots/RQ_Histogram.png', dpi=400, bbox_inches='tight')
plt.close()
