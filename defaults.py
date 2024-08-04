# Importing Libraries
import numpy as np

## Encoder Settings
# Codecs
codecs = ["libx265", "libx264"]

# Presets
presets = ["veryslow", "slow", "medium", "fast", "veryfast", "ultrafast"]

# Resolutions
# resolutions = [(3840,2160),(2560,1440),(1920,1080),(1280,720),(960,540),(768,432),(640,360),(512,288)]
resolutions = [(3840,2160),(2560,1440),(1920,1080),(1280,720),(960,540),(768,432)]

## Rate-Control Parameters
# Bitrate in kbps
bitrates = [125,250,500,750,1000,1500,2000,2500,3000,3500,4000,4500,5000,6000,7000,10000,20000,40000,100000]

# CRFs
# CRFs = [15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]
CRFs = [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,37,39,41]

# QPs
QPs = [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]


# Thresholds for RQ-points
min_bitrate = -np.inf
max_bitrate = np.inf
min_quality = 15
max_quality = 95


## Paths to Datasets
rq_points_dataset_path = "/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/dataset/rq_points_dataset"
llf_features_path = "/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/dataset/features_dataset/low_level_features"
vif_information_path = "/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/dataset/features_dataset/vif"


# Evaluation Bitrates i.e Steps in Bitrate Ladder
evaluation_bitrates = [500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10500,12000,15000]
evaluation_bitrates = np.log2(1000*np.asarray(evaluation_bitrates))
evaluation_bitrates = list(np.round(evaluation_bitrates, decimals=4))

# Evaluation Qualities i.e Steps in Quality Ladder
evaluation_qualities = np.asarray([25,35,45,50,55,60,65,70,75,80,85,90,92.5])
evaluation_qualities = list(np.round(evaluation_qualities/100.0, decimals=4))


## Video Filenames in BVT-100 4K dataset
Video_Titles = ['aerial_3840x2160_10bit_420_60fps_frames1-64', 'air-acrobatics-scene1_3840x2160_10bit_420_60fps_frames1-64', 'american-football-scene2_3840x2160_10bit_420_60fps_frames1-64', 'american-football-scene3_3840x2160_10bit_420_60fps_frames1-64', 'american-football-scene4_3840x2160_10bit_420_60fps_frames1-64', 'american-football-scene5_3840x2160_10bit_420_60fps_frames1-64', 'american-football-scene6_3840x2160_10bit_420_60fps_frames1-64', 'animals-scene10_3840x2160_10bit_420_60fps_frames1-64', 'animals-scene11_3840x2160_10bit_420_60fps_frames1-64', 'animals-scene1_3840x2160_10bit_420_60fps_frames1-64', 'animals-scene3_3840x2160_10bit_420_60fps_frames1-64', 'animals-scene5_3840x2160_10bit_420_60fps_frames1-64', 'animals-scene7_3840x2160_10bit_420_60fps_frames1-64', 'animals-scene9_3840x2160_10bit_420_60fps_frames1-64', 'asian-fusion-scene3_3840x2160_10bit_420_60fps_frames1-64', 'asian-fusion-scene5_3840x2160_10bit_420_60fps_frames1-64', 'asian-fusion-scene7_3840x2160_10bit_420_60fps_frames1-64', 'barscene_3840x2160_10bit_420_60fps_frames1-64', 'bosphorus_3840x2160_10bit_420_120fps_frames1-64', 'boxingpractice_3840x2160_10bit_420_60fps_frames1-64', 'bundnightscape_3840x2160_10bit_420_60fps_frames1-64', 'campfireparty_3840x2160_10bit_420_60fps_frames1-64', 'coastguard_3840x2160_10bit_420_60fps_frames1-64', 'constructionfield_3840x2160_10bit_420_60fps_frames1-64', 'costa-rica-scene1_3840x2160_10bit_420_60fps_frames1-64', 'costa-rica-scene2_3840x2160_10bit_420_60fps_frames1-64', 'costa-rica-scene3_3840x2160_10bit_420_60fps_frames1-64', 'costa-rica-scene4_3840x2160_10bit_420_60fps_frames1-64', 'crosswalk_3840x2160_10bit_420_60fps_frames1-64', 'dinnerscene-scene1_3840x2160_10bit_420_60fps_frames1-64', 'dinnerscene-scene2_3840x2160_10bit_420_60fps_frames1-64', 'drivingpov_3840x2160_10bit_420_60fps_frames1-64', 'fjords-scene1_3840x2160_10bit_420_60fps_frames1-64', 'fjords-scene2_3840x2160_10bit_420_60fps_frames1-64', 'fjords-scene3_3840x2160_10bit_420_60fps_frames1-64', 'foodmarket-scene1_3840x2160_10bit_420_60fps_frames1-64', 'fountains_3840x2160_10bit_420_60fps_frames1-64', 'honeybee_3840x2160_10bit_420_120fps_frames1-64', 'hong-kong-scene1_3840x2160_10bit_420_60fps_frames1-64', 'hong-kong-scene2_3840x2160_10bit_420_60fps_frames1-64', 'hong-kong-scene3_3840x2160_10bit_420_60fps_frames1-64', 'hong-kong-scene4_3840x2160_10bit_420_60fps_frames1-64', 'india-buildings-scene2_3840x2160_10bit_420_60fps_frames1-64', 'india-buildings-scene3_3840x2160_10bit_420_60fps_frames1-64', 'india-buildings-scene4_3840x2160_10bit_420_60fps_frames1-64', 'jockey_3840x2160_10bit_420_120fps_frames1-64', 'library_3840x2160_10bit_420_60fps_frames1-64', 'marathon_3840x2160_10bit_420_60fps_frames1-64', 'mobile_3840x2160_10bit_420_60fps_frames1-64', 'myanmar-scene3_3840x2160_10bit_420_60fps_frames1-64', 'myanmar-scene4_3840x2160_10bit_420_60fps_frames1-64', 'myanmar-scene5_3840x2160_10bit_420_60fps_frames1-64', 'myanmar-scene6_3840x2160_10bit_420_60fps_frames1-64', 'myanmar-scene7_3840x2160_10bit_420_60fps_frames1-64', 'narrator_3840x2160_10bit_420_60fps_frames1-64', 'pierseaside-scene1_3840x2160_10bit_420_60fps_frames1-64', 'pierseaside-scene2_3840x2160_10bit_420_60fps_frames1-64', 'raptors-scene1_3840x2160_10bit_420_60fps_frames1-64', 'raptors-scene2_3840x2160_10bit_420_60fps_frames1-64', 'readysetgo_3840x2160_10bit_420_120fps_frames1-64', 'red-rock-vol-scene6_3840x2160_10bit_420_60fps_frames1-64', 'red-rock-vol3-scene1_3840x2160_10bit_420_60fps_frames1-64', 'red-rock-vol3-scene3_3840x2160_10bit_420_60fps_frames1-64', 'red-rock-vol3-scene4_3840x2160_10bit_420_60fps_frames1-64', 'red-rock-vol3-scene5_3840x2160_10bit_420_60fps_frames1-64', 'residentialbuilding_3840x2160_10bit_420_60fps_frames1-64', 'ritualdance_3840x2160_10bit_420_60fps_frames1-64', 'rollercoaster_3840x2160_10bit_420_60fps_frames1-64', 'runners_3840x2160_10bit_420_60fps_frames1-64', 'rushhour_3840x2160_10bit_420_60fps_frames1-64', 'scarf_3840x2160_10bit_420_60fps_frames1-64', 'shakendry_3840x2160_10bit_420_120fps_frames1-64', 'skateboarding-scene12_3840x2160_10bit_420_60fps_frames1-64', 'skateboarding-scene7_3840x2160_10bit_420_60fps_frames1-64', 'skateboarding-scene8_3840x2160_10bit_420_60fps_frames1-64', 'snow-monkeys-scene1_3840x2160_10bit_420_60fps_frames1-64', 'snow-monkeys-scene2_3840x2160_10bit_420_60fps_frames1-64', 'snow-monkeys-scene3_3840x2160_10bit_420_60fps_frames1-64', 'snow-monkeys-scene5_3840x2160_10bit_420_60fps_frames1-64', 'snow-monkeys-scene6_3840x2160_10bit_420_60fps_frames1-64', 'squareandtimelapse_3840x2160_10bit_420_60fps_frames1-64', 'streets-of-india-scene1_3840x2160_10bit_420_60fps_frames1-64', 'streets-of-india-scene2_3840x2160_10bit_420_60fps_frames1-64', 'streets-of-india-scene3_3840x2160_10bit_420_60fps_frames1-64', 'tallbuildings_3840x2160_10bit_420_60fps_frames1-64', 'tango_3840x2160_10bit_420_60fps_frames1-64', 'toddlerfountain_3840x2160_10bit_420_60fps_frames1-64', 'trafficandbuilding_3840x2160_10bit_420_60fps_frames1-64', 'trafficflow_3840x2160_10bit_420_60fps_frames1-64', 'treeshade_3840x2160_10bit_420_60fps_frames1-64', 'tunnelflag-scene1_3840x2160_10bit_420_60fps_frames1-64', 'tunnelflag-scene2_3840x2160_10bit_420_60fps_frames1-64', 'venice-scene1_3840x2160_10bit_420_60fps_frames1-64', 'venice-scene2_3840x2160_10bit_420_60fps_frames1-64', 'venice-scene3_3840x2160_10bit_420_60fps_frames1-64', 'venice-scene4_3840x2160_10bit_420_60fps_frames1-64', 'windandnature-scene1_3840x2160_10bit_420_60fps_frames1-64', 'windandnature-scene2_3840x2160_10bit_420_60fps_frames1-64', 'wood_3840x2160_10bit_420_60fps_frames1-64', 'yachtride_3840x2160_10bit_420_120fps_frames1-64']


# Train, Validation and Test Splits of Video files.
"""
- The videos are split into non-overlapping training,validation and test dataset.
- The videos are split in such a way that each video in validation and testing sets are not scenes before/after an other video in the training dataset i.e each video is the only scene considered/taken from the video. 
- The training set contains 70% of the data, validation set contains 10% of the data and 20% of the data is allocated to test set.
- Due to small size of dataset, cross-validation should be prefered by joining training and validation sets.
"""
"""
def Create_Splits():
	import random

	Non_Scene_Videos = []
	Rest = []
	for f in Video_Titles:
		if "scene" in f:
			Rest.append(f)
		else:
			Non_Scene_Videos.append(f)

	random.shuffle(Non_Scene_Videos)

	Test = Non_Scene_Videos[:20]
	Valid = Non_Scene_Videos[20:30]
	Train = Rest + Non_Scene_Videos[30:]

	Train.sort()
	Valid.sort()
	Test.sort()
	print (Test)
	print ()
	print (Valid)
	print ()
	print (Train)
	
Create_Splits()
"""


# Video Filenames in BVT-100 4K dataset used for training
Train_Video_Titles = ['air-acrobatics-scene1_3840x2160_10bit_420_60fps_frames1-64', 'american-football-scene2_3840x2160_10bit_420_60fps_frames1-64', 'american-football-scene3_3840x2160_10bit_420_60fps_frames1-64', 'american-football-scene4_3840x2160_10bit_420_60fps_frames1-64', 'american-football-scene5_3840x2160_10bit_420_60fps_frames1-64', 'american-football-scene6_3840x2160_10bit_420_60fps_frames1-64', 'animals-scene10_3840x2160_10bit_420_60fps_frames1-64', 'animals-scene11_3840x2160_10bit_420_60fps_frames1-64', 'animals-scene1_3840x2160_10bit_420_60fps_frames1-64', 'animals-scene3_3840x2160_10bit_420_60fps_frames1-64', 'animals-scene5_3840x2160_10bit_420_60fps_frames1-64', 'animals-scene7_3840x2160_10bit_420_60fps_frames1-64', 'animals-scene9_3840x2160_10bit_420_60fps_frames1-64', 'asian-fusion-scene3_3840x2160_10bit_420_60fps_frames1-64', 'asian-fusion-scene5_3840x2160_10bit_420_60fps_frames1-64', 'asian-fusion-scene7_3840x2160_10bit_420_60fps_frames1-64', 'barscene_3840x2160_10bit_420_60fps_frames1-64', 'bosphorus_3840x2160_10bit_420_120fps_frames1-64', 'costa-rica-scene1_3840x2160_10bit_420_60fps_frames1-64', 'costa-rica-scene2_3840x2160_10bit_420_60fps_frames1-64', 'costa-rica-scene3_3840x2160_10bit_420_60fps_frames1-64', 'costa-rica-scene4_3840x2160_10bit_420_60fps_frames1-64', 'dinnerscene-scene1_3840x2160_10bit_420_60fps_frames1-64', 'dinnerscene-scene2_3840x2160_10bit_420_60fps_frames1-64', 'fjords-scene1_3840x2160_10bit_420_60fps_frames1-64', 'fjords-scene2_3840x2160_10bit_420_60fps_frames1-64', 'fjords-scene3_3840x2160_10bit_420_60fps_frames1-64', 'foodmarket-scene1_3840x2160_10bit_420_60fps_frames1-64', 'hong-kong-scene1_3840x2160_10bit_420_60fps_frames1-64', 'hong-kong-scene2_3840x2160_10bit_420_60fps_frames1-64', 'hong-kong-scene3_3840x2160_10bit_420_60fps_frames1-64', 'hong-kong-scene4_3840x2160_10bit_420_60fps_frames1-64', 'india-buildings-scene2_3840x2160_10bit_420_60fps_frames1-64', 'india-buildings-scene3_3840x2160_10bit_420_60fps_frames1-64', 'india-buildings-scene4_3840x2160_10bit_420_60fps_frames1-64', 'mobile_3840x2160_10bit_420_60fps_frames1-64', 'myanmar-scene3_3840x2160_10bit_420_60fps_frames1-64', 'myanmar-scene4_3840x2160_10bit_420_60fps_frames1-64', 'myanmar-scene5_3840x2160_10bit_420_60fps_frames1-64', 'myanmar-scene6_3840x2160_10bit_420_60fps_frames1-64', 'myanmar-scene7_3840x2160_10bit_420_60fps_frames1-64', 'pierseaside-scene1_3840x2160_10bit_420_60fps_frames1-64', 'pierseaside-scene2_3840x2160_10bit_420_60fps_frames1-64', 'raptors-scene1_3840x2160_10bit_420_60fps_frames1-64', 'raptors-scene2_3840x2160_10bit_420_60fps_frames1-64', 'red-rock-vol-scene6_3840x2160_10bit_420_60fps_frames1-64', 'red-rock-vol3-scene1_3840x2160_10bit_420_60fps_frames1-64', 'red-rock-vol3-scene3_3840x2160_10bit_420_60fps_frames1-64', 'red-rock-vol3-scene4_3840x2160_10bit_420_60fps_frames1-64', 'red-rock-vol3-scene5_3840x2160_10bit_420_60fps_frames1-64', 'rushhour_3840x2160_10bit_420_60fps_frames1-64', 'skateboarding-scene12_3840x2160_10bit_420_60fps_frames1-64', 'skateboarding-scene7_3840x2160_10bit_420_60fps_frames1-64', 'skateboarding-scene8_3840x2160_10bit_420_60fps_frames1-64', 'snow-monkeys-scene1_3840x2160_10bit_420_60fps_frames1-64', 'snow-monkeys-scene2_3840x2160_10bit_420_60fps_frames1-64', 'snow-monkeys-scene3_3840x2160_10bit_420_60fps_frames1-64', 'snow-monkeys-scene5_3840x2160_10bit_420_60fps_frames1-64', 'snow-monkeys-scene6_3840x2160_10bit_420_60fps_frames1-64', 'streets-of-india-scene1_3840x2160_10bit_420_60fps_frames1-64', 'streets-of-india-scene2_3840x2160_10bit_420_60fps_frames1-64', 'streets-of-india-scene3_3840x2160_10bit_420_60fps_frames1-64', 'tunnelflag-scene1_3840x2160_10bit_420_60fps_frames1-64', 'tunnelflag-scene2_3840x2160_10bit_420_60fps_frames1-64', 'venice-scene1_3840x2160_10bit_420_60fps_frames1-64', 'venice-scene2_3840x2160_10bit_420_60fps_frames1-64', 'venice-scene3_3840x2160_10bit_420_60fps_frames1-64', 'venice-scene4_3840x2160_10bit_420_60fps_frames1-64', 'windandnature-scene1_3840x2160_10bit_420_60fps_frames1-64', 'windandnature-scene2_3840x2160_10bit_420_60fps_frames1-64']

# Video Filenames in BVT-100 4K dataset used for validation
Valid_Video_Titles = ['boxingpractice_3840x2160_10bit_420_60fps_frames1-64', 'campfireparty_3840x2160_10bit_420_60fps_frames1-64', 'coastguard_3840x2160_10bit_420_60fps_frames1-64', 'constructionfield_3840x2160_10bit_420_60fps_frames1-64', 'marathon_3840x2160_10bit_420_60fps_frames1-64', 'narrator_3840x2160_10bit_420_60fps_frames1-64', 'ritualdance_3840x2160_10bit_420_60fps_frames1-64', 'shakendry_3840x2160_10bit_420_120fps_frames1-64', 'squareandtimelapse_3840x2160_10bit_420_60fps_frames1-64', 'trafficflow_3840x2160_10bit_420_60fps_frames1-64']

# Video Filenames in BVT-100 4K dataset used for testing
Test_Video_Titles = ['bundnightscape_3840x2160_10bit_420_60fps_frames1-64', 'crosswalk_3840x2160_10bit_420_60fps_frames1-64', 'drivingpov_3840x2160_10bit_420_60fps_frames1-64', 'fountains_3840x2160_10bit_420_60fps_frames1-64', 'honeybee_3840x2160_10bit_420_120fps_frames1-64', 'jockey_3840x2160_10bit_420_120fps_frames1-64', 'library_3840x2160_10bit_420_60fps_frames1-64', 'mobile_3840x2160_10bit_420_60fps_frames1-64', 'readysetgo_3840x2160_10bit_420_120fps_frames1-64', 'residentialbuilding_3840x2160_10bit_420_60fps_frames1-64', 'rollercoaster_3840x2160_10bit_420_60fps_frames1-64', 'runners_3840x2160_10bit_420_60fps_frames1-64', 'rushhour_3840x2160_10bit_420_60fps_frames1-64', 'scarf_3840x2160_10bit_420_60fps_frames1-64', 'tango_3840x2160_10bit_420_60fps_frames1-64', 'toddlerfountain_3840x2160_10bit_420_60fps_frames1-64', 'trafficandbuilding_3840x2160_10bit_420_60fps_frames1-64', 'treeshade_3840x2160_10bit_420_60fps_frames1-64', 'wood_3840x2160_10bit_420_60fps_frames1-64', 'yachtride_3840x2160_10bit_420_120fps_frames1-64']

# For Testing, we combine validation and testing datasets for a bigger sample size
Test_Video_Titles = Test_Video_Titles + Valid_Video_Titles


# Feature Names

# GLCM Features
glcm_features = ['kurt_GLCM_contrast_mean',
'kurt_GLCM_contrast_std',
'mean_GLCM_contrast_mean',
'mean_GLCM_contrast_std',
'skew_GLCM_contrast_mean',
'skew_GLCM_contrast_std',
'std_GLCM_contrast_mean',
'std_GLCM_contrast_std',
'kurt_GLCM_correlation_mean',
'kurt_GLCM_correlation_std',
'mean_GLCM_correlation_mean',
'mean_GLCM_correlation_std',
'skew_GLCM_correlation_mean',
'skew_GLCM_correlation_std',
'std_GLCM_correlation_mean',
'std_GLCM_correlation_std',
'kurt_GLCM_energy_mean',
'kurt_GLCM_energy_std',
'mean_GLCM_energy_mean',
'mean_GLCM_energy_std',
'skew_GLCM_energy_mean',
'skew_GLCM_energy_std',
'std_GLCM_energy_mean',
'std_GLCM_energy_std',
'kurt_GLCM_homogeneity_mean',
'kurt_GLCM_homogeneity_std',
'mean_GLCM_homogeneity_mean',
'mean_GLCM_homogeneity_std',
'skew_GLCM_homogeneity_mean',
'skew_GLCM_homogeneity_std',
'std_GLCM_homogeneity_mean',
'std_GLCM_homogeneity_std'
]

per_frame_glcm_features = ['GLCM_contrast_mean',
'GLCM_contrast_std',
'GLCM_correlation_mean',
'GLCM_correlation_std',
'GLCM_energy_mean',
'GLCM_energy_std',
'GLCM_homogeneity_mean',
'GLCM_homogeneity_std'
]

# Temporal Coherence Features
tc_features = ['mean_TC_kurt',
'mean_TC_mean',
'mean_TC_skew',
'mean_TC_std',
'std_TC_kurt',
'std_TC_mean',
'std_TC_skew',
'std_TC_std'
]

per_frame_tc_features = ['TC_kurt',
'TC_mean',
'TC_skew',
'TC_std'
]

# Spatial Information
si_features = ['kurt_SI_mean',
'kurt_SI_std',
'mean_SI_mean',
'mean_SI_std',
'skew_SI_mean',
'skew_SI_std',
'std_SI_mean',
'std_SI_std'
]

per_frame_si_features = ['SI_mean',
'SI_std'
]

# Temporal Information
ti_features = ['kurt_TI_mean',
'kurt_TI_std',
'mean_TI_mean',
'mean_TI_std',
'skew_TI_mean',
'skew_TI_std',
'std_TI_mean',
'std_TI_std'
]

per_frame_ti_features = ['TI_mean',
'TI_std'
]

# Contrast Information
cti_features = ['kurt_CTI_mean',
'kurt_CTI_std',
'mean_CTI_mean',
'mean_CTI_std',
'skew_CTI_mean',
'skew_CTI_std',
'std_CTI_mean',
'std_CTI_std'
]

per_frame_cti_features = ['CTI_mean',
'CTI_std'
]

# Colofulness
cf_features = ['kurt_CF',
'mean_CF',
'skew_CF',
'std_CF'
]

per_frame_cf_features = ['CF']

# Chromiance Information
ci_features = ['kurt_CI_U_mean',
'kurt_CI_U_std',
'mean_CI_U_mean',
'mean_CI_U_std',
'skew_CI_U_mean',
'skew_CI_U_std',
'std_CI_U_mean',
'std_CI_U_std',
'kurt_CI_V_mean',
'kurt_CI_V_std',
'mean_CI_V_mean',
'mean_CI_V_std',
'skew_CI_V_mean',
'skew_CI_V_std',
'std_CI_V_mean',
'std_CI_V_std'
]

per_frame_ci_features = ['CI_U_mean',
'CI_U_std',
'CI_V_mean',
'CI_V_std'
]

# DCT Texture Energy
dct_features = ['mean_E_Y',
'mean_h_Y',
'mean_L_Y',
'mean_E_U',
'mean_h_U',
'mean_L_U',
'mean_E_V',
'mean_h_V',
'mean_L_V'
]

per_frame_dct_features = ['E_Y',
'h_Y',
'L_Y',
'E_U',
'h_U',
'L_U',
'E_V',
'h_V',
'L_V'
]

# Customized Bitrate dependent Texture Features
bitrate_texture_features = {
"log2(sqrt(mean_h_Y/mean_E_Y)) + 2b": ["mean_E_Y", "mean_h_Y"],
"log2(sqrt(mean_h_U/mean_E_U)) + 2b": ["mean_E_U", "mean_h_U"],
"log2(sqrt(mean_h_V/mean_E_V)) + 2b": ["mean_E_V", "mean_h_V"],
}

per_frame_bitrate_texture_features = {
"log2(sqrt(h_Y/E_Y)) + 2b": ["E_Y", "h_Y"],
"log2(sqrt(h_U/E_U)) + 2b": ["E_U", "h_U"],
"log2(sqrt(h_V/E_V)) + 2b": ["E_V", "h_V"],
}

# Customized Quality dependent Texture Features
quality_texture_features = {
"0.5*(q - log2(sqrt(mean_h_Y/mean_E_Y)))": ["mean_E_Y", "mean_h_Y"],
"0.5*(q - log2(sqrt(mean_h_U/mean_E_U)))": ["mean_E_U", "mean_h_U"],
"0.5*(q - log2(sqrt(mean_h_V/mean_E_V)))": ["mean_E_V", "mean_h_V"],
}

per_frame_quality_texture_features = {
"0.5*(q - log2(sqrt(h_Y/E_Y)))": ["mean_E_Y", "mean_h_Y"],
"0.5*(q - log2(sqrt(h_Y/E_Y)))": ["mean_E_U", "mean_h_U"],
"0.5*(q - log2(sqrt(h_Y/E_Y)))": ["mean_E_V", "mean_h_V"],
}

# Compute Time
compute_time_features = [
'GLCM_compute_time',
'TC_compute_time',
'SI_compute_time',
'TI_compute_time',
'CTI_compute_time',
'CF_compute_time',
'CI_U_compute_time',
'CI_V_compute_time',
'EhL_Y_compute_time',
'EhL_U_compute_time',
'EhL_V_compute_time'
]