"""
Calculating VIF features on uncompressed videos
"""
# Importing Libraires
import numpy as np
import cv2

import os,sys,warnings
warnings.filterwarnings("ignore")
sys.path.append("/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity")
import argparse
import joblib
import features.VIF as VIF
import functions.IO_functions as IO_functions



# Feature Extraction from Video
def extract_vif_features(uncompressed_video_path, reference_features_save_path):
	"""
	Extract VIF features from uncompressed video.
	Args:
		uncompressed_video_path (str): Path to uncompressed video.
		reference_features_save_path (str): Path to save extracted features.
	"""
	# Reference Video
	yuv_reader = IO_functions.YUV_Reader(
		filepath=uncompressed_video_path,
		width=3840,
		height=2160,
		yuv_type="yuv420p10le"
	)
	video = yuv_reader.get_RGB_video()

	# Initializing VIF
	VIF_Function = VIF.Compute_VIF()

	# Computing Reference Video Features
	Reference_Video_Features = []

	# Iterating for each frame
	for i in range(video.shape[0]):
		# Calculating VIF features
		frame = np.copy(video[i])

		# Luma Component of current frame
		# Converting to int32 to avoid overflow during operations.
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)[:,:,0]
		if frame.shape != (2160,3840):
			frame = cv2.resize(frame, dsize=(3840, 2160), interpolation=cv2.INTER_LANCZOS4).astype(np.int32)


		# Decomposation
		vif_pyr_ref, vif_subband_keys = VIF_Function.Decomposation(frame)
		vif_subband_keys.sort(reverse=True)

		# GSM Model
		[vif_S_squared_all, vif_EigenValues_all] = VIF_Function.GSM_Model(vif_pyr_ref, vif_subband_keys)

		# Information in each subband along each eigen value
		vif_features_reference = VIF_Function.Reference_Subband_Eigen_Information_Matrix(
			subband_keys=vif_subband_keys, S_squared_all=vif_S_squared_all, EigenValues_all=vif_EigenValues_all
		)


		# Calculating Diff-VIF (T-VIF) Features
		if i == 0:
			current_frame = np.zeros(video[i].shape, dtype=np.uint8)
			previous_frame = np.zeros(video[i].shape, dtype=np.uint8)
		else:
			current_frame = np.copy(video[i])
			previous_frame = np.copy(video[i-1])

		# Luma Component of current frame
		# Converting to int32 to avoid overflow during operations.
		current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2YUV)[:,:,0]
		if current_frame.shape != (2160,3840):
			current_frame = cv2.resize(current_frame, dsize=(3840, 2160), interpolation=cv2.INTER_LANCZOS4).astype(np.int32)
			
		# Luma Component of previous frame
		# Converting to int32 to avoid overflow during operations.
		previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_RGB2YUV)[:,:,0]
		if previous_frame.shape != (2160,3840):
			previous_frame = cv2.resize(previous_frame, dsize=(3840, 2160), interpolation=cv2.INTER_LANCZOS4).astype(np.int32)

		# Frame Difference
		diff_frame = np.copy(current_frame - previous_frame)


		# Decomposation
		diff_vif_pyr_ref, diff_vif_subband_keys = VIF_Function.Decomposation(diff_frame)
		diff_vif_subband_keys.sort(reverse=True)

		# GSM Model
		[diff_vif_S_squared_all, diff_vif_EigenValues_all] = VIF_Function.GSM_Model(diff_vif_pyr_ref, diff_vif_subband_keys)

		# Information in each subband along each eigen value
		diff_vif_features_reference = VIF_Function.Reference_Subband_Eigen_Information_Matrix(
			subband_keys=diff_vif_subband_keys, S_squared_all=diff_vif_S_squared_all, EigenValues_all=diff_vif_EigenValues_all
		)

		# Appending reference video features and all other parameters
		Reference_Video_Features.append({"vif_info":vif_features_reference, "diff_vif_info":diff_vif_features_reference, "mean_abs_frame_diff":np.mean(np.abs(diff_frame))})


	# Saving computed features
	np.save(reference_features_save_path, np.array(Reference_Video_Features))
	print ("Saved", reference_features_save_path)



def process_video(uncompressed_video_path):
	"""
	Process Video
	"""
	# Filename
	filename = os.path.basename(uncompressed_video_path)
	print ("-"*75 + "\n" + filename[:-4] + "\n" + "-"*75, flush=True)

	# Path
	reference_features_save_path = os.path.join(args.vif_information_path, os.path.splitext(filename)[0] + ".npy")

	# Extract VIF features
	if os.path.exists(reference_features_save_path):
		# Print
		print (reference_features_save_path, "already exists. Skipping to next uncompressed-video.", flush=True)
	else:
		# Print
		print(f'Processing {filename}', flush=True)
		print ("\n"*2, flush=True)

		# Extract
		extract_vif_features(uncompressed_video_path, reference_features_save_path)



def main(args):
	# Assertions of Paths
	assert os.path.exists(args.raw_videos_path), "Invalid path to raw videos"
	assert os.path.exists(args.vif_information_path), "Invalid path to save VIF information of compressed videos"

	# Calculating VIF information for each uncompressed video
	filenames = os.listdir(args.raw_videos_path)
	joblib.Parallel(n_jobs=args.n_jobs)(joblib.delayed(process_video)(os.path.join(args.raw_videos_path, filename)) for filename in filenames)



# Calling Main function
if __name__ == '__main__':
	root_dir = os.path.dirname(os.path.realpath(__file__))

	# Get Arguments
	parser = argparse.ArgumentParser(description='Estimating compressed video information')

	# Dataset Paths
	parser.add_argument('--raw_videos_path', default='/home/krishna/Nebula/krishna/BVT-100_4K', help='Path to dataset.')
	parser.add_argument('--vif_information_path', default='/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/dataset/features_dataset/vif', help='Path to information of various parameters computed during VIF quality estimation of compressed videos.')
	
	# Main Path
	parser.add_argument('--main_path', default='/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/generate', type=str, help='Path to main folder')

	# Number of Parallel Jobs
	parser.add_argument('--n_jobs', default=4, type=int, help='Number of parallel jobs. Each jobs handles one video. Recommended value ~ 0.5 * number of cores to number of cores. -1 uses n_jobs = number of cores')

	# Parse Arguments
	args = parser.parse_args()

	main(args)