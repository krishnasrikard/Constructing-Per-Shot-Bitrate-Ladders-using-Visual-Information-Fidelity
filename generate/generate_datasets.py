"""
Generating RQ points after downscaling, compression and quality estimation.
Saving compressed videos
"""

import os, sys
sys.path.append("/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity")
import argparse
import functions.rate_quality_estimations as rate_quality_estimations
import defaults


def main(args):
	# Assertions of Paths
	assert os.path.exists(args.main_path), "Path to main directory does not exist."
	assert os.path.exists(args.raw_videos_path), "Path to raw-videos does not exist."
	assert os.path.exists(args.ffmpeg_path), "Path to ffmpeg does not exits."
	
	if args.rq_points_dataset != "None":
		assert os.path.exists(args.rq_points_dataset), "Path to RQ-points does not exist."
	else:
		args.rq_points_dataset = None
		
	if args.compressed_videos_dataset != "None":
		assert os.path.exists(args.compressed_videos_dataset), "Path to compressed videos dataset does not exist."
	else:
		args.compressed_videos_dataset = None


	# Assertions
	assert args.codec in defaults.codecs, 'Invalid codec provided. Codecs: ["libx265", "libx264"]'
	assert args.preset in defaults.presets, 'Invalid preset provided. Presets: ["veryslow", "slow", "medium", "fast", "veryfast", "ultrafast"]'
	assert args.rate_control in ["crf", "bitrate", "qp"], 'Provide a valid rate-control. Rate-Control: ["crf", "bitrate", "qp"]'


	# Creating a temporary paths
	temp_folders_path = os.path.join(args.rq_points_dataset, "temp_folders")
	if os.path.exists(temp_folders_path) == False:
		os.mkdir(temp_folders_path)
	temp_path = os.path.join(temp_folders_path, "temp_{}_{}_{}".format(args.codec, args.preset, args.rate_control))
	if os.path.exists(temp_path) == False:
		os.mkdir(temp_path)


	# Selected Rate-Control
	if args.rate_control == "qp":
		selected_qps = defaults.QPs
		selected_crfs = None
		selected_bitrates = None
	elif args.rate_control == "crf":
		selected_qps = None
		selected_crfs = defaults.CRFs
		selected_bitrates = None
	elif args.rate_control == "bitrate":
		selected_qps = None
		selected_crfs = None
		selected_bitrates = defaults.bitrates
	else:
		None


	for filename in sorted(os.listdir(args.raw_videos_path)):
		print ("-"*75 + "\n" + filename[:-4] + "\n" + "-"*75, flush=True)

		rate_quality_estimations.estimate_rate_quality_points(
			input_yuv_path=os.path.join(args.raw_videos_path,filename),
			output_resolutions=defaults.resolutions,
			codec=args.codec,
			preset=args.preset,
			QPs=selected_qps,
			CRFs=selected_crfs,
			bitrates=selected_bitrates,
			rq_points_output_path=args.rq_points_dataset,
			compressed_videos_output_path=args.compressed_videos_dataset,
			temp_path=temp_path,
			vmaf_resolution=(3840,2160),
			quality_metrics={"PSNR":True, "SSIM":False, "MS_SSIM":False},
			num_threads=args.num_threads,
			ffmpeg_path=args.ffmpeg_path
		)


# Calling Main function
if __name__ == '__main__':
	# -----------------------------------------------------------------
	# Flushing Output
	import functools
	print = functools.partial(print, flush=True)

	# Saving stdout
	sys.stdout = open('{}.log'.format(os.path.basename(__file__)[:-3]), 'w')

	# -----------------------------------------------------------------

	root_dir = os.path.dirname(os.path.realpath(__file__))

	# Get Arguments
	parser = argparse.ArgumentParser(description='Estimating compressed video information')

	# Dataset Paths
	parser.add_argument('--raw_videos_path', default='/home/krishna/Nebula/krishna/BVT-100_4K', help='Path to dataset.')
	parser.add_argument('--rq_points_dataset', default='/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/dataset/rq_points_dataset', help='Path to RQ points dataset.')
	parser.add_argument('--compressed_videos_dataset', default='None', help='Path to compressed videos dataset.')
	parser.add_argument('--ffmpeg_path', default="/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/ffmpeg/ffmpeg-6.0-amd64-static", help='Path to ffmpeg.')	

	# Main Path
	parser.add_argument('--main_path', default='/home/krishna/Constructing-Per-Shot-Bitrate-Ladders-using-Visual-Information-Fidelity/generate', type=str, help='Path to main folder')

	# Processing
	parser.add_argument('--num_threads', default=8, type=int, help='No.of threads used to run ffmpeg commands. Generally no.of threads are set between 4-8 for ffmpeg. (Default: 8)')

	# Select Codec and Preset
	parser.add_argument('--codec', default="libx265", type=str, help='Index of the codec that needs to be selected. Options: ["libx265", "libx264"]. (Default: "libx265")')
	parser.add_argument('--preset', default="medium", type=str, help='Index of the preset that needs to be selected. Options: ["veryslow", "slow", "medium", "fast", "veryfast", "ultrafast"]. (Default: "medium")')
	parser.add_argument('--rate_control', default='crf', type=str, help='Index of the rate-control method that needs to be selected. Options: ["crf", "bitrate", "qp"]. (Default: "crf")')

	# Parse Arguments
	args = parser.parse_args()

	main(args)