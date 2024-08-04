"""
Function to generate ffmpeg commmands.
References:
- https://ffmpeg.org/ffmpeg-all.html
- https://trac.ffmpeg.org/wiki/Encode/H.264
- https://trac.ffmpeg.org/wiki/Encode/H.265
- https://ottverse.com/hevc-encoding-using-ffmpeg-crf-cbr-2-pass-lossless/
- https://ottverse.com/calculate-psnr-vmaf-ssim-using-ffmpeg/
- https://ottverse.com/vmaf-easyvmaf/
"""

import os
import subprocess, shlex
from typing import Type, Callable, Tuple, Optional, Set, List, Union

def compression_command(
	raw_video:bool=False,
	input_resolution:Tuple=None,
	frame_rate:float=None,
	pixel_format:str=None,
	input_time_limit:float=None,
	input_video_path:str=None,
	output_resolution:Tuple=(1920,1080),
	scaling_algo:str="lanczos",
	video_codec:str=None,
	preset:str=None,
	QP:int=None,
	CRF:int=None,
	bitrate:float=None,
	output_video_path:str=None,
	num_threads:int=8,
	ffmpeg_path:str="ffmpeg/ffmpeg-6.0-amd64-static"
):
	"""
	Function to generate ffmpeg command for videos for the specified compression settings.

	Notes:

		- For rawvideo inputs i.e inputs in YUV format, video-format, input-resolution, frame-rate and pixel-format are neccessary as input. 
		- For most of the common video formats, the ffmpeg auto-detects video-format, input-resolution, frame-rate and pixel-format from the metadata and the container.
		- For a rawvideo, all the compression settings video-codec, preset, QP or CRF or bitrate must be provided.

	Args:
		raw_video (bool): Whether input video is YUV format or not. (Default: None)
		input_resolution (tuple): The input resolution needs to provided when video format is yuv as (width, height) as frame size is not stored in the input file. 
			(Default: None)
		frame_rate (float): The frame-rate of the video. (Default: None)
		pixel_format (str): The input video pixel format. Use command `ffmpeg -pix_fmts` to get all the options. (Default: None)
		input_time_limit (float): The limit of the time duration of data read from the input file. (Default: None)
		input_video_path (str): The path to the input file. (Default: None)
		output_resolution (tuple): The resolution of the output. "-1" in for height or width maintains the ascept ratio of wrt corresponding width or height 
			respectively. If output-resolution is not provided, output-resolution will be same as input resolution. Default: (1920,1080)
		scaling_algo (str): The scaling algorithm. Refer to this link for all the options. Link: https://ffmpeg.org/ffmpeg-scaler.html#scaler_005foptions. 
			(Default: "lanczos")
		video_codec (str): The encoder/decoder video codec. Use command `ffmpeg -codecs` to get all the options. (Default: None)
		preset (str): A preset is a collection of options that will provide a certain encoding speed to compression ratio. If None, the default value of 
			preset depends on the codec.(Default: None)
		QP (int): The quantization-paramter (QP) setting. The value of QP lies between [0,51] (only integers) with "0": highest quality and "51": worst quality. 
			When QP is provided, the encoding process is constant-QP encoding. (Default: None)
		CRF (int): The range of the CRF scale is 0-51 i.e lossless to worst quality possible. The 0-51 CRF quantizer scale mentioned on this page only applies 
			to 8-bit x264. When compiled with 10-bit support, x264's quantizer scale is 0-63. When CRF is provided, the encoding process is constant-quality encoding. 
			(Default: None)
		bitrate (float): The bitrate of the output video. When bitrate is provided, the encoding process is constant-bitrate encoding i.e max-bitrate, min-bitrate 
			and buffer-size will be set as bitrate. (Default: None) 
		output_video_path (str): The path to the output file. (Default: None)
		num_threads (int): No.of threads used to run ffmpeg commands. Generally no.of threads are set between 4-8 for ffmpeg. (Default: 8)
		ffmpeg_path (str): The path to ffmpeg. (Default: "ffmpeg/ffmpeg-6.0-amd64-static")
	Returns:
		cmd (str): The ffmpeg command as per the mentioned specifications.
	"""
	
	# Input video settings
	if raw_video:
		# The following are necessary for rawvideo inputs.
		assert input_resolution is not None, "For a 'YUV' format video input, resolution is required are inputs."
		assert frame_rate is not None, "For a 'YUV' format video input, frame-rate is required are inputs."
		assert pixel_format is not None,  "For a 'YUV' format video input, pixel_format is required are inputs."
		
		cmd_raw_video = "-f rawvideo -vcodec rawvideo"
		cmd_input_resolution = "-s " + str(input_resolution[0]) + "x" + str(input_resolution[1])
		cmd_frame_rate = "-r " + str(frame_rate)
		cmd_pixel_format = "-pix_fmt " + str(pixel_format)
		cmd_input_video_settings = " ".join([cmd_raw_video, cmd_input_resolution, cmd_frame_rate, cmd_pixel_format])
	else:
		# The ffmpeg auto-detects video-format, input-resolution, frame-rate and pixel-format from the metadata and the container.
		cmd_input_video_settings = ""

	if input_time_limit is not None:
		cmd_input_time_limit = "-t " + str(input_time_limit)
		cmd_input_video_settings = cmd_input_video_settings + " " + cmd_input_time_limit

	# The input video
	assert os.path.exists(input_video_path), "Provide a valid input_video_path."
	cmd_input_video_path = "-i " + input_video_path

	# The output video or compression settings
	check_compression_settings = (video_codec is not None) and (preset is not None) and ((QP is not None) or (CRF is not None) or (bitrate is not None))
	valid_rate_settings = ((QP is not None) and (CRF is None) and (bitrate is None)) or ((QP is None) and (CRF is not None) and (bitrate is None)) or ((QP is None) and (CRF is None) and (bitrate is not None))

	# Settings Assertions
	if raw_video:
		# YUV raw file to video
		assert check_compression_settings, "Input video is in YUV format. All compression settings must be provided."
		assert valid_rate_settings, "Two of QP or CRF or bitrate should be 'None'."

	# Downscaling
	if output_resolution is not None and output_resolution != input_resolution:
		assert scaling_algo is not None, "Please provide the scaling algorithm i.e remove its input as 'None' as output resolution is not same as input resolution."

		cmd_output_resolution_scaling_algo = '-vf "scale=' + str(output_resolution[0]) + 'x' + str(output_resolution[1]) + ':' + "flags=" + scaling_algo + '"'
	else:
		# No downscaling
		cmd_output_resolution_scaling_algo = ""

	
	# Compression
	if check_compression_settings:
		assert valid_rate_settings, "Two of QP or CRF or bitrate should be 'None'."

		cmd_video_codec = "-codec:v " + video_codec
		cmd_preset = "-preset " + preset

		if QP is not None:
			# Constant-QP encoding
			cmd_rate_setting = "-qp " + str(QP)
		elif CRF is not None:
			# Constant-Quality encoding
			cmd_rate_setting = "-crf " + str(CRF)
		else:
			# Constant-Bitrate encoding
			cmd_rate_setting = "-b:v {}k -minrate {}k -maxrate {}k -bufsize {}k".format(bitrate,bitrate,bitrate,2*bitrate)
	else:
		# No compression
		cmd_video_codec = ""
		cmd_preset = ""
		cmd_rate_setting = ""

	cmd_output_video_settings = " ".join([cmd_output_resolution_scaling_algo, cmd_video_codec, cmd_preset, cmd_rate_setting])

	# No.of threads
	if num_threads is not None:
		cmd_threads = "-threads {}".format(num_threads)
	else:
		cmd_threads = ""

	# Output video path
	output_path = "/".join(output_video_path.split("/")[0:-1])
	assert os.path.exists(output_path), "Provided output path '" + str(output_path) + "' Provide a valid output_video_path."
	cmd_output_video_path = "-y " + output_video_path

	# Final Command
	cmd_prefix = "/usr/bin/time"
	cmd = ffmpeg_path + "/ffmpeg"
	cmd = " ".join([cmd_prefix, cmd, cmd_input_video_settings, cmd_input_video_path, cmd_output_video_settings, cmd_threads, cmd_output_video_path])

	return cmd



def bitrate_estimation_command(
	input_video_path:str=None,
	num_threads:int=8,
	ffmpeg_path:str="ffmpeg/ffmpeg-6.0-amd64-static"
):
	"""
	Function to generate ffmpeg command for bitrate estimation of a video.
	Args:
		input_video_path (str): The path to the input file. (Default: None)
		num_threads (int): No.of threads used to run ffmpeg commands. Generally no.of threads are set between 4-8 for ffmpeg. (Default: 8)
		ffmpeg_path (str): The path to ffmpeg. (Default: "ffmpeg/ffmpeg-6.0-amd64-static")
	Returns:
		cmd (str): The ffmpeg command to estimate bitrate of the given video.
	"""
	
	# The input video
	assert os.path.exists(input_video_path), "Provide a valid input_video_path."

	# No.of threads
	if num_threads is not None:
		cmd_threads = "-threads {}".format(num_threads)
	else:
		cmd_threads = ""

	# Final Command
	cmd_prefix = "/usr/bin/time"
	cmd = ffmpeg_path + "/ffprobe -v error -select_streams v:0 -show_entries format=bit_rate -of default=noprint_wrappers=1:nokey=1"
	cmd = " ".join([cmd_prefix, cmd, cmd_threads, input_video_path])

	return cmd


def quality_estimation_command(
		raw_video:bool=False,
		input_resolution:Tuple=None,
		frame_rate:float=None,
		pixel_format:str=None,
		reference_video_path:str=None,
		distored_video_path:str=None,
		vmaf_resolution:tuple=(3840,2160),
		scaling_algo:str='lanczos',
		PSNR:bool=True,
		SSIM:bool=True,
		MS_SSIM:bool=True,
		logfile_path:str=None,
		num_threads:int=8,
		ffmpeg_path:str="ffmpeg/ffmpeg-6.0-amd64-static"
):
	"""
	Function to generate ffmpeg command for video-quality estimations. The VMAF value is calculated as default.
	Args:
		raw_video (bool): Whether input video is YUV format or not. (Default: None)
		input_resolution (tuple): The input resolution needs to provided when video format is yuv as (width, height) as frame size is not stored in the input file. 
			(Default: None)
		frame_rate (float): The frame-rate of the video. (Default: None)
		pixel_format (str): The input video pixel format. Use command `ffmpeg -pix_fmts` to get all the options. (Default: None)
		reference_video_path (str): The path to the reference video. (Default: None)
		distored_video_path (str): The path to the distored video. (Default: None)
		vmaf_resolution(tuple): The resolution at which VMAF is calculated. (Default: (3840,2160))
		scaling_algo (str): The scaling algorithm. Refer to this link for all the options. Link: https://ffmpeg.org/ffmpeg-scaler.html#scaler_005foptions. 
			(Default: "lanczos")
		PSNR (bool): Whether to estimate PSNR or not. (Default: True)
		SSIM (bool): Whether to estimate SSIM or not. (Default: True)
		MS_SSIM (bool): Whether to estimate MS_SSIM or not. (Default: True)
		logfile_path (str): File to logfile which contains the estimated quality values. (Default: None)
		num_threads (int): No.of threads used to run ffmpeg commands. Generally no.of threads are set between 4-8 for ffmpeg. (Default: 8)
		ffmpeg_path (str): The path to ffmpeg. (Default: "ffmpeg/ffmpeg-6.0-amd64-static")
	Returns:
		cmd (str): The ffmpeg command to estimate quality of the given video.
	"""
	# Display Resolution: VMAF resolution i.e VMAF model used for predicting Quality of compressed video
	if vmaf_resolution == (3840,2160):
		vmaf_model_path = ffmpeg_path + "/model/vmaf_4k_v0.6.1.json"
	elif vmaf_resolution == (1920,1080):
		vmaf_model_path = ffmpeg_path + "/model/vmaf_v0.6.1.json"
	else:
		assert False, 'Select a valid VMAF resolution i.e (1920,1080) or (3840,2160).'

	# ffmpeg path
	cmd_ffmpeg = ffmpeg_path + '/ffmpeg'

	# Input reference and distored videos
	if raw_video:
		# Assertions: The following are necessary for rawvideo inputs.
		assert input_resolution is not None, "For a 'YUV' format video input, resolution is required are inputs."
		assert frame_rate is not None, "For a 'YUV' format video input, frame-rate is required are inputs."
		assert pixel_format is not None,  "For a 'YUV' format video input, pixel_format is required are inputs."
		
		cmd_raw_video = "-f rawvideo -vcodec rawvideo"
		cmd_input_resolution = "-s " + str(input_resolution[0]) + "x" + str(input_resolution[1])
		cmd_frame_rate = "-r " + str(frame_rate)
		cmd_pixel_format = "-pix_fmt " + str(pixel_format)
		cmd_reference_video_settings = " ".join([cmd_raw_video, cmd_input_resolution, cmd_frame_rate, cmd_pixel_format])
	else:
		cmd_reference_video_settings = ""

	cmd_inputs = '-i {} '.format(distored_video_path) + cmd_reference_video_settings + ' -i {}'.format(reference_video_path)

	# Upscaling of reference/distored videos
	cmd_upscale= '-lavfi "[0:v]scale={}x{}:flags={}[distorted]; [1:v]scale={}x{}:flags={}[reference];[distorted][reference]'.format(vmaf_resolution[0],vmaf_resolution[1],scaling_algo,vmaf_resolution[0],vmaf_resolution[1],scaling_algo)
	
	# Multi-Threading
	if num_threads is not None:
		cmd_quality = 'libvmaf=log_fmt=json:log_path={}:model_path={}:n_threads={}:psnr={}:ssim={}:ms_ssim={}" -f null -'.format(logfile_path,vmaf_model_path,num_threads,int(PSNR),int(SSIM),int(MS_SSIM))
	else:
		cmd_quality = 'libvmaf=log_fmt=json:log_path={}:model_path={}:psnr={}:ssim={}:ms_ssim={}" -f null -'.format(logfile_path,vmaf_model_path,int(PSNR),int(SSIM),int(MS_SSIM))

	# Final Command
	cmd_prefix = "/usr/bin/time"
	cmd = " ".join([cmd_prefix, cmd_ffmpeg, cmd_inputs, cmd_upscale, cmd_quality])

	return cmd