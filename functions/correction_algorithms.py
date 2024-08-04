# Importing Libraries

def Top_Bottom(
	ladder:dict
):
	"""
	Corrects a quality/bitrate-ladder i.e making sure that resolutions are in non-increasing order for decrease in quality/bitrate.
	Args:
		ladder (dict): The quality/bitrate-ladder
	"""
	steps = list(ladder.keys())
	steps.sort(reverse=True)

	max_resolution = ladder[steps[0]]

	corrected_ladder = {}
	for i in range(len(steps)):
		if ladder[steps[i]][1] > max_resolution[1]:
			corrected_ladder[steps[i]] = max_resolution
		else:
			max_resolution = ladder[steps[i]]
			corrected_ladder[steps[i]] = ladder[steps[i]]

	return corrected_ladder


def Bottom_Top(
	ladder:dict
):
	"""
	Corrects a quality/bitrate-ladder i.e making sure that resolutions are in non-increasing order for decrease in quality/bitrate.
	Args:
		ladder (dict): The quality/bitrate-ladder
	"""
	steps = list(ladder.keys())
	steps.sort()

	min_resolution = ladder[steps[0]]

	corrected_ladder = {}
	for i in range(len(steps)):
		if ladder[steps[i]][1] < min_resolution[1]:
			corrected_ladder[steps[i]] = min_resolution
		else:
			min_resolution = ladder[steps[i]]
			corrected_ladder[steps[i]] = ladder[steps[i]]

	return corrected_ladder