"""
Paper: Bjøntegaard Delta (BD): A Tutorial Overview of the Metric, Evolution, Challenges, and Recommendations
Source: https://github.com/Anserw/Bjontegaard_metric/blob/master/bjontegaard_metric.py
"""

import numpy as np
import scipy.interpolate

def BD_Rate(R1, Q1, R2, Q2, piecewise=False):
	lR1 = np.log(R1)
	lR2 = np.log(R2)

	# Rate Method
	p1 = np.polyfit(Q1, lR1, 3)
	p2 = np.polyfit(Q2, lR2, 3)

	# Integration Interval
	min_int = max(min(Q1), min(Q2))
	max_int = min(max(Q1), max(Q2))

	# Calculating Integral
	if piecewise == False:
		p_int1 = np.polyint(p1)
		p_int2 = np.polyint(p2)

		int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
		int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
	else:
		lin = np.linspace(min_int, max_int, num=100, retstep=True)
		interval = lin[1]
		samples = lin[0]
		v1 = scipy.interpolate.pchip_interpolate(np.sort(Q1), lR1[np.argsort(Q1)], samples)
		v2 = scipy.interpolate.pchip_interpolate(np.sort(Q2), lR2[np.argsort(Q2)], samples)
		
		# Calculating the integral using the trapezoid method on the samples.
		int1 = np.trapz(v1, dx=interval)
		int2 = np.trapz(v2, dx=interval)

	# Finding average difference
	avg_exp_diff = (int2-int1)/(max_int-min_int)
	avg_diff = (np.exp(avg_exp_diff)-1)*100
	
	return avg_diff


def BD_Quality(R1, Q1, R2, Q2, piecewise=False):
	lR1 = np.log(R1)
	lR2 = np.log(R2)

	Q1 = np.array(Q1)
	Q2 = np.array(Q2)

	p1 = np.polyfit(lR1, Q1, 3)
	p2 = np.polyfit(lR2, Q2, 3)

	# Integration Interval
	min_int = max(min(lR1), min(lR2))
	max_int = min(max(lR1), max(lR2))

	# Calculating Integral
	if piecewise == False:
		p_int1 = np.polyint(p1)
		p_int2 = np.polyint(p2)

		int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
		int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
	else:
		# See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
		lin = np.linspace(min_int, max_int, num=100, retstep=True)
		interval = lin[1]
		samples = lin[0]
		v1 = scipy.interpolate.pchip_interpolate(np.sort(lR1), Q1[np.argsort(lR1)], samples)
		v2 = scipy.interpolate.pchip_interpolate(np.sort(lR2), Q2[np.argsort(lR2)], samples)
		
		# Calculating the integral using the trapezoid method on the samples.
		int1 = np.trapz(v1, dx=interval)
		int2 = np.trapz(v2, dx=interval)

	# Finding average difference
	avg_diff = (int2-int1)/(max_int-min_int)

	return avg_diff