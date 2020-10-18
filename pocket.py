import numpy as np
# import pandas as pd

NUMBER_UNCHANGED_POCKET_RUNS = np.inf
MAXIMUM_TOTAL_STEPS = 10000

ETA = 1

def termination_criteria(runs, steps):
	return runs > NUMBER_UNCHANGED_POCKET_RUNS or steps > MAXIMUM_TOTAL_STEPS

def perceptron(data):

	w = np.zeros(data.shape[1]) ## weight vector should be # features + 1, assume data is x_1,x_2,...,x_p,y, ndim of weights = p + 1
	w_pocket = np.zeros(data.shape[1])
	run = 0
	run_pocket = 0
	total_steps = 0

	while termination_criteria(runs, total_steps):
		
		row = data[total_steps % data.shape[0],:]
		x = row[:-1]
		y = row[-1]

		if y*np.dot(w, x) <= 0:
			if run > run_pocket:
				w_pocket = w
				run_pocket = run
			w = w + ETA*y*x
			run = 0
		else:
			run = run + 1

		total_steps = total_steps + 1
		
	if run > run_pocket:
		w_pocket = w

	return w_pocket