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
		
		row_idx = total_steps % data.shape[0]
		row = data[row_idx,:]
		x = row[:-1]
		y = row[-1]

		if y*np.dot(w, x) <= 0: # if misclassified
			if run > run_pocket: # if this is the most in a row we've seen
				w_pocket = w # update weights
				run_pocket = run # update number of best in a row
			w = w + ETA*y*x # adjust current weight vector
			run = 0 # reset run count
		else: # if correctly classified just increment counter and move to next example
			run = run + 1

		total_steps = total_steps + 1

	if run > run_pocket:
		w_pocket = w

	return w_pocket

def polynomial_deg2_kernel(u, v):
	return (np.dot(u, v) + 1)**2

def cosine_kernel(u, v):
	return np.dot(u, v)/np.linalg.norm(u)/np.linalg.norm(v)

def yangs_putative_deg2_kernel(u, v):
	u_greater = 0
	v_greater = 0
	denom = 0
	for u_i, v_i in zip(u, v):
		if u_i > v_i:
			u_greater = u_greater + u_i - v_i
			if 
		else:
			v_greater = v_greater + v_i - u_i
		denom = denom + max(abs(u_i), abs(v_i), abs(u_i - v_i))
	return 1 - np.sqrt(u_greater**2 + v_greater**2)/denom


def kernelized_perceptron(data, kernel):

	alpha = np.zeros()
	alpha_pocket = np.zeros()
	run = 0
	run_pocket = 0
	total_steps = 0

	while termination_criteria(runs, total_steps):

		row_idx = total_steps % data.shape[0]
		row = data[row_idx,:]
		x = row[:-1]
		y = row[-1]

		if np.dot(np.dot(alpha, y), np.apply_along_axis(lambda x_i: kernel(x_i, x), 1, data[:,:-1])) <= 0:
			if run > run_pocket:
				alpha_pocket = alpha
				run_pocket = run
			alpha[i] = alpha[i] + 1
			run = 0
		else:
			run = run + 1

		total_steps = total_steps + 1

	if run > run_pocket:
		alpha_pocket = alpha

	return alpha_pocket
