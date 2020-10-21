import numpy as np
import pandas as pd

NUMBER_UNCHANGED_POCKET_RUNS = np.inf
MAXIMUM_TOTAL_STEPS = 100000

ETA = 1

def termination_criteria(runs, steps):
	return runs > NUMBER_UNCHANGED_POCKET_RUNS or steps > MAXIMUM_TOTAL_STEPS

def perceptron(data):

	w = np.zeros(data.shape[1]) ## weight vector should be # features + 1, assume data is x_1,x_2,...,x_p,y, ndim of weights = p + 1
	w_pocket = np.zeros(data.shape[1])
	run = 0
	run_pocket = 0
	total_steps = 0

	while not termination_criteria(run, total_steps):
		
		row_idx = total_steps % data.shape[0]
		row = data[row_idx,:]
		x = np.concatenate(([1], row[:-1]))
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
		else:
			v_greater = v_greater + v_i - u_i
		denom = denom + max(abs(u_i), abs(v_i), abs(u_i - v_i))
	return 1 - np.sqrt(u_greater**2 + v_greater**2)/denom


def kernelized_perceptron(data, kernel):

	alpha = np.zeros(data.shape[1])
	alpha_pocket = np.zeros(data.shape[1])
	run = 0
	run_pocket = 0
	total_steps = 0

	while not termination_criteria(runs, total_steps):

		row_idx = total_steps % data.shape[0]
		row = data[row_idx,:]
		x = row[:-1]
		y = row[-1]

		if y*np.dot(alpha, np.apply_along_axis(lambda x_i: kernel(x_i, x), 1, data[:,:-1])) <= 0:
			if run > run_pocket:
				alpha_pocket = alpha
				run_pocket = run
			alpha[row_idx] = alpha[row_idx] + ETA
			run = 0
		else:
			run = run + 1

		total_steps = total_steps + 1

	if run > run_pocket:
		alpha_pocket = alpha

	return alpha_pocket

def read_data(dataset_name):
	filepath = {"sonar": "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"}
	data = pd.read_csv(filepath[dataset_name], header=None)
	data.iloc[:,-1] = data.iloc[:,-1].replace({"R": -1, "M": 1})
	return data.to_numpy()

def train_test_split(data):
	neg_idxs, = np.where(data[:,-1]==-1)
	pos_idxs, = np.where(data[:,-1]==1)
	np.random.shuffle(neg_idxs)
	np.random.shuffle(pos_idxs)
	neg_idxs_test, neg_idxs_train = np.array_split(neg_idxs, [neg_idxs.shape[0]//5])
	pos_idxs_test, pos_idxs_train = np.array_split(pos_idxs, [pos_idxs.shape[0]//5])
	train = data[np.concatenate((neg_idxs_train, pos_idxs_train)),:]
	test = data[np.concatenate((neg_idxs_test, pos_idxs_test)),:]
	np.random.shuffle(train)
	np.random.shuffle(test)
	return train, test

def get_accuracy(data, weights):
	X = data[:,:-1]
	y = data[:,-1]
	X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
	predictions = np.matmul(X, weights)
	correctly = np.multiply(predictions, y) > 0
	return sum(correctly) / correctly.shape[0]

def main():
	data = read_data("sonar")
	train, test = train_test_split(data)
	weights = perceptron(train)
	print(weights)
	train_accuracy = get_accuracy(train, weights)
	print(round(train_accuracy, 2))
	test_accuracy = get_accuracy(test, weights)
	print(round(test_accuracy, 2))

if __name__ == "__main__":
	main()
