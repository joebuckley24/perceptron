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

	alpha = np.zeros(data.shape[0])
	alpha_pocket = np.zeros(data.shape[0])
	b = 0
	b_pocket = 0

	run = 0
	run_pocket = 0
	total_steps = 0

	while not termination_criteria(run, total_steps):

		row_idx = total_steps % data.shape[0]
		row = data[row_idx,:]
		x = row[:-1]
		y = row[-1]

		if y*np.dot(alpha, np.apply_along_axis(lambda x_i: kernel(x_i, x), 1, data[:,:-1])) + b <= 0:
			if run > run_pocket:
				alpha_pocket = alpha
				run_pocket = run
				b_pocket = b
			alpha[row_idx] = alpha[row_idx] + ETA
			b = b + y*ETA
			run = 0
		else:
			run = run + 1

		total_steps = total_steps + 1

	if run > run_pocket:
		alpha_pocket = alpha
		b_pocket = b

	return alpha_pocket, b_pocket

def read_data(name):
	info = {
		"sonar": {
			"filepath": "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data", 
			"labels": {"R": -1, "M": 1}
		},
		"cleveland": {
			"filepath": "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", 
			"labels": {0: -1, 1: 1, 2: 1, 3: 1, 4: 1}
		},
		"transfusion": {
			"filepath": "https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data",
			"labels": {1: 1, 0: -1}
		}
	} 
	#above is data structure (dict of dicts) to hold info for each dataset: 
	#"sonar" detecting rock or metal cylinder with sonar, 
	#"cleveland" detecting heart disease, 
	#"transfusion" detect whether donor from database donated blood March 2007
	data = pd.read_csv(info[name]["filepath"], header="infer" if name == "transfusion" else None, sep=",", na_values = "?")
	data.iloc[:,-1] = data.iloc[:,-1].replace(info[name]["labels"]) #converts y column (last column) to be in {-1, 1} 
	data = data.to_numpy() #pandas DataFrame to numpy ndarray
	return data[~np.isnan(data).any(axis=1)] #remove rows with missing values

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

def get_accuracy_kernelized(train_data, test_data, alpha, b, kernel):
	train_X = train_data[:,:-1]
	train_y = train_data[:,-1]
	test_X = test_data[:,:-1]
	test_y = test_data[:,-1]
	point_weights = alpha*train_y
	correct = 0
	for i, y in enumerate(test_y):
		if y*np.dot(point_weights, np.apply_along_axis(lambda x_i: kernel(x_i, test_X[i]), 1, train_X)) + b <= 0:
			correct = correct + 1
	return correct/test_y.shape[0]


def main():

	for dataset in ["transfusion"]:#["sonar", "cleveland", "transfusion"]:

		print("")
		print("##################")
		print("DATASET = " + dataset)
		print("##################")
		print("")
		
		data = read_data(dataset)
		train, test = train_test_split(data)

		print("perceptron:")
		weights = perceptron(train)
		print(weights)

		train_accuracy = get_accuracy(train, weights)
		test_accuracy = get_accuracy(test, weights)
		print("train accuracy: " + str(round(train_accuracy, 2)))
		print("test accuracy: " + str(round(test_accuracy, 2)))

		print("\nkernelized_perceptron:")
		alpha, bias = kernelized_perceptron(train, np.dot)

		train_accuracy_kernelized = get_accuracy_kernelized(train, train, alpha, bias, np.dot)
		test_accuracy_kernelized = get_accuracy_kernelized(train, test, alpha, bias, np.dot)
		print("train accuracy: " + str(round(train_accuracy_kernelized, 2)))
		print("test accuracy: " + str(round(test_accuracy_kernelized, 2)))



if __name__ == "__main__":
	main()
