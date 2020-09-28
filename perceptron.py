#################################
# Your name: Ruben Wolhandler
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import matplotlib.pyplot as plt
import numpy.random
import sklearn
from sklearn.datasets import fetch_openml
from sklearn import preprocessing
from tabulate import tabulate


"""
Assignment 3 question 1 skeleton.

Please use the provided function signature for the perceptron implementation.
Feel free to add functions and other code, and submit this file with the name perceptron.py
"""

def helper():
	mnist = fetch_openml('mnist_784')
	data = mnist['data']
	labels = mnist['target']

	neg, pos = "0", "8"
	train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
	test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

	train_data_unscaled = data[train_idx[:6000], :].astype(float)
	train_labels = (labels[train_idx[:6000]] == pos)*2-1

	validation_data_unscaled = data[train_idx[6000:], :].astype(float)
	validation_labels = (labels[train_idx[6000:]] == pos)*2-1

	test_data_unscaled = data[60000+test_idx, :].astype(float)
	test_labels = (labels[60000+test_idx] == pos)*2-1

	# Preprocessing
	train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
	validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
	test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
	return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

def perceptron(data, labels):
	"""
	returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the perceptron classifier
    """
	w = np.array([0 for i in range(784)]) #initialisation of the w vector
	data = preprocessing.normalize(data) #normalize the data

	for i in range(len(data)):
		if np.sign(np.dot(w,data[i])) != labels[i]: #update w if needed
			w = w + labels[i]*data[i]
	return w


def calc_accuracy(w,data,labels):
	errors = 0
	for i in range(len(data)):
		if np.sign(np.dot(w,data[i])) != labels[i]:
			errors += 1
	return (len(data)- errors)/len(data)


#################################
train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()

#Question a
samplesNumber = [5, 10, 50, 100, 500, 1000, 5000]
accuracy_mean = []
percentile_5 = []
percentile_95 = []
train_data_labels = np.column_stack((train_data,train_labels))
for n in samplesNumber:
	accuracy_n = []
	accuracy_sum = 0
	for i in range(100):
		np.random.shuffle(train_data_labels)
		w = perceptron([train_data_labels[i][:-1] for i in range(n)], [train_data_labels[i][-1] for i in range(n)])
		accuracy = calc_accuracy(w,train_data,train_labels)
		accuracy_sum += accuracy
		accuracy_n.append(accuracy)
	percentile_5.append(np.percentile(accuracy_n,5))
	percentile_95.append(np.percentile(accuracy_n,95))
	accuracy_mean.append(accuracy_sum/100)
headers = ["n","Accuracy mean","5% Percentile","95% percentile"]
table = [[samplesNumber[i],accuracy_mean[i],percentile_5[i],percentile_95[i]] for i in range(len(samplesNumber))]
print(tabulate(table,headers=headers,tablefmt='grid'))

#Question b
w = perceptron(train_data,train_labels)
plt.imshow(np.reshape(w,(28,28)),interpolation = 'nearest')
plt.show()

#question c
w = perceptron(train_data,train_labels)
print("the accuracy of the classifier trained on the full training set applied on the test set is:\n" + str(calc_accuracy(w,test_data,test_labels)))

#question d
j = 0
for i in range(len(test_data)):
	if np.sign(np.dot(w, test_data[i])) != test_labels[i]:
		plt.imshow(np.reshape(test_data[i], (28, 28)), interpolation='nearest')
		if np.sign(np.dot(w, test_data[i])) == 1:
			plt.title("False prediction: prediction is 8 but labels is 0")
		else:
			plt.title("False prediction: prediction is 0 but labels is 8")

		plt.show()


#################################

