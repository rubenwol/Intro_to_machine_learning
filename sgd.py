#################################
# Your name:Ruben Wolhandler
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt
import math

"""
Assignment 3 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper_hinge():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def helper_ce():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:8000] != 'a'))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[8000:10000] != 'a'))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = labels[train_idx[:6000]]

    validation_data_unscaled = data[train_idx[6000:8000], :].astype(float)
    validation_labels = labels[train_idx[6000:8000]]

    test_data_unscaled = data[8000 + test_idx, :].astype(float)
    test_labels = labels[8000 + test_idx]

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
	Implements Hinge loss using SGD.
	"""
    w = np.array([0 for i in range(784)])  # initialisation of the w vector
    for t in range(1, T + 1):
        i = np.random.randint(len(data))
        if (np.dot(labels[i] * w, data[i]) < 1):
            w = (1 - eta_0 / t) * w + (eta_0 / t) * C * labels[i] * data[i]
        else:
            w = (1 - eta_0 / t) * w
    return w


def calc_accuracy(w, data, labels):
    errors = 0
    for i in range(len(data)):
        if np.sign(np.dot(w, data[i])) != labels[i]:
            errors += 1
    return (len(data) - errors) / len(data)


def predict_ce(v1, w_arr):
    z = [np.dot(v1,w_arr[j]) for j in range(10)]
    return np.argmax(z)


def SGD_ce(data, labels, eta_0, T):
    """Implements multi-class cross entropy loss using SGD."""
    data = sklearn.preprocessing.normalize(data)
    w = np.zeros((10,784))
    for t in range(1, T + 1):
        i = np.random.randint(len(data))
        w = w - (eta_0) * grad(w, data[i], labels[i])
    return w


# question 1a
train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
etas = np.arange(-5, 6)
eta_acc = []
for eta in etas:
    accuracy = 0
    for i in range(10):
        w = SGD_hinge(train_data, train_labels, 1, 10.0 ** eta, 1000)
        accuracy += calc_accuracy(w, validation_data, validation_labels) / 10
    eta_acc.append(accuracy)
plt.plot(etas, eta_acc, 'x-')
plt.xlabel('10 power:')
plt.ylabel('accuracy_avg')
plt.title('average accuracy for each eta')
plt.show()

# question 1b
C_arr = np.arange(-5, 6)
C_acc = []
for C in C_arr:
    accuracy = 0
    for i in range(10):
        w = SGD_hinge(train_data, train_labels, 10.0 ** C, 1, 1000)
        accuracy += calc_accuracy(w, validation_data, validation_labels) / 10
    C_acc.append(accuracy)
plt.plot(C_arr, C_acc, 'x-')
plt.xlabel('10 power:')
plt.ylabel('accuracy_avg')
plt.title('average accuracy for each C')
plt.show()

# question 1c
w = SGD_hinge(train_data, train_labels, 10.0 ** (-4), 1, 20000)
plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
plt.show()

# question 1d
print("the accuracy of the best classifier on the test set is :" + str(
calc_accuracy(w, test_data, test_labels)))

def softmax(z):
    e_z = np.exp(z)
    return e_z / e_z.sum(axis=0)


def grad(w, x, y):
    wt = np.zeros((10, len(x)))
    z = [np.dot(w[i],x) for i in range(10)]
    soft = softmax(z)
    for i in range(10):
        if i == int(y):
            wt[i] = soft[i] * x - x
        else:
            wt[i] = soft[i] * x
    return wt


def calc_accuracy_ce(w_arr, data, labels):
    errors = 0
    for i in range(len(data)):
        if int(labels[i]) != int(predict_ce(data[i], w_arr)):
            errors += 1
    return (len(data) - errors) / len(data)


# question 2a
train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_ce()
etas = np.arange(-1, 1, 0.1)
eta_acc = []
for eta in etas:
    accuracy = 0
    for i in range(10):
        w = SGD_ce(train_data, train_labels, 10.0 ** eta, 100)
        accuracy += calc_accuracy_ce(w, validation_data, validation_labels) / 10
    eta_acc.append(accuracy)
plt.plot(etas, eta_acc, 'x-')
plt.xlabel('10 power:')
plt.ylabel('accuracy_avg')
plt.title('average accuracy for eta, Multi class')
plt.show()

# question 2b
w = SGD_ce(train_data, train_labels, 10**0.50, 20000)
for k in range(10):
    plt.imshow(np.reshape(w[k], (28, 28)), interpolation='nearest')
    plt.title('the image of the w['+str(k)+']')
    plt.show()

#question 2c
print("the accuracy of the best classifier on the test set is :" + str(calc_accuracy_ce(w, test_data, test_labels)))