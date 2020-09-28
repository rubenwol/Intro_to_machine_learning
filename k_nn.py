
#ruben wolhandler
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml


def k_nn(data, labels, image, k):
    dist = np.linalg.norm(data - image, axis=1)
    _, n_labels = zip(*sorted(zip(dist, labels)))
    count_sort = kcount_sort(n_labels, k)
   # print(count_sort.index(max(count_sort)))
    return count_sort.index(max(count_sort))


# list of integers between 0 and 9
def kcount_sort(l, k):
    array = [0 for i in range(10)]
    for i in range(0, k):
        array[int(l[i])] += 1
    return array

def accuracy_knn(k,n):
    error = 0
    for i in range(1000):
        labels = int(test_labels[i])
        knnLabels = k_nn(train[:n,:],train_labels[:n],test[i],k)
        if labels!= knnLabels:
            error += 1
    if error == 0:
     #   print("the accuracy knn is: " + str(1))
        return 1
    #print("the accuracy knn is: " + str(1-(error/n)))
    return 1-(error/n)


mnist = fetch_openml('mnist_784')
data = mnist['data']
labels = mnist['target']

idx = np.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]].astype(int)
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]].astype(int)


mnist = fetch_openml('mnist_784')
data = mnist['data']
labels = mnist['target']

idx = np.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

#Question b
print("the accuracy knn is: " + str(accuracy_knn(10,1000)))

#Question c
x = [i for i in range(1,100)]
y = [accuracy_knn(i,1000) for i in range(1,100)]
plt.plot(x,y)
plt.show()

#Question d
x = [i for i in range(100, 5001,100)]
y = [accuracy_knn(1,i) for i in range(100,5001,100)]

plt.plot(x,y)
plt.show()

