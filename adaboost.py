#################################
# Your name: Ruben Wolhandler
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
import sys
import math
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)


def calc_h(x, h):
    if h[0] == 1:
        if x[int(h[1])]<= h[2]:
            return 1
        return -1
    else:
        if x[int(h[1])]<= int(h[2]):
            return -1
        return 1


def update_D(D, w, h, X_train, y_train,epsilon):
    D = [(D[i] * math.exp(-w*y_train[i]*calc_h(X_train[i],h)))/(2*math.sqrt(epsilon*(1-epsilon))) for i in range(len(X_train))]
    return D


def empError(D, X_train, y_train, h):
    return sum([D[i] for i in range(len(D)) if y_train[i] != calc_h(X_train[i],h)])




def run_adaboost(X_train, y_train, T):
    """
    Returns:

        hypotheses :
            A list of T tuples describing the hypotheses chosen by the algorithm.
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals :
            A list of T float values, which are the alpha values obtained in every
            iteration of the algorithm.
    """
    #initialise Distribution with uniform
    D = np.zeros(len(X_train))
    alphas = []
    hyp = []
    D = [1/len(D) for i in range(len(D))]
    for t in range(T):
        print("iteration number {}".format(t))
        h = bestWL(D,X_train,y_train)
        epsilon = empError(D,X_train, y_train ,h)
        w = (1/2)*np.log((1-epsilon)/epsilon)
        D = update_D(D,w,h,X_train,y_train,epsilon)
        alphas.append(w)
        hyp.append(h)
    return hyp,alphas

##############################################
# You can add more methods here, if needed.
def bestWL_k(k,D,X_train,y_train):
    F_star = sys.maxsize
    for j in range(len(X_train[0])):
        new_train = [(i, X_train[i][j], y_train[i], D[i]) for i in range(len(X_train))]
        new_train.sort(key = lambda tup: tup[1])
        new_train.append((0,new_train[len(new_train)-1][1]+1,0,0))
        sum = 0
        for i in range(len(new_train)-1):
            if new_train[i][2] == k:
                sum += new_train[i][3]
        F = sum
        if F<F_star:
            F_star = F
            theta_star = new_train[0][1]-1
            j_star = j
        for i in range(len(new_train)-1):
            F = F - (k)*new_train[i][2]*new_train[i][3]
            if F < F_star and new_train[i][1]!=new_train[i+1][1]:
                F_star = F
                theta_star = 0.5*(new_train[i][1]+new_train[i+1][1])
                j_star = j
    return k,j_star,theta_star,F_star


def bestWL(D,X_train, y_train):
    h1_pred,h1_index,h1_theta ,F1 = bestWL_k(1,D,X_train, y_train)
    h2_pred,h2_index,h2_theta ,F2 = bestWL_k(-1,D,X_train, y_train)
    if F1 < F2:
        return (h1_pred,h1_index,h1_theta)
    return (h2_pred,h2_index,h2_theta)

##############################################


def error(hypotheses, alpha_vals, X_train, y_train):
    err = []
    for t in range(len(hypotheses)):
        error = 0
        for i in range(len(X_train)):
            pred = prediction(hypotheses[:t+1], alpha_vals[:t+1], X_train[i])
            if pred != y_train[i]:
                error += 1
        err.append(error/len(X_train))
    return err

def prediction(hypotheses, alpha_vals, x):
    g = sum([calc_h(x,hypotheses[i])*alpha_vals[i] for i in range(len(hypotheses))])
    if g >= 0:
        return 1
    else:
        return -1

def error_exp_loss(hypotheses, alpha_vals, X, y):
    return sum([math.exp(
        -y[i] * sum([alpha_vals[j] * calc_h(X[i],hypotheses[j]) for j in range(len(hypotheses))])) for
        i in range(len(X))]) / len(X)

def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data
    T = 80
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    train_error = error(hypotheses,alpha_vals,X_train,y_train)
    test_error = error(hypotheses, alpha_vals,X_test,y_test)
    train_errors_exp_loss = []
    test_errors_exp_loss = []
    for t in range(T):
        train_errors_exp_loss.append(error_exp_loss(hypotheses[:t+1], alpha_vals[:t+1], X_train, y_train))
        test_errors_exp_loss.append(error_exp_loss(hypotheses[:t+1], alpha_vals[:t+1], X_test, y_test))
    x = [i for i in range(T)]
    plt.plot(x,train_error,label = 'train_error')
    plt.plot(x, test_error, label='test_error')
    plt.xlabel('number of iterations t')
    plt.ylabel('error')
    plt.legend()
    plt.show()
    plt.plot(x, train_errors_exp_loss, label = "train_errors_exp_loss")
    plt.plot(x,test_errors_exp_loss , label = "test_errors_exp_loss")
    plt.xlabel('number of iterations t')
    plt.ylabel('error')
    plt.legend()
    plt.show()
    for i in range(10):
        print(
             "hyphothesis is {} Word is {} and his weight is {}".format(hypotheses[i], vocab[hypotheses[i][1]], alpha_vals[i]))


if __name__ == '__main__':
    main()

