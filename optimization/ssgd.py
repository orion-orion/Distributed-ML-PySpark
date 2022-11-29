'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-05-26 21:02:38
LastEditors: ZhangHongYu
LastEditTime: 2022-07-02 11:49:57
'''
from sklearn.datasets import load_breast_cancer
import numpy as np
from pyspark.sql import SparkSession
from operator import add
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import sys
import os

os.environ['PYSPARK_PYTHON'] = sys.executable

n_slices = 4  # Number of Slices
n_iterations = 1500  # Number of iterations
eta = 0.1
mini_batch_fraction = 0.1 # the fraction of mini batch sample 
lam = 0 # coefficient of regular term

def logistic_f(x, w):
    return 1 / (np.exp(-x.dot(w)) + 1)


def gradient(point: np.ndarray, w: np.ndarray):
    """ Compute linear regression gradient for a matrix of data points
    """
    y = point[-1]    # point label
    x = point[:-1]   # point coordinate
    # For each point (x, y), compute gradient function, then sum these up
    return - (y - logistic_f(x, w)) * x


def reg_gradient(w, reg_type="l2", alpha=0):
    """ gradient for reg_term
    """ 
    assert(reg_type in ["none", "l2", "l1", "elastic_net"])
    if reg_type == "none":
        return 0
    elif reg_type == "l2":
        return w
    elif reg_type == "l1":
        return np.sign(w)
    else:
        return alpha * np.sign(w) + (1 - alpha) * w


def draw_acc_plot(accs, n_iterations):
    def ewma_smooth(accs, alpha=0.9):
        s_accs = np.zeros(n_iterations)
        for idx, acc in enumerate(accs):
            if idx == 0:
                s_accs[idx] = acc
            else:
                s_accs[idx] = alpha * s_accs[idx-1] + (1 - alpha) * acc
        return s_accs

    s_accs = ewma_smooth(accs, alpha=0.9)
    plt.plot(np.arange(1, n_iterations + 1), accs, color="C0", alpha=0.3)
    plt.plot(np.arange(1, n_iterations + 1), s_accs, color="C0")
    plt.title(label="Accuracy on test dataset")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.savefig("ssgd_acc_plot.png")


if __name__ == "__main__":

    X, y = load_breast_cancer(return_X_y=True)

    D = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, shuffle=True)
    n_train, n_test = X_train.shape[0], X_test.shape[0]

    spark = SparkSession\
        .builder\
        .appName("SSGD")\
        .getOrCreate()

    matrix = np.concatenate(
        [X_train, np.ones((n_train, 1)), y_train.reshape(-1, 1)], axis=1)

    points = spark.sparkContext.parallelize(matrix, n_slices).cache()

    # Initialize w to a random value
    w = 2 * np.random.ranf(size=D + 1) - 1
    print("Initial w: " + str(w))

    accs = []
    for t in range(n_iterations):
        print("On iteration %d" % (t + 1))
        w_br = spark.sparkContext.broadcast(w)
        
        (g, mini_batch_size) = points.sample(False, mini_batch_fraction, 42 + t)\
            .map(lambda point: gradient(point, w_br.value))\
            .treeAggregate(
                (0.0, 0),\
                    seqOp=lambda res, g: (res[0] + g, res[1] + 1),\
                        combOp=lambda res_1, res_2: (res_1[0] + res_2[0], res_1[1] + res_2[1])
            )

        w -= eta * (g/mini_batch_size + lam * reg_gradient(w, "l2"))
        
        y_pred = logistic_f(np.concatenate(
            [X_test, np.ones((n_test, 1))], axis=1), w)
        pred_label = np.where(y_pred < 0.5, 0, 1)
        acc = accuracy_score(y_test, pred_label)
        accs.append(acc)
        print("iterations: %d, accuracy: %f" % (t, acc))

    print("Final w: %s " % w)
    print("Final acc: %f" % acc)

    spark.stop()

    draw_acc_plot(accs, n_iterations)


# Final w: [ 3.58216967e+01  4.53599397e+01  2.07040135e+02  8.52414269e+01
#   4.33038042e-01 -2.93986236e-01  1.43286366e-01 -2.95961229e-01
#  -7.63362321e-02 -3.93180625e-01  8.19325971e-01  3.30881477e+00
#  -3.25867503e+00 -1.24769634e+02 -8.52691792e-01 -5.18037887e-01
#  -1.34380402e-01 -7.49316038e-01 -8.76722455e-01  9.23748261e-01
#   3.81531205e+01  5.56880612e+01  2.04895002e+02 -1.17586430e+02
#   8.92355523e-01 -9.40611324e-01 -9.24082612e-01 -1.16210791e+00
#   7.10117706e-01 -7.62921434e-02  4.48389687e+00] 
# Final acc: 0.929825