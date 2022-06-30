'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-05-26 21:02:38
LastEditors: ZhangHongYu
LastEditTime: 2022-06-29 20:20:00
'''
from sklearn.datasets import load_breast_cancer
import numpy as np
from pyspark.sql import SparkSession
from operator import add
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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
    plt.savefig("dsgd_acc_plot.png")


if __name__ == "__main__":

    X, y = load_breast_cancer(return_X_y=True)

    D = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, shuffle=True)
    n_train, n_test = X_train.shape[0], X_test.shape[0]

    spark = SparkSession\
        .builder\
        .appName("SGD")\
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

        w -= eta * g/mini_batch_size + lam * reg_gradient(w, "l2")
        
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


# Final w: [ 2.47050494e+03  4.50175117e+03  1.49694229e+04  1.48496465e+04
#   2.39865542e+01  5.39857896e+00 -2.14847453e+01 -9.38178616e+00
#   4.71167820e+01  1.98715626e+01  1.20180738e+00  3.26319136e+02
#  -1.30502268e+01 -6.92270842e+03  2.83258511e+00  1.70759297e+00
#   2.18407260e+00  1.93251469e+00  6.01895680e+00  2.62176741e-01
#   2.45764325e+03  5.82117047e+03  1.47596541e+04 -1.50274740e+04
#   3.33873472e+01  4.36492228e+00 -2.75599841e+01 -6.11206766e+00
#   6.85433184e+01  2.01329301e+01  3.09835463e+02] 
# Final acc: 0.912281