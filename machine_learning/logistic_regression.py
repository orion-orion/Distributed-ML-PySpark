'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-05-26 21:02:38
LastEditors: ZhangHongYu 
LastEditTime: 2022-07-01 16:22:53
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
eta = 0.1  # iteration step_size

def logistic_f(x, w):
    return 1 / (np.exp(-x.dot(w)) + 1)


def gradient(point: np.ndarray, w: np.ndarray) -> np.ndarray:
    """ Compute linear regression gradient for a matrix of data points
    """
    y = point[-1]    # point label
    x = point[:-1]   # point coordinate
    # For each point (x, y), compute gradient function, then sum these up
    return - (y - logistic_f(x, w)) * x

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
    plt.savefig("logistic_regression_acc_plot.png")


if __name__ == "__main__":

    X, y = load_breast_cancer(return_X_y=True)

    D = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    n_train, n_test = X_train.shape[0], X_test.shape[0]

    spark = SparkSession\
        .builder\
        .appName("Logistic Regression")\
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
        
        # g = points.map(lambda point: gradient(point, w)).reduce(add)
        # g = points.map(lambda point: gradient(point, w_br.value)).reduce(add)
        g = points.map(lambda point: gradient(point, w_br.value))\
            .treeAggregate(0.0, add, add)

        w -= eta * g
        
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

# Final w: [ 1.16200213e+04  1.30671054e+04  6.53960395e+04  2.13003287e+04
#   8.92852998e+01 -1.09553416e+02 -2.98667851e+02 -1.26433988e+02
#   1.59947852e+02  7.85600857e+01 -3.90622568e+01  8.09490631e+02
#  -1.29356637e+03 -4.02060982e+04  4.22124893e+00 -2.30863864e+01
#  -4.22144623e+01 -9.06373487e+00  1.16047444e+01  9.14892224e-01
#   1.25920286e+04  1.53120086e+04  6.48615769e+04 -3.23661608e+04
#   1.00625479e+02 -3.98123440e+02 -6.89846039e+02 -1.77214836e+02
#   1.95991193e+02  5.96495248e+01  1.53245784e+03] 
# Final acc: 0.941520

