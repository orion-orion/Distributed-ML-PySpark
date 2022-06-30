'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-05-26 21:02:38
LastEditors: ZhangHongYu
LastEditTime: 2022-06-30 16:22:29
'''
from functools import reduce
from typing import Tuple
from sklearn.datasets import load_breast_cancer
import numpy as np
from pyspark.sql import SparkSession
from operator import add
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

n_slices = 4  # Number of Slices
n_iterations = 1500  # Number of iterations 300
eta = 0.1
mini_batch_fraction = 0.1 # the fraction of mini batch sample 
rho = 0.1 # penalty constraint coefficient
alpha = eta * rho # iterative constraint coefficient
beta = n_slices * alpha # the parameter of history information

def logistic_f(x, w):
    return 1 / (np.exp(-x.dot(w)) + 1 +1e-6)


def gradient(pt_w: Tuple):
    """ Compute linear regression gradient for a matrix of data points
    """
    idx, (point, w) = pt_w
    y = point[-1]    # point label
    x = point[:-1]   # point coordinate
    # For each point (x, y), compute gradient function, then sum these up
    return  (idx, (w, - (y - logistic_f(x, w)) * x))


def update_local_w(iter, w):
    iter = list(iter)
    idx, (local_w, _) = iter[0]
    g_mean = np.mean(np.array([ g for _, (_, g) in iter]), axis=0) 
    return  [(idx, local_w - eta * g_mean - alpha * (local_w - w))]


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
    plt.savefig("easgd_acc_plot2.png")


if __name__ == "__main__":

    X, y = load_breast_cancer(return_X_y=True)

    D = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, shuffle=True)
    n_train, n_test = X_train.shape[0], X_test.shape[0]

    spark = SparkSession\
        .builder\
        .appName("Model Average")\
        .getOrCreate()

    matrix = np.concatenate(
        [X_train, np.ones((n_train, 1)), y_train.reshape(-1, 1)], axis=1)

    points = spark.sparkContext.parallelize(matrix, n_slices).cache()
    points = points.mapPartitionsWithIndex(lambda idx, iter: [ (idx, arr) for arr in iter])

    ws = spark.sparkContext.parallelize(2 * np.random.ranf(size=(n_slices, D + 1)) - 1, n_slices).cache()
    ws = ws.mapPartitionsWithIndex(lambda idx, iter: [(idx, next(iter))])

    w = 2 * np.random.ranf(size=D + 1) - 1
    print("Initial w: " + str(w))
    
    accs = []
    for t in range(n_iterations):
        print("On iteration %d" % (t + 1))
        w_br = spark.sparkContext.broadcast(w)
                            
        ws = points.sample(False, mini_batch_fraction, 42 + t)\
            .join(ws, numPartitions=n_slices)\
                .map(lambda pt_w: gradient(pt_w))\
                    .mapPartitions(lambda iter: update_local_w(iter, w=w_br.value)) 
            
        par_w_sum = ws.mapPartitions(lambda iter: [iter[0][1]]).treeAggregate(0.0, add, add)           
  
        w  = (1 - beta) * w + beta * par_w_sum / n_slices 

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


# Final w: [ 4.75633325e+01  7.05407657e+01  2.79006876e+02  1.45465411e+02
#   4.54467492e-01 -2.10142380e-01 -6.30421903e-01  4.53977048e-01
#   1.01717057e-01 -2.14420411e-01 -2.94989128e-01  4.89426514e+00
#  -3.05999725e+00 -1.62456459e+02  1.27772367e-01 -4.68403685e-02
#  -8.63345165e-03  2.15800745e-01  5.77719463e-01 -1.83278567e-02
#   5.01647916e+01  8.80774672e+01  2.79145194e+02 -1.81621547e+02
#   2.14490664e-01 -8.83817758e-01 -1.43244912e+00 -5.96750910e-01
#   1.04627441e+00  4.37109225e-01  6.04818129e+00] 
# Final acc: 0.929825
