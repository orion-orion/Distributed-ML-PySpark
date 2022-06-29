'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-05-26 21:02:38
LastEditors: ZhangHongYu
LastEditTime: 2022-06-29 21:40:21
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
n_iterations = 300  # Number of iterations 300
eta = 0.1
mini_batch_fraction = 0.1 # the fraction of mini batch sample 
n_local_iterations = 5 # the number local epochs 5
mu = 0.5
zeta = 0.5

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


def update_local_w(iter):
    iter = list(iter)
    idx, (w, _) = iter[0]
    g_mean = np.mean(np.array([ g for _, (_, g) in iter]), axis=0) 
    return  [(idx, w - eta * g_mean)]


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
    plt.savefig("bmuf_acc_plot.png")


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
    
    # weight update
    delta_w = 2 * np.random.ranf(size=D + 1) - 1
        
    accs = []
    for t in range(n_iterations):
        print("On iteration %d" % (t + 1))
        w_br = spark.sparkContext.broadcast(w)
        ws = ws.mapPartitions(lambda iter: [(iter[0][0], w_br.value)])
                            
        for local_t in range(n_local_iterations):
            ws = points.sample(False, mini_batch_fraction, 42 + t)\
                .join(ws, numPartitions=n_slices)\
                    .map(lambda pt_w: gradient(pt_w))\
                        .mapPartitions(update_local_w) 
            
        par_w_sum = ws.mapPartitions(lambda iter: [iter[0][1]]).treeAggregate(0.0, add, add)           
  
        w_avg  = par_w_sum / n_slices 

        delta_w = mu * delta_w + zeta * (w_avg - w)
        w = w + delta_w
        
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


# Final w: [ 3.58817936e+01  5.16085051e+01  2.06738276e+02  1.01696450e+02
#   4.16141169e-01  5.91462408e-01 -1.29360142e+00 -1.93766314e-01
#   4.85939825e-01  3.60555394e-01 -4.84201024e-01  4.04012280e+00
#  -1.91641993e+00 -1.25408757e+02 -1.27251919e+00 -1.11473622e+00
#  -2.51673394e-01 -5.11414585e-01 -6.31052871e-01  5.44476435e-01
#   3.65063589e+01  6.46285082e+01  2.09258512e+02 -1.32287680e+02
#  -4.05273314e-01  4.24595449e-01 -5.97314646e-01 -1.32357538e+00
#   6.60142707e-01  6.43625602e-01  6.50917716e+00] 
# Final acc: 0.918129
