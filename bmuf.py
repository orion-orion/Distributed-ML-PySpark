'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-05-26 21:02:38
LastEditors: ZhangHongYu
LastEditTime: 2022-06-29 21:07:15
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


# Final w: [ 3.56746113e+01  5.23773783e+01  2.10458066e+02  1.13411434e+02
#   9.03684544e-01 -1.64116119e-01 -8.50221455e-01 -1.02747339e-01
#   1.01249316e+00  5.86541201e-01 -7.95885004e-01  4.08388583e+00
#  -3.72622203e+00 -1.22789161e+02  7.15869286e-01 -9.59608820e-01
#   5.85750752e-01  9.52410298e-01 -4.74590169e-01  8.01080392e-01
#   3.72102291e+01  6.54164219e+01  2.11443556e+02 -1.34266020e+02
#   1.19299917e+00 -9.48551465e-01 -3.02582118e-01 -9.50371027e-01
#   1.30280295e+00  5.02509358e-01  5.34567937e+00] 
# Final acc: 0.923977