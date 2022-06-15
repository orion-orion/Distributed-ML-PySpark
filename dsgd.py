'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-05-26 21:02:38
LastEditors: ZhangHongYu
LastEditTime: 2022-06-15 20:53:34
'''
from sklearn.datasets import load_breast_cancer
import numpy as np
from pyspark.sql import SparkSession
from operator import add
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

n_slices = 3  # Number of Slices
n_iterations = 300  # Number of iterations
alpha = 10  # iteration step_size, because gradient sum is divided by minibatch size, it shoulder be larger
mini_batch_fraction = 0.1 # the fraction of mini batch sample 

def logistic_f(x, w):
    return 1 / (np.exp(-x.dot(w)) + 1)


def gradient(point: np.ndarray, w: np.ndarray):
    """ Compute linear regression gradient for a matrix of data points
    """
    y = point[-1]    # point label
    x = point[:-1]   # point coordinate
    # For each point (x, y), compute gradient function, then sum these up
    # notice thet we need to compute minibatch size, so return(g, 1)
    return - (y - logistic_f(x, w)) * x


if __name__ == "__main__":

    X, y = load_breast_cancer(return_X_y=True)

    D = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
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

    
    for t in range(n_iterations):
        w_br = spark.sparkContext.broadcast(w)
        print("On iteration %d" % (t + 1))

        (g, mini_batch_size) = points.sample(False, mini_batch_fraction, 42 + t)\
            .map(lambda point: gradient(point, w_br.value))\
            .treeAggregate(
                (0.0, 0),\
                    seqOp=lambda res, g: (res[0] + g, res[1] + 1),\
                        combOp=lambda res_1, res_2: (res_1[0] + res_2[0], res_1[1] + res_2[1])
            )
        print(g, mini_batch_size)
        w -= alpha * g/mini_batch_size
        
        y_pred = logistic_f(np.concatenate(
            [X_test, np.ones((n_test, 1))], axis=1), w)
        pred_label = np.where(y_pred < 0.5, 0, 1)
        acc = accuracy_score(y_test, pred_label)
        print("iterations: %d, accuracy: %f" % (t, acc))

    print("Final w: %s " % w)
    print("Final acc: %f" % acc)

    spark.stop()

# Final w: [ 2.47106772e+03  4.53539140e+03  1.49658271e+04  1.49085211e+04
#   2.49189579e+01  3.86567135e+00 -2.10764497e+01 -9.70220892e+00
#   4.66690447e+01  1.98304608e+01  5.39929228e-03  3.18391480e+02
#  -3.33336179e+01 -6.97806573e+03  1.85282484e+00  2.47784246e+00
#   2.90481836e+00  1.34441817e+00  4.78931649e+00 -4.90105658e-02
#   2.46075335e+03  5.85553211e+03  1.47565943e+04 -1.50819870e+04
#   3.35107010e+01  4.44383010e+00 -2.74355495e+01 -5.50744484e+00
#   6.77728432e+01  2.07616117e+01  3.09770050e+02] 
# Final acc: 0.912281