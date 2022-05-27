'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-05-26 21:02:38
LastEditors: ZhangHongYu
LastEditTime: 2022-05-27 16:43:20
'''
from sklearn.datasets import load_breast_cancer
import numpy as np
from pyspark.sql import SparkSession
from operator import add
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

n_slices = 3  # Number of Slices
n_iterations = 300  # Number of iterations
alpha = 0.01  # iteration step_size


def logistic_f(x, w):
    return 1 / (np.exp(-x.dot(w)) + 1)


def gradient(point: np.ndarray, w: np.ndarray) -> np.ndarray:
    """ Compute linear regression gradient for a matrix of data points
    """
    y = point[-1]    # point label
    x = point[:-1]   # point coordinate
    # For each point (x, y), compute gradient function, then sum these up
    return - (y - logistic_f(x, w)) * x


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

    for t in range(n_iterations):
        print("On iteration %d" % (t + 1))
        g = points.map(lambda point: gradient(point, w)).reduce(add)
        w -= alpha * g

        y_pred = logistic_f(np.concatenate(
            [X_test, np.ones((n_test, 1))], axis=1), w)
        pred_label = np.where(y_pred < 0.5, 0, 1)
        acc = accuracy_score(y_test, pred_label)
        print("iterations: %d, accuracy: %f" % (t, acc))

    print("Final w: %s " % w)
    print("Final acc: %f" % acc)

    spark.stop()

# Final w: [ 8.24868738e+02  1.49115758e+03  4.98938172e+03  4.41818270e+03
#   8.76940491e+00  6.86876222e-01 -6.73606185e+00 -3.62177163e+00
#   1.66678293e+01  7.01869229e+00  1.28927662e+00  1.10511838e+02
#  -5.71687775e+00 -2.30300514e+03  4.34909364e-01  1.14213057e+00
#   4.83840944e-01 -4.46209845e-01  2.73957723e+00 -6.58003032e-01
#   8.33382242e+02  1.91120121e+03  4.95887133e+03 -4.68168721e+03
#   1.09020526e+01  2.41567929e-02 -1.13417669e+01 -1.55230036e+00
#   2.35182245e+01  6.32931729e+00  1.03941431e+02]
# Final acc: 0.929825
