'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-05-26 21:02:38
LastEditors: ZhangHongYu
LastEditTime: 2022-05-27 16:34:41
'''
from sklearn.datasets import load_breast_cancer
import numpy as np
from pyspark.sql import SparkSession
from operator import add
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

numSlices = 3 # Number of Slices
iterations = 300 # Number of iterations
alpha = 0.01 # iteration step_size

def logistic_f(x, w):
    return 1 / (np.exp(-x.dot(w)) + 1)

# Compute linear regression gradient for a matrix of data points
def gradient(point: np.ndarray, w: np.ndarray) -> np.ndarray:
    y = point[-1]    # point label
    x = point[:-1]   # point coordinate
    # For each point (x, y), compute gradient function, then sum these up
    return  - (y - logistic_f(x, w)) * x

if __name__ == "__main__":

    X, y = load_breast_cancer(return_X_y=True)

    D = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    n_train, n_test = X_train.shape[0], X_test.shape[0]

    spark = SparkSession\
        .builder\
        .appName("Logistic Regression")\
        .getOrCreate()

    matrix = np.concatenate([X_train, np.ones((n_train, 1)), y_train.reshape(-1, 1)], axis=1)

    points = spark.sparkContext.parallelize(matrix, numSlices).cache()


    # Initialize w to a random value
    w = 2 * np.random.ranf(size = D + 1) - 1
    print("Initial w: " + str(w))

    for i in range(iterations):
        print("On iteration %i" % (i + 1))
        g = points.map(lambda point: gradient(point, w)).reduce(add)
        w -= alpha * g

        y_pred = logistic_f(np.concatenate([X_test, np.ones((n_test, 1))], axis=1), w)
        pred_label = np.where(y_pred < 0.5, 0, 1)
        acc = accuracy_score(y_test, pred_label)
        print("iterations: %d, accuracy: %f" % (i, acc))


    print("Final w: %s " % w)
    print("Final acc: %f" % acc)
    
    spark.stop()
    
# Final w: [ 8.24963374e+02  1.49061653e+03  4.99129841e+03  4.41557347e+03
#   8.61406180e+00  6.75207334e-01 -6.42839586e+00 -3.56389089e+00
#   1.62278467e+01  5.52493855e+00  2.35922112e+00  1.10248851e+02
#  -6.27003990e+00 -2.30316915e+03  1.11515026e+00  1.83001336e-01
#   6.53404674e-01  8.31347239e-01  2.73936848e+00 -6.85365798e-01
#   8.33326788e+02  1.91290094e+03  4.96153939e+03 -4.67970066e+03
#   1.14797829e+01  8.62281501e-01 -1.07984126e+01 -3.03795729e+00
#   2.28165961e+01  7.60731769e+00  1.03868750e+02] 
# Final acc: 0.929825