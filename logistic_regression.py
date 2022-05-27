'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-05-26 21:02:38
LastEditors: ZhangHongYu
LastEditTime: 2022-05-27 11:27:46
'''
from sklearn.datasets import load_breast_cancer
import numpy as np
from pyspark.sql import SparkSession
from operator import add
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

numSlices = 3 # Number of Slices
iterations = 400 # Number of iterations
step_size = 0.01 # iteration step

def logistic_f(x, w):
    return 1 / (np.exp(-x.dot(w)) + 1)

# Compute linear regression gradient for a matrix of data points
def gradient(point: np.ndarray, w: np.ndarray) -> np.ndarray:
    y = point[-1]    # point label
    x = point[:-1]   # point coordinate
    # For each point (x, y), compute gradient function, then sum these up
    return  - x * (y - logistic_f(x, w))

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
        grad = points.map(lambda point: gradient(point, w)).reduce(add)
        w -= step_size * grad / n_train

        y_pred = logistic_f(np.concatenate([X_test, np.ones((n_test, 1))], axis=1), w)
        pred_label = np.where(y_pred < 0.5, 0, 1)
        acc = accuracy_score(y_test, pred_label)
        print("iterations: %d, accuracy: %f" % (i, acc))


    print("Final w: %s " % w)
    print("Final acc: %f" % acc)
    
    spark.stop()
    
# Final w: [  2.16071074   3.83393988  12.31919527   9.22327535  -0.13195555
#    0.11412315  -0.62970909  -0.97013437   0.84985948   0.17285233
#    0.09301001  -0.54307295   0.87885498  -6.1623771    0.37821874
#    0.81860751   0.43138264  -0.73978801   0.11298207  -0.87324957
#    2.52234183   4.28098502  12.52925899 -10.31221014   0.89696044
#   -0.36258326  -0.75073835  -0.98280347   0.39573432  -0.47147882
#    1.19939714] 
# Final acc: 0.92982