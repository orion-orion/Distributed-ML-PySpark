'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-06-30 19:32:44
LastEditors: ZhangHongYu
LastEditTime: 2022-07-02 11:51:14
'''
import numpy as np
from pyspark.sql import SparkSession
import sys
import os

os.environ['PYSPARK_PYTHON'] = sys.executable

lam = 0.01   # regularization coefficient
m = 100 # number of users
n = 500 # number of items
k = 10 # dim of the latent vectors of users and items
n_iterations = 5 # number of iterations
n_threads = 4  # Number of local threads

def rmse(R: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.float64:
    diff = R - U @ V.T
    return np.sqrt(np.sum(np.power(diff, 2)) / (m * n))


def update(i: int, mat: np.ndarray, ratings: np.ndarray) -> np.ndarray:
    X_dim = mat.shape[0]
    
    XtX = mat.T @ mat
    Xty = mat.T @ ratings[i, :].T

    for i in range(k):
        XtX[i, i] += lam * X_dim

    return np.linalg.solve(XtX, Xty)


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("Matrix Decomposition")\
        .master("local[%d]" % n_threads)\
        .getOrCreate()

    R = np.random.rand(m, k) @ (np.random.rand(n, k).T)
    U = np.random.rand(m, k)
    V = np.random.rand(n, k)

    R_br = spark.sparkContext.broadcast(R)
    U_br = spark.sparkContext.broadcast(U)
    V_br = spark.sparkContext.broadcast(V)

    # we use the alternating least squares (ALS) to solve the SVD problem
    for t in range(n_iterations):
        U_ = spark.sparkContext.parallelize(range(m)) \
            .map(lambda x: update(x, V_br.value, R_br.value)) \
            .collect()

        # collect() returns a list, so we need to convert it to a 2-d array
        U = np.array(U_)
        U_br = spark.sparkContext.broadcast(U)

        V_ = spark.sparkContext.parallelize(range(n)) \
            .map(lambda x: update(x, U_br.value, R_br.value.T)) \
            .collect()
        V = np.array(V_)
        V_br = spark.sparkContext.broadcast(V)

        error = rmse(R, U, V)
        print("iterations: %d, rmse: %f" % (t, error))

    spark.stop()