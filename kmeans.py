'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-06-30 21:53:37
LastEditors: ZhangHongYu
LastEditTime: 2022-06-30 22:12:23
'''
import sys
from typing import List
import numpy as np
from pyspark.sql import SparkSession

K = 2
convergeDist = 0.1
n_slices = 2

def closestPoint(p: np.ndarray, centers: List[np.ndarray]) -> int:
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(centers)):
        tempDist = np.sum((p - centers[i]) ** 2)
        if tempDist < closest:
            closest = tempDist
            bestIndex = i
    return bestIndex


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("KMeans")\
        .getOrCreate()

    matrix = np.array([[1, 2], [1, 4], [1, 0],
                  [10, 2], [10, 4], [10, 0]])
    points = spark.sparkContext.parallelize(matrix, n_slices).cache()

    kPoints = points.takeSample(False, K, 42)
    tempDist = 1.0

    while tempDist > convergeDist:
        closest = points.map(
            lambda p: (closestPoint(p, kPoints), (p, 1)))
        pointStats = closest.reduceByKey(
            lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
        newPoints = pointStats.map(
            lambda st: (st[0], st[1][0] / st[1][1])).collect()

        tempDist = sum(np.sum((kPoints[iK] - p) ** 2) for (iK, p) in newPoints)

        for (iK, p) in newPoints:
            kPoints[iK] = p

    print("Final centers: " + str(kPoints))

    spark.stop()