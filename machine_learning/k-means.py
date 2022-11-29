'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-06-30 21:53:37
LastEditors: ZhangHongYu
LastEditTime: 2022-07-02 11:52:42
'''
import random
from typing import List, Tuple
import numpy as np
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import sys
import os

os.environ['PYSPARK_PYTHON'] = sys.executable

k = 2
convergeDist = 0.1
n_slices = 2
n_iterations = 5

def closest_center(p: np.ndarray, centers: List[np.ndarray]) -> int:
    closest_cid = 0
    min_dist = float("+inf")
    for cid in range(len(centers)):
        dist = np.sqrt(np.sum((p - centers[cid]) ** 2))
        if dist < min_dist:
            min_dist = dist
            closest_cid = cid
    return closest_cid

def display_clusters(center_to_point: List[Tuple]):    
    clusters = dict([ (c_id, []) for c_id in range(k)])
    for c_id, (p, _) in center_to_point:
        clusters[c_id].append(p)

    for c_id, points in clusters.items():
        points = np.array(points)
        color = "#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)])
        plt.scatter(points[:, 0], points[:, 1], c=color)
    
    plt.savefig("kmeans_clusters_display.png")
        

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("K-means")\
        .getOrCreate()

    matrix = np.array([[1, 2], [1, 4], [1, 0],
                  [10, 2], [10, 4], [10, 0]])
    points = spark.sparkContext.parallelize(matrix, n_slices).cache()

    k_centers = points.takeSample(False, k, 42)

    for t in range(n_iterations):
        # assign each point to the center closest to it.
        center_to_point = points.map(
            lambda p: (closest_center(p, k_centers), (p, 1)))

        # for each cluster(points shareing the some center),
        # compute the sum of vecters in it and the size of it.
        cluster_stats = center_to_point.reduceByKey(
            lambda p1_cnt1, p2_cnt2: (p1_cnt1[0] + p2_cnt2[0], p1_cnt1[1] + p2_cnt2[1]))

        # for each cluster, compute the mean vecter.
        mean_vecters = cluster_stats.map(
            lambda stat: (stat[0], stat[1][0] / stat[1][1])).collect()

        # update the centers.
        for (c_id, mean_vecter) in mean_vecters:
            k_centers[c_id] = mean_vecter

    print("Final centers: " + str(k_centers))
    
    if matrix.shape[1] == 2: 
        display_clusters(center_to_point.collect())

    spark.stop()