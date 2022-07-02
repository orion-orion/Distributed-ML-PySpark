'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-07-01 22:04:00
LastEditors: ZhangHongYu
LastEditTime: 2022-07-02 10:16:02
'''
from pyspark.sql import SparkSession
n_slices = 2  # Number of Slices

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("PythonTransitiveClosure")\
        .getOrCreate()
     
    paths = spark.sparkContext.parallelize([(1, 2), (1, 3), (2, 3), (3, 1)], n_slices).cache()

    # Linear transitive closure: each round grows paths by one edge,
    # by joining the the already-discovered paths with graph's edges. 
    # e.g. join the path (y, z) from the paths with the edge (x, y) from
    # the graph to obtain the new path (x, z).
    

    # The edges are stored in reversed order because they are about to be joined.
    edges = paths.map(lambda x_y: (x_y[1], x_y[0]))

    old_cnt = 0
    next_cnt = paths.count()
    while True:
        old_cnt = next_cnt
        # Perform the join, obtaining an RDD of (y, (z, x)) pairs,
        # then map the result to obtain the new (x, z) paths.
        new_paths = paths.join(edges).map(lambda vertexes: (vertexes[1][1], vertexes[1][0]))
        # union new paths
        paths = paths.union(new_paths).distinct().cache()
        next_cnt = paths.count()
        if next_cnt == old_cnt:
            break

    print("The original graph has %i paths" % paths.count())

    spark.stop()