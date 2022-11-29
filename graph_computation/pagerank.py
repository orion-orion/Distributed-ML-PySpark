'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-05-31 14:14:35
LastEditors: ZhangHongYu
LastEditTime: 2022-07-02 11:48:23
'''
import sys
from operator import add
from typing import Iterable, Tuple
from pyspark.resultiterable import ResultIterable
from pyspark.sql import SparkSession
import os

os.environ['PYSPARK_PYTHON'] = sys.executable

n_threads = 4  # Number of local threads
n_iterations = 10  # Number of iterations
q = 0.15 #the default value of q is 0.15

def computeContribs(neighbors: ResultIterable[int], rank: float) -> Iterable[Tuple[int, float]]:
    # Calculates the contribution(rank/num_neighbors) of each vertex, and send it to its neighbours.
    num_neighbors = len(neighbors)
    for vertex in neighbors:
        yield (vertex, rank / num_neighbors)

if __name__ == "__main__":
    # Initialize the spark context.
    spark = SparkSession\
        .builder\
        .appName("PageRank")\
        .master("local[%d]" % n_threads)\
        .getOrCreate()

    # link: (source_id, dest_id)
    links = spark.sparkContext.parallelize(
        [(1, 2), (1, 3), (2, 3), (3, 1)],
    )                       

    # drop duplicate links and convert links to an adjacency list.
    adj_list = links.distinct().groupByKey().cache()

    # count the number of vertexes
    n_vertexes = adj_list.count()

    # init the rank of each vertex, the default is 1.0/n_vertexes
    ranks = adj_list.map(lambda vertex_neighbors: (vertex_neighbors[0], 1.0/n_vertexes))

    # Calculates and updates vertex ranks continuously using PageRank algorithm.
    for t in range(n_iterations):
        # Calculates the contribution(rank/num_neighbors) of each vertex, and send it to its neighbours.
        contribs = adj_list.join(ranks).flatMap(lambda vertex_neighbors_rank: computeContribs(
            vertex_neighbors_rank[1][0], vertex_neighbors_rank[1][1]  # type: ignore[arg-type]
        ))

        # Re-calculates rank of each vertex based on the contributions it received
        ranks = contribs.reduceByKey(add).mapValues(lambda rank: q/n_vertexes + (1 - q)*rank)

    # Collects all ranks of vertexs and dump them to console.
    for (vertex, rank) in ranks.collect():
        print("%s has rank: %s." % (vertex, rank))

    spark.stop()
    
    
# 1 has rank: 0.38891305880091237.  
# 2 has rank: 0.214416470596171.
# 3 has rank: 0.3966704706029163.