'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-05-31 14:14:35
LastEditors: ZhangHongYu
LastEditTime: 2022-06-02 20:26:49
'''
import re
import sys
from operator import add
from typing import Iterable, Tuple

from pyspark.resultiterable import ResultIterable
from pyspark.sql import SparkSession

n_slices = 3  # Number of Slices
n_iterations = 10  # Number of iterations


def computeContribs(neighbors: ResultIterable[int], rank: float) -> Iterable[Tuple[int, float]]:
    """Calculates the average rank(rank/num_neighbors) of each url, and send it to its neighbours."""
    num_neighbors = len(neighbors)
    for neighbor in neighbors:
        yield (neighbor, rank / num_neighbors)

if __name__ == "__main__":
    # Initialize the spark context.
    spark = SparkSession\
        .builder\
        .appName("PythonPageRank")\
        .getOrCreate()

    # link: (source_id, dest_id)
    links = spark.sparkContext.parallelize(
        [(1, 2), (1, 3), (2, 3), (3, 1)],
        n_slices
    )                       

    # drop duplicate links and convert links to an adjacency list.
    links = links.distinct().groupByKey().cache()

    # init the rank of each url, the default is 1.0
    ranks = links.map(lambda url_neighbors: (url_neighbors[0], 1.0))

    # Calculates and updates link ranks continuously using PageRank algorithm.
    for t in range(n_iterations):
        # Calculates the average rank(rank/num_neighbors) of each url, and send it to its neighbours.
        contribs = links.join(ranks).flatMap(lambda url_neighbors_rank: computeContribs(
            url_neighbors_rank[1][0], url_neighbors_rank[1][1]  # type: ignore[arg-type]
        ))

        # Re-calculates rank of each url based on its neighbors' rank.
        ranks = contribs.reduceByKey(add).mapValues(lambda rank: rank * 0.85 + 0.15)
        # 0.15 is the miminum rank of each url
    print(ranks)
    # Collects all ranks of urls and dump them to console.
    for (url, rank) in ranks.collect():
        print("%s has rank: %s." % (url, rank))

    spark.stop()