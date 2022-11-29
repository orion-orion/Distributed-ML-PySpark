'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-07-01 21:28:32
LastEditors: ZhangHongYu
LastEditTime: 2022-07-01 21:48:31
'''
from random import random
from operator import add
from pyspark.sql import SparkSession
import sys
import os

os.environ['PYSPARK_PYTHON'] = sys.executable

n_threads = 4  # Number of local threads
# times of sampling
n = 100000 * n_threads
    
def is_accept(_: int) -> int:
    x = random() * 2 - 1
    y = random() * 2 - 1
    return 1 if x ** 2 + y ** 2 <= 1 else 0

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("monte_carlo")\
        .master("local[%d]" % n_threads)\
        .getOrCreate()

    count = spark.sparkContext.parallelize(range(n)).map(is_accept).reduce(add)

    # equation for the ratio of the area of a circle to a square： count/n = pi/4.
    print("Pi is roughly %f" % (4.0 * count / n))

    spark.stop()