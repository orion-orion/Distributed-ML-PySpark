'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-06-14 22:04:48
LastEditors: ZhangHongYu
LastEditTime: 2022-06-14 22:05:57
'''
import os
import sys
import numpy as np
import time


def readRating(line):
    line = line.split(",")
    return [int(line[0]), int(line[1]),float(line[2])]

def genperms(numworkers):
    return np.random.permutation(numworkers)+1

#function to assign blocks given the number of rows, number of workers, and index of element
def assignBlockIndex (index, numData, numWorkers):
    blockSize = numData/numWorkers
    if(numData % numWorkers != 0): blockSize = blockSize + 1
    return int(np.floor(index/np.ceil(blockSize)))+1

def main(numFactors, numWorkers, maxIter, beta, lam):
    #conf = SparkConf().setAppName('DSGD-MF')
    #sc = SparkContext(conf=conf)
    numWorkers = int(numWorkers)
    maxIter = int(maxIter)
    numFactors = int(numFactors)
    beta = float(beta)
    lam = sc.broadcast(lam)
    
    
    #load data consisting of triples (user, movie, rating)
    #M = sc.textFile("/FileStore/tables/wdb42krq1465126026181/TrainingRatings.txt").map(readRating).persist()
    #M = sc.textFile("/FileStore/tables/8j3z7jgi1465112138947/RatingsShuf.txt").map(readRating).persist()
    M = sc.textFile("/FileStore/tables/7dn7w4ew1465114785307/RatingsShuf.txt").map(readRating).persist()
    #M = sc.textFile("/FileStore/tables/9lkfxx6x1464574874711/ratings_tiny-993e4.txt").map(readRating)
  
    #get basic statistics
    start = time.time()
    numRows = M.max(lambda x : x[0])[0] +1
    numCols = M.max(lambda x : x[1])[1] +1
    avgRating = M.map(lambda x: x[2]).mean()
    
    #compute the scaling factor that the randomly initialized W and H matrices need to be scaled by so that dot(W_0,H_0) results in values that are similar to the average ratings
    scaleRating = np.sqrt(avgRating / numFactors)
    
    tau = 100
    
    
    mseList = []
    times = []
    #block the M matrix by assigning blocks so that rows and columns are divided into equal chunks of size numRows/numWorkers and numCols/numWorkers
    Mblocked = M.keyBy(lambda x: assignBlockIndex(x[0], numRows, numWorkers)).partitionBy(numWorkers)

    #print Mblocked.collect()
    #initialize the factor matrices and attach with them the count of the number of non-zero elements
    W = M.map(lambda x: tuple([int(x[0]),1])).reduceByKey(lambda x,y : x+y).map(lambda x: tuple([x[0],tuple([x[1],scaleRating*np.random.rand(1, numFactors).astype(np.float32)])])).persist()
    H = M.map(lambda x: tuple([int(x[1]),1])).reduceByKey(lambda x,y : x+y).map(lambda x: tuple([x[0],tuple([x[1],scaleRating*np.random.rand(numFactors,1).astype(np.float32)])])).persist()
    
    #main loop through iterations
    for it in range(maxIter):
        mse = sc.accumulator(0.0)
        nUpdates = sc.accumulator(0.0)
        #broadcast the stepsize
        stepSize = sc.broadcast(np.power(tau + it, -beta))
        #generate random strata
        perms = genperms(numWorkers)
        #filter the data matrix to have only the entries in the blocks from the current strata
        Mfilt = Mblocked.filter(lambda x: perms[x[0]-1]==assignBlockIndex(x[1][1],numCols,numWorkers)).persist()
        #block the W and H matrices using the block number as a key
        Hblocked = H.keyBy(lambda x: perms[assignBlockIndex(x[0], numRows, numWorkers)-1])
        Wblocked = W.keyBy(lambda x: assignBlockIndex(x[0], numRows, numWorkers))
        #group the RDDs together
        groupRDD = Mfilt.groupWith(Hblocked, Wblocked).partitionBy(numWorkers)
        Mfilt.unpersist()
        
        #run SGD on each block
        WH = groupRDD.mapPartitions(lambda x: SGD(x, stepSize, numFactors,lam, mse, nUpdates,scaleRating)).reduceByKey(lambda x,y: x+y).persist()

        W = WH.filter(lambda x: x[0]=='W').flatMap(lambda x: x[1]).persist()
        H = WH.filter(lambda x: x[0]=='H').flatMap(lambda x: x[1]).persist()
        Wvec = W.collect()
        Hvec = H.collect()
        mseCur = mse.value / nUpdates.value
        #check convergence: if just testing scalability comment this section out and run for fixed iterations
        if len(mseList) > 1:
          if abs(mseCur[0][0] - mseList[it-1]) < 0.001:
            mseList.append(mseCur[0][0])
            times.append(time.time()-start)
            break
        mseList.append(mseCur[0][0])
        times.append(time.time()-start)
        


    print times
    print mseList
    


#function to run SGD on a given block with input iterable of (Vblock, (Wblock, Hbloc))
def SGD(keyed_iterable, stepSize, numFactors,lam, mse, nUpdates, scaleRating):
    iterlist = (keyed_iterable.next())
    Miter = iterlist[1][0]
    Hiter = iterlist[1][1]
    Witer = iterlist[1][2]
    
    Wdict = {}
    Hdict = {}
    
    Wout = {}
    Hout = {}
    
    #iterate through H and W and create dictionary of elements
    for h in Hiter:
        Hdict[h[0]] = h[1]
    
    for w in Witer:
        Wdict[w[0]] = w[1]
    
    #iterate through entries of M and compute L2-loss
    for m in Miter:
        (i,j,rat) = m
        
        if i not in Wdict:
            Wdict[i] = tuple([i,scaleRating*np.random.rand(1,numFactors).astype(np.float32)])
        if j not in Hdict:
            Hdict[j] = tuple([j,scaleRating*np.random.rand(numFactors,1).astype(np.float32)])

        (Nw, Wprev) = Wdict[i]
        (Nh, Hprev) = Hdict[j]

        delta = -2*(rat - Wprev.dot(Hprev))
        mse += (rat - Wprev.dot(Hprev))**2

        Wnew = Wprev - stepSize.value*(delta*Hprev.T + (2.0*lam.value/Nh)*Wprev)
        Hnew = Hprev - stepSize.value*(delta*Wprev.T + (2.0*lam.value/Nw)*Hprev)

        nUpdates += 1

        Wout[i] = tuple([Nw, Wnew])
        Hout[j] = tuple([Nh, Hnew])
        
    return (tuple(['W',Wout.items()]), tuple(['H',Hout.items()]))


    #Run by calling main(numFactors, numWorkers, maxIter, beta, lam)