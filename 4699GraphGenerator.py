#2699 baseline code attempt 2

#standard imports
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#dataset generation
def datasetGenerator(yMin, yMax, zMin, zMax, amount):
    ourList = []
    yList = []
    zList = []
    for i in range(amount):
        y = random.uniform(yMin, yMax)
        z = random.uniform(zMin, zMax)
        ourList.append((y,z))
        yList.append(y)
        zList.append(z)
    #print(ourList)
    #plt.scatter(yList, zList) - commented out for testing purposes
    return ourList, yList, zList

#logistic classifier test
"""so this is actually just straight up a kmeans thing., I was just too lazy to rename it. tbh i have no real justification for doing anything else,
besides laziness. """

def logisticClassifier(ourList1, ourList2):
    neoList = ourList1 + ourList2
    points = np.array(neoList)
    points1 = np.array(ourList1)
    points2 = np.array(ourList2)
    #print(points)
    #print(neoList)
    #kmeans = KMeans(n_clusters=2, random_state=0).fit(points)
    #print("AHEM AHEM AHEM")
    #print(kmeans.cluster_centers_)
    #print("AHEM AHEM AHEM")
    plt.scatter(*zip(*ourList1)) #you can kinda see the clustering here. be sure to turn this off when building SVMs.
    plt.scatter(*zip(*ourList2))
    plt.plot([2, 6], [4 , 10], 'k-')
    plt.show()
    #2 * (point2[y] - point1[y])/(point2[x] - point1[x]) + point1[y] - (point1[x] * m)
    return neoList


    

#testing 1
def test1(yMin1, yMax1, zMin1, zMax1, amount1, yMin2, yMax2, zMin2, zMax2, amount2):
    list1 = datasetGenerator(yMin1, yMax1, zMin1, zMax1, amount1)[0]
    list2 = datasetGenerator(yMin2, yMax2, zMin2, zMax2, amount2)[0]
    logisticClassifier(list1, list2)


#test1(2, 3, 4, 5, 6, 12, 13, 14, 15, 6)
test1(2, 3, 4, 5, 1000, 2.75, 3.75, 4.75, 5.75, 1000)

"""potential features - cluster a certain point. this is unsupervised, probably want a supervised version of this too to shell."""
