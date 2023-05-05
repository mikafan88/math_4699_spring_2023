#2699 baseline code attempt 5.5
#mikafan88

#imports
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn import * 


#phase 1: data fun

#I. Data Generation

#rewrite dataset generator to talk about which group the thing is from. 
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
    return ourList, yList, zList

#II. KMeans to cluster
def kMeans(ourList1, ourList2):
    neoList = ourList1 + ourList2
    points = np.array(neoList)
    points1 = np.array(ourList1)
    points2 = np.array(ourList2)
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(points)
    centroids = kmeans.cluster_centers_
    return (((centroids[0][0], centroids[0][1]), (centroids[1][0], centroids[1][1])))

#III. Draw a line between the centers of the clusters
def mxPlusB(point1, point2): #points 1 and 2 are tuples
    y = 1
    x = 0
    m = 2 * (point2[y] - point1[y])/(point2[x] - point1[x])
    b = point1[y] - (point1[x] * m)
    return (m, b)

#if slope is less than 1, the rate gets altered in favor of one group
#if slope is more than 2, the rate gets altered in favor of another group

#IV. Assign ratios based off the line
def trueguilt(ourList1, ourList2, ratio1, ratio2, m, b):
    for i in range(len(ourList1)):
        ourList1[i] = ourList1[i] + ("white",)
    for i in range(len(ourList2)):
        ourList2[i] = ourList2[i] + ("black",)
    ourList3 = ourList1 + ourList2
    neoList = []
    for i in range(len(ourList3)):
        a = random.random()
        if ourList3[i][1] > ((m * ourList3[i][0]) + b):
            if a > ratio1:
                neoList.append(ourList3[i] + (1,)) #guilty+ ("above", )
            else:
                neoList.append(ourList3[i]  + (0,)) #innocent + ("above",)
        else:
            if a > ratio2:
                neoList.append(ourList3[i] + (1,)) #guilty + ("below",)
            else:
                neoList.append(ourList3[i] + (0,)) #innocent  + ("below",)
    #print(neoList)
    return neoList #list of 4-tuples

#what's being returned is: (x, y, i {above if i = 1 below if i = 0}, guilty {guilty if i = 1 innocent if i = 0})

#phase 2: determining guilt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

def logisticGuiltDeterminer(guiltList):
    guiltArray = np.array(guiltList)
    X = np.delete(guiltArray, 3, 1)
    y = np.delete(guiltArray, 0, 1)
    y = np.delete(y, 0, 1)
    y = np.delete(y, 0, 1)
    Y = np.array(y, dtype = float)
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
    TrueX_train = np.array(np.delete(X_train, 2, 1), dtype = float)
    TrueX_test = np.array(np.delete(X_test, 2, 1), dtype = float)

    # Define the loss function for logistic regression
    def loss(w):
        lr = LogisticRegression(solver='liblinear', C=1/w[0])
        lr.fit(TrueX_train, y_train)
        y_pred = lr.predict(TrueX_test)
        return np.mean(np.abs(y_test - y_pred)) + w[0] * 0.01

    # Initialize the weights with random values
    w0 = np.random.rand(TrueX_train.shape[1] + 1)

    # Optimize the weights using the L-BFGS-B algorithm
    res = minimize(loss, w0, method='L-BFGS-B', bounds=[(0, None)] * (TrueX_train.shape[1] + 1))

    # Create the logistic regression model using the optimized weights
    lr = LogisticRegression(solver='liblinear', C=1/res.x[0])
    lr.fit(TrueX_train, y_train)
    y_pred = lr.predict(TrueX_test)

    #above, true guilt, predicted guilt
    abomination = list(zip(np.ravel(X_test[:,2:]), np.array(np.ravel(y_test), dtype = float), y_pred))
    return abomination


#phase 3 : checking false positive rate:

def falsePositiveChecker(abomination):
    #split lists
    belowList = []
    aboveList = []
    for i in range(len(abomination)):
        if abomination[i][0] == 'white':
            aboveList.append(abomination[i])
        else:
            belowList.append(abomination[i])
    return (falsePositiveCheckerSupplement(aboveList), falsePositiveCheckerSupplement(belowList))
    
def falsePositiveCheckerSupplement(aList): #a supplement to falsepositivechecker
    listSize = len(aList)
    counter = 0
    negativeCounter = 0
    for i in range(len(aList)):
        if (aList[i][1] == 0) and (aList[i][2] == 1):
            counter = counter + 1
        if (aList[i][1] == 0) and (aList[i][2] == 0):
            negativeCounter = negativeCounter + 1
    return (counter/(counter + negativeCounter))

def falseNegativeChecker(abomination):
    #split lists
    belowList = []
    aboveList = []
    for i in range(len(abomination)):
        if abomination[i][0] == 'white':
            aboveList.append(abomination[i])
        else:
            belowList.append(abomination[i])
    return (falseNegativeCheckerSupplement(aboveList), falseNegativeCheckerSupplement(belowList))

def falseNegativeCheckerSupplement(aList): #a supplement to falsenegativechecker
    listSize = len(aList)
    counter = 0
    positiveCounter = 0
    for i in range(len(aList)):
        if (aList[i][1] == 1) and (aList[i][2] == 0):
            counter = counter + 1
        if (aList[i][1] == 1) and (aList[i][2] == 1):
            positiveCounter = positiveCounter + 1
    return (counter/(counter + positiveCounter))

def accuracyChecker(abomination):
    #split lists
    belowList = []
    aboveList = []
    for i in range(len(abomination)):
        if abomination[i][0] == 'white':
            aboveList.append(abomination[i])
        else:
            belowList.append(abomination[i])
    return (accuracyCheckerSupplement(aboveList), accuracyCheckerSupplement(belowList))

def accuracyCheckerSupplement(aList):
    listSize = len(aList)
    counter = 0
    for i in range(len(aList)):
        if (aList[i][1] == 0) and (aList[i][2] == 0):
            counter = counter + 1
        if (aList[i][1] == 1) and (aList[i][2] == 1):
            counter = counter + 1
    return (counter/listSize)
    
"""

What I want to do is get my data into the format of two numpy arrays:
an "above" array, and a "below" array. I then want to find the percent of falsely guilty in each.  


"""

#tests
def test1(yMin1, yMax1, zMin1, zMax1, amount1, yMin2, yMax2, zMin2, zMax2, amount2, ratio1, ratio2):
    list1 = datasetGenerator(yMin1, yMax1, zMin1, zMax1, amount1)[0]
    list2 = datasetGenerator(yMin2, yMax2, zMin2, zMax2, amount2)[0]
    points = kMeans(list1, list2)
    mxplusb = mxPlusB(points[0], points[1])
    m = mxplusb[0]
    b = mxplusb[1]
    guiltList = (trueguilt(list1, list2, ratio1, ratio2, m, b))
    abomination = logisticGuiltDeterminer(guiltList)
    #abomination2 = logisticGuiltDeterminer2(guiltList)
    falsePositiveRate = falsePositiveChecker(abomination)
    #falsePositiveChecker(abomination2)
    falseNegativeRate = falseNegativeChecker(abomination)
    #accuracy
    accuracyRate = accuracyChecker(abomination)
    return falsePositiveRate, falseNegativeRate, accuracyRate



whiteFalsePositive = []
blackFalsePositive = []
whiteFalseNegative = []
blackFalseNegative = []
whiteAccuracy = []
blackAccuracy = []

whiteX1 = 2
whiteX2 = 3
whiteY1 = 4
whiteY2 = 5

blackX1 = 2.75
blackX2 = 3.75
blackY1 = 4.75
blackY2 = 5.75
for i in range(1000):
    a = test1(whiteX1, whiteX2, whiteY1, whiteY2, 1000, blackX1, blackX2, blackY1, blackY2, 1000, .99, .3)
    whiteFalsePositive.append(a[0][0])
    blackFalsePositive.append(a[0][1])
    whiteFalseNegative.append(a[1][0])
    blackFalseNegative.append(a[1][1])
    whiteAccuracy.append(a[2][0])
    blackAccuracy.append(a[2][1])
    
print("black false positive rate is", sum(blackFalsePositive) / len(blackFalsePositive))
print("white false positive rate is", sum(whiteFalsePositive) / len(whiteFalsePositive))
print("black false negative rate is", sum(blackFalseNegative) / len(blackFalseNegative))
print("white false negative rate is", sum(whiteFalseNegative) / len(whiteFalseNegative))
print("white accuracy rate is", sum(whiteAccuracy) / len(whiteAccuracy))
print("black accuracy rate is", sum(blackAccuracy) / len(blackAccuracy))




