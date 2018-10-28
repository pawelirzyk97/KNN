import pandas as pd
import numpy as np
import scipy.spatial as sp
import operator

class dataProcessor:
    def __init__(self):
        return None

    def processData(self,path):
        df = pd.read_csv(path, header=None)
        array = np.array(df)
        return array

    def deleteLabels(self,array):
        newArr = []
        for item in array:
            newArr.append(item[0:len(item)-1])
        return np.array(newArr)

    def getLabels(self,array):
        newArr =[]
        for item in array:
            newArr.append(item[len(item)-1])
        return np.array(newArr)


def getEuclideanDistance(item1,item2):
    return sp.distance.euclidean(item1[0:3],item2[0:3])

def lookForLabel(neighbors):
    labels = {}
    for i in range(len(neighbors)):
        label = neighbors[i][-1]
        if label in labels:
            labels[label] += 1
        else:
            labels[label] = 1
    sortedLabels = sorted(labels.__iter__(), key=operator.itemgetter(1))
    return sortedLabels[0]
