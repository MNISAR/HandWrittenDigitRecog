# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:51:43 2019

@author: monil
"""
from sklearn.neighbors import KNeighborsClassifier

trainFile = "./data/train.csv"
testFile = "./data/test.csv"


def loadData(file, l=0):
    data = []
    label = []
    with open(file, 'r') as f:
        lines = f.readlines()[1:l]
        for line in lines:
            allCols = line.split(",")
            label.append(int(allCols[0]))
            data.append(list(map(int,allCols[1:])))
    return data, label


### load data and train model
def main():
    trainData, trainLabel = loadData(trainFile, 1000)
    k = 8
    print("Train data loaded!", len(trainData))
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData, trainLabel)
    
    testData, testLabel = loadData(testFile, 1000)
    print("Test data loaded!", len(testData))
    
    print(model.score(testData, testLabel))
    
    #prediction = model.predict(testData)
    
main()