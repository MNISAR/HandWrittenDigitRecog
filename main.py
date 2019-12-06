# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:51:43 2019

@author: monil
"""
from sklearn.neighbors import KNeighborsClassifier

trainFile = "./data/train.csv"
testFile = "./data/test.csv"


def loadData(file):
    data = []
    label = []
    with open(file, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            allCols = line.split(",")
            label.append(int(allCols[0]))
            data.append(list(map(int,allCols[1:])))
    return data, label


### load data and train model
def main():
    trainData, trainLabel = loadData(trainFile)
    k = 4
    
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData, trainLabel)
    
    testData, testLabel = loadData(testFile)
    
    print(model.score(testData, testLabel))
    
    #prediction = model.predict(testData)
    
main()