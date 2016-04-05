import numpy as np
from scipy.misc import comb
from functions import *

def readClass(fileName):
    cls = [-1]
    filePath = "Data/Spam/" + fileName 
    f = open(filePath)
    for line in f.readlines():
        row = (line.strip().split("\n")) 
        cls.append(int(row[0]))
    cls = np.array(cls)
    return cls
    
def readData(fileName):
    filePath = "Data/Spam/" + fileName 
    f = open(filePath)
    data = [{},{}]

    docId = 1
    for line in f.readlines(): 
        row = (line.strip().split(" "))
        
        if(int(row[0]) == docId):
            data[docId][int(row[1])] = int(row[2])
        else:
            data.append({})
            docId += 1
            data[docId][int(row[1])] = int(row[2])   

    data = np.array(data)
    
    return data


def train(data,cls, labels):
    p = [] #total number of words in each document
    wordKeys = [] #all word ID's
    mean = [{},{}] #mean of f
    
    
    for row in data:
        count = 0  
        for key,value in row.iteritems():
            count += value
            wordKeys.append(key)
        p.append(count)
    
    wordKeys = list(set(wordKeys))
    
    sCount = 0
    nsCount = 0
    n = float((len(cls) - 1))/2  
    
    for wk in wordKeys:
        sCount  = 0
        nsCount = 0
        index = -1
        for row in data:
            count = 0
            index += 1
            try:
                count = row[wk]
            except KeyError:
                pass
            if(cls[index] == 1):
                sCount += count
            elif(cls[index] == 0):
                nsCount += count
                
        mean[0][wk] = nsCount/ n
        mean[1][wk] = sCount / n
           
    return (p,mean)

def classify(x, mean, p, data, cls, labels):
    total = 0
    p = 0
    wordKeys = []
    for key,value in x.iteritems():
        p += value
        wordKeys.append(key)
        
    for cl in labels:
        total = 0
        for key,value in x.iteritems():
            try:
                alpha = mean[cl][key]
            except KeyError:
                alpha = 0
  
            total += np.log(comb(p, value)) + np.log( pow(alpha,value)) + np.log(pow((1 - alpha), (p - value)))
            print total
            
    return 0
         
def genLearning(data,cls):
    labels = list(set(cls[1:]))
    n = len(data) #length of data 
    predicted = []
    
    p, mean = train(data, cls, labels)
    
    testData = readData("test-features.txt")

    for x in testData:
        predicted.append(classify(x, mean, p, data, cls, labels))
                          
if __name__ == "__main__":   
    
    fileName = "train-features.txt"
    data = readData(fileName)
    cls = readClass("train-labels.txt")
    genLearning(data, cls)
            
    