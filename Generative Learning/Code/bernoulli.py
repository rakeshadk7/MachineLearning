from sklearn import preprocessing
import numpy as np
import operator
from functions import *

def train(data, cls, labels): #Train function
    mean = [0,0]
    prior = [0,0]
    y = cls.astype(bool)
    notSpam = data[np.invert(y)]
    spam = data[y]
    
    mean[0] = np.mean(notSpam,axis=0)
    mean[1] = np.mean(spam,axis=0)
    
    prior[0] = float(notSpam.shape[0]) /data.shape[0]
    prior[1] = float(spam.shape[0]) /data.shape[0]
    return mean,prior

def getMembership(x, mean, prior, cl): #Calculate membership 
    total = 0
    for j in range(0,len(x)):
        total += x[j] * np.log(mean[cl][j]) + (1-x[j]) * np.log(1 - mean[cl][j]) + np.log(prior[cl])
    return total
    
def classify(x, mean, prior, data, cls, labels): #Classify 
    membership = {}
    
    for i in xrange(0, len(labels)):
        membership[labels[i]] = getMembership(x, mean, prior, int(labels[i]))
        
    return max(membership.iteritems(), key=operator.itemgetter(1))[0] 

def genLearning(data, cls): 
    labels = list(set(cls))
    n = len(data) #length of data 
    predicted = []
    p = np.random.permutation(len(data))
    data = data[p]
    cls = cls[p]
    
    mean, prior = train(data, cls, labels)
    for row in data:
        predicted.append(classify(row, mean, prior, data, cls, labels))
    
    evaluate(predicted, cls, labels)
    return predicted, cls

if __name__ == "__main__":   
    
    fileName = "spambase.data"
    data, cls = readData(fileName)
    cls = cls.astype(int)
    data = data[:,range(0,48)] #Get first 48 attributes which is word freq
    binarizer = preprocessing.Binarizer().fit(data)
    data = binarizer.transform(data) #Convert frequency to binary values, 1s and 0's
    predicted, cls = genLearning(data, cls) # Train and classify and evaluate 
    plot_matrix(cls, predicted)
    
    
    
    
    