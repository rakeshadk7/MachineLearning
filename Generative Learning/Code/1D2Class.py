# -*- coding: utf-8 -*-
import numpy as np
import operator
import itertools
from sklearn.cross_validation import KFold
from functions import *

def mean(data,cls,cl): #claculate the parameters for training mean of feature
    total = 0 
    count = 0
    for index in xrange(0,data.size):
         if cls[index] == cl:
             count += 1                      
             total += data[index]
    return float(total)/count
  
def sigma(data,cls,cl,mean): #claculate the parameters for training, variance of feature
    count = 0
    total = 0.0
    for index in xrange(0,data.size):
        if cls[index] == cl:
            count += 1
            total += pow((data[index] - mean),2)
    return (total/count)  
    
def train(data,cls): #function to train 
    classes = list(set(cls))
    Mu = {} #Using dictionary to store the mean values, because it's the smart thing to do
    S = {}  #Dictionaries again
    
    for cl in classes:
        m = mean(data,cls,cl)
        Mu[cl] = m                   
        S[cl] = sigma(data,cls,cl,m)
    
    return (Mu,S)

def classify(data, cls, classes, Mu, S):
    predicted = []
    membership = {}
    for x in data:
        for cl in classes: #For each unique class
            membership[cl] =  (-1 * np.log(S[cl])) - (pow((x - Mu[cl]),2)/(2 * S[cl])) #Membership value
        predicted.append(max(membership.iteritems(), key=operator.itemgetter(1))[0]) #Select max from 
    return predicted
                  
def genLearning(data, cls):    
    classes = list(set(cls))
    n = len(data) #length of data 
    predicted = []
    p = np.random.permutation(len(data))
    data = data[p]
    cls = cls[p]
      
    kf = KFold(n, n_folds = 10, shuffle = False) #Cross Validation
    for trainIndex, testIndex in kf:
        Mu, S = train(data[trainIndex],cls[trainIndex]) #train the data, calculate Mu and Sigma for train data
        predicted.append(classify(data[testIndex], cls[testIndex], classes, Mu, S)) #Classify
        
    predicted = list(itertools.chain(*predicted))
    evaluate(predicted,cls, classes)
    
    return (predicted,cls)
    
if __name__ == "__main__":       
    fileName = "iris.csv"
    data, cls = readData(fileName)
    
    data = data[:100] #uses first 2 classes, 50 of each
    cls = cls[:100]   #uses first 2 classes, 50 of each
    data = data[:,0]  #use one column to make x 1D
    predicted, cls = genLearning(data, cls) #Train, predict & evaluate
    plot_matrix(cls, predicted)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 
    
'''   
 You didn't expect to find Doge here, but Doge is everywhere.   
░░░░░░░░░▄░░░░░░░░░░░░░░▄
░░░░░░░░▌▒█░░░░░░░░░░░▄▀▒▌
░░░░░░░░▌▒▒█░░░░░░░░▄▀▒▒▒▐
░░░░░░░▐▄▀▒▒▀▀▀▀▄▄▄▀▒▒▒▒▒▐
░░░░░▄▄▀▒░▒▒▒▒▒▒▒▒▒█▒▒▄█▒▐
░░░▄▀▒▒▒░░░▒▒▒░░░▒▒▒▀██▀▒▌
░░▐▒▒▒▄▄▒▒▒▒░░░▒▒▒▒▒▒▒▀▄▒▒▌
░░▌░░▌█▀▒▒▒▒▒▄▀█▄▒▒▒▒▒▒▒█▒▐
░▐░░░▒▒▒▒▒▒▒▒▌██▀▒▒░░░▒▒▒▀▄▌
░▌░▒▄██▄▒▒▒▒▒▒▒▒▒░░░░░░▒▒▒▒▌
▌▒▀▐▄█▄█▌▄░▀▒▒░░░░░░░░░░▒▒▒▐
▐▒▒▐▀▐▀▒░▄▄▒▄▒▒▒▒▒▒░▒░▒░▒▒▒▒▌
▐▒▒▒▀▀▄▄▒▒▒▄▒▒▒▒▒▒▒▒░▒░▒░▒▒▐
░▌▒▒▒▒▒▒▀▀▀▒▒▒▒▒▒░▒░▒░▒░▒▒▒▌
░▐▒▒▒▒▒▒▒▒▒▒▒▒▒▒░▒░▒░▒▒▄▒▒▐
░░▀▄▒▒▒▒▒▒▒▒▒▒▒░▒░▒░▒▄▒▒▒▒▌
░░░░▀▄▒▒▒▒▒▒▒▒▒▒▄▄▄▀▒▒▒▒▄▀
░░░░░░▀▄▄▄▄▄▄▀▀▀▒▒▒▒▒▄▄▀
░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▀▀
Program by Rakesh Adhikesavan

'''