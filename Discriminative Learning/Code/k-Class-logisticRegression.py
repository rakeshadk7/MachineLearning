# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cross_validation import KFold
import random
from math import exp
from sklearn import datasets
from sklearn import preprocessing
import itertools
import operator
from functions import *
    
def H(t, x): 
    return 1 / (1 + exp(-np.dot(np.transpose(t),x)))
    
def softmax(t, x, theta ):
    denom = 0
    for Ti in theta.itervalues():
        denom += exp(np.dot(Ti.T, x))
    
    return ( exp(np.dot(t.T, x))/denom )


def train(X, Y, classes, delta = 9999, treshold = 0.01, learningRate = 0.001, maxIterations = 3000):
    theta = {} #theta to be returned
    iteration = 0 # to count total number of passes of training
    #inital random guess for Theta values 
    
    for cl in classes:
        initial = []
        for i in xrange(0, len(X[0])):
            initial.append(random.randrange(1,10)/float(1000)) #Values closer to zero is desired to be less decisive
        theta[cl] = np.array(initial)
    
    while(iteration < maxIterations ): #Stopping condition to be updated
        #print theta
        iteration += 1
        for cl in classes:
            total = 0
            for i in xrange(len(X)):
                if(Y[i] == cl): #indicator function
                    y = 1
                else:
                    y = 0
                    
                s = softmax(theta[cl],X[i], theta)
                total += ((s - y) * X[i])
            
            theta[cl]  -= (learningRate * total) #update theta[cl]
        
    return theta
       
def classify(theta, X, classes):
    predicted = []
    d = {}
    for row in X:
        for cl in classes:
            d[cl] = np.dot(theta[cl].T, row)
        predicted.append(max(d.iteritems(), key=operator.itemgetter(1))[0]) #return index(class label) of max value 
    
    return predicted   
       
if __name__ == "__main__":  
    #Import the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    predicted = [] #List to store the predicted classes
    
    poly = preprocessing.PolynomialFeatures(1,interaction_only=True)
    X = poly.fit_transform(X)
    
    classes = np.unique(Y)
    
    p = np.random.permutation(len(X))
    X = X[p]
    Y = Y[p]
    
    
    kf = KFold(len(X), n_folds = 10, shuffle = False) #Cross Validation
    for trainIndex, testIndex in kf:
        theta = train(X[trainIndex],Y[trainIndex], classes) #train to estimate parameters
        predicted.append(classify(theta, X[testIndex], classes)) #predict the classes 
        
    predicted = list(itertools.chain(*predicted))

    evaluate(predicted, Y, np.unique(Y)) #evaluate results by comparing predicted vs actual Y 
    plot_matrix(Y, predicted)
   









































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
