# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cross_validation import KFold
import random
from math import exp
from sklearn import datasets
from sklearn import preprocessing
import itertools
from functions import *
    
def H(theta, x):
    return 1 / (1 + exp(-np.dot(np.transpose(theta),x)))

def train(X, Y, delta = 9999, learningRate = 0.01, maxIterations = 2000):
    theta = []
    #inital random guess for Theta values 
    for i in xrange(0, len(X[0])):
        theta.append(random.randrange(1,10)/float(1000)) #Values closer to zero is desired to be less decisive
    iteration = 0
    while( (np.sum(np.absolute(delta)) > 0.01) or (iteration < maxIterations) ):
        iteration += 1
        total = 0
        for i in xrange(len(X)):
            total += ((H(theta,X[i]) - Y[i]) * X[i])
        #print "total = " , total
        old = theta    
        theta = theta - (learningRate * total)
        delta = old - theta
        
    return theta
       
def classify(theta, X):
    predicted = []
    for row in X:
        predicted.append( int(H(theta, row) >= 0.5) )
    
    return predicted   
           
if __name__ == "__main__":  
    #Import the Iris dataset
    iris = datasets.load_iris()
    X = iris.data[:, :2]  #Taking only first two features
    Y = iris.target
    #Let's take only first two classes to build a 2-class classifier
    X = X[:100]
    Y = Y[:100]
    predicted = [] #List to store the predicted classes
    
    poly = preprocessing.PolynomialFeatures(2,interaction_only=True)
    X = poly.fit_transform(X)
    
    p = np.random.permutation(len(X))
    X = X[p]
    Y = Y[p]
    
    kf = KFold(len(X), n_folds = 10, shuffle = False) #Cross Validation
    for trainIndex, testIndex in kf:
        theta = train(X[trainIndex],Y[trainIndex]) #train to estimate parameters
        predicted.append(classify(theta, X[testIndex])) #predict the classes 
        
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
