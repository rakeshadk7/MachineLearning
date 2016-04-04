# -*- coding: utf-8 -*-
from sklearn import datasets
import numpy as np
from sklearn.cross_validation import KFold
import random
from sklearn import preprocessing
from math import exp
import operator
import itertools
from functions import *
import sys

Beta = 0

def sigmoid(theta, x):
    return float(1) / (1 + exp(-np.dot(np.transpose(theta),x)))
    
def softmax(t, x, theta ):
    denom = 0
  #  print theta
    for Ti in theta.itervalues():
        denom += exp(np.dot(Ti.T, x))
    
    return ( exp(np.dot(t.T, x))/denom )
        
def train(model, x, y, classes, maxIterations = 2000, learningRate = 0.001):
    
    iteration = 0
    W = model['W']
    V = model['V']
    
    while(iteration < maxIterations):
       
        iteration += 1
        gradientV = [[0 for i in xrange(len(W) + 1)] for j in xrange(len(V))]
        gradientW = [[0 for i in xrange(len(x[0]))] for j in xrange(len(W))]
        
        #Compute Zi and Yi
        for rowIndex in xrange(len(x)):
            Z = np.array([1])
            Y = np.array([])
            for i in xrange(len(W)):
                Z = np.append(Z, sigmoid(W[i], x[rowIndex]))
            for i in xrange(len(V)):
                Y = np.append(Y, softmax(V[i], Z, V))
            
            #Update V, W
            for i in xrange(len(V)):   
                gradientV[i] += (Y[i] - y[rowIndex][i]) * Z 
            sum1 = 0
            for j in xrange(len(W)):
                for i in xrange(len(V)):
                    sum1 += (Y[i] - y[rowIndex][i]) * V[i][j]
                gradientW[j] += sum1 * Z[j] * (1 - Z[j]) * x[rowIndex]  
            
         
        for i in xrange(len(V)):
            V[i] -=learningRate * gradientV[i]
            
        for i in xrange(len(W)):
            old = W[i]
            new = old - learningRate * gradientW[i] 
            W[i] -= learningRate * gradientW[i]  + (Beta * (old - new))
            

        
    model['V'] = V
    model['W'] = W
    return model

def classify(model, x, classes):
    V = model['V']
    W = model['W']
    predicted = []
    for rowIndex in xrange(len(x)):
        Z = np.array([1])
        Y = []
        
        for i in xrange(len(W)):
            Z = np.append(Z, sigmoid(W[i], x[rowIndex]))
        for i in xrange(len(V)):
            Y.append(softmax(V[i], Z, V))
        
        predicted.append(Y.index(max(Y)))
            
    return predicted
    
    
#nnHidden : Number of nodes in hidden layer
#maxIterations : Number of passes through the training data for gradient descent
#classes : set of all unique classes
def buildModel(x, nnHidden, classes):
    model = {}
    k = len(classes)
    
    #initial guess for weights W
    W = {}
    for j in xrange(nnHidden):
        initial = []
        for i in xrange(len(x[0])):
            initial.append(random.randrange(1,10)/float(1000)) #Values closer to zero is desired to be less decisive
        W[j] = np.array(initial)
    
    #initial guess for weights V 
    V = {}
    for j in xrange(k):   
        initial = []
        for i in xrange(nnHidden + 1): 
            initial.append(random.randrange(1,10)/float(1000)) #Values closer to zero is desired to be less decisive
        V[j] = np.array(initial)
    
    model['W'] = W
    model['V'] = V
        
    return model

if __name__ == "__main__":  
    # The digits dataset
    digits = datasets.load_digits()
    x = digits.data 
    y = digits.target
    predicted = [] #to store the predicted classes
    
    #Let's make the problem a 3-class problem, because the assignment says so
    mask = [ index for index in range(len(y)) if y[index] in [0,1,2] ]
    
    x = x[mask]
    y = y[mask]
    x = preprocessing.scale(x)
    
    poly = preprocessing.PolynomialFeatures(1,interaction_only=True)
    x = poly.fit_transform(x)
    
    #shuffle the data
    p = np.random.permutation(len(x))
    x = x[p]
    y = y[p]
    Y = y
    classes = np.unique(y)
    
    y = [np.array([1 if i == j else 0 for i in xrange(len(classes))]) for j in y]
    
    m = len(x) # training set s
   
    model = buildModel(x, 6, classes)
    
    model = train(model, x, y, classes) #train to estimate parameters
    predicted = classify(model, x, classes) #predict the classes 
    
   
    count = 0
    kf = KFold(len(x), n_folds = 10, shuffle = False) #Cross Validation
    for trainIndex, testIndex in kf:
        
        yy = y[trainIndex]
        yy = [np.array([1 if i == j else 0 for i in xrange(len(classes))]) for j in yy]
        model = train(model, x[trainIndex],yy, classes) #train to estimate parameters
        predicted.append(classify(model, x[testIndex], classes)) #predict the classes 
        
        count += 1
        print count
        sys.stdout.flush()
    predicted = list(itertools.chain(*predicted))

    #evaluate(predicted, Y, np.unique(Y)) #evaluate results by comparing predicted vs actual Y 
    plot_matrix(Y, predicted)
    
 
