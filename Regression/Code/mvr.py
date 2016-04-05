import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv
from os import listdir
from sklearn.cross_validation import KFold
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import PolynomialFeatures
import random
import scipy as scip


def readData(fileName): #Function to read Data
    filePath = "Data/mvr/"
    data = np.genfromtxt(filePath + fileName, delimiter=" ", skip_header=True)
    return data
    
def gradientDesc(z, y, weight = 0.000001, treshold = 0.000001, diff = 9999):

    n = len(z[0])
    #x = data[:, 0:len(data[0])-1]
    #y = data[:, -1]
    #ones = np.ones((m, 1), dtype=np.float) #Column of ones
    #z = np.column_stack((ones, x)) 
    theta = []
    for i in xrange(0,n):
        theta.append(random.random()) #random theta to start with
    while(diff > treshold): #stopping criteria
        newTheta = theta - weight * np.dot(z.T, (np.dot(z,theta) - y))
        diff = np.mean(np.absolute(newTheta - theta))
        theta = newTheta
    
    #print("Intercept     :   {} \nCoefficients : {}\n".format(theta[0], theta[1:]))   
    return theta
    
def evaluateGradientDesc(data): #Function to evaluate Gradient Descent Algorithm in terms of testing Error
     poly = PolynomialFeatures(1)
     x = data[:, 0:len(data[0])-1]
     y = data[:, -1]
     z = poly.fit_transform(x)
     testingError, sol = evaluateModel(z,y, True)
     
     print "Iterative solution\nTesting error : ", testingError , "\n"
        

def dualRegression(data): #Solve dual problem
    x = data[:, 0:len(data[0])-1]
    y = data[:, -1]
    pairwise_dists = squareform(pdist(x, 'euclidean'))
    G = scip.exp(-pairwise_dists ** 2 / 1 ** 2) #Gramm Matrix
    
    alpha = np.dot(inv(G),y) #Alpha values
    sol = np.dot(alpha.T, x) #Soltuion
    return sol
    
    
def fitModel(z,y):
    return np.dot(pinv(z),y)
          
def evaluateModel(z,y, gradDesc):  #Function to evaluate model
    n = len(z)
    sol = np.empty(y.shape) #initial
    testingError = [] #List to gather all testing error from k fold validation
    
    kf = KFold(n, n_folds = 10, shuffle = False) #Cross Validation
    
    for trainIndex, testIndex in kf: #K fild Validation
        Z_train, Z_test = z[trainIndex], z[testIndex]
        Y_train, Y_test = y[trainIndex], y[testIndex]
        
        if(gradDesc):
            theta = gradientDesc(Z_train, Y_train)
        else:
            theta = np.dot(pinv(Z_train), Y_train)
        Y_est = np.dot(theta, Z_test.T)
        sol[testIndex] = Y_est
            
        mat = (Y_test - Y_est)
        testingError.append(np.dot(mat.T,mat)/len(testIndex))
    
    testMean = sum(testingError)/len(testingError)
    
    return [testMean,sol]
    
def mvr(data): 
    x = data[:, 0:len(data[0])-1]
    y = data[:, -1]
    
    minTestingError = np.inf
    for dim in xrange(1,3):
        if(dim > 1):
            print("Mapping into higher dimension of {} \n".format(dim))
        else:
            evaluateGradientDesc(data)
            print("Explicit solution\n")
        poly = PolynomialFeatures(dim)
        z = poly.fit_transform(x)
        
        theta = fitModel(z , y)
        
        print("Intercept     :   {} \nCoefficients : {}\n".format(theta[0], theta[1:]))
        testingError, sol = evaluateModel(z,y, False)
        
        if(dim == 1):
            print "Testing Error :", testingError
        
        if (testingError < minTestingError):
            minTestingError = testingError
            optimalDimension = dim
            optSol = sol
         
    print "Optimal Dimension : {}, Testing Error : {} ".format(optimalDimension, minTestingError)
    return optSol
        
if __name__ == "__main__":
    
    mvrFiles = listdir("Data/mvr")
    
    for fileName in mvrFiles:
        print ("\n \t {} \t \n".format(fileName))
        data = readData(fileName)
        mvr(data)
        if(fileName == "mvar-set1.dat"):
            sol = dualRegression(data)
        
       
        
        
        
        
        
    
    
    
    
    

    
    
    
    

    
    
    
    
    
   