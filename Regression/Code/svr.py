import numpy as np
from numpy.linalg import pinv
from os import listdir
import pylab as plt
from sklearn.cross_validation import KFold
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures


def readData(fileName):
    filePath = "C:\\Users\\Rakesh\\Documents\\Personal\\IIT\\ML\\Assignment 1\\Data\\svr"
    data = np.genfromtxt(filePath + fileName, delimiter=" ", skip_header=True)
    return data
    
def linearModel(fileName):
    data = readData(fileName)
    n = len(data)
    ones = np.ones((n, 1), dtype=np.float) #Column of ones
    X = data[:, 0] #All the x values, column matrix
    Y = data[:, 1] #Corresponding Y values, column matrix
    Z = np.column_stack((ones, X))  
    sol = np.empty(X.shape)
    testingE = []
    trainingE = []
    intercept = []
    coef = []
    
    kf = KFold(n, n_folds = 10, shuffle = False) #Kfold Validation function from sklearn
    print "\n File Name: ", fileName
    for trainIndex, testIndex in kf:
        
        Z_train, Z_test = Z[trainIndex], Z[testIndex]
        Y_train, Y_test = Y[trainIndex], Y[testIndex]
        
        theta = np.dot(pinv(Z_train), Y_train)
        intercept.append(theta[0])
        coef.append(theta[1])
        
        Y_est = np.dot(theta, Z_test.T)
        sol[testIndex] = Y_est
        Y_train_est = np.dot(theta, Z_train.T)
        
        trainingSize = len(trainIndex)
        testingSize = len(testIndex)
       
        mat = (Y_test - Y_est)
        testingError = np.dot(mat.T, mat)/testingSize
        mat = (Y_train - Y_train_est)
        trainingError = np.dot(mat.T, mat)/trainingSize
        
        testingE.append(testingError)
        trainingE.append(trainingError) 
    
    ################# Print Results #############################################
    print "Intercept : ", np.mean(intercept), "\t", "Coeffecient :", np.mean(coef)
    print "Testing Error : ", np.mean(testingE)," \t", "Training Error : ", np.mean(trainingE)
    ################# End of Print #############################################
    
    sol = sol.reshape((200,1))
    orig = plt.scatter(X,Y, color = 'red')
    pred = plt.scatter(X, sol,  color='green', s=100)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(fileName)

    
    plt.show()
    return [sol, orig, pred]
    
    
def inbuiltModel(fileName):
    data = readData(fileName)
    X = data[:, 0] #All the x values, column matrix
    Y = data[:, 1] #Corresponding Y values, column matrix
    X = X.reshape((200,1))
    Y = Y.reshape((200,1))
    lr = linear_model.LinearRegression()
    #sol = cross_val_predict(lr, X, Y, cv=10)
    

    sol = lr.fit(X,Y)
    YPred = sol.predict(X)
    inbuilt = plt.scatter(X, YPred,  color='blue', s=10)
    print "Inbuilt Function\nIntercept :" , sol.intercept_ , "Coeffecients: ", sol.coef_
    print "Training Error : ", np.mean((YPred - Y) ** 2)
  
    return [lr, sol, inbuilt]


def nonLinearModel(fileName, maxDegree, reduceData):
    data = readData(fileName)
    n = len(data)
    X = data[:, 0] #All the x values, column matrix
    Y = data[:, 1] #Corresponding Y values, column matrix
    
    sol = np.empty(Y.shape)
    kf = KFold(n, n_folds = 10, shuffle = False) #Cross Validation
    
    minTestingError = np.inf
    
    for degree in xrange(1,maxDegree + 1): #loop through different degrees
        poly = PolynomialFeatures(degree)
        Z = poly.fit_transform(X.reshape(200,1))
        
        for trainIndex, testIndex in kf:
            if(reduceData == True): 
                trainIndex = trainIndex[:(int((0.10)*len(trainIndex)))]
            Z_train, Z_test = Z[trainIndex], Z[testIndex]
            Y_train, Y_test = Y[trainIndex], Y[testIndex]
            
            theta = np.dot(pinv(Z_train), Y_train)
            Y_est = np.dot(theta, Z_test.T)
            sol[testIndex] = Y_est
            
        mat = (Y_test - Y_est)
        testingError = np.dot(mat.T,mat)/len(testIndex)
        
        if(testingError < minTestingError):
           minTestingError =  testingError
           optimalDegree = degree
           #optTheta 
           predicted = sol
           
    
    print "Optimal Degree : " , optimalDegree
    print("Intercept    :   {} \nCoefficients : {}\nTesting Error : {}".format(theta[0], theta[1:], minTestingError))
    
    
    orig = plt.scatter(X,Y, color = 'red')
    pred = plt.scatter(X, predicted,  color='blue', s=10)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend((orig, pred),
           ('Original', 'Predicted'),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=8)
    plt.show()
  
    
    
if __name__ == "__main__":
    
    svarFiles = listdir("Data/svr")
    
    for fileName in svarFiles:
        '''
        sol1, orig, pred = linearModel(fileName) #linear Model function written by me
        lr, sol2, inbuilt = inbuiltModel(fileName) #Applies inbuilt sklearn function
        
        plt.legend((orig, pred, inbuilt),
           ('Original', 'Predicted', 'Inbuilt Function'),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=8)
        
           
        plt.savefig("Plots/" + fileName[:-3] + "png")

        fig, ax = plt.subplots() 
        '''
        print "\nFileName : ", fileName
        print "Fitting Polynomial model"
        
        reduceData = False
        pred = nonLinearModel(fileName,6, reduceData)
        plt.title(fileName)
        plt.savefig("Plots/" + "poly-" +fileName[:-3] + "png")
        
        fig, ax = plt.subplots() 
        
        print "\n reducing training data \n"
        reduceData = True
        pred = nonLinearModel(fileName,6, reduceData)
        plt.title(fileName)
        plt.savefig("Plots/" + "reduced-poly-" +fileName[:-3] + "png")
        fig, ax = plt.subplots() 
        

    
    