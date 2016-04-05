import numpy as np
import operator
from numpy.linalg import inv, det
import itertools
import functions as f
from sklearn.cross_validation import KFold

def mMean(data,cls,cl): #Parameter calulated by train
    m = []
    for column in data.T:
        m.append(f.mean(column,cls,cl))
    return m

def mSigma(data,cls,cl,m): #parameter calculated by train 
    count = 0
    n = len(m)
    s = np.zeros((n,n))
    for index in xrange(0, len(data)):
        if(cls[index] == cl):
            count += 1
            mat = map(operator.sub, data[index], m)
            mat = np.array(mat)
            mat = mat.reshape(n,1)
            prod = np.dot(mat, mat.T) 
            
            s += prod
            
    return s/count
          
def train(data,cls): #train function 
    S= {}
    Mu = {}
    
    clsLabels = list(set(cls))
    for cl in clsLabels:
        m = mMean(data,cls,cl)
        Mu[cl] = m
        S[cl] = mSigma(data,cls,cl,m)
        
    return (Mu, S)

def classify(data, cls, classes, Mu, S):
    predicted = []
    n = len(data[0]) #number of features
    membership = {}
    for x in data:
        for cl in classes: #For each unique class
            mat = np.array(map(operator.sub, x, Mu[cl]))
            mat = mat.reshape(n,1)
            membership[cl] =  (- np.log(det(S[cl]))) - ((mat.T).dot(inv(S[cl])).dot(mat))
        predicted.append(max(membership.iteritems(), key=operator.itemgetter(1))[0])
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
    f.evaluate(predicted,cls, classes)
    return (predicted,cls)

if __name__ == "__main__":     
    reload(f)  
    fileName = "iris.csv"
    data, cls = f.readData(fileName)
    
    data = data[:100] #uses first 2 classes, 50 of each
    cls = cls[:100]   #uses first 2 classes, 50 of each
    predicted, cls = genLearning(data, cls) #Train, predict & evaluate
    f.plot_matrix(cls, predicted)
    
    
    
    