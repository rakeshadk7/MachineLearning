import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def readData(fileName): # Read data from file
    filePath = "Data/" + fileName 
    f = open(filePath)
    data = []
    cls = []
    for line in f.readlines(): 
        row = (line.strip().split(","))
        try:
            n = len(row) - 1
            if(n > 0):
                value = row[:n]
                cl = row[n]
        except IndexError:
            break
        data.append(value)
        cls.append(cl)
    data = np.array(data, dtype = 'float_')
    cls = np.array(cls)
    f.close()
    return (data,cls)


def getTP(predicted, actual, cl):
    total = 0
    for i in xrange(0,len(predicted)):
        if(predicted[i] == cl and actual[i] ==  cl):
            total += 1
    return total
    
def getTN(predicted, actual, cl):
    total = 0
    for i in xrange(0,len(predicted)):
        if(predicted[i] != cl and actual[i] !=  cl):
            total += 1
    return total
    
def getFP(predicted, actual, cl):
    total = 0
    for i in xrange(0,len(predicted)):
        if(predicted[i] == cl and actual[i] !=  cl):
            total += 1
    return total
    
def getFN(predicted, actual, cl):
    total = 0
    for i in xrange(0,len(predicted)):
        if(predicted[i] != cl and actual[i] ==  cl):
            total += 1
    return total
    
def mean(data,cls,cl):
    total = 0
    count = 0
    for index in xrange(0,data.shape[0]):
         if cls[index] == cl:
             count += 1                      
             total += data[index]
    return float(total)/count
  
def sigma(data,cls,cl,mean):
    count = 0
    total = 0.0
    for index in xrange(0,data.shape[0]):
        if cls[index] == cl:
            count += 1
            total += pow((data[index] - mean),2)
    return (total/count)
      
    
def evaluate(predicted, actual, cLabels):
    n = len(predicted)
    TP = getTP(predicted, actual, cLabels[0])
    TN = getTN(predicted, actual, cLabels[0])
    FP = getFP(predicted, actual, cLabels[0])
    FN = getFN(predicted, actual, cLabels[0])
    
    #print n,TP,TN,FP, FN
    accuracy = float(TP + TN) /n
    precision = float(TP)/ float(TP + FP)
    recall = float(TP)/float(TP + FN)
    fMeasure = (precision + recall) / 2
    
    print "accuracy : ", accuracy
    print "precision : ", precision
    print "recall : ",recall
    print "F-Measure", fMeasure
    
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def plot_matrix(cls, predicted):
    cm = confusion_matrix(cls, predicted)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm, target_names = list(set(cls)))
    
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized,list(set(cls)), title='Normalized confusion matrix')
    
    plt.show()
   

