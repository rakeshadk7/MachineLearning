import numpy as np
from os import listdir
import matplotlib.pyplot as plt

def readData(fileName): # Read data from file
    filePath = "Data/svr/" + fileName 
    f = open(filePath)
    data = []
    for line in f.readlines()[5:]: #Skip Headers
        data.append(line.strip().split(" "))
    data = np.array(data, dtype="f")
    f.close()
    return data
    
def scatterPlots(files): 
    index = 1
    for fileName in files: #Create a scatterplot matrix of all 4 data sets
        data = readData(fileName)
        plt.subplot(2,2, index)
        plt.title(fileName)
        #plt.xlabel("X")
       # plt.ylabel("Y")
        plt.scatter(data[:, 0], data[:, 1], color="blue")
        index += 1
        plt.savefig("Plots/scatterPlots.png")
    plt.show()
    
if __name__ == "__main__":
    svarFiles = listdir("Data/svr")
    scatterPlots(svarFiles)
    
    