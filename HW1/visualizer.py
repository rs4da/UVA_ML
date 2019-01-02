# Roman Sharykin
# rs4da

# Machine Learning CS 4501

import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy import stats


def loadDataSet(filename):
    # Used to read in the data from given data file
    ones = []
    X = []
    Y = []
    data = open(filename)
    openData = csv.reader(data, delimiter='\t')
    for one_col, x, y in openData:
        ones.append(float(one_col))
        X.append(float(x))
        Y.append(float(y))
    X = np.asarray(X)
    Y = np.asarray(Y)
    ones = np.asarray(ones)
    new = np.vstack([ones, X]).T
    
    return new, Y


# here we load the dataset and plot the points
X, Y = loadDataSet('Q2data.txt')
plt.scatter(X[:,1], Y)
plt.show()


# plot the model
x = X[:,1]
y = Theta[0] + Theta[1] * x
plt.scatter(x, Y)
plt.plot(x, y, '-', color="red")
plt.show()

