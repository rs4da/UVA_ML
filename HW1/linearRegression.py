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


def standRegresOpt1(xVal, yVal):
    # uses normal equations
    XmulX = np.matmul(xVal.T, xVal)
    invXmulX = np.linalg.inv(XmulX)
    invXtrans = np.matmul(invXmulX, xVal.T)
    theta = np.matmul(invXtrans, yVal)
    
    return theta

def standRegresOpt2(X, Y, alpha, epochs):
    # uses gradient descent
    xVal, yVal = shuffle(X, Y)
    theta = np.asarray([1, 1])
    m = len(xVal[:,1])
    loss_arr = []
    for iter in range(0, epochs):
        hypothesis = np.dot(xVal, theta)
        loss = hypothesis - yVal
        loss_arr.append(np.mean(loss))
        J = np.sum(loss ** 2) / (2 * m)  # cost
        gradient = np.dot(xVal.T, loss) / m
        theta = theta - alpha * gradient  # update
    
    return theta, loss_arr

def standRegresOpt3(X, Y, alpha, epochs):
    # uses mini-batch gradient descent
    xVal, yVal = shuffle(X, Y)
    theta = np.asarray([1, 1])
    # Split into batches
    xBatches = np.split(xVal, 100)
    yBatches = np.split(yVal, 100)
    # iterate over each batch and perform GD
    for i in range(len(yBatches)-1):
        m = len(xBatches[i][:,1])
        loss_arr = []
        for iter in range(0, epochs):
            hypothesis = np.dot(xBatches[i], theta)
            loss = hypothesis - yBatches[i]
            J = np.sum(loss ** 2) / (2 * m)  # cost
            gradient = np.dot(xBatches[i].T, loss) / m
            theta = theta - alpha * gradient  # update
    
    return theta

def shuffle(matA, matB):
    # this function performs a random shuffle of X and Y while keeping the rows alligned
    assert len(matA) == len(matB)
    newA = np.empty(matA.shape, dtype=matA.dtype)
    newB = np.empty(matB.shape, dtype=matB.dtype)
    perm = np.random.permutation(len(matA))
    for i, j in enumerate(perm):
        newA[j] = matA[i]
        newB[j] = matB[i]
    return newA, newB

# here we load the dataset and plot the points
X, Y = loadDataSet('Q2data.txt')
plt.scatter(X[:,1], Y)
plt.show()

# we find the first theta using normal equations
Theta = standRegresOpt1(X, Y)
print("Standard Theta: ", Theta)

# plot the model
x = X[:,1]
y = Theta[0] + Theta[1] * x
plt.scatter(x, Y)
plt.plot(x, y, '-', color="red")
plt.show()

# we find the first theta using gradient descent
epochs = 10000 # number of runs we do over the whole dataset
alpha = 0.005 # learning rate
theta2, loss = standRegresOpt2(X, Y, alpha, epochs)
print("Gradient Descent theta: ", theta2)

# plot the model
x = X[:,1]
y = theta2[0] + theta2[1] * x
plt.scatter(x, Y)
plt.plot(x, y, '-', color="green")
plt.show()

# plot epoch against loss
x = range(epochs)
y = loss
plt.plot(x, y, '--', color="blue")
plt.show()

# we find the first theta using mini batch gradient descent
epochs = 10
theta3 = standRegresOpt3(X, Y, 0.005, epochs)
print("Mini-Batch Gradient Descent: ", theta3)

# plot the model
x = X[:,1]
y = theta3[0] + theta3[1] * x
plt.scatter(x, Y)
plt.plot(x, y, '-', color="pink")
plt.show()
