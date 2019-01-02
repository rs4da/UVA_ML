# Roman Sharykin
# rs4da

# Machine Learning CS 4501

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
from scipy import stats


def loadDataSet(filename):
    # Used to read in the data from given data file
    ones = []
    X1 = []
    X2 = []
    Y = []
    data = open(filename)
    openData = csv.reader(data, delimiter=' ')
    
    for one_col, x1, x2, y in openData:
        ones.append(float(one_col))
        X1.append(float(x1))
        X2.append(float(x2))
        Y.append(float(y))
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    Y = np.asarray(Y)
    ones = np.asarray(ones)
    new = np.vstack([ones, X1, X2]).T
    return new, Y

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

def ridgeRegress(xVal, yVal, lambdaV):
    # uses normal equations
    XmulX = np.matmul(xVal.T, xVal)
    XmulX = XmulX + (lambdaV * np.identity(len(XmulX)))
    invXmulX = np.linalg.inv(XmulX)
    invXtrans = np.matmul(invXmulX, xVal.T)
    beta = np.matmul(invXtrans, yVal)
    
    return beta

def cv(xVal, yVal):
	
	lambdas = [0.02*i for i in range(1, 51)]
	x_splits = np.split(xVal, 4)
	y_splits = np.split(yVal, 4)

	errors = []
	for lamb in lambdas:
		fold_errors = []
		i = 0
		while i < 4:
			x_test = x_splits[0]
			y_test = y_splits[0]

			del x_splits[0]
			del y_splits[0]

			x_train = np.concatenate((x_splits[0], x_splits[1], x_splits[2]))
			y_train = np.concatenate((y_splits[0], y_splits[1], y_splits[2]))

			beta = ridgeRegress(x_train, y_train, lamb)

			y_hat = np.matmul(beta, x_test.T)
			error = y_hat - y_test
			error_sq = np.square(error)
			error_sum = np.sum(error_sq, axis=0)
			error_av = error_sum/len(x_test)
			fold_errors.append(error_av)

			x_splits.append(x_test)
			y_splits.append(y_test)
			i += 1

		fold_errors = np.asarray(fold_errors)
		avg_error = np.average(fold_errors)
		errors.append(avg_error)
	
	errors = np.asarray(errors)
	ind = np.argmin(errors)
	lambdaBest = lambdas[ind]

	plt.plot(lambdas, errors, "green")
	plt.title("Lamba vs Mean Squared Error (4 Folds)")
	plt.show()
	return lambdaBest


def standRegress(xVal, yVal):
	# uses normal equations
	XmulX = np.matmul(xVal.T, xVal)
	invXmulX = np.linalg.inv(XmulX)
	invXtrans = np.matmul(invXmulX, xVal.T)
	theta = np.matmul(invXtrans, yVal)
	
	return theta



X, Y = loadDataSet('RRdata.txt')

betaLR = ridgeRegress(X, Y, lambdaV = 0)
print("betaLR: ", betaLR)

x1 = X[:,1]

x2 = X[:,2]
learned_y = betaLR[0] + (betaLR[1] * x1) + (betaLR[2] * x2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, Y)
ax.plot_trisurf(x1, x2, learned_y,  color="red")
plt.title("Ridge Regression: lambda = 0")
plt.show()



lambdaBest = cv(X, Y)
print("lambdaBest: ", lambdaBest)

betaRR = ridgeRegress(X, Y, lambdaBest)
print("betaRR: ", betaRR)

best_learned = betaRR[0] + (betaRR[1] * x1) + (betaRR[2] * x2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, Y)
#ax.hold(True)
ax.plot_trisurf(x1, x2, best_learned, color="purple")
plt.title("Ridge Regression: lambda = 0.5")
plt.show()

theta = standRegress(X[:,:2], X[:,2])
print("Theta: ", theta)

x = X[:,1]
y = theta[0] + theta[1] * x
plt.scatter(x, X[:,2])
plt.plot(x, y, '-', color="pink")
plt.show()
