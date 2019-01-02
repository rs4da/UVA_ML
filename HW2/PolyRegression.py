# Roman Sharykin
# rs4da

# CS 4501 - Machine Learning

# HW2


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

def polyRegresTrain(xVal, yVal, max):
    # your code

    x = xVal[:, 1]
    arr = [xVal[:, 0]]
    for i in range(max):
        arr.append(np.power(x, i+1))
    phi = np.vstack(arr).T
    phiTphi = np.matmul(phi.T, phi)
    inverse = np.linalg.inv(phiTphi)
    invPhiT = np.matmul(inverse, phi.T)
    theta = np.matmul(invPhiT, yVal)
    prediction = theta[0]
    for i in range(len(theta)-1):
        prediction += np.power(x,i+1)*theta[i+1]

    loss = yVal - prediction
    loss_len = len(loss)
    MSE_t_N = 0
    for i in loss:
        MSE_t_N += np.square(i)
    MSE = MSE_t_N/loss_len


    return theta, MSE
def polyRegresTest(xVal,yVal, ptheta):
    x = xVal[:, 1]

    prediction = ptheta[0]
    for i in range(len(ptheta) - 1):
        prediction += np.power(x, i + 1) * ptheta[i + 1]

    loss = yVal - prediction
    loss_len = len(loss)
    MSE_t_N = 0
    for i in loss:
        MSE_t_N += np.square(i)
    MSE = MSE_t_N / loss_len

    return MSE

def TrainValidatePolyRegress(X_train, Y_train, X_test, Y_test):
    d_set = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    MSE_train = []
    MSE_test = []
    ptheta_arr = []

    for d in d_set:
        ptheta, x = polyRegresTrain(X_train, Y_train, d)
        ptheta_arr.append(ptheta)
        MSE_train.append(x)
        y = polyRegresTest(X_test, Y_test, ptheta)
        MSE_test.append(y)

    plt.plot(d_set, MSE_train, '-', color="red")
    plt.plot(d_set, MSE_test, '-', color="blue")
    plt.show()

    best_order = MSE_test.index(min(MSE_test))
    ptheta = ptheta_arr[best_order]

    x_bin = np.linspace(0, 7, 1000)
    y_poly = ptheta[0]
    for i in range(len(ptheta) - 1):
        y_poly += ptheta[i + 1] * np.power(x_bin, i + 1)

    plt.scatter(X_test[:,1], Y_test)
    plt.show()

    plt.scatter(X_test[:,1], Y_test)
    plt.plot(x_bin, y_poly, '-', color="orange")
    plt.show()

    print("Min train loss: ", min(MSE_train))
    print("Min test loss: ", min(MSE_test))

    print("Train loss when test loss is min: ", MSE_train[best_order])
    print("Poly of order: ", best_order)
    print("Best poly order if using train data: ", MSE_train.index(min(MSE_train)))
    print("Test loss when train loss is min: ", MSE_test[MSE_train.index((min(MSE_train)))])

    return ptheta



# here we load the dataset and plot the points
X, Y = loadDataSet('polyRegress_train.txt')

X_tst, Y_tst = loadDataSet('polyRegress_validation.txt')
plt.scatter(X[:,1], Y)
plt.show()

x = X[:,1]
ptheta, MSE = polyRegresTrain(X, Y, 3)

x_bin = np.linspace(0, 7, 1000)
y_poly = ptheta[0]
for i in range(len(ptheta) - 1):
    y_poly += ptheta[i+1]*np.power(x_bin, i+1)

plt.scatter(x, Y)
plt.plot(x_bin, y_poly, '-', color="red")
plt.show()

print("Poly Theta: ", ptheta)
print("MSE: ", MSE)

MSE_test = polyRegresTest(X_tst, Y_tst, ptheta)
print("MSE Test Loss: ", MSE_test)

pthetaBest = TrainValidatePolyRegress(X, Y, X_tst, Y_tst)



