# UVA CS 4501 Machine Learning- KNN

__author__ = 'rs4da'

# Collaborated with Jed Barson

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def read_csv(filename):
    data = []
    with open(filename) as datafile:
        next(datafile)
        for line in datafile:
            l = np.array(line.split('\t')).astype(np.float)
            data.append(l)
    data = np.asarray(data)
    np.random.shuffle(data)
    return data

def fold(data, currenti, kfold):
    foldSize = len(data) // kfold
    start = currenti * foldSize
    end = start + foldSize
    testI = list(range(start, end))
    trainI = list(range(0, start)) + list(range(end, len(data)))
    training = data[trainI]
    testing = data[testI]
    return (training, testing)


def dist(x, y):
    sq_sum = 0
    for i in range(len(x) - 1):
        diff = x[i] - y[i]
        diff = diff ** 2
        sq_sum += diff
    sq_root = sq_sum ** .5
    return sq_root


def classify(training, testing, K):
    predict = []

    for t in testing:
        k_closest = []
        distances = []
        for i in range(K):
            k_closest.append(training[i])
            distances.append(np.inf)

        for y in training:
            d = dist(t, y)
            if d < max(distances):
                j = distances.index(max(distances))
                k_closest[j] = y
                distances[j] = d
        total = 0
        for point in k_closest:
            total += point[-1]
        avg = total / len(k_closest)
        if avg >= .5:
            predict.append(1)
        else:
            predict.append(0)

    return predict


def calc_accuracy(predictions, labels):
    right = 0
    for x in range(len(predictions)):
        if predictions[x] == labels[x]:
            right += 1
    accuracy = right / len(predictions)
    return accuracy

def findBestK(data, kfold):

    k_arr = [3, 5, 7, 9, 11, 13]
    accuracy = []
    for k in k_arr:
        sum = 0
        print("Current K: ", k)
        for i in range(0, kfold):
            training, testing = fold(data, i, kfold)
            predictions = classify(training, testing, k)
            labels = testing[:,-1]
            sum += calc_accuracy(predictions, labels)
        print("Accuracy: ", sum / kfold)
        accuracy.append(sum/kfold)

    most_accurate = max(accuracy)
    index = accuracy.index(most_accurate)

    print("Best K: ", k_arr[index])
    print("Accuracy: ", most_accurate)
    return k_arr, accuracy, index


if __name__ == "__main__":
    filename = "Movie_Review_Data.txt"
    data = np.asarray(read_csv(filename))
    kfold = 4
    k_arr, accuracy, best_index = findBestK(data, kfold)

    plt.bar(k_arr, accuracy)
    plt.xlabel('K', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.show()


