#!/usr/bin/python

# Roman Sharykin
# rs4da


import sys
import random
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Collaborated with Jed Barson

def loadData(fileDj):
    data = []
    with open(fileDj) as f:
        for i in f:
            i = i.strip().split(" ")
            ex = i
            data.append(ex)

    data = np.asarray(data).astype(float)
    return data

def getInitialCentroids(X, k):
    initialCentroids = []
    min = np.amin(X[:, :-1], axis=0)
    max = np.amax(X[:, :-1], axis=0)

    for i in range(k):
        centroid = []
        for x in range(len(min)):
            value = random.uniform(min[x], max[x])
            centroid.append(value)
        initialCentroids.append(centroid)

    initialCentroids = np.asarray(initialCentroids)

    return initialCentroids


def getDistance(pt1, pt2):
    s = 0
    for x in range(len(pt1)):
        diff = pt1[x] - pt2[x]
        sq = diff ** 2
        s += sq

    dist = s ** .5
    return dist


def allocatePoints(X, centroids):
    clusters = [[] for k in range(len(centroids))]

    for point in X:
        minD = np.inf
        i = 0
        for x in range(len(centroids)):
            d = getDistance(point[:-1], centroids[x])
            if d < minD:
                minD = d
                i = x
        clusters[i].append(point)

    clusters = np.asarray(clusters)

    for i in range(len(clusters)):
        clusters[i] = np.array(clusters[i])
    return clusters


def updateCentroids(clusters):
    new_centroids = []

    for cluster in clusters:
        centroid = np.mean(cluster[:, :-1], axis=0)
        new_centroids.append(centroid.tolist())

    return new_centroids


def visualizeClusters(clusters, centroids):
    color_map = ['r', 'b']

    for i in range(len(clusters)):
        for point in clusters[i]:
            x = point[0]
            y = point[1]
            plt.scatter(x, y, color=color_map[i])

    for centroid in centroids:
        x = centroid[0]
        y = centroid[1]
        plt.scatter(x, y, color='black', marker='*')

    plt.show()


def kmeans(X, k, maxIter=1000):

    centroids = getInitialCentroids(X, k)
    clusters = []

    for x in range(maxIter):
        clusters = allocatePoints(X, centroids)
        centroids = updateCentroids(clusters)

    return clusters, centroids


# There are some rare occasions when the code crashes (usually around k = 5 on my machine)
# I can't figure out what's causing this issue, so if my code crashes please just give it another chance and rerun it.
def kneeFinding(X, kList):
    print("Knee Finding")
    SSE = []
    try:
        for k in kList:
            clusters, centroids = kmeans(X, k, maxIter=200)
            total_sum = 0
            print("k: ", k)
            for cluster in clusters:
                cluster_sum = 0
                for pt1 in cluster:
                    for pt2 in cluster:
                        dist = getDistance(pt1, pt2)
                        dist_sq = dist ** 2
                        cluster_sum += dist_sq
                total_sum += cluster_sum
            SSE.append(total_sum)

        plt.plot(kList, SSE)
        plt.xlabel("K")
        plt.ylabel("SSE")
        plt.show()

    except IndexError:
        print("Index error occurred, please rerun the program.")


def purity(X, clusters):
    purities = [0 for i in range(len(clusters))]
    for x in range(len(clusters)):
        numP = len(clusters[x])
        l = clusters[x][:, -1]
        counts = Counter(l)
        max_count = max(counts.values())
        p = max_count / numP
        purities[x] = p
    return purities


def main():
    datadir = sys.argv[1]

    pathDataset1 = datadir + '/humanData.txt'
    dataset1 = loadData(pathDataset1)

    ### I commented out the commands to load in data set 2 since it's never used ###
    # pathDataset2 = datadir + '/audioData.txt'
    # dataset2 = loadData(pathDataset2)

    # Q4
    kneeFinding(dataset1, range(1, 7))

    # Q5
    clusters, centroids = kmeans(dataset1, 2, maxIter=1000)
    purities = purity(dataset1, clusters)
    print("...Purities...")
    print(purities)
    visualizeClusters(clusters, centroids)


if __name__ == "__main__":
    main()
