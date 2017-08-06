"""
Ayse Bilge Gunduz
Dominant Color
K-means and histogram calculation is written in manually.
Opencv, numpy, matplotlib is used
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

class KMeansClustering():
    #initial values
    def __init__(self):
        self.array = None
        self.k = None
        self.centroids = None
        self.cluster_to_labels = None
        self.clusters = None
        self.previous_k_points = None
        self.new_k_points = None
        self.iteration_no = 20

    def initialise_img_from_img(self, img, kval):
        self.array = img
        self.k = kval

    def k_means(self):

        self.centroids = []
        centroids_points = self.random_initialise_cluster()  # initialise cluster centroids
        for i in centroids_points:
            self.centroids.append(self.array[i])
        self.k_means_cluster()
        return self.cluster_to_labels, self.centroids

    def k_means_cluster(self):
        i = 1
        while True:
            i = i + 1
            previous_k_points = self.centroids[:]
            self.find_clusters()
            self.update_centroids()
            cur_k_points = self.centroids[:]
            v = np.array(previous_k_points) == np.array(cur_k_points)
            if (v.all() == True):
                break

    def find_clusters(self):
        #put values for each cluster
        self.cluster_to_labels = {}
        for i in range(self.k):
            self.cluster_to_labels[i] = []
        for i in range(len(self.array)):
            cluster_no = 0
            min_dist = 0
            for j in range(self.k):  # calculate euclidean distance with each k means point
                dist = self.calculate_euclidean_distance(self.array[i], self.centroids[j])
                if j == 0:
                    min_dist = dist
                elif dist < min_dist:
                    min_dist = dist
                    cluster_no = j
            self.cluster_to_labels[cluster_no].append(i)
        self.iteration_no = self.iteration_no + 1
        return self.cluster_to_labels

    def update_centroids(self):
        #find new centroids for each turn
        column_size = self.array.shape[1]
        for i in range(self.k):
            sum = np.zeros(column_size)
            length = len(self.cluster_to_labels[i])
            for index in self.cluster_to_labels[i]:
                sum = np.add(sum, self.array[index])
            self.centroids[i] = np.divide(sum, length)

    def random_initialise_cluster(self):
        # random initialisation of k centroids
        no_of_rows = self.array.shape[0]
        d = [x for x in range(no_of_rows)]
        random_points = random.sample(d, self.k)
        return random_points

    def calculate_euclidean_distance(self, x1, x2):
        #euclidean distance
        distance = np.linalg.norm(x1 - x2)
        return distance

def paint_it_black(hist, cent):
    start = 0
    end = 0
    myRect = np.zeros((50, 300, 3), dtype="uint8")
    tmp = hist[0]
    tmpC = cent[0]
    for (percent, color) in zip(hist, cent):
        if(percent > tmp):
            tmp = percent
            tmpC = color
    end = start + (tmp * 300) # try to fit my rectangle 50*300 shape
    cv2.rectangle(myRect, (int(start), 0), (int(end), 50),
                  tmpC.astype("uint8").tolist(), -1)
    start = end
    #rest will be black. Convert to black
    for (percent,color) in zip(hist, cent):
        end = start + (percent * 300)  # try to fit my rectangle 50*300 shape
        if(percent != tmp):
            color = [0, 0, 0]
            cv2.rectangle(myRect, (int(start), 0), (int(end), 50),
                      color, -1) #draw in a rectangle
            start = end
    return myRect

k=8
img = cv2.imread("pic/img2.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure()
plt.axis("off")
plt.imshow(img)

img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
km = KMeansClustering()
km.initialise_img_from_img(img, k)
clt,centers = km.k_means()
#total number of points

count = sum(len(v) for v in clt.values())
#find labels for histogram
clt_labels = np.zeros(count, dtype=np.int)
for i in range(k):
    for j in clt[i]:
        clt_labels[j] = i
hist = []
for i in range(k):
    hist.append(len(clt[i])/count)

bar = paint_it_black(hist, centers)
#paint'em all =)
plt.axis("off")
plt.imshow(bar)
plt.show()
