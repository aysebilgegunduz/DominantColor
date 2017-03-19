import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def find_histogram(clt):
    """
    create a histogram with k clusters
    :param clt:
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors(hist, cent):
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


def plot_colors2(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


img = cv2.imread("pic/img2.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure()
plt.axis("off")
plt.imshow(img)

img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
clt = KMeans(n_clusters=3)
clt.fit(img)

hist = find_histogram(clt)
bar = plot_colors2(hist, clt.cluster_centers_)

plt.axis("off")
plt.imshow(bar)
plt.show()

