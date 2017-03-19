import operator
import logging
from random import randint

def distance(a, b):
    """ Get the distance between two vectors a and b """
    sum = 0
    if len(a) == len(b):
        for i, a_i in enumerate(a):
            diff = a_i - b[i]
            sum += diff * diff
        return sum
    return 0

def cluster(means, data, n):
    """ Cluster the data around the means. Vectors are of length n"""
    num_pixels = len(data) / n
    k = len(means) / n
    sums = [0 for i in range(len(means))]
    counts = [0 for i in range(k)]
    for i in range(num_pixels):
        offset = i * n
        pixel = data[offset : offset + n]
        j_m = 0
        smallest_d = 9999999
        # Find the closest mean
        for j in range(k):
            m = j * n
            d = distance(pixel, means[m : m + n])
            if d < smallest_d:
                j_m = j
                smallest_d = d
        counts[j_m] += 1
        m = j_m * n
        # Add the pixel to the running count for that mean's cluster
        sums[m : m + n] = map(operator.add, sums[m : m + n], pixel)
    # Find the new average for all the data points in each cluster
    means = [-1 if counts[i/n] == 0 else x / (counts[i/n]) for i, x in enumerate(sums)]
    return (means, counts)

def get_random_mean(data, n, max_val):
    """ Generate a random mean from the dataset """
#    new_means = [randint(0, max_val) for i in range(k * n)]
    num_pixels = len(data) / n
    start = randint(0, num_pixels)
    # Pick a random pixel and add a random perturbation, clamping
    return map(lambda x: min(max_val, max(0, x + randint(-1,1))), data[start : start + n])

def kmeans(k, data, n, max_val, t):
    """ Run k means on some data with vector length n with max value max
        and convergence threshold t"""
    new_means = []
    for i in range(k):
        new_means += get_random_mean(data, n, max_val)
    logging.debug(new_means)
    means = [-max_val for i in range(len(new_means))] # dummy means
    safety = 0
    # Run until means converge or we run this too many times
    while distance(means, new_means) > t and safety < 1000:
        means = new_means
        (new_means, counts) = cluster(means, data, n)
        safety += 1
    logging.debug("Ran cluster %d times for threshold  %d" % (safety, t))
    return (new_means, counts)