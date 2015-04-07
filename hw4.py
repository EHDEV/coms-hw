__author__ = 'eliashussen'

import numpy as np
import pandas as p
import sys, os
from collections import Counter
from matplotlib import pyplot as plt

def generate_data():
    mu1 = [0, 0]
    mu2 = [3, 0]
    mu3 = [0, 3]
    sigma = [[1, 0], [0, 1]]

    first = np.concatenate(
        (np.random.normal(0, 1, 100), np.random.normal(3, 1, 250), np.random.normal(0, 1, 150)), 0)

    second = np.concatenate(
        (np.random.normal(0, 1, 100), np.random.normal(0, 1, 250), np.random.normal(3, 1, 150)), 0)

    data = np.vstack((first, second)).T

    return data


def kmeans_clustering(bigK, iteration=20):
    data = generate_data()
    np.random.shuffle(data)

    mu_idx = np.random.random_integers(0, 500, bigK).tolist()
    # randomly initializing the centroids of each cluster
    mu = data[mu_idx, :]
    ci = []
    colors = ['#b2182b',
                '#d6604d',
                '#f4a582',
                '#fddbc7',
                '#d1e5f0',
                '#92c5de',
                '#4393c3',
                '#2166ac']

    for iterate in range(iteration):
        distances = []
        for littlek in range(bigK):

            distances.append(np.sqrt(np.sum((data - mu[littlek, :]) ** 2, 1)))

        ci = np.argmin(np.array(distances), 0)

        for littlek in range(bigK):

            xs_in_clust_k = np.where(ci == littlek)[0]
            nk = xs_in_clust_k.shape[0]
            mu[littlek, :] = np.sum(data[xs_in_clust_k, :], 0) / nk

    ax1 = plt.subplot(211)
    ax1.scatter(data[:, 0], data[:, 1], c=ci)
    ax1.scatter(mu[:, 0], mu[:, 1], c=colors, s=60, marker='+')
    ax1.set_title("Plot of 500 Points and Their Clusters. K = " + str(bigK))
    ax1.legend(loc='upper right', shadow=True, fontsize='small')
    plt.savefig('./out/kmeans_points_plot.png')
    return ci, mu

if __name__ == '__main__':
   c, m = kmeans_clustering(5)
   print c
   print m