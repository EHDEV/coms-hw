__author__ = 'eliashussen'

import numpy as np
import pandas as p
import sys
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from numpy import linalg as LA


def get_data():
    """ fetches data from file into a pandas dataframe which will then
        be converted into a numpy matrix
    """

    trainpath = sys.argv[1]
    legendpath = sys.argv[2]

    traindata = p.read_csv(trainpath, names=["t1idx", "t1score", "t2idx", "t2score"])

    teams = np.array(np.unique(traindata.t1idx))
    team_names = p.read_csv(legendpath, names=["team_name"])
    team_names = np.array(team_names.values).reshape((teams.shape[0],))

    return traindata.values, teams.tolist(), team_names


def transition_matrix(iteration=10):
    size = 759
    pj1 = 1
    pj2 = 3

    rankings = []

    wt_eig_norm = []

    colors = ['#b2182b',
              '#d6604d',
              '#f4a582',
              '#fddbc7',
              '#d1e5f0',
              '#92c5de',
              '#4393c3',
              '#2166ac']

    wt = np.matrix(np.repeat(1.0 / size, size).reshape((1, size)))

    game_data, teams, team_names = get_data()

    M = np.matrix(np.zeros((size, size)))

    iteration = 100

    for row in range(game_data.shape[0]):

        try:
            j1 = teams.index(game_data[row, 0])
            j2 = teams.index(game_data[row, 2])

        except IndexError:
            print "Index Error "
            break

        if game_data[row, pj1] > game_data[row, pj2]:

            M[j1, j1] += (1 + float(game_data[row, pj1]) / (game_data[row, pj1] + game_data[row, pj2]))

            M[j2, j1] += (1 + float(game_data[row, pj1]) / (game_data[row, pj1] + game_data[row, pj2]))

        elif game_data[row, pj2] > game_data[row, pj1]:

            M[j2, j2] += (1 + float(game_data[row, pj2]) / (game_data[row, pj1] + game_data[row, pj2]))

            M[j1, j2] += (1 + float(game_data[row, pj2]) / (game_data[row, pj1] + game_data[row, pj2]))

    # Normalizing the Matrix rows
    for mr in range(M.shape[0]):
        M[mr] /= np.sum(M[mr])

    eigvalue, eigvec = LA.eig(M.T)

    for row in range(iteration):

        wt *= M

        rankings = sorted(range(size), key=np.array(wt).reshape(size, ).tolist().__getitem__, reverse=True)

        l1_err = LA.norm(wt.T - (eigvec[:, 0] / np.sum(eigvec[:, 0])), ord=1)
        wt_eig_norm += [l1_err]

        # print team_names[rankings]

    plt.plot(range(iteration), wt_eig_norm, c=colors[1])
    plt.title("Plot of M.T Eigenvector L1 Distance from W_t")
    plt.savefig("./out/hw5/eig_wt_err.png")
    plt.close()

    return rankings, wt, team_names[rankings]


if __name__ == '__main__':
    transition_matrix()




