__author__ = 'eliashussen'

import numpy as np
import pandas as p
import sys
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from numpy import linalg as LA


class Problem1:
    def __init__(self):
        pass

    def get_data(self):
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

    def transition_matrix(self, iteration=10):
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
        # wt = np.matrix(np.random.uniform(0, 1, size).reshape((1, size)))

        game_data, teams, team_names = self.get_data()

        M = np.matrix(np.zeros((size, size)))

        for row in range(game_data.shape[0]):

            try:
                j1 = teams.index(game_data[row, 0])
                j2 = teams.index(game_data[row, 2])

            except IndexError:
                print "Index Error "
                break

            if game_data[row, pj1] > game_data[row, pj2]:

                M[j1, j1] += 1 + (float(game_data[row, pj1]) / (game_data[row, pj1] + game_data[row, pj2]))

                M[j2, j2] += (float(game_data[row, pj2]) / (game_data[row, pj1] + game_data[row, pj2]))

                M[j1, j2] += (float(game_data[row, pj2]) / (game_data[row, pj1] + game_data[row, pj2]))

                M[j2, j1] += 1 + (float(game_data[row, pj1]) / (game_data[row, pj1] + game_data[row, pj2]))


            else:  # game_data[row, pj2] > game_data[row, pj1]:

                M[j1, j1] += (float(game_data[row, pj1]) / (game_data[row, pj1] + game_data[row, pj2]))

                M[j2, j2] += 1 + (float(game_data[row, pj2]) / (game_data[row, pj1] + game_data[row, pj2]))

                M[j1, j2] += 1 + (float(game_data[row, pj2]) / (game_data[row, pj1] + game_data[row, pj2]))

                M[j2, j1] += (float(game_data[row, pj1]) / (game_data[row, pj1] + game_data[row, pj2]))

        # Normalizing the Matrix rows
        for mr in range(M.shape[0]):
            M[mr] /= np.sum(M[mr])

        eigvalue, eigvec = LA.eig(M.T)
        first_eigvec = eigvec[:, np.argmax(eigvalue)]

        for row in range(iteration):
            rankings = sorted(range(size), key=np.array(wt).reshape(size, ).tolist().__getitem__, reverse=True)

            l1_err = LA.norm(wt.T - (first_eigvec / np.sum(first_eigvec)), ord=1)
            wt_eig_norm += [l1_err]

            wt *= M
            # print team_names[rankings[0:10]]

        print l1_err

        plt.plot(range(iteration), wt_eig_norm, c='#b2182b')
        plt.title("Plot of M.T Eigenvector L1 Distance from W_t")
        plt.ylabel('L1 Distance from wt')
        plt.xlabel('Iteration')
        plt.savefig("./out/hw5/eig_wt_err.png")
        plt.close()

        for i in range(20):
            print i, ' ', team_names[rankings[i]], ' - ',
            print np.array(wt.T[rankings[i]])[0][0]

        return rankings, wt, team_names[rankings]


class Problem2:
    def __init__(self):
        pass

    def get_data(self):
        """ fetches data from file into a pandas dataframe which will then
            be converted into a numpy matrix
        """

        file_path = sys.argv[3]

        faces = p.read_csv(file_path, header=None)

        faces_m = faces.values

        return faces_m

    def nmf(self, iteration=20):
        K = 25

        penality = []

        # X is the data matrix and W an H are th factorizations of X. X ~= W*H
        X = self.get_data()

        n, m = X.shape

        W = np.random.uniform(0, 1, n * K).reshape((n, K))
        H = np.random.uniform(0, 1, K * m).reshape((K, m))

        for iterate in range(iteration):
            # Updating H
            h_num = H * np.dot(W.T, X)
            h_den = np.dot(np.dot(W.T, W), H)

            H = h_num / h_den

            # Updating W
            w_num = W * np.dot(X, H.T)
            w_den = np.dot(np.dot(W, H), H.T)

            W = w_num / w_den

            # Calculating penalty per iteration and adding to list
            WH = np.dot(W, H)
            penality += [np.sum((X - WH) ** 2)]

        three_images = [4, 1, 2]

        for th in three_images:
            # Getting the weight that is largest from the 25 weights in H
            hix = np.argmax(H[:, th])

            im_mx = W[:, hix].reshape(32, 32)
            self.plot_image(X[:, th].reshape(32, 32), im_mx, th, hix)

        # for tx in three_images:
        #     im_mx = X[:, tx].reshape(32, 32)
        #     self.plot_image(im_mx, tx, 'x')

        plt.plot(range(iteration), penality, c='#9e0142')
        plt.title("Plot of Objective as a Function of Iteration")
        plt.ylabel('Squared Error Objective')
        plt.xlabel('Iteration')
        plt.savefig("./out/hw5/nmf_euclidean_obj_plot.png")
        plt.close()

    def plot_image(self, x_array, w_array, th, hix):

        plt.imshow(x_array, cmap="Greys_r")
        plt.axis('off')
        plt.savefig('./out/hw5/orig_x_img_' + str(th) + '.png')
        plt.close()
        plt.imshow(w_array, cmap="Greys_r")
        plt.axis('off')
        plt.savefig('./out/hw5/image_nmf_w' + str(th) + '.png')
        plt.close()


class Problem2P2:
    def get_data(self):

        docs_path = sys.argv[4]
        vocab_path = sys.argv[5]

        with open(docs_path, 'r') as doc_file:
            doc_freq = doc_file.readlines()

        vocab = p.read_csv(vocab_path, header=None)

        return doc_freq, vocab

    def nmf_divergence(self, iteration=20):

        doc_freq, vocab = self.get_data()

        K = 25
        m = len(doc_freq)
        n = vocab.shape[0]
        obj_list = []

        div_penalties = []

        X = np.zeros((n, m))
        W = np.random.uniform(0, 1, n * K).reshape((n, K))
        H = np.random.uniform(0, 1, K * m).reshape((K, m))

        # Constructing the data matrix X line by line, word by word
        for didx, line in enumerate(doc_freq):
            words_and_counts = line.split(',')

            for item in words_and_counts:
                widx, count = tuple(item.split(':'))
                X[int(widx) - 1, didx] = count

        for iterate in range(iteration):

            WH = np.dot(W, H) # Approximation of X
            xwh_purple = X / (WH + 1e-16) # Purple Matrix
            Wtn = W.T / np.sum(W.T, 1).reshape(W.T.shape[0], 1) # Row normalized W matrix (cyan)

            H = H * np.dot(Wtn, xwh_purple) # Updating H

            WH = np.dot(W, H)
            xwh_purple = X / (WH + 1e-16) # Creating the purple matrix again for W update
            Htn = H.T / np.sum(H.T, 0).reshape(1, H.T.shape[1]) # Column normalizaing H

            W = W * np.dot(xwh_purple, Htn) # Updating W

            # Matrix multiplying for the objective function then sum ober all the values of the matrix
            obj_m = -(X * np.nan_to_num(np.log(WH)) - WH)
            obj_list += [np.sum(obj_m)]

        # Normalizing the columns of W to sum to 1

        W = W / np.sum(W, 0).reshape(1, W.shape[1])

        # five_cols = np.random.random_integers(0, W.shape[1], 5)

        five_cols = [22, 12, 7, 10, 17]

        tmtop = []
        for col in five_cols:

            topten = np.argsort(W[:, col])[::-1][0:10]
            print 'For Column ', col, ' the top ten words are: '
            ttmx = np.concatenate((vocab.loc[topten].values, W[topten, col].reshape(len(topten), 1)), 1)
            print ttmx

        plt.plot(range(iteration), obj_list, c='#d53e4f')
        plt.title("Plot of Divergence Objective as a Function of Iteration")
        plt.ylabel('Divergence Objective')
        plt.xlabel('Iteration')
        plt.savefig("./out/hw5/nmf_divergence_obj_plot.png")
        plt.close()

if __name__ == '__main__':
    p1 = Problem1()
    p1.transition_matrix(1000)

    # p2 = Problem2()
    # p2.nmf(200)

    # pp2 = Problem2P2()
    # pp2.nmf_divergence(200)


