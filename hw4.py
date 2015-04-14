__author__ = 'eliashussen'

import numpy as np
import pandas as p
import sys
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def get_matrices():
    """ fetches data from file into a pandas dataframe which will then
        be converted into a numpy matrix
    """

    trainpath = sys.argv[1]
    testpath = sys.argv[2]
    movies = sys.argv[2]

    traindata = p.read_csv(trainpath, names=["userid", "movieid", "rating"])
    testdata = p.read_csv(testpath, names=["userid", "movieid", "rating"])
    movies = p.read_csv('/Users/eliashussen/code/ML/coms-hw/data/movies_csv/movies.txt', names=['title'])

    traindata = traindata.pivot(index="userid", columns="movieid", values="rating")
    # testdata = testdata.pivot(index="userid", columns="movieid", values="rating")

    traindata = traindata.fillna(0)
    testdata = testdata.fillna(0)

    train_mx = traindata.values
    test_mx = testdata.values
    movies = np.array(movies.values)[:, 0]

    return train_mx, test_mx, list(traindata.index), list(traindata.columns), \
           list(testdata.index), list(testdata.columns), movies


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

    mu_idx = np.random.choice(500, bigK, replace=False).tolist()
    # randomly initializing the centroids of each cluster
    mu = data[mu_idx, :]
    ci = []
    sumdist = []
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
        ob = 0
        print iterate
        for littlek in range(bigK):
            distances.append(np.sum(np.square(data - mu[littlek, :]), 1))

        ci = np.argmin(np.array(distances), 0)

        for i in range(len(ci)):
            ob += np.array(distances)[ci[i], i]

        for littlek in range(bigK):
            xs_in_clust_k = np.where(ci == littlek)[0]
            nk = xs_in_clust_k.shape[0]
            mu[littlek, :] = np.sum(data[xs_in_clust_k, :], 0) / nk

        sumdist += [ob]

    plt.plot(range(iteration), sumdist, label="Objective Function", c=colors[1])
    plt.title("Plot of K Means Objective Function by Iteration")
    plt.savefig("./out/kmeans_objective_plot_k" + str(bigK) + ".png")
    plt.close()

    plt.scatter(data[:, 0], data[:, 1], c=ci)
    plt.scatter(mu[:, 0], mu[:, 1], c=colors, s=60, marker='+')
    plt.title("Plot of 500 Points and Their Clusters. K = " + str(bigK))
    plt.legend(loc='upper right', shadow=True, fontsize='small')

    plt.savefig('./out/kmeans_points_plot_k' + str(bigK) + '.png')
    return ci, mu


def matrix_factorize(iteration=100):
    m_train, m_test, train_userids, train_movieids, test_userids, test_movieids, movies_list = get_matrices()

    lmbd = 10
    sigmasq = .25
    d = 20
    u_size = m_train.shape[0]
    v_size = m_train.shape[1]

    rmse_list = []
    # generating random values for all of v
    objlist = []
    u = np.matrix(np.random.multivariate_normal(np.zeros(d), np.identity(d) * sigmasq, u_size))
    v = np.matrix(np.random.multivariate_normal(np.zeros(d), np.identity(d) * sigmasq, v_size))

    np.random.seed(100)

    for itr in range(iteration):
        print itr
        for urow_idx in range(u_size):
            idx = np.where(m_train[urow_idx, :] > 0)[0]
            if len(idx) > 0:
                u[urow_idx, :] = np.array(
                    np.linalg.inv(lmbd * sigmasq * np.identity(d) + (v[idx, :].T * v[idx, :])) * np.sum(
                        m_train[urow_idx, idx].T * v[idx, :], 0).T).T

        for vrow_idx in range(v_size):
            idx = np.where(m_train[:, vrow_idx] > 0)[0]
            if len(idx) > 0:
                v[vrow_idx, :] = np.array(
                    np.linalg.inv(10 * .25 * np.identity(20) + (u[idx, :].T * u[idx, :])) * np.sum(
                        m_train[idx, vrow_idx].T * u[idx, :], 0).T).T

        # The prediction at iteration itr rounded to the nearest integer
        uvt = u * v.T

        squared_error = 0
        # Calculating squared error (the first term in the objective funtion)
        for row in range(uvt.shape[0]):
            iidx = np.where(m_train[row] > 0)[0]
            squared_error += np.sum(
                (1 / (2 * sigmasq)) * np.square(m_train[row, iidx] - uvt[row, iidx]))
        # for row in range(uvt.shape[0]):
        # iidx = np.where(m_train[row] != 0)[0]
        # for iid in iidx:
        # squared_error += (1/(2*sigmasq)) * np.square(m_train[row, iid] - uvt[row, iid])

        obj = np.sum(- squared_error - np.sum(lmbd / 2 * np.square(u)) - np.sum(lmbd / 2 * np.square(v)))
        objlist.append(obj)

        pred_at_iter = np.round(u * v.T)

        iz = np.where(pred_at_iter == 0)
        for i in iz[0]:
            for j in iz[1]:
                pred_at_iter[i, j] = 1

        rmse_list += [get_rmse(pred_at_iter, m_test, train_movieids, train_userids)]

    fc_idx = np.random.random_integers(0, 30, 5).tolist()

    closest_movies(movies_list, v)
    centroids, clusters = kmeans_movies(u)
    km_sklrn = KMeans(30)
    km_sklrn.fit(u)
    sklrn_cen = km_sklrn.cluster_centers_
    sklrn_clust = km_sklrn.labels_
    five_topten = movie_user_characterize(centroids[fc_idx], clusters, v)
    sklrn_five_tt = movie_user_characterize(sklrn_cen[fc_idx], sklrn_clust, v)

    for l in range(len(five_topten)):
        print 'For the ', str(l), 'the cluster, we have the following 10 movies with the largest dot product'
        for m in five_topten[l]:
            print '|--> ', movies_list[m]

    for l in range(len(sklrn_five_tt)):
        print 'For the ', str(l), 'sklearn'
        for m in sklrn_five_tt[l]:
            print '<>-> ', movies_list[m]

    # get a copy of the prediction and set the cells in the copy that original matrix's nan/0 cells to 0
    # pr_sparse = np.round(pr_sparse, 1)
    # for urow_idx in range(mmx.shape[0]):
    # idx = np.where(mmx[urow_idx, :] == 0)[0]
    # pr_sparse[urow_idx, idx] = np.tile(0, pr_sparse[urow_idx, idx].shape)
    plt.plot(range(1, iteration + 1), np.array(rmse_list), '#3288bd', label="RMSE by Iteration")
    plt.savefig('./out/mxf_rmse.png')
    plt.close()
    plt.plot(range(1, iteration + 1), objlist, c='r', label="Log Likelihood by Iteration")
    plt.savefig('./out/mxf_obj_plt.png')

    # indices of pred_at_iter that are rounded to zero. To be set to 1


def get_rmse(preds, test_data, train_movie_ids, train_user_ids):
    size = test_data.shape[0]
    rmse = 0.0
    sse = 0.0

    for i in range(size):
        mid = test_data[i, 1]
        cid = test_data[i, 0]
        tsrating = test_data[i, 2]
        try:
            trmid = train_movie_ids.index(mid)
            truid = train_user_ids.index(cid)
        except ValueError:
            print mid, '|', cid
            continue

        prrating = preds[truid, trmid]

        sse += (tsrating - prrating) ** 2

    rmse = np.sqrt(sse / size)

    return rmse


def closest_movies(movies_list, v, n=5):
    # three_movies = np.random.choice(v.shape[0], 3, replace=False).tolist()
    three_movies = [1617, 1015, 28]
    for i in three_movies:
        dist = []
        five_movies = []
        for j in range(v.shape[0]):
            if j != i:
                dist += [np.linalg.norm(v[i] - v[j]) ** 2]
            else:
                continue

        top_distances = sorted(dist)[0:5]

        for a in top_distances:

            try:
                five_movies += [dist.index(a)]

            except ValueError:
                print a
                continue

        print 'For Movie: ', movies_list[i], 'the closest five movies are: '
        for i, f in enumerate(five_movies):
            print movies_list[f], '\t ', top_distances[i]

def kmeans_movies(udata, bigK=30, iteration=20):

    mu_idx = np.random.choice(udata.shape[0], bigK, replace=False).tolist()
    # randomly initializing the centroids of each cluster
    mu = udata[mu_idx, :]
    ci = []
    sumdist = []
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
        ob = 0
        print iterate
        for littlek in range(bigK):
            distances.append(np.array(np.sum(np.square(udata - mu[littlek, :]), 1))[:, 0])

        ci = np.argmin(np.array(distances), 0)

        for i in range(len(ci)):
            ob += np.array(distances)[ci[i], i]

        for littlek in range(bigK):
            xs_in_clust_k = np.where(ci == littlek)[0]
            nk = xs_in_clust_k.shape[0]
            mu[littlek, :] = np.sum(udata[xs_in_clust_k, :], 0) / nk

        sumdist += [ob]

    plt.plot(range(iteration), sumdist, label="Objective Function", c=colors[1])
    plt.title("Plot of K Means Objective Function by Iteration")
    plt.savefig("./out/mxf_objective_plot_k" + str(bigK) + ".png")
    plt.close()

    # plt.scatter(udata[:, 0], udata[:, 1], c=ci)
    # plt.scatter(mu[:, 0], mu[:, 1], c=colors, s=60, marker='+')
    # plt.title("Plot of 500 Points and Their Clusters. K = " + str(bigK))
    # plt.legend(loc='upper right', shadow=True, fontsize='small')
    #
    # plt.savefig('./out/mxf_points_plot_k' + str(bigK) + '.png')

    return mu, ci

def movie_user_characterize(centroids, clusters, vmovies):

    dp = np.dot(vmovies, centroids.T)
    topten = []

    # for center in range(centroids.shape[0]):
    #     centroids[center] * vmovies.T

    for col in range(dp.shape[1]):
        # copying the column of centroid col
        mlst = dp[:, col].tolist()
        tt = []
        for r in range(10):
            idx = np.argmax(mlst)
            mlst[idx] = -1
            tt += [idx]

        topten.append(tt)

    return topten

if __name__ == '__main__':
    c, m = kmeans_clustering(5)
    # matrix_factorize(100)

