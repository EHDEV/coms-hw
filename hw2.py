__author__ = 'eliashussen'

import numpy as np
import pandas as p
import sys, os
from collections import Counter
# from matplotlib import pyplot as plt

def get_data():
    """ fetches data from file into a pandas dataframe which will then
        be converted into a numpy matrix
    """

    pathxtrain = sys.argv[1]
    pathxtest = sys.argv[2]
    pathlabeltrain = sys.argv[3]
    pathlabeltest = sys.argv[4]

    xtrain = p.read_csv(pathxtrain, header=None)
    xtest = p.read_csv(pathxtest, header=None)
    label_train = p.read_csv(pathlabeltrain, header=None)
    label_test = p.read_csv(pathlabeltest, header=None)

    xtrain_mx = xtrain.values
    xtest_mx = xtest.values

    label_train = label_train.values.reshape(label_train.shape[0])
    label_test = label_test.values.reshape(label_test.shape[0])

    return xtrain_mx, xtest_mx, label_train, label_test


def get_data2():
    pathxtrain = sys.argv[1]
    pathlabeltrain = sys.argv[2]

    xtrain = p.read_csv(pathxtrain, header=None)
    label_train = p.read_csv(pathlabeltrain, header=None)
    x = xtrain.values
    yclass = label_train.values.reshape(label_train.shape[0])

    size = 183

    xtest = x[0:size, :]
    xtrain = x[size:, :]

    yclass_test = yclass[0:size]
    yclass_train = yclass[size:]

    return xtrain, xtest, yclass_train, yclass_test


def knn_manage(k):
    """
    Manages the functions and calls them accordingly
    :param repeat:
    :return:
    """

    xtrain, xtest, label_train, label_test = get_data()
    pred = knn_classify(xtrain, xtest, label_train, k)
    conf_mat, accuracy, misclassified = confusion_matrix_accuracy(pred, label_test)
    print accuracy
    print conf_mat


def vote_xstar_label(label_train, indices, k):
    votes = []
    for ki in range(k):
        votes.append(label_train[indices[ki]])

    return Counter(votes).most_common(1)[0][0]


def knn_classify(xtrain, xtest, label_train, k):
    pred_label = []
    for xstar in xtest:
        dist_xi, indices = eu_dist(xtrain, xstar)
        xstar_label = vote_xstar_label(label_train, indices, k)
        pred_label.append(xstar_label)
    return pred_label


def confusion_matrix_accuracy(pred_label, label_test):
    conf_mx = np.zeros((10, 10))
    misclassified_idxs = []
    for num in range(len(pred_label)):
        conf_mx[pred_label[num], label_test[num]] += 1
        if pred_label[num] != label_test[num]:
            misclassified_idxs += [num]
    accuracy = np.sum(np.diag(conf_mx)) / len(label_test)
    return conf_mx, accuracy, misclassified_idxs


def conf_misc(pred_label, label_test):
    count_right = 0.0
    misclassified = []

    for i in range(len(pred_label)):
        if pred_label[i] == label_test[i]:
            count_right += 1
        else:
            misclassified += [i]

    accuracy = count_right / len(pred_label)

    return accuracy, misclassified


def eu_dist(xtrain, x_istar):
    # diffsq = (xtrain - x_*)^2
    # dist = diffsq.sum(axis=1) ** .5
    distance = ((xtrain - x_istar) ** 2).sum(axis=1) ** .5
    indices = np.argsort(distance)
    return distance, indices.tolist()


def bayes_manage():
    mu_mx = []
    cov_list = []
    pred_labels = []
    pi_prior = []
    class_probs_all = []

    xtrain_mx, xtest_mx, label_train, label_test = get_data()
    classes_set_y = set(label_train)  # unique classes in y

    for cls in classes_set_y:
        print "Class: ", cls
        class_indices = np.where(label_train == cls)[0].tolist()
        pi_prior += [prior_prob_y(label_train, class_indices)]
        mu_vec = mle_mean(xtrain_mx[class_indices, :])
        mu_mx += [mu_vec]
        cov_mx = mle_covariance(xtrain_mx[class_indices, :], mu_vec)
        cov_list += [cov_mx]

    # mu_mx = np.matrix(np.concatenate(mu_mx, axis=0))
    # cov_mx = np.matrix(np.concatenate(cov_mx, axis=0))
    pi_prior = np.array(pi_prior).reshape((10, 1))

    for xstar in xtest_mx:
        probs = []

        for cls in classes_set_y:
            probs += [nb_classify(pi_prior[cls], mu_mx[cls], cov_list[cls], np.matrix(xstar))]

        pred_labels += [np.argmax(probs)]
        class_probs_all += [probs]
    print pred_labels

    conf, accuracy, misclassified_idxs = confusion_matrix_accuracy(pred_labels, label_test)
    print conf

    # plot_gaussian(mu_mx)
    show_misclassified(misclassified_idxs, label_test, pred_labels, class_probs_all)


def prior_prob_y(label_train, cls_idx):
    pi_y = len(cls_idx) / float(len(label_train))

    return pi_y


def mle_mean(xtrain_sub):
    mu_y = np.sum(xtrain_sub, 0) / xtrain_sub.shape[0]

    return mu_y.reshape((1, xtrain_sub.shape[1]))


def mle_covariance(xtrain_sub, mu_y):
    cov = (1 / float(xtrain_sub.shape[0])) * np.matrix(xtrain_sub - np.tile(mu_y, 1)).T * np.matrix(
        xtrain_sub - np.tile(mu_y, 1))

    return cov


def nb_classify(pi_prior, mle_mu, mle_cov, xstar):
    # Probabilities
    # expx = np.exp(-1/2 * (np.power(np.tile(xstar, 1)-mle_mu, 2).T * np.matrix(1/mle_cov)))
    expx = np.exp((-1.0 / 2.0 * (xstar - mle_mu) * np.linalg.inv(np.matrix(mle_cov)) * (xstar - mle_mu).T)[0, 0])
    fx = (np.matrix(pi_prior).T * (1 / np.power(np.linalg.det(mle_cov), 0.5))) * expx

    return fx[0, 0]


def plot_gaussian(mu):
    pathQ = sys.argv[5]
    q = p.read_csv(pathQ, header=None).values
    idx = 0
    # for xbar in mu:
    # pc = np.matrix(xbar) * np.matrix(q).T
    #     pc = np.array(pc.reshape((28, 28)))
    #     plt.imshow(pc, cmap = plt.cm.gray_r)
    #     plt.savefig(str(idx) + '.png')
    #     idx+=1


def show_misclassified(miscl_x, label_test, pred_label, xprobs):
    for idx in miscl_x:
        print 'Actual label: ', label_test[idx], 'Predicted label: ', pred_label[idx]
        print 'Probabilities: ', '\t|\t', xprobs[idx]


def logit_manage(iteration):
    xtrain_mx, xtest_mx, label_train, label_test = get_data2()
    classes_set_y = set(label_train)  # unique classes in y

    # x0 = np.matrix(np.ones((xtrain_mx.shape[0], 1)))
    # xtrain_mx = np.matrix(np.concatenate((np.array(x0), np.array(xtrain_mx)), axis=1))
    # x0 = np.matrix(np.ones((xtest_mx.shape[0], 1)))
    # xtest_mx = np.matrix(np.concatenate((np.array(x0), np.array(xtest_mx)), axis=1))
    w_shape = (len(classes_set_y), xtrain_mx.shape[1])
    w = np.matrix(np.zeros(w_shape))

    eta = .1  # /xtrain_mx.shape[0]

    for cls in classes_set_y:

        print "Class: ", cls
        class_indices = np.where(label_train == cls)[0].tolist()

        for i in range(iteration):
            sum_one_to_n = np.zeros((1, xtrain_mx.shape[1]))
            # for row in range(xtrain_mx.shape[0]):
            sum_one_to_n += sofmax(w, xtrain_mx, label_train, cls)
            w[cls] += (eta * sum_one_to_n)
        print w

    pred_label = logit_classify(w, xtest_mx)

    conf = conf_misc(pred_label, label_test)
    print conf


def logit_classify(w, xtest):
    res = xtest * w.T
    pred = []

    for row in res:
        pred += [np.argmax(row, 1)[0, 0]]

    return pred


def sofmax(w, x, y, cls):
    numerator = np.exp(x * w[cls].T)
    denominator = np.sum(np.exp(x * w.T), 1)
    hx = np.matrix(np.array(numerator) / np.array(denominator))
    cls_idx = np.where(y == cls)[0].tolist()
    not_cls_idx = np.where(y != cls)[0].tolist()
    # for i in range(x.shape[0]):
    # if i not in cls_idx:
    #         not_cls_idx.append(i)

    hxi = (1 - hx[cls_idx]).T * x[cls_idx]
    minus_hxi = -hx[not_cls_idx].T * x[not_cls_idx]

    result = np.matrix(np.array(hxi) + np.array(minus_hxi))

    return result


if __name__ == '__main__':
    # knn_manage(5)
    # bayes_manage()
    logit_manage(1000)

