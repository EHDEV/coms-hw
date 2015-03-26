__author__ = 'eliashussen'

import numpy as np
import pandas as p
import sys, os
from collections import Counter
from matplotlib import pyplot as plt


class Util:
    # paths = {'pathxtrain': '', 'pathxtest': '', 'pathlabeltrain': '', 'pathlabeltest': ''}

    def __init__(self):
        self.pathxtrain = sys.argv[1]
        self.pathlabel = sys.argv[2]
        # self.pathlabeltrain = sys.argv[3]
        # self.pathlabeltest = sys.argv[4]


    def get_data(self):
        """ fetches data from file into a pandas dataframe which will then
            be converted into a numpy matrix
        """
        size = 183
        x = p.read_csv(self.pathxtrain, header=None)
        yclass = p.read_csv(self.pathlabel, header=None)

        x = x.values
        yclass = yclass.values.reshape(yclass.shape[0])

        xtest = x[0:size, :]
        xtrain = x[size:, :]

        yclass_test = yclass[0:size]
        yclass_train = yclass[size:]

        return xtrain, xtest, yclass_train, yclass_test


class Prediction:
    def __init__(self):
        pass

    def vote_xstar_label(self, label_train, indices, k):
        votes = []
        for ki in range(k):
            votes.append(self, label_train[indices[ki]])

        return Counter(votes).most_common(1)[0][0]


    def confusion_matrix_accuracy(self, pred_label, label_test):
        count_right = 0.0
        misclassified = []

        for i in range(len(pred_label)):
            if pred_label[i] == label_test[i]:
                count_right += 1
            else:
                misclassified += [i]

        accuracy = count_right / len(pred_label)
        # conf_mx = np.zeros((2, 2))
        # misclassified_idxs = []
        # for num in range(len(pred_label)):
        #     conf_mx[pred_label[num], label_test[num]] += 1
        #     if pred_label[num] != label_test[num]:
        #         misclassified_idxs += [num]
        # accuracy = np.sum(np.diag(conf_mx)) / len(label_test)
        return accuracy, misclassified


class LogisticRegression:
    def logit_manage(self, iteration):

        ut = Util()
        predict = Prediction()
        xtrain_mx, xtest_mx, label_train, label_test = ut.get_data
        classes_set_y = set(label_train)  # unique classes in y

        x0 = np.matrix(np.ones((xtrain_mx.shape[0], 1)))
        xtrain_mx = np.matrix(np.concatenate((np.array(x0), np.array(xtrain_mx)), axis=1))
        x0 = np.matrix(np.ones((xtest_mx.shape[0], 1)))
        xtest_mx = np.matrix(np.concatenate((np.array(x0), np.array(xtest_mx)), axis=1))
        w_shape = (len(classes_set_y), xtrain_mx.shape[1])
        w = np.matrix(np.zeros(w_shape))

        eta = .1 / xtrain_mx.shape[0]

        for cls in classes_set_y:

            print "Class: ", cls
            class_indices = np.where(label_train == cls)[0].tolist()

            for i in range(iteration):
                sum_one_to_n = np.zeros((1, xtrain_mx.shape[1]))
                # for row in range(xtrain_mx.shape[0]):
                sum_one_to_n += self.sofmax(w, xtrain_mx, label_train, cls)
                w[cls] += (eta * sum_one_to_n)
            print w

        pred_label = self.logit_classify(w, xtest_mx)

        conf = predict.confusion_matrix_accuracy(pred_label, label_test)
        print conf

    def logit_classify(self, w, xtest):

        res = xtest * w.T
        pred = []

        for row in res:
            pred += [np.argmax(row, 1)[0, 0]]

        return pred


    def sofmax(self, w, x, y, cls):
        numerator = np.exp(x * w[cls].T)
        denominator = np.sum(np.exp(x * w.T), 1)
        hx = np.matrix(np.array(numerator) / np.array(denominator))
        cls_idx = np.where(y == cls)[0].tolist()
        not_cls_idx = np.where(y != cls)[0].tolist()
        # for i in range(x.shape[0]):
        # if i not in cls_idx:
        # not_cls_idx.append(i)

        hxi = (1 - hx[cls_idx]).T * x[cls_idx]
        minus_hxi = -(hx[not_cls_idx].T) * x[not_cls_idx]

        result = np.matrix(np.array(hxi) + np.array(minus_hxi))

        return result


class LDA:
    def __init__(self):
        pass

    def lda_manage(self):
        mu_list = []
        cov_list = []
        pred_labels = []
        pi_prior = []
        class_probs_all = []
        ut = Util()
        predict = Prediction()

        xtrain_mx, xtest_mx, label_train, label_test = ut.get_data()
        classes_set_y = set(label_train)  # unique classes in y

        size_n = label_train.shape[0]
        classes_k = len(classes_set_y)  # unique classes in y

        for cls in classes_set_y:
            class_indices = np.where(label_train == cls)[0].tolist()
            print "Class: ", cls
            pi_prior += [self.prior_prob_y(label_train, class_indices)]
            mu_vec = self.mle_mean(xtrain_mx[class_indices, :])
            mu_list += [mu_vec]
            cov = self.mle_covariance(xtrain_mx[class_indices, :], mu_vec, label_train, size_n, classes_k)
            cov_list += [cov]
        cov = cov_list[0] + cov_list[1]

        w0, wd = self.calc_coefs(pi_prior, mu_list, cov)
        pred_labels = self.lda_classify(xtest_mx, w0, wd)

        acc, mis = predict.confusion_matrix_accuracy(pred_labels, label_test)

        print acc
        print mis
        # print cov
        # print mu_list
        # print label_train
        # print pred_labels


    def prior_prob_y(self, label_train, cls_idx):
        pi_y = len(cls_idx) / float(len(label_train))

        return pi_y


    def mle_mean(self, xtrain_sub):
        mu_y = np.sum(xtrain_sub, 0) / float(xtrain_sub.shape[0])

        return mu_y.reshape((1, xtrain_sub.shape[1]))


    def mle_covariance(self, xtrain, mu, label_train, size_n, classes_k):
        cov = (np.matrix(xtrain - np.tile(mu, 1)).T * np.matrix(xtrain - np.tile(mu, 1))) / float(size_n - classes_k)

        return cov

    def calc_coefs(self, prior_list, mu_list, cov):
        pi1 = prior_list[0]
        pi0 = prior_list[1]
        mu1 = mu_list[0]
        mu0 = mu_list[1]

        w0 = np.log(pi1 / pi0) - .5 * (mu1 + mu0) * np.linalg.pinv(cov) * (mu1 - mu0).T
        wd = np.linalg.pinv(cov) * (mu1 - mu0).T

        return np.array(w0)[0][0], wd


    def lda_classify(self, x, w0, wd):

        class_predictions = np.zeros((x.shape[0],))

        gx = np.array(w0 + (x * wd))
        gx = gx.reshape((gx.shape[0], ))

        idx_pos = np.where(gx > 0)[0].tolist()
        idx_neg = np.where(gx < 0)[0].tolist()

        class_predictions[idx_pos] = 1
        class_predictions[idx_neg] = -1

        return class_predictions

class Boosting:
    def __init__(self):
        pass

    def boosting_manage(self, prob, plot=False):
        sample_size = [100, 200, 300, 400, 500]
        histograms = []
        np.random.seed(1000)
        for n in sample_size:
            u = np.random.sample(n)
            random_sample = self.rand_sampler(n, prob, u)

            if plot:
                self.plot_histogram(random_sample)
                plt.savefig('../out/n_' + str(n) + '.png')

    def rand_sampler(self, n, prob, u):
        """
        Generates a random sample of k numbers according to the discrete probability distribution
        n: sample size
        pro: probability distribution
        """
        cdf = []
        np.random.seed(100)
        for k in range(0, len(prob)):
            cdf.append(sum([prob[k] for k in range(0, k + 1)]))
        # u = np.random.sample(n)
        rds = []

        for j in range(0, len(cdf)):
            if j == 0:

                tmp = np.where(u <= cdf[j])[0]
                rds += np.repeat(j, len(tmp)).tolist()
            else:
                print j
                tmp = np.where(np.logical_and(u <= cdf[j], u > cdf[j - 1]))[0]
                rds += np.repeat(j, len(tmp)).tolist()

        return rds


    def plot_histogram(self, samples):
        f, ax = plt.subplots()
        ax.hist(samples)
        ax.set_title('n = ' + str(len(samples)) + ' - Histogram')

    # def weighted_values(values, probabilities, size):
    # bins = np.add.accumulate(probabilities)
    # return values[np.digitize(np.random.random_sample(size), bins)]

    def boost_binary(self, xtrain, ylabel):
        n = xtrain.shape[0]
        wt = np.repeat(1.0 / n, n)
        er = 0
        alpha = 0

        for t in range(1000):
            bt = Boosting.rand_sampler(n, wt)
            # classification step


if __name__ == '__main__':
    ld = LDA()

    ld.lda_manage()
    # naive_bayes = NaiveBayes()
    # naive_bayes.bayes_manage()
    # boost = Boosting()
    # boost.boosting_manage([.2, .19, .07, .14, .4], True)
    # logreg = LogisticRegression()
    # logreg.logit_manage(1000)

