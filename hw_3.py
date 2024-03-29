__author__ = 'eliashussen'

import numpy as np
import pandas as p
import sys, os
from collections import Counter
from matplotlib import pyplot as plt


class Util:
    def __init__(self):
        self.pathxtrain = sys.argv[1]
        self.pathlabel = sys.argv[2]


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

        return accuracy, misclassified


class LogisticRegression:
    def logit_manage(self, iteration):

        ut = Util()
        predict = Prediction()
        xtrain_mx, xtest_mx, label_train, label_test = ut.get_data()
        classes_set_y = set(label_train)  # unique classes in y

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

        pred_label = self.logit_softmax_classify(w, xtest_mx)

        conf, misc = predict.confusion_matrix_accuracy(pred_label, label_test)
        print conf

    def logit_softmax_classify(self, w, xtest):

        res = xtest * w.T
        pred = []

        for row in res:
            pred += [np.argmax(row, 1)[0, 0]]

        return pred

    def logit_classify(self, x, w):

        pred = np.zeros(x.shape[0])
        extw = np.exp(x * w.T)
        res = np.array(extw / (1 + extw)).reshape(x.shape[0])

        ones = np.where(res >= 0.5)[0].tolist()
        minus_ones = np.where(res < 0.5)[0].tolist()

        pred[ones] = 1
        pred[minus_ones] = -1

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
        minus_hxi = -hx[not_cls_idx].T * x[not_cls_idx]

        result = np.matrix(np.array(hxi) + np.array(minus_hxi))

        return result

    def logit_noboost(self):

        conf_acc = Prediction()
        ut = Util()
        xtr, xts, ytr, yts = ut.get_data()

        coef = self.online_log_reg(xtr, ytr)

        pred = self.logit_classify(xts, coef)

        acc, misc = conf_acc.confusion_matrix_accuracy(pred, yts)

        print 'logreg', acc, misc

    def online_log_reg(self, x, y):

        indices = np.arange(x.shape[0])
        np.random.permutation(indices)

        xtrain_mx = np.matrix(x[indices, :])
        label_train = y[indices, ]

        eta = .2
        n, d = xtrain_mx.shape
        w = np.matrix(np.zeros(d))

        for i in range(n):
            sigmoid_ywx = 1 / (1 + np.exp(-label_train[i] * xtrain_mx[i, :] * w.T))
            w += eta * (1 - sigmoid_ywx) * label_train[i] * xtrain_mx[i]

        return w


class LDA:
    def __init__(self):
        """

        """
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
        print 'lda', acc, mis
        return acc, mis

    def lda_boost_classify(self, x, y):
        mu_list = []
        cov_list = []
        pred_labels = []
        pi_prior = []
        class_probs_all = []
        ut = Util()
        predict = Prediction()

        classes_set_y = set(y)  # unique classes in y

        size_n = y.shape[0]
        classes_k = len(classes_set_y)  # unique classes in y

        for cls in classes_set_y:
            class_indices = np.where(y == cls)[0].tolist()

            pi_prior += [self.prior_prob_y(y, class_indices)]
            mu_vec = self.mle_mean(x[class_indices, :])
            mu_list += [mu_vec]
            cov = self.mle_covariance(x[class_indices, :], mu_vec, y, size_n, classes_k)
            cov_list += [cov]
        cov = cov_list[0] + cov_list[1]

        w0, wd = self.calc_coefs(pi_prior, mu_list, cov)
        # pred_labels = self.lda_classify(xtest_mx, w0, wd)
        #
        # acc, mis = predict.confusion_matrix_accuracy(pred_labels, label_test)

        return w0, wd


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

    def plot_rand_hist(self, prob, plot=False):

        sample_size = [100, 200, 300, 400, 500]
        for n in sample_size:
            u = np.random.sample(n)
            random_sample = self.rand_sampler(n, prob)

            if plot:
                self.plot_histogram(random_sample)
                plt.savefig('./out/n_' + str(n) + '.png')

    def boost_manage(self, iteration=1000):
        ut = Util()
        predict = Prediction()
        lda = LDA()

        boost_classifier_params = []
        alphas = []
        test_error = []
        train_error = []
        err_list = []  # Misclassified errors
        three_points_prob = []

        xtrain_mx, xtest_mx, label_train, label_test = ut.get_data()
        classes_set_y = set(label_train)  # unique classes in y
        boost_final_prediction = np.zeros(xtest_mx.shape[0])

        size_n = label_train.shape[0]
        boost_predict = np.zeros(size_n)
        classes_k = len(classes_set_y)  # unique classes in y

        dist_t = np.repeat(1.0 / size_n, size_n)

        for t in range(iteration):
            # generating random sample
            bt_sample_idx = self.rand_sampler(size_n, dist_t)
            # get the parameters to pass to classifier
            bias, coef = lda.lda_boost_classify(xtrain_mx[bt_sample_idx, :], label_train[bt_sample_idx])
            # get prediction, accuracy and misclassified rows for test and train data
            pred_labels_tr = lda.lda_classify(xtrain_mx, bias, coef)
            pred_labels_ts = lda.lda_classify(xtest_mx, bias, coef)
            acc_tr, misclassified_tr = predict.confusion_matrix_accuracy(pred_labels_tr, label_train)
            acc_ts, misclassified_ts = predict.confusion_matrix_accuracy(pred_labels_ts, label_test)
            train_error.append(len(misclassified_tr) / float(label_train.shape[0]))
            test_error.append(len(misclassified_ts) / float(label_test.shape[0]))
            # get error and alpha values
            err_t = np.sum(dist_t[misclassified_tr])
            err_list.append(err_t)
            alpha_t = .5 * np.log((1 - err_t) / err_t)
            # update the prob distribution and normalize
            dist_t *= np.exp(-1 * alpha_t * label_train * pred_labels_tr)
            dist_t /= np.sum(dist_t)

            three_points_prob.append([dist_t[7], dist_t[33], dist_t[498]])
            alphas.append(alpha_t)
            # collect bias and coefficient parameters in a list to use later
            boost_classifier_params.append((bias, coef))
            # print t, ' ', alpha_t, misclassified_tr

        three_points_prob = np.array(three_points_prob)
        print boost_classifier_params
        for i in range(iteration):
            w0 = boost_classifier_params[i][0]
            wd = boost_classifier_params[i][1]
            preds = lda.lda_classify(xtest_mx, w0, wd)
            boost_final_prediction += (preds * alphas[i])

        fig1 = plt.figure()
        ax1 = plt.subplot(211)
        ax1.plot(range(1, iteration + 1), np.array(train_error), '#3288bd', label="Training Error")
        ax1.plot(range(1, iteration + 1), np.array(test_error), '#d53e4f', label="Testing Error")
        ax1.set_title("Training and Testing Error per Iteration")
        legend = ax1.legend(loc='upper right', shadow=True, fontsize='small')
        plt.savefig('./out/lda_train_test_err.png')

        fig2 = plt.figure()
        ax2 = plt.subplot(211)
        ax2.plot(range(1, iteration + 1), np.array(alphas), '#99d594', label="alpha_t")
        ax2.plot(range(1, iteration + 1), np.array(err_list), '#fc8d59', label="error_t")
        ax2.set_title("alpha and Error per Iteration")
        legend = ax2.legend(loc='upper right', shadow=True, fontsize='small')
        plt.savefig('./out/lda_alpha_err_plot.png')

        fig3 = plt.figure()
        ax3 = plt.subplot(211)
        ax3.plot(range(1, iteration + 1), three_points_prob[:, 0], '#99d594', label="w_t(7)")
        ax3.plot(range(1, iteration + 1), three_points_prob[:, 1], '#d6604d', label="w_t(33)")
        ax3.plot(range(1, iteration + 1), three_points_prob[:, 2], '#4393c3', label="w_t(498)")
        ax3.set_title("Weights of Three Points as a Function of their Iteration")
        legend = ax3.legend(loc='upper center', shadow=True, fontsize='x-small')
        plt.savefig('./out/lda_three_points_weight_plot.png')

        pred_vals = self.sign(boost_final_prediction)
        acc, misc = predict.confusion_matrix_accuracy(pred_vals, label_test)

        print 'boost-lda', acc, misc

    def boost_manage_logreg(self, iteration=1000):
        ut = Util()
        predict = Prediction()
        lr = LogisticRegression()

        boost_classifier_params = []
        alphas = []
        test_error = []
        train_error = []
        err_list = []  # Misclassified errors
        acc_list = []
        three_points_prob = []

        xtrain_mx, xtest_mx, label_train, label_test = ut.get_data()
        classes_set_y = set(label_train)  # unique classes in y
        boost_final_prediction = np.zeros(xtest_mx.shape[0])

        size_n = label_train.shape[0]
        boost_predict = np.zeros(size_n)
        classes_k = len(classes_set_y)  # unique classes in y

        dist_t = np.repeat(1.0 / size_n, size_n)

        for t in range(iteration):
            # generating random sample
            bt_sample_idx = self.rand_sampler(size_n, dist_t)
            # get the parameters to pass to classifier
            coef = lr.online_log_reg(xtrain_mx[bt_sample_idx, :], label_train[bt_sample_idx])
            # get prediction, accuracy and misclassified rows for test and train data
            pred_labels_tr = lr.logit_classify(xtrain_mx, coef)
            pred_labels_ts = lr.logit_classify(xtest_mx, coef)
            pred_labels_t = lr.logit_classify(xtrain_mx[bt_sample_idx, :], coef)
            acc_tr, misclassified_tr = predict.confusion_matrix_accuracy(pred_labels_tr, label_train)
            acc_ts, misclassified_ts = predict.confusion_matrix_accuracy(pred_labels_ts, label_test)
            acc_t, misclassified_t = predict.confusion_matrix_accuracy(pred_labels_t, label_train[bt_sample_idx])
            train_error.append(len(misclassified_tr) / float(label_train.shape[0]))
            test_error.append(len(misclassified_ts) / float(label_test.shape[0]))
            # get error and alpha values
            err_t = np.sum(dist_t[misclassified_t])
            err_list.append(err_t)
            alpha_t = .5 * np.log((1 - err_t ) / err_t )
            # update the prob distribution and normalize
            dist_t *= np.exp(-alpha_t * label_train[bt_sample_idx] * pred_labels_t)
            dist_t /= np.sum(dist_t)

            alphas.append(alpha_t)
            boost_classifier_params.append(coef)
            acc_list.append(acc_ts)
            three_points_prob.append([dist_t[7], dist_t[33], dist_t[498]])

            print t, acc_ts, acc_tr, alpha_t, err_t, misclassified_t
            #print t, np.sum(dist_t)
        three_points_prob = np.array(three_points_prob)
        # print boost_classifier_params

        for i in range(iteration):
            w = boost_classifier_params[i]
            preds = lr.logit_classify(xtest_mx, w)
            boost_final_prediction += (preds * alphas[i])

        fig1 = plt.figure()
        ax1 = plt.subplot(211)
        ax1.plot(range(1, iteration + 1), np.array(train_error), '#3288bd', label="Training Error")
        ax1.plot(range(1, iteration + 1), np.array(test_error), '#d53e4f', label="Testing Error")
        ax1.set_title("Training and Testing Error by Iteration")
        legend = ax1.legend(loc='upper right', shadow=True, fontsize='small')
        plt.savefig('./out/logreg_train_test_err.png')

        fig2 = plt.figure()
        ax2 = plt.subplot(211)
        ax2.plot(range(1, iteration + 1), np.array(alphas), '#99d594', label="alpha_t")
        ax2.plot(range(1, iteration + 1), np.array(err_list), '#fc8d59', label="error_t")
        ax2.set_title("alpha and Error per Iteration")
        legend = ax2.legend(loc='upper right', shadow=True, fontsize='small')
        plt.savefig('./out/logreg_alpha_err_plot.png')

        fig3 = plt.figure()
        ax3 = plt.subplot(211)
        ax3.plot(range(1, iteration + 1), three_points_prob[:, 0], '#99d594', label="w_t(7])")
        ax3.plot(range(1, iteration + 1), three_points_prob[:, 1], '#d6604d', label="w_t(33)")
        ax3.plot(range(1, iteration + 1), three_points_prob[:, 2], '#4393c3', label="w_t(498)")
        ax3.set_title("Weights of Three Points as a Function of their Iteration")
        legend = ax3.legend(loc='upper center', shadow=True, fontsize='x-small')
        plt.savefig('./out/logreg_three_points_weight_plot.png')

        pred_vals = self.sign(boost_final_prediction)
        acc, misc = predict.confusion_matrix_accuracy(pred_vals, label_test)

        print 'boost-logreg', acc, misc

    def sign(self, arr_var):

        vals = np.zeros(arr_var.shape[0])

        idx_pos = np.where(arr_var > 0)[0].tolist()
        idx_neg = np.where(arr_var <= 0)[0].tolist()

        vals[idx_pos] = 1
        vals[idx_neg] = -1

        return vals

    def rand_sampler(self, n, prob):
        """
        Generates a random sample of k numbers according to the discrete probability distribution
        n: sample size
        pro: probability distribution
        """
        cdf = []
        np.random.seed(100)
        u = np.random.sample(n)
        for k in range(0, len(prob)):
            cdf.append(sum([prob[k] for k in range(0, k + 1)]))
        # u = np.random.sample(n)
        rds = []

        for j in range(0, len(cdf)):
            if j == 0:

                tmp = np.where(u <= cdf[j])[0]
                rds += np.repeat(j, len(tmp)).tolist()
            else:

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
    logreg = LogisticRegression()
    logreg.logit_noboost()
    boost = Boosting()
    # boost.boost_manage(1000)
    boost.boost_manage_logreg(500)

    # logreg.online_log_reg()

