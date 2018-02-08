from sklearn.mixture import GMM
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from sklearn.metrics import accuracy_score
import itertools
from scipy import linalg

def computeUBM(ubm_model, data):
    ###########################################
    # ubm_model - gmm-represent distribution of our model
    # data - samples, which will correct ubm-model
    ###############################################

    xdim = data.shape[1]
    M = ubm_model.n_components

    ###############################################################
    #    ubm_means: means of the ubm <number array>               #
    #    ubm_covars: covariances of the ubm <number array>        #
    #    ubm_weights: weights of the ubm <number array>           #
    #    new_means: means adapted from the ubm <number array>     #
    #    new_weights: weights adapted from the ubm <number array> #
    ###############################################################

    # Copy parameters GMM-model
    ubm_weights = ubm_model.weights_
    ubm_means = ubm_model.means_
    ubm_covars = ubm_model.covars_

    ###################################################################
    # for X = {x_1, ..., x_T}                                         #
    # P(i|x_t) = w_i * p_i(x_t) / sum_j=1_M(w_j * P_j(x_t))           #
    ###################################################################

    posterior_prob = ubm_model.predict_proba(data)
    pr_i_xt = (ubm_weights * posterior_prob) / np.asmatrix(np.sum(ubm_weights \
                                                                  * posterior_prob, axis=1)).T

    n_i = np.asarray(np.sum(pr_i_xt, axis=0)).flatten()  # [M, ]

    # Then we can compute E(x) and E(x2) and calculate new parameters of
    # our model

    E_x = np.asarray([(np.asarray(pr_i_xt[:, i]) * data).sum(axis=0) / n_i[i] for i in range(M)])  # [M x xdim]
    E_x2 = np.asarray([(np.asarray(pr_i_xt[:, i]) * (data ** 2)).sum(axis=0) / n_i[i] for i in range(M)])  # [M x xdim]

    ################################################################
    #    T: scaling factor, number of samples                      #
    #    relevance_factor: factor for scaling the adapted means    #
    #    scaleparam - scale parameter for weights matrix estimation#
    ################################################################

    T = data.shape[0]
    relevance_factor = 16
    scaleparam = 1

    ################################################################
    # compute alpha_i: data-depentend apaptation coefficient       #
    # alpha_w = alpha_m = alpha_v                                  #
    # alpha_i = n_i/ (n_i + relevance factor)                      #
    ################################################################

    alpha_i = n_i / (n_i + relevance_factor)

    ###############################
    # Parqameter`s adaptation
    ##############################
    new_weights = (alpha_i * n_i / T + (1.0 - alpha_i) * ubm_weights) * scaleparam

    alpha_i = np.asarray(np.asmatrix(alpha_i).T)
    new_means = (alpha_i * E_x + (1. - alpha_i) * ubm_means)
    new_covars = alpha_i * E_x2 + (1. - alpha_i) * (ubm_covars + (ubm_means ** 2)) - (new_means ** 2)

    #############################################
    # if we want compute `full` covariance matrix - comment code here
    # new_covars = np.zeros([M, xdim, xdim])
    # for j in range(M):
    #    new_covars[j] = alpha_i[j]*E_x2[j] +(1. - alpha_i[j]).flatten()*(ubm_covars[j] + (new_means[j]**2))- (ubm_means[j]**2)
    #    new_covars[i] = np.where(new_covars[i] < MIN_VARIANCE, MIN_VARIANCE, new_covars[i])
    ####################################################################
    #   `covars_` : array
    #    Covariance parameters for each mixture component.  The shape
    #    depends on `covariance_type`::
    #        (n_components, n_features)             if 'spherical',
    #        (n_features, n_features)               if 'tied',
    #        (n_components, n_features)             if 'diag',
    #        (n_components, n_features, n_features) if 'full'
    #####################################################################

    ubm_model.means_ = new_means
    ubm_model.weights_ = new_weights
    ubm_model.covars_ = new_covars

    return ubm_model

def plotGMM(mixture, data, Y_):

    color_iter = itertools.cycle(['r', 'g', 'm', 'b', 'r'])
    plt.figure(figsize=(20,20))
    splot = plt.subplot(2, 1, 1)

    for i, (mean, covar, color) in enumerate(zip(
            mixture.means_, mixture._get_covars(), color_iter)):

        v, w = linalg.eigh(covar)
        u = w[0] / linalg.norm(w[0])

        if not np.any(Y_ == i):
                continue

        plt.scatter(x_test[Y_ == i, 0], x_test[Y_ == i, 1], 2, color=color)

        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
        plt.xlim(-5, 5)
        plt.ylim(-3, 3)
    plt.show()