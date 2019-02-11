#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:51:46 2017

@author: oghinde

This should be rewritten as a notebook.

Check these githubs:
    https://github.com/tansey/regression_mixtures
    https://github.com/victorkristof/linear-regressions-mixture
    
This should go in a notebook.
"""
# Add our module to the path
import sys
sys.path.append('/Users/oghinde/Dropbox/Work/Clusterwise_LM/code/')

import numpy as np
from cwlm.clusterwise_linear_model import ClusterwiseLinModel as CWLM
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt
import pickle

# Parameter definition
figpath = 'figures/'
save_path = '/Users/oghinde/Dropbox/Work/Clusterwise_LM/test_sets/'
save_figs = False
save_data = False

n1 = 200
n2 = 200
n3 = 200
n1_tst = 100
n2_tst = 100
n3_tst = 100
n_tr = n1 + n2 + n3
n_tst = n1_tst + n2_tst + n3_tst
n_outliers = 10

W =  np.array([[1, -2],
               [0, 0],
               [1, 2]])
precisions = np.array([0.7, 0.7, 0.7]) 
X_var = 3

c = ['r', 'g', 'b']

# Data generation
X = np.ones((n_tr, 2))
X_tst = np.ones((n_tst, 2))

X[:n1, 1] = -4 + np.random.randn(n1,) * X_var
X[n1:n1+n2, 1] = 5 + np.random.randn(n1,) * X_var
X[n1+n2:, 1] = 10 + np.random.randn(n1,) * X_var

X_tst[:n1_tst, 1] = -4 + np.random.randn(n1_tst,) * X_var
X_tst[n1_tst:n1_tst+n2_tst, 1] = 5 + np.random.randn(n1_tst,) * X_var
X_tst[n1_tst+n2_tst:, 1] = 10 + np.random.randn(n1_tst,) * X_var

y = np.zeros((n_tr, 1))
y_tst = np.zeros((n_tst, 1)) 

y[:n1] = np.dot(X[:n1, :], W[0, :].T)[:, np.newaxis] + np.random.randn(n1, 1)*precisions[0]
y[n1:n1+n2] = np.dot(X[n1:n1+n2, :], W[1, :].T)[:, np.newaxis] + np.random.randn(n1, 1)*precisions[1]
y[n1+n2:] = np.dot(X[n1+n2:, :], W[2, :].T)[:, np.newaxis] + np.random.randn(n1, 1)*precisions[2]

y_tst[:n1_tst] = np.dot(X_tst[:n1_tst, :], W[0, :].T)[:, np.newaxis] + np.random.randn(n1_tst, 1)*precisions[0]
y_tst[n1_tst:n1_tst+n2_tst] = np.dot(X_tst[n1_tst:n1_tst+n2_tst, :], W[1, :].T)[:, np.newaxis] + np.random.randn(n1_tst, 1)*precisions[1]
y_tst[n1_tst+n2_tst:] = np.dot(X_tst[n1_tst+n2_tst:, :], W[2, :].T)[:, np.newaxis] + np.random.randn(n1_tst, 1)*precisions[2]

# Insert some outliers
outlier_idx = np.random.choice(n_tr, n_outliers, replace=False)
outlier_mag = 10
y[outlier_idx] = (y[outlier_idx].T + np.random.randn(n_outliers) * outlier_mag).T

X = X[:, 1][:, np.newaxis]
X_tst = X_tst[:, 1][:, np.newaxis]
X_ext = np.concatenate((np.ones((n_tr, 1)), X), axis=1)

# Plot the dataset
xlims = [-20, 25]
ylims = [-20, 50]

fig0 = plt.figure()
plt.scatter(X, y, color='b', marker='x')
plt.xlabel('X')
plt.ylabel('y', rotation=0)
axes = plt.gca()
axes.set_ylim(ylims)
plt.title('Training dataset')
axes.set_xlim(xlims)
plt.show()
if save_figs:
    fig0.savefig(figpath + 'dataset.png', 
                bbox_inches='tight', 
                pad_inches=0, 
                dpi=500)




# CWLM MODEL TRAINING
K = 3
cwlm = CWLM(n_components=K, eta=7, plot=True, n_init=20, tol=1e-10, 
            init_params='kmeans', smoothing=False)
cwlm.fit(X, y)
mu_est = cwlm.means_
mu_ext = np.concatenate((np.ones((K, 1)), mu_est), axis=1)
W_clust = cwlm.reg_weights_
y_, score = cwlm.predict_score(X_tst, y_tst)

print('\nR2 score = {}'.format(score))

# GMM TRAINING
gmm = GMM(n_components=K)
gmm.fit(X)

# Depict the likelyhoods on Y for the clusterwise linear model
rx = np.ones((100, 2))
r = np.linspace(X.min(), X.max(), 100)
rx[:, 1] = r
c = ['r', 'g', 'b']

fig = plt.figure(1)
y_est = []


# Depict all clustering approaches for clusterwise linear model

fig = plt.figure(2)
plt.title('Responsability Clustering')
lab = [2, 1, 0]
for k in range(K):
    idx = np.where(cwlm.labels_ == lab[k])[0]
    plt.scatter(X[idx], y[idx], color=c[k], marker='x')
    w = W_clust[:, k]
    y_est = np.dot(mu_ext[k,:], w)
    plt.plot(r, np.dot(rx, w), color=c[k])
    plt.scatter(mu_est[k], y_est, color=c[k], marker='x')
axes = plt.gca()
axes.set_ylim(ylims)
axes.set_xlim(xlims)
plt.xlabel('X')
plt.ylabel('y', rotation=0)
plt.tight_layout()
plt.show()
if save_figs:
    fig.savefig(figpath + 'clusterwise_RESP_clust.png', 
                bbox_inches='tight', 
                pad_inches=0, 
                dpi=500)

fig = plt.figure(3)
plt.title('Input dimension clustering')
for k in range(K):
    idx = np.where(cwlm.labels_X_ == k)[0]
    plt.scatter(X[idx], y[idx], color=c[k], marker='x')
    w = W_clust[:, k]
    y_est = np.dot(mu_ext[k,:], w)
    plt.plot(r, np.dot(rx, w), color=c[k])
    plt.scatter(mu_est[k], y_est, color=c[k], marker='x')
axes = plt.gca()
axes.set_ylim(ylims)
axes.set_xlim(xlims)
plt.xlabel('X')
plt.ylabel('y', rotation=0)
plt.tight_layout()
plt.show()
if save_figs:
    fig.savefig(figpath + 'clusterwise_X_clust.png', 
                bbox_inches='tight', 
                pad_inches=0, 
                dpi=500)

fig = plt.figure(4)
plt.title('Output dimension clustering')
for k in range(K):
    idx = np.where(cwlm.labels_y_ == k)[0]
    plt.scatter(X[idx], y[idx], color=c[k], marker='x')
    w = W_clust[:, k]
    y_est = np.dot(mu_ext[k,:], w)
    plt.plot(r, np.dot(rx, w), color=c[k])
    plt.scatter(mu_est[k], y_est, color=c[k], marker='x')
axes = plt.gca()
axes.set_ylim(ylims)
axes.set_xlim(xlims)
plt.xlabel('X')
plt.ylabel('y', rotation=0)
plt.tight_layout()
plt.show()
if save_figs:
    fig.savefig(figpath + 'clusterwise_Y_clust.png', 
                bbox_inches='tight', 
                pad_inches=0, 
                dpi=500)

# Depict prediction results for the clusterwise linear model

fig = plt.figure(3)
plt.scatter(X_tst, y_tst, color='b', label='Real targets', marker='x')
plt.scatter(X_tst, y_, color='r', label='Predicted targets', s=1.8)
plt.legend(loc=2)
plt.title('Prediction results for Clusterwise Linear Regressor model')
plt.xlabel('X')
plt.ylabel('y', rotation=0)
axes = plt.gca()
axes.set_ylim(ylims)
axes.set_xlim(xlims)
plt.show()
if save_figs:
    fig.savefig(figpath + 'clusterwise_prediction.png', 
                bbox_inches='tight', 
                pad_inches=0, 
                dpi=500)

# Save the dataset

if save_data:
    with open(save_path + 'test7.pickle', 'w') as f:
        pickle.dump((X, y, cwlm), f)