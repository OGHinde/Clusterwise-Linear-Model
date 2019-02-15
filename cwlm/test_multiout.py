#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 10:48:25 2019

@author: oghinde
"""

import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# Add our module to the path
import sys
sys.path.append('/Users/oghinde/Git/Clusterwise_Linear_Model/')

from cwlm.clusterwise_linear_model import ClusterwiseLinModel as CWLM
from cwlm.gmm_regressor import GMMRegressor
from cwlm.kmeans_regressor import KMeansRegressor

n_tr = 10  # number of samples
n_tst = 5
d = 4   # number of input dimensions
t = 2  # number of tasks
K = 4   # number of clusters
labels_tr = np.random.randint(0, K, (n_tr, ))
labels_tst = np.random.randint(0, K, (n_tst, ))

X_tr = np.random.randn(n_tr, d)
y_tr = np.empty((n_tr, t))
X_tst = np.random.randn(n_tst, d)
y_tst = np.empty((n_tst, t))

intercepts = np.empty((t, K))
coefs = np.empty((t, d, K))

intercepts = np.random.randint(-2, 2, size=(t, K))
coefs = np.random.randint(-3, 3, size=(t, d, K))

for k in range(K):
    idx_tr = (labels_tr == k)
    idx_tst = (labels_tst == k)
    y_tr[idx_tr, :] = intercepts[:, k] + np.dot(X_tr[idx_tr, :], coefs[:, :, k].T) + np.random.randn(sum(idx_tr), t)
    y_tst[idx_tst, :] = intercepts[:, k] + np.dot(X_tst[idx_tst, :], coefs[:, :, k].T) + np.random.randn(sum(idx_tst), t)

reg = Ridge()
reg.fit(X_tr, y_tr)

Kreg = KMeansRegressor(n_components=K)
Kreg.fit_predict(X_tr, y_tr)