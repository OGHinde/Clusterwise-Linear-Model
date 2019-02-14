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

n = 10 # number of samples
d = 2 # number of input dimensions
t = 1 # number of tasks
K = 3 # number of clusters
labels = np.random.randint(0, K, (n, ))

X = np.random.randn(n, d)
y = np.empty((n, t))

intercepts = np.empty((t, K))
coefs = np.empty((t, d, K))

intercepts = np.random.randint(-2, 2, size=(t, K))
coefs = np.random.randint(-3, 3, size=(t, d, K))

for k in range(K):
    idx = (labels == k)
    y[idx, :] = intercepts[:, k] + np.dot(X[idx, :], coefs[:, :, k].T) + np.random.randn(sum(idx), t)

reg = Ridge()
reg.fit(X, y)

Kreg = KMeansRegressor(n_components=K)