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
sys.path.append('/Users/oghinde/Dropbox/Work/Clusterwise_LM/model/')
from cwlm.clusterwise_linear_model import ClusterwiseLinModel as CWLM
from cwlm.gmm_regressor import GMMRegressor
from cwlm.kmeans_regressor import KMeansRegressor

n = 10 # number of samples
d = 2 # number of input dimensions
t = 2 # number of tasks
K = 3 # number of clusters
labels = np.random.randint(0, K, (n, ))

X = np.random.randn(n, d)
y = np.empty((n, t))
intercepts = []
coefs = []

for k in range(K):
    intercepts.append(np.random.randint(-2, 2, size=(t,)))
    coefs.append(np.random.randint(-3, 3, size=(t, d)))

for k in range(K):
    idx = (labels == k)
    y[idx, :] = intercepts[k] + np.dot(X[idx, :], coefs[k].T) + np.random.randn(sum(idx), t)




reg = Ridge()
reg.fit(X, y)



print('Real intercept = ', intercept)
print('Estimated intercept = ', reg.intercept_)
print('Real coef = ', coef)
print('Estimated coef = ', reg.coef_)