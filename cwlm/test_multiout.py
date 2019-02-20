"""
Created on Thu Feb 14 10:48:25 2019

@author: oghinde
"""

import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# Add our module to PATH
from pathlib import Path
home = str(Path.home())
import sys
sys.path.append(home + '/Git/Clusterwise_Linear_Model/')

from cwlm.clusterwise_linear_model import ClusterwiseLinModel as CWLM
from cwlm.gmm_regressor import GMMRegressor
from cwlm.kmeans_regressor import KMeansRegressor

print('Generating data')
def compute_targets(X, coefs, intercepts, noise_var=0.5):
    n, d = X.shape
    t = coefs.shape[0]
    noise = noise_var*np.random.randn(n, t)
    dot_product = np.dot(X, coefs.T)
    return intercepts + dot_product + noise

n_tr = 500  # number of training samples
n_tst = 100 # number of testsamples
d = 1       # number of input dimensions
t = 2       # number of tasks
K = 2       # number of clusters
plot = True

if d > 1:
    # Don't plot if inputs have more than 1d.
    plot = False

labels_tr = np.random.randint(0, K, (n_tr, ))
labels_tst = np.random.randint(0, K, (n_tst, ))

X_tr = np.empty((n_tr, d))
X_tst = np.empty((n_tst, d))
y_tr = np.empty((n_tr, t))
y_tst = np.empty((n_tst, t))

displace = np.random.randint(-25, 25, size=(K, d))
for k in range(K):
    idx_tr = labels_tr == k
    idx_tst = labels_tst == k
    X_tr[idx_tr, :] = 2*np.random.randn(sum(idx_tr), d) + displace[k, :]
    X_tst[idx_tst, :] = 2*np.random.randn(sum(idx_tst), d) + displace[k, :]

intercepts = np.empty((t, K))
coefs = np.empty((t, d, K))

intercepts = np.random.randint(-2, 2, size=(t, K))
coefs = np.random.randint(-4, 4, size=(t, d, K))

for k in range(K):
    idx_tr = (labels_tr == k)
    idx_tst = (labels_tst == k)
    y_tr[idx_tr, :] = compute_targets(X_tr[idx_tr, :], 
        coefs[:, :, k], intercepts[:, k])
    y_tst[idx_tst, :] = compute_targets(X_tst[idx_tst, :], 
         coefs[:, :, k], intercepts[:, k])

model = KMeansRegressor(n_components=K)
y_ = model.fit(X_tr, y_tr)
est_weights = model.reg_weights_

print('Done')

if plot:
    for task in range(t):
        figure = plt.figure(task)
        for k in range(K):
            idx = labels_tr == k
            aux_y = compute_targets(X_tr[idx, :], est_weights[:, 1:, k], 
                                est_weights[:, 0, k], noise_var=0)
            plt.scatter(X_tr[idx, :], y_tr[idx, task])
            plt.plot(X_tr[idx, :], aux_y[:, task], 'r')
        plt.title('Fitted model for task %d'%task)
        plt.show()

