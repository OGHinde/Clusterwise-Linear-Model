""" TEST MODULE FOR THE CLUSTERWISE REGRESSION MODEL

    Author: Óscar García Hinde <oghinde@tsc.uc3m.es>
    Python Version: 3.6
"""

from time import time
import numpy as np
#from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# Add our module to PATH
from pathlib import Path
home = str(Path.home())
import sys
sys.path.append(home + '/Git/Clusterwise_Linear_Model/')

from cwlm.clusterwise_linear_model import ClusterwiseLinModel as CWLM
from cwlm.clusterwise_linear_model_mt import ClusterwiseLinModel as MT_CWLM
from cwlm.gmm_regressor import GMMRegressor
from cwlm.kmeans_regressor import KMeansRegressor


def compute_targets(X, coefs, intercepts, RandomState, noise_var=0.5):
    n, d = X.shape
    t = coefs.shape[0]
    noise = noise_var*RandomState.randn(n, t)
    dot_product = np.dot(X, coefs.T)
    return intercepts + dot_product + noise

print('MULTIOUTPUT CLUSTERED REGRESSION TEST.\n')

n_tr = 500      # number of training samples
n_tst = 100     # number of testsamples
d = 1           # number of input dimensions
t = 2           # number of tasks
K = 3           # number of clusters
plot = True
#model = 'KMeansRegressor'
#model = 'GMMRegressor'
#model = 'CWLM'
model = 'MT_CWLM'
seed = None

print('Test parameters:')
print('\t- Training samples = ', n_tr)
print('\t- Test samples = ', n_tst)
print('\t- Input dimensions =', d)
print('\t- Regression tasks =', t)
print('\t- Clusters =', K)
print('\t- Selected model:', model)

if model == 'KMeansRegressor':
    model = KMeansRegressor(n_components=K)
elif model == 'GMMRegressor':
    model = GMMRegressor(n_components=K)
elif model == 'CWLM':
    model = CWLM(n_components=K, 
                 init_params='kmeans', 
                 plot=plot,
                 smoothing=True,
                 tol=1e-10, 
                 n_init=10)
elif model == 'MT_CWLM':
    model = MT_CWLM(n_components=K, 
                    init_params='gmm', 
                    plot=plot,
                    smoothing=True,
                    tol=1e-10, 
                    n_init=10)
else:
    print('\nIncorrect model specified')
    sys.exit(0)

RandomState = (np.random.RandomState(seed) if seed != None 
               else np.random.RandomState())

if d > 1 & plot == True:
    print('\nWarning: Too many dimensions to plot. Plotting defaulted to False.')
    plot = False

# DATA GENERATION
print('\nGenerating data...')
labels_tr = RandomState.randint(0, K, (n_tr, ))
labels_tst = RandomState.randint(0, K, (n_tst, ))

X_tr = np.empty((n_tr, d))
X_tst = np.empty((n_tst, d))
y_tr = np.empty((n_tr, t))
y_tst = np.empty((n_tst, t))

displace = RandomState.randint(-25, 25, size=(K, d))
for k in range(K):
    idx_tr = labels_tr == k
    idx_tst = labels_tst == k
    X_tr[idx_tr, :] = 2*RandomState.randn(sum(idx_tr), d) + displace[k, :]
    X_tst[idx_tst, :] = 2*RandomState.randn(sum(idx_tst), d) + displace[k, :]

intercepts = np.empty((t, K))
coefs = np.empty((t, d, K))

intercepts = RandomState.randint(-2, 2, size=(t, K))
coefs = RandomState.randint(-4, 4, size=(t, d, K))

for k in range(K):
    idx_tr = (labels_tr == k)
    idx_tst = (labels_tst == k)
    y_tr[idx_tr, :] = compute_targets(X=X_tr[idx_tr, :], 
        coefs=coefs[:, :, k], 
        intercepts=intercepts[:, k], 
        RandomState=RandomState)
    y_tst[idx_tst, :] = compute_targets(X_tst[idx_tst, :], 
        coefs[:, :, k], 
        intercepts[:, k],  
        RandomState=RandomState)

# MODEL EVALUATION
start = time()
print('Fitting model...')
model.fit(X_tr, y_tr)
stop = time()
print('Training time = ', stop - start)
y_pred, scores = model.predict_score(X_tst, y_tst, metric='all')
print('\nTest scores:')
for key, value in scores.items():
    print('\t- ', key, '=', value)

print('\nDone!')
if plot:    
    est_weights = model.reg_weights_
    labels_tr = model.labels_tr_
    labels_tst = model.labels_tst_
    X_tr_ext = np.concatenate((np.ones((n_tr, 1)), X_tr), axis=1)
    if est_weights.ndim == 2:
        # Make sure we can iterate even if there's only one task.
        est_weights = est_weights[np.newaxis, :, :]
        labels_tr = labels_tr[:, np.newaxis]
        labels_tst = labels_tst[:, np.newaxis]        
        y_pred = y_pred[:, np.newaxis]

    for task in range(t):
        for k in range(K):
            idx = labels_tr[:, task] == k
            idx = idx.squeeze()
            aux_y = np.dot(X_tr_ext[idx, :], est_weights[task, :, k])
            plt.scatter(X_tr[idx, :], y_tr[idx, task])
            plt.plot(np.sort(X_tr[idx, :]), aux_y, c='r')
        plt.title('Fitted model for task %d'%task)
        plt.show()
        
    for task in range(t):
        for k in range(K):
            idx = labels_tst == k
            idx = idx.squeeze()
            plt.scatter(X_tst[idx, :], y_tst[idx, task])
            plt.scatter(X_tst[idx, :], y_pred[idx, task], c='r', marker='.')
        plt.title('Model predictions for task %d'%task)
        plt.show()

