""" TEST MODULE FOR THE CLUSTERWISE REGRESSION MODEL

    Author: Óscar García Hinde <oghinde@tsc.uc3m.es>
    Python Version: 3.6
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt
import os

# Add our module to PATH
from pathlib import Path
home = str(Path.home())
import sys
sys.path.append(home + '/Git/Clusterwise_Linear_Model/')

import pickle
from cwlm.clusterwise_linear_model_mixed import ClusterwiseLinModel as CWLM
from cwlm.gmm_regressor_mixed import GMMRegressor
from cwlm.kmeans_regressor_mixed import KMeansRegressor

def time_format(seconds):
    """Display a time given in seconds in a more pleasing manner.
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return '{:.0f} hours, {:.0f} minutes {:.3f} seconds'.format(h, m, s)

def compute_targets(X, coefs, intercepts, RandomState, noise_var=0.5):
    n, d = X.shape
    t = coefs.shape[0]
    noise = noise_var*RandomState.randn(n, t)
    dot_product = np.dot(X, coefs.T)
    return intercepts + dot_product + noise

print('MULTIOUTPUT CLUSTERED REGRESSION TEST.\n')
n_tr = 500      # number of training samples
n_tst = 100     # number of testsamples
d_gmm = 10       # number of input dimensions for the gmm view input matrix
d_reg = 5       # number of input dimensions for the reg view input matrix
t = 1           # number of tasks
K = 3           # number of clusters
seed = None
plot_data = True
load_data = True
save_data = True
plot_bounds = False
quick = True

#model = 'KMeansRegressor'
#model = 'GMMRegressor'
model = 'CWLM'

print('Test parameters:')
print('\t- Training samples = {}'.format(n_tr))
print('\t- Test samples = {}'.format(n_tst))
print('\t- Input dimensions for gmm = {}'.format(d_gmm))
print('\t- Input dimensions for gmm = {}'.format(d_reg))
print('\t- Regression tasks = {}'.format(t))
print('\t- Number of clusters = {}'.format(K))
print('\t- Selected model: {}'.format(model))

if model == 'KMeansRegressor':
    model = KMeansRegressor(n_components=K)
elif model == 'GMMRegressor':
    model = GMMRegressor(n_components=K)
elif model == 'CWLM':
    model = CWLM(n_components=K, 
                 init_params='gmm', 
                 plot=plot_bounds,
                 smoothing=True,
                 tol=1e-10, 
                 n_init=10,
                 random_seed=1)
else:
    print('\nIncorrect model specified')
    sys.exit(0)

RandomState = (np.random.RandomState(seed) if seed != None 
               else np.random.RandomState())

# DATA GENERATION

# Generate new dataset
print('\nGenerating data...')
labels_tr = RandomState.randint(0, K, (n_tr, ))
labels_tst = RandomState.randint(0, K, (n_tst, ))
    
X_gmm_tr = np.empty((n_tr, d_gmm))
X_reg_tr = np.empty((n_tr, d_reg))
X_gmm_tst = np.empty((n_tst, d_gmm))
X_reg_tst = np.empty((n_tst, d_reg))
y_tr = np.empty((n_tr, t))
y_tst = np.empty((n_tst, t))
    
displace_gmm = RandomState.randint(-10, 10, size=(K, d_gmm))
displace_reg = RandomState.randint(-10, 10, size=(K, d_reg))
for k in range(K):
    idx_tr = labels_tr == k
    idx_tst = labels_tst == k
    X_gmm_tr[idx_tr, :] = 2*RandomState.randn(sum(idx_tr), d_gmm) + displace_gmm[k, :]
    X_reg_tr[idx_tr, :] = 2*RandomState.randn(sum(idx_tr), d_reg) + displace_reg[k, :]
    X_gmm_tst[idx_tst, :] = 2*RandomState.randn(sum(idx_tst), d_gmm) + displace_gmm[k, :]
    X_reg_tst[idx_tst, :] = 2*RandomState.randn(sum(idx_tst), d_reg) + displace_reg[k, :]

X_reg_tr = np.hstack((X_gmm_tr, X_reg_tr))
X_reg_tst = np.hstack((X_gmm_tst, X_reg_tst))

intercepts = np.empty((t, K))
coefs = np.empty((t, d_gmm + d_reg, K))
    
intercepts = RandomState.randint(-2, 2, size=(t, K))
coefs = RandomState.randint(-4, 4, size=(t, d_gmm + d_reg, K))
    
for k in range(K):
    idx_tr = (labels_tr == k)
    idx_tst = (labels_tst == k)
    y_tr[idx_tr, :] = compute_targets(X=X_reg_tr[idx_tr, :],
                                      coefs=coefs[:, :, k],
                                      intercepts=intercepts[:, k],
                                      RandomState=RandomState)
    y_tst[idx_tst, :] = compute_targets(X_reg_tst[idx_tst, :],
                                        coefs[:, :, k],
                                        intercepts[:, k],
                                        RandomState=RandomState)

# MODEL EVALUATION
start = time()
print('Fitting model...')
model.fit(X_gmm_tr, X_reg_tr, y_tr)
stop = time()
elapsed_time = time_format(stop-start)
print('Training time =', elapsed_time)

y_pred, scores = model.predict_score(X_gmm_tst, X_reg_tst, y_tst, metric='all')
print('\nTest scores:')
for key, value in scores.items():
    print('\t- {} = {:.3f}'.format(key, value))
print('\nDone!')
'''
# PLOTTING
if plot_data:    
    est_weights = model.reg_weights_
    labels_tr = model.labels_tr_
    labels_tst = model.labels_tst_
        
    if est_weights.ndim == 2:
        # Make sure we can iterate even if there's only one task.
        est_weights = est_weights[np.newaxis, :, :]
        labels_tr = labels_tr[:, np.newaxis]
        labels_tst = labels_tst[:, np.newaxis]
        y_pred = y_pred[:, np.newaxis]
    
    for task in range(t):
        figure, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False, figsize=(6, 10))
        for k in range(K):
            idx_tr = (labels_tr[:, task] == k).squeeze()
            idx_tst = (labels_tst == k).squeeze()
            
            ax1.scatter(X_tr, y_tr[:, task])
            ax1.set_xlabel('X')
            ax1.set_ylabel('y')
            
            aux_X = np.sort(X_tr[idx_tr, :], axis=0)[[0, -1]]
            aux_X_ext = np.hstack((np.ones((2, 1)), aux_X))
            aux_y = np.dot(aux_X_ext, est_weights[task, :, k])
            ax2.scatter(X_tr[idx_tr, :], y_tr[idx_tr, task])
            ax2.plot(aux_X, aux_y, 'k--')
            
            ax3.scatter(X_tst[idx_tst, :], y_tst[idx_tst, task])
            ax3.scatter(X_tst[idx_tst, :], y_pred[idx_tst, task], c='r', marker='.', label='Preditions')
        ax1.set_title('Training set', loc='left', fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('y')
        ax2.set_title('Fitted model on the training set', loc='left', fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('y')
        ax3.set_title('Model predictions for the test set', loc='left', fontweight='bold')
        ax3.set_xlabel('X')
        ax3.set_ylabel('y')
        figure.tight_layout()
        #st = figure.suptitle('Example Dataset'.format(task+1), fontsize=15, fontweight='bold')
        #st.set_y(0.98)
        figure.subplots_adjust(top=0.92)
        figure.show()
          
# SAVING MODEL
if save_data:
    data = {'X_tr': X_tr,
            'y_tr': y_tr,
            'X_tst': X_tr,
            'y_tst': y_tr,
            'Trained model': model}
    while True:
        challenge = input('Save dataset? y/n: ')
        if challenge == 'y':    
            name = input('Specify file name: ')    
            print('Saving dataset as', name + '.pickle')
            plt.savefig('example_datasets/' + name + '.png', 
                        format='png', 
                        dpi=500, 
                        bbox_inches='tight')
            with open('example_datasets/' + name + '.pickle', 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            print('Done!')
            break
        elif challenge == 'n':
            print('Dataset discarded.')
            break
        else:
            print("Wrong input. Please type 'y' or 'n'.")            
'''