"""K-MEANS REGRESSION

@author: oghinde
"""

import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

class KMeansRegressor(object):
    def __init__(self, n_components=8, alpha=1, n_init=10, verbose=False, random_state=None):
        self.n_components_ = n_components
        self.alpha_ = alpha
        self.n_init = n_init
        self.kmeans_ = KMeans(n_clusters=self.n_components_, n_init=n_init)
        self.verbose = verbose
        self.regs_ = []
        self.random_state = random_state
        for k in range(self.n_components_):
            self.regs_.append(Ridge(alpha=self.alpha_))
    
    def _check_data(self, X, y):
        """Check that the input data is correctly formatted.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        y : array, shape (n_samples, n_targets)

        Returns
        -------
        t : int
            The total number of targets.
        
        n : int
            The total number of samples.

        d : int
            The total number of features (dimensions)

        """

        if y.ndim == 1:
            y = y[:, np.newaxis]

        n_x, d = X.shape
        n_y, t = y.shape

        if n_x == n_y:
            n = n_x
        else:
            print('Data size error. Number of samples in X and y must match:')
            print('X n_samples = %d, y n_samples = %d'%(n_x, n_y))
            print('Exiting.')
            sys.exit()

        return t, n, d, X, y


    def fit(self, X_tr, y_tr):
        """TODO: 
                - Add docstring.
                - Complete multioutput.
        """
        self.is_fitted_ = False
        t, n, d, X_tr, y_tr = self._check_data(X_tr, y_tr)
        labels_tr = self.kmeans_.fit_predict(X_tr, random_state=self.random_state)
        reg_weights = np.empty((t, d+1, self.n_components_))
        reg_precisions = np.zeros((t, self.n_components_))

        for k in range(self.n_components_):
            idx = (labels_tr == k)
            self.regs_[k].fit(X_tr[idx, :], y_tr[idx, :])
            reg_weights[:, 0, k] = self.regs_[k].intercept_
            reg_weights[:, 1:, k] = self.regs_[k].coef_
            reg_vars = np.var(y_tr[idx, :], axis=0)
            eps = 10 * np.finfo(reg_vars.dtype).eps
            reg_precisions[:, k] = 1/(reg_vars + eps)

        self.n_tasks_ = t
        self.n_input_dims_ = d
        self.labels_tr_ = labels_tr
        self.reg_weights_ = reg_weights.squeeze()
        self.reg_precisions_ = reg_precisions.squeeze()
        self.is_fitted = True
        self.cluster_centers_ = self.kmeans_.cluster_centers_
    
    def predict(self, X_tst):
        if not self.is_fitted: 
            print("Model isn't fitted.")
            return

        n, d = X_tst.shape
        if d != self.n_input_dims_:
            print('Incorrect dimensions for input data.')
            sys.exit(0)

        labels_tst = self.kmeans_.predict(X_tst)
        targets = np.empty((n, self.n_tasks_))
        for k in range(self.n_components_):
            idx = (labels_tst == k)
            # Check for empty clusters
            if sum(idx) != 0:
                targets[idx, :] = self.regs_[k].predict(X_tst[idx, :])
            else:
                if self.verbose:
                    print("No test samples in cluster %d"%k)
                pass

        self.labels_tst_ = labels_tst
        
        return targets
    
    def score(self, X_tst, y_tst):
        '''TODO: this needs extra metrics.
        '''
        targets = self.predict(X_tst)
        score = r2_score(y_tst, targets)
        return  score