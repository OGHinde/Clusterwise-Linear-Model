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
        self.kmeans_ = KMeans(n_clusters=self.n_components_, n_init=n_init, random_state=random_state)
        self.verbose = verbose
        self.regs_ = []
        self.random_state = random_state
        for k in range(self.n_components_):
            self.regs_.append(Ridge(alpha=self.alpha_))
    
    def _check_data(self, X_km, X_reg, y):
        """Check that the input data is correctly formatted.

        Parameters
        ----------
        X_km : array-like, shape (n_samples, n_km_features)
        X_reg : array-like, shape (n_samples, n_reg_features)
        y : array, shape (n_samples, n_targets)

        Returns
        -------
        t : int, the total number of targets
        n : int, the total number of samples
        d_km : int, the total number of GMM features
        d_reg : int, the total number of regression features
        """

        if y.ndim == 1:
            y = y[:, np.newaxis]

        n_x_km, d_km = X_km.shape
        n_x_reg, d_reg = X_reg.shape
        n_y, t = y.shape

        if n_x_km == n_x_km:
            n_x = n_x_km
        else:
            error_report = 'Both views of the input data must have 1 to 1 sample correspondence.'
            error_details = '\nX_km has {} samples while X_reg has {} samples.'.format(n_x_km, n_x_reg) 
            raise ValueError(error_report, error_details)

        if n_x == n_y:
            n = n_x
        else:
            error_report = 'Data size error. Number of samples in X and y must match:'
            error_details = 'X n_samples = {}, y n_samples = {}'.format(n_x, n_y)
            raise ValueError(error_report, error_details)

        return t, n, d_km, d_reg, X_km, X_reg, y


    def fit(self, X_km, X_reg, y):
        """TODO: 
                - Add docstring.
                - Complete multioutput.
        """
        self.is_fitted_ = False
        t, n, d_km, d_reg, X_km, X_reg, y = self._check_data(X_km, X_reg, y)
        labels_tr = self.kmeans_.fit_predict(X_km)
        reg_weights = np.empty((t, d_reg + 1, self.n_components_))
        reg_precisions = np.zeros((t, self.n_components_))

        for k in range(self.n_components_):
            idx = (labels_tr == k)
            self.regs_[k].fit(X_reg[idx, :], y[idx, :])
            reg_weights[:, 0, k] = self.regs_[k].intercept_
            reg_weights[:, 1:, k] = self.regs_[k].coef_
            reg_vars = np.var(y[idx, :], axis=0)
            eps = 10 * np.finfo(reg_vars.dtype).eps
            reg_precisions[:, k] = 1/(reg_vars + eps)

        self.n_tasks_ = t
        self.n_km_dims_ = d_km
        self.n_reg_dims_ = d_reg
        self.labels_tr_ = labels_tr
        self.reg_weights_ = reg_weights.squeeze()
        self.reg_precisions_ = reg_precisions.squeeze()
        self.is_fitted = True
        self.cluster_centers_ = self.kmeans_.cluster_centers_
    
    def predict(self, X_km, X_reg):
        if not self.is_fitted:  
            raise RuntimeError("Model isn't fitted.")

        n, d_km = X_km.shape
        _, d_reg = X_reg.shape
        if d_km != self.n_km_dims_:
            raise ValueError('Incorrect dimensions for the KM input data.')
        if d_reg != self.n_reg_dims_:
            raise ValueError('Incorrect dimensions for the regression input data.')

        labels_tst = self.kmeans_.predict(X_km)
        targets = np.empty((n, self.n_tasks_))
        for k in range(self.n_components_):
            idx = (labels_tst == k)
            # Check for empty clusters
            if sum(idx) != 0:
                targets[idx, :] = self.regs_[k].predict(X[idx, :])
            else:
                if self.verbose:
                    print("No test samples in cluster %d"%k)
                pass

        self.labels_tst_ = labels_tst
        
        return targets
    
    def score(self, X_km, X_reg, y):
        '''TODO: this needs extra metrics.
        '''
        targets = self.predict(X_km, X_reg)
        score = r2_score(y, targets)
        return  score