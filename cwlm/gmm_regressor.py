"""GAUSSIAN MIXTURE REGRESSION

@author: oghinde
"""

import numpy as np
from scipy.linalg import solve
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import r2_score

class GMMRegressor(object):
    
    def __init__(self, n_components=8, alpha=1, n_init=20, covariance_type='diag', verbose=0):
        self.n_components_ = n_components
        self.alpha_ = alpha
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.verbose = verbose
        
    def fit(self, X_tr, y_tr):
        self.is_fitted_ = False
        eps = np.finfo(float).eps
        n, d = X_tr.shape
        
        # Determine training sample/component posterior probability
        gmm = GMM(n_components=self.n_components_, n_init=self.n_init)
        gmm.fit(X_tr)
        resp_tr = gmm.predict_proba(X_tr)
        
        # Calculate weights conditioned on posterior probabilities
        reg_weights = np.zeros((d+1, self.n_components_))
        X_ext = np.concatenate((np.ones((n, 1)), X_tr), axis=1)
        for k in range(self.n_components_):
            R_k = np.diag(resp_tr[:, k] + eps)
            R_kX = R_k.dot(X_ext)
            L = R_kX.T.dot(X_ext) + np.eye(d+1) * self.alpha_
            R = R_kX.T.dot(y_tr)
            reg_weights[:, k] = np.squeeze(solve(L, R, sym_pos=True))

        means = np.dot(X_ext, reg_weights)
        err = (np.tile(y_tr[:, np.newaxis], (1, self.n_components_)) - means) ** 2
        reg_precisions = n * gmm.weights_ / np.sum(resp_tr * err)

        self.resp_ = resp_tr 
        self.reg_precisions_ = reg_precisions
        self.reg_weights_ = reg_weights
        self.gmm_ = gmm
        self.is_fitted = True
        return resp_tr
    
    def predict(self, X_tst):
        if self.is_fitted:
            n, d = X_tst.shape
        
            # Determine test sample/component posterior probability
            resp_tst = self.gmm_.predict_proba(X_tst)
        
            # Predict test targets
            X_ext = np.concatenate((np.ones((n, 1)), X_tst), axis=1)
            dot_prod = np.dot(X_ext, self.reg_weights_)
            targets = np.sum(resp_tst * dot_prod, axis=1)

            return targets
        else:
            print("Model isn't fitted.")
            return
    
    def fit_predict(self, X_tr, y_tr, X_tst):
        self.fit(X_tr, y_tr)
        targets = self.predict(X_tst)
        return targets
    
    def score(self, X_tst, y_tst):
        targets = self.predict(X_tst)
        score = r2_score(y_tst, targets)
        return  score