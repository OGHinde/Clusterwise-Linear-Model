"""K-MEANS REGRESSION

@author: oghinde
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

class KMeansRegressor(object):
    
    def __init__(self, n_components=8, alpha=1, verbose=False):
        self.n_components_ = n_components
        self.alpha_ = alpha
        self.kmeans_ = KMeans(n_clusters=self.n_components_)
        self.regs_ = list()
        self.verbose = verbose
        for k in range(self.n_components_):
            self.regs_.append(Ridge(alpha=self.alpha_))
        
    def fit(self, X_tr, y_tr):
        self.is_fitted_ = False
        n, d = X_tr.shape
        self.labels_ = self.kmeans_.fit_predict(X_tr)
        reg_weights = np.zeros((d+1, self.n_components_))
        reg_precisions = np.zeros((self.n_components_, ))
        for k in range(self.n_components_):
            idx = self.labels_ == k
            self.regs_[k].fit(X_tr[idx, :], y_tr[idx])
            reg_weights[1:, k] = self.regs_[k].coef_
            reg_weights[0, k] = self.regs_[k].intercept_
            reg_vars = np.var(y_tr[idx])
            eps = 10 * np.finfo(reg_vars.dtype).eps
            reg_precisions[k] = 1/(reg_vars + eps)
        self.reg_weights_ = reg_weights
        self.reg_precisions_ = reg_precisions
        self.is_fitted = True
        self.cluster_centers_ = self.kmeans_.cluster_centers_
    
    def predict(self, X_tst):
        if not self.is_fitted: 
            print("Model isn't fitted.")
            return

        n, d = X_tst.shape
        labels_tst = self.kmeans_.predict(X_tst)
        targets = np.zeros_like(labels_tst)
        for k in range(self.n_components_):
            idx = labels_tst == k
            if sum(idx) != 0:
                targets[idx] = np.squeeze(self.regs_[k].predict(X_tst[idx, :]))
            else:
                if self.verbose:
                    print("Empty cluster")
                pass
        self.labels_tst = labels_tst
        self.targets = targets
        return targets
    
    def fit_predict(self, X_tr, y_tr, X_tst):
        self.fit(X_tr, y_tr)
        targets = self.predict(X_tst)
        return targets
    
    def score(self, X_tst, y_tst):
        targets = self.predict(X_tst)
        score = r2_score(y_tst, targets)
        return  score