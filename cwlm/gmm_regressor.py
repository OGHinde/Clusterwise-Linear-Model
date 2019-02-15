"""GAUSSIAN MIXTURE REGRESSION

    Author: Óscar García Hinde <oghinde@tsc.uc3m.es>
    Python Version: 3.6
"""

import numpy as np
from scipy.linalg import solve
from sklearn.mixture import GaussianMixture as GMM
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

def _estimate_regression_weights(X, y, resp_k, alpha):
    """Estimate the regression weights for the output space for component k.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    y : array-like, shape (n_samples, )

    resp_k : array-like, shape (n_samples, )

    alpha : float

    Returns
    -------
    reg_weights : array, shape (n_features, )
        The regression weights for component k.
    """
    _, d = X.shape
    eps = 10 * np.finfo(resp_k.dtype).eps
    reg_weights_k = np.zeros((d+1,))
    
    solver = Ridge(alpha=alpha)
    solver.fit(X, y, sample_weight=resp_k + eps)
    reg_weights_k[0] = solver.intercept_
    reg_weights_k[1:] = solver.coef_

    return reg_weights_k

class GMMRegressor(object):
    """Linear regression on Gaussian Mixture components.

    Combination of a Gaussian mixture model for input clustering with a 
    per-component linear regression.
    
    Te likelyhoods for each sample are used as sample-weights in the 
    reggression stage.

    Parameters
    ----------
    n_components : int,  defaults to 1.
        The number of mixture components.

    alpha : int, defaults to 1.
        The regression L2 regularization term

    n_init : int, defaults to 1.
        The number of EM initializations to perform. The best results are kept.

    covariance_type : {'full' (default), 'tied', 'diag', 'spherical'}
        String describing the type of covariance parameters to use.
        Must be one of:

        'full'
            each component has its own general covariance matrix
        'tied'
            all components share the same general covariance matrix
        'diag'
            each component has its own diagonal covariance matrix
        'spherical'
            each component has its own single variance

    Attributes
    ----------

    TODO

    weights_ : array-like, shape (n_components, )
        The weights of each mixture components.

    reg_weights_ : array-like, shape ( n_features + 1, n_components)
        The linear regressor weights fo each mixture component.

    precisions_ : array-like, shape (n_components, )
        The precisions of each mixture component. The precision is the inverse 
        of the variance. 

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    lower_bound_ : float
        Log-likelihood of the best fit of EM.

    """
    def __init__(self, n_components=8, alpha=1, n_init=10, covariance_type='diag', verbose=0):
        self.n_components = n_components
        self.alpha = alpha
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.verbose = verbose
        
    def _check_data(X, y):
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
            print('X n_samples = {}, y n_samples = {}'.format(n_x, n_y))
            print('Exiting.')
            sys.exit()

        return t, n, d, X, y

    def fit(self, X, y):

        self.is_fitted_ = False
        t, n, d, X, y = self._check_data(X, y)
        eps = 10 * np.finfo(float).eps
        
        # Determine training sample/component posterior probability
        gmm = GMM(n_components=self.n_components, n_init=self.n_init)
        gmm.fit(X)
        resp_tr = gmm.predict_proba(X)
        
        # Calculate weights conditioned on posterior probabilities
        reg_weights = np.zeros((d+1, self.n_components))
        X_ext = np.concatenate((np.ones((n, 1)), X), axis=1)

        for k in range(self.n_components):
            
            #R_k = np.diag(resp_tr[:, k] + eps)
            #R_kX = R_k.dot(X_ext)
            #L = R_kX.T.dot(X_ext) + np.eye(d+1) * self.alpha
            #R = R_kX.T.dot(y)
            #reg_weights[:, k] = np.squeeze(solve(L, R, sym_pos=True))  

            reg_weights[:, k] = _estimate_regression_weights(X, y, 
                resp_k=resp_tr[:, k], alpha=self.alpha)

        means = np.dot(X_ext, reg_weights)
        err = (np.tile(y[:, np.newaxis], (1, self.n_components)) - means) ** 2
        reg_precisions = n * gmm.weights_ / np.sum(resp_tr * err)

        self.resp_ = resp_tr 
        self.reg_precisions_ = reg_precisions
        self.reg_weights_ = reg_weights
        self.gmm_ = gmm
        self.is_fitted = True
        return resp_tr
    
    def predict(self, X):
        if self.is_fitted:
            n, d = X.shape
        
            # Determine test sample/component posterior probability
            resp_tst = self.gmm_.predict_proba(X)
        
            # Predict test targets
            X_ext = np.concatenate((np.ones((n, 1)), X), axis=1)
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
    
    def score(self, X, y):
        targets = self.predict(X)
        score = r2_score(y, targets)
        return  score