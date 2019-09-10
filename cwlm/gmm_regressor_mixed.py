"""GAUSSIAN MIXTURE REGRESSION

    Author: Óscar García Hinde <oghinde@tsc.uc3m.es>
    Python Version: 3.6
"""

import numpy as np
from scipy.linalg import solve
from sklearn.mixture import GaussianMixture as GMM
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

def _estimate_regression_params_k(X, y, resp_k, alpha, weight_k):
    """Estimate the regression weights for the output space for component k.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    y : array-like, shape (n_samples, )

    resp_k : array-like, shape (n_samples, )

    alpha : float

    Returns
    -------
    reg_weights : array, shape (n_tasks, n_features + 1)
        The regression weights for component k.
    
    reg_precisions_k : array, shape (n_tasks, )
        The regression precisions for component k.
    """
    n, d = X.shape
    _, t = y.shape
    eps = 10 * np.finfo(resp_k.dtype).eps
    X_ext = np.concatenate((np.ones((n, 1)), X), axis=1)
    reg_weights_k = np.empty((t, d+1))
    reg_precisions_k = np.empty((t, ))
    
    # Compute regression weights using Ridge
    solver = Ridge(alpha=alpha)
    solver.fit(X, y, sample_weight=resp_k + eps)
    reg_weights_k[:, 0] = solver.intercept_
    reg_weights_k[:, 1:] = solver.coef_

    # Compute regression precisions
    means = np.dot(X_ext, reg_weights_k.T)
    err = (y - means) ** 2
    reg_precisions_k = n * weight_k / np.sum(resp_k[:, np.newaxis] * err)

    return reg_weights_k, reg_precisions_k

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
    def __init__(self, n_components=8, alpha=1, n_init=10, covariance_type='diag', 
                 verbose=0, random_state=None):
        self.n_components = n_components
        self.alpha = alpha
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        
    def _check_data(self, X_gmm, X_reg, y):
        """Check that the input data is correctly formatted.

        Parameters
        ----------
        X_gmm : array-like, shape (n_samples, n_gmm_features)
        X_reg : array-like, shape (n_samples, n_reg_features)
        y : array, shape (n_samples, n_targets)

        Returns
        -------
        t : int, the total number of targets
        n : int, the total number of samples
        d_gmm : int, the total number of GMM features
        d_reg : int, the total number of regression features
        """

        if y.ndim == 1:
            y = y[:, np.newaxis]

        n_x_gmm, d_gmm = X_gmm.shape
        n_x_reg, d_reg = X_reg.shape
        n_y, t = y.shape

        if n_x_gmm == n_x_gmm:
            n_x = n_x_gmm
        else:
            error_report = 'Both views of the input data must have 1 to 1 sample correspondence.'
            error_details = '\nX_gmm has {} samples while X_reg has {} samples.'.format(n_x_gmm, n_x_reg) 
            raise ValueError(error_report, error_details)

        if n_x == n_y:
            n = n_x
        else:
            error_report = 'Data size error. Number of samples in X and y must match:'
            error_details = 'X n_samples = {}, y n_samples = {}'.format(n_x, n_y)
            raise ValueError(error_report, error_details)

        return t, n, d_gmm, d_reg, X_gmm, X_reg, y

    def fit(self, X_gmm, X_reg y):
        self.is_fitted_ = False
        t, n, d_gmm, d_reg, X_gmm, X_reg, y = self._check_data(X, y)
        eps = 10 * np.finfo(float).eps
        
        # Determine training sample/component posterior probability
        gmm = GMM(n_components=self.n_components, 
                  n_init=self.n_init, 
                  random_state=self.random_state)
        gmm.fit(X_gmm)
        resp_tr = gmm.predict_proba(X_gmm)
        labels_tr = np.argmax(resp_tr, axis=1)
        
        # Calculate regression weights & precisions conditioned on 
        # posterior probabilities
        reg_weights = np.empty((t, d_reg + 1, self.n_components))
        reg_precisions = np.zeros((t, self.n_components))
        for k in range(self.n_components):
            (reg_weights[:, :, k], 
            reg_precisions[:, k]) = _estimate_regression_params_k(X_reg, y, 
                resp_k=resp_tr[:, k], alpha=self.alpha, weight_k=gmm.weights_[k])
        
        self.n_tasks_ = t
        self.n_gmm_dims_ = d_gmm
        self.n_reg_dims_ = d_reg
        self.resp_tr_ = resp_tr
        self.labels_tr_ = labels_tr
        self.reg_weights_ = reg_weights.squeeze()
        self.reg_precisions_ = reg_precisions.squeeze()
        self.gmm_ = gmm
        self.is_fitted_ = True
    
    def predict(self, X_gmm, X_reg):
        if not self.is_fitted:  
            raise RuntimeError("Model isn't fitted.")

        n, d_gmm = X_gmm.shape
        _, d_reg = X_reg.shape
        if d_gmm != self.n_gmm_dims_:
            raise ValueError('Incorrect dimensions for the GMM input data.')
        if d_reg != self.n_reg_dims_:
            raise ValueError('Incorrect dimensions for the regression input data.')
        
        # Make sure we can iterate when n_tasks = 1
        if self.n_tasks_ == 1:
            reg_weights = self.reg_weights_[np.newaxis, :, :]
        else:
            reg_weights = self.reg_weights_

        # Determine test sample posterior probability
        resp_tst = self.gmm_.predict_proba(X_gmm)
        labels_tst = np.argmax(resp_tst, axis=1)

        # Predict test targets
        X_ext = np.concatenate((np.ones((n, 1)), X_reg), axis=1)
        targets = np.zeros((n, self.n_tasks_))
        for k in range(self.n_components):
            dot_prod = np.dot(X_ext, reg_weights[:, :, k].T)            
            targets += resp_tst[:, k][:, np.newaxis] * dot_prod

        self.resp_tst_ = resp_tst
        self.labels_tst_ = labels_tst

        return targets
    
    def score(self, X_gmm, X_reg, y):
        targets = self.predict(X_gmm, X_reg)
        score = r2_score(y, targets)
        return  score