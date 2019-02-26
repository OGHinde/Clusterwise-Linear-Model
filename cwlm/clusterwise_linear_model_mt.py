"""CLUSTERWISE LINEAR REGRESSION MODEL - MULTI-TASK

    Author: Óscar García Hinde <oghinde@tsc.uc3m.es>
    Python Version: 3.6

TODO:
    - Implement parallelization with MPI.
    - Implement other input covariances.
    - Revisit RandomState
    - Update Attributes docstring.

ISSUES:
    - Weirdness in the lower bound results indicates that something's not
      quite right.
    - Turns out this version is a little faster?? LOL Wut?  

NEXT STEPS:
    - Pruning of weak clusters.
"""

import sys
import numpy as np
from numpy.random import RandomState
from scipy import linalg
from scipy.stats import norm
from scipy.misc import logsumexp
from sklearn.utils import check_array
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from cwlm.kmeans_regressor import KMeansRegressor
from cwlm.gmm_regressor import GMMRegressor

from mpi4py import MPI

import warnings
from sklearn.exceptions import ConvergenceWarning

import matplotlib.pyplot  as plt

def mean_absolute_percentage_error(y_true, y_pred, multitarget=None):
    """Mean absolute precentage error regression loss.
    TODO: multi target.
    ----------
    y_true : array-like, shape = (n_samples) or (n_samples, n_targets)
        Ground truth (correct) target values.
    y_pred : array-like, shape = (n_samples) or (n_samples, n_targets)
        Estimated target values.

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    y_true = check_array(y_true)
    y_pred = check_array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

###############################################################################
# USED IN THE E STEP 

def _estimate_log_prob_X(X, means, precisions_cholesky):
    """Estimate the log Gaussian probability of the input space,
    i.e. the log probability factor for each sample in X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    means : array-like, shape (n_components, n_features)

    precisions_cholesky : array-like
        Cholesky decompositions of the precision matrices.

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n, d = X.shape
    k, _ = means.shape
    # det(precision_chol) is half of det(precision)
    log_det = _compute_log_det_cholesky(precisions_cholesky, d)

    log_prob = np.empty((n, k))
    for k, (mu, prec_chol) in enumerate(zip(means, precisions_cholesky)):
        y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
        log_prob[:, k] = np.sum(np.square(y), axis=1)

    return -.5 * (d * np.log(2 * np.pi) + log_prob) + log_det

def _compute_log_det_cholesky(matrix_chol, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.

    Parameters
    ----------
    matrix_chol : array-like
        Cholesky decompositions of the matrices.

    n_features : int
        Number of features.

    Returns
    -------
    log_det_precision_chol : array-like, shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    k, _, _ = matrix_chol.shape
    log_det_chol = (np.sum(np.log(
        matrix_chol.reshape(
            k, -1)[:, ::n_features + 1]), 1))

    return log_det_chol

def _estimate_log_prob_y_k(X, y, reg_weights_k, reg_precisions_k):
    """Estimate the log Gaussian probability of the output space,
    i.e. the log probability factor for each sample in y for 
    component k.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
    y : array, shape (n_samples, n_targets)
    reg_weights_k : array, shape (n_targets, n_features)
    reg_precisions_k : array, shape (n_targets, )

    Returns
    -------
    log_prob : array, shape (n_targets, n_samples, n_components)
    """
    n, d = X.shape
    _, t = y.shape

    # Extend X with a column of ones 
    X_ext = np.concatenate((np.ones((n, 1)), X), axis=1)
    
    means = np.dot(X_ext, reg_weights_k.T)
    std_devs = np.sqrt(reg_precisions_k ** -1)

    return norm.logpdf(y, loc=means, scale=std_devs)


###############################################################################
# USED IN THE M STEP 

def _estimate_gaussian_parameters(X, resp, reg_covar):
    """Estimate the Gaussian distribution parameters for the input space.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data array.

    resp : array-like, shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.

    Returns
    -------
    nk : array-like, shape (n_components,)
        The numbers of data samples in the current components.

    means : array-like, shape (n_components, n_features)
        The centers of the current components.

    covariances : array-like
        The covariance matrix of the current components. 
    """
    
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = _estimate_gaussian_covariances(resp, X, nk, means, reg_covar)

    return nk, means, covariances

def _estimate_gaussian_covariances(resp, X, nk, means, reg_covar):
    """Estimate the full covariance matrices for the input space.

    Parameters
    ----------
    resp : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    means : array-like, shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[::n_features + 1] += reg_covar
    return covariances

def _compute_precision_cholesky(covariances):
    """Compute the Cholesky decomposition of the precisions for 
    the input space.

    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.

    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    """

    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar.")
    
    n_components, n_features, _ = covariances.shape
    precisions_chol = np.empty((n_components, n_features, n_features))
    for k, covariance in enumerate(covariances):
        try:
            cov_chol = linalg.cholesky(covariance, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol[k] = linalg.solve_triangular(cov_chol, 
                                                     np.eye(n_features), 
                                                     lower=True).T
    
    return precisions_chol

def _estimate_regression_params_k(X, y, resp_k, reg_term_k, weight_k):
    """Estimate the regression parameters for the output space for component k.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)

    y : array, shape (n_samples, n_targets)

    resp_k : array, shape (n_samples, n_targets)

    reg_term_k : array, shape (n_targets, )

    Returns
    -------
    reg_weights : array, shape (n_targets, n_features)
        The regression weights for component k.
    """
    n, d = X.shape
    _, n_targets = y.shape
    X_ext = np.concatenate((np.ones((n, 1)), X), axis=1)
    eps = 10 * np.finfo(resp_k.dtype).eps
    reg_weights_k = np.empty((n_targets, d+1))
    reg_precisions_k = np.empty((n_targets, ))
    
    # Compute the output space regression weights using Ridge
    # We'll iterate over all targets for now. Not sure if this can be
    # done without the loop if n_targets > 1.
    for t in range(n_targets):
        solver = Ridge(alpha=reg_term_k[t])
        solver.fit(X, y[:, t], sample_weight=resp_k[:, t] + eps)
        reg_weights_k[t, 0] = solver.intercept_
        reg_weights_k[t, 1:] = solver.coef_

    # Compute the output space precision terms
    means = np.dot(X_ext, reg_weights_k.T)
    err = (y - means) ** 2
    product = np.multiply(resp_k + eps, err)
    reg_precisions_k = n * weight_k / np.sum(product, axis=0)
    
    return reg_weights_k, reg_precisions_k


###############################################################################
# MAIN CLASS

class ClusterwiseLinModel():
    """Clusterwise Linear Regressor Mixture.

    Representation of a coupled Gaussian and linear regression mixture 
    probability distribution. It handles multi-target regression, albeit
    assuming full independence of the targets.
    
    This class estimates the parameters of said mixture distribution using
    the EM algorithm.
    
    Parameters
    ----------
    n_components : int,  defaults to 5.
        The number of mixture components.

    eta : int, defaults to 1.
        The regression L2 regularization term

    tol : float, defaults to 1e-10.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, defaults to 200.
        The number of EM iterations to perform.

    n_init : int, defaults to 10.
        The number of EM initializations to perform. The best results are kept.

    init_params : str, defaults to 'gmm'.
        The method used to initialize the Gaussian mixture weights, means and
        precisions; and the linear regressor weights and precisions.
            'gmm' : a Gaussian mixture + Ridge Regression model is used.
            'kmeans' : a K-Means + Ridge Regression model is used.
            'random' : weights are initialized randomly.
    
    smoothing : bool, defaults to False.
        Wether or not to perform average smoothing on the evolution  of the 
        lower bound. Can help converge quicker in ill conditioned databases.

    smooth_window : int, defaults to 20.
        The averaging window size in case lower bound smoothing is used.

    weights_init : array-like, shape (n_components, ), optional
        The user-provided initial weights, defaults to None.
        If it None, weights are initialized using the `init_params` method.

    means_init : array-like, shape (n_components, n_features), optional.
        The user-provided initial means for the Gaussian Mixture model at input 
        space.
    
    precisions_init : array-like, optional.
        The user-provided initial precisions (inverse of the covariance matrices)
        for the Gaussian Mixture model at input space.
        
    reg_weights_init : array-like, shape (n_targets, n_components, n_features + 1), 
        optional The user-provided initial means, defaults to None,
        If it None, means are initialized using the `init_params` method.
    
    reg_precisions_init : array-like, shape (n_targets, n_components), optional.
        The user-provided initial precisions.
        (The precision is the inverse of the label noise variance).

    random_seed : int, defaults to None.
        The random number generator seed. To be used when init_params is set 
        to 'random'.

    plot : bool, defaults to False.
        Enable plotting of the lower bound's evolution for each initialisation.

    Attributes
    ----------
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

    def __init__(self, n_components=5, eta=1, tol=1e-10, reg_covar=1e-6,
                 max_iter=200, n_init=10, init_params='gmm', smoothing=False, 
                 smooth_window=20, weights_init=None, means_init=None, 
                 covariances_init=None, reg_weights_init=None, 
                 reg_precisions_init=None, random_seed=None, plot=False):
        self.weights_init = weights_init                # Pi_k in the notes
        self.means_init = means_init                    # mu_k in the notes
        self.covariances_init = covariances_init        # Sigma_k in the notes              
        self.reg_weights_init = reg_weights_init        # W_k in the notes
        self.reg_precisions_init = reg_precisions_init  # Beta_k in the notes
        self.eta = eta                                  # Eta in the notes
        self.n_init = n_init
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.init_params = init_params
        self.smoothing = smoothing
        self.smooth_window = smooth_window
        self.random_seed = random_seed
        self.plot = plot

    def _check_data(self, X, y):
        """Check that the input data is correctly formatted.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array, shape (n_samples, n_targets)

        Returns
        -------
        t : int, the total number of targets.
        n : int, the total number of samples.
        d : int, the total number of features (dimensions)

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

    def _initialise(self, X, y, RandomState):
        """Initialization of the Clusterwise Linear Model parameters.    

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples, n_targets)

        resp : array-like, shape (n_samples, n_components)
        """
        
        n, d = X.shape

        if self.init_params == 'kmeans':
            initializer = KMeansRegressor(n_components=self.n_components, 
                                          alpha=self.eta)
            initializer.fit(X, y)
            resp = np.zeros((n, self.n_components))
            resp[np.arange(n), initializer.labels_tr_] = 1
            reg_weights = initializer.reg_weights_
            reg_precisions = initializer.reg_precisions_

        elif self.init_params == 'gmm':
            initializer = GMMRegressor(n_components=self.n_components, 
                                       alpha=self.eta, 
                                       n_init=1, 
                                       covariance_type='full')
            initializer.fit(X, y)
            resp = initializer.resp_tr_
            reg_weights = initializer.reg_weights_
            reg_precisions = initializer.reg_precisions_
        
        elif self.init_params == 'random':
            # This tends to work like crap.
            # TODO: adapt it to multitarget.
            resp = RandomState.rand(n, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
            reg_weights = RandomState.randn(d + 1, self.n_components)
            reg_precisions = np.zeros((self.n_components, )) + 1 / np.var(y)
        else:
            raise ValueError("Unimplemented initialization method '%s'"
                             % self.init_params)

        (weights, 
        means, 
        covariances) = _estimate_gaussian_parameters(X, resp, self.reg_covar)
        weights /= n

        self.weights_ = weights if self.weights_init is None else self.weights_init
        self.means_ = means if self.means_init is None else self.means_init

        if self.covariances_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(covariances)
        else:
            self.covariances_ = self.covariances_init
            self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_)

        if self.reg_weights_init is None:
            self.reg_weights_ = reg_weights  
        else: 
            self.reg_weights_ = self.reg_weights_init
        
        if self.reg_precisions_init is None:
            self.reg_precisions_ = reg_precisions  
        else: 
            self.reg_precisions_ = self.reg_precisions_init

    def fit(self, X, y):
        """Fit the clustered linear regressor model for a training 
        data set using the EM algorithm.

        It does n_init instances of the algorithm and keeps the one with the
        highest complete log-likelihood.
        
        Each initialization of the algorithm runs until convergence or max_iter
        times.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array, shape (n_samples, n_targets)

        Returns
        -------
        self
        """
        n, d = X.shape
        max_lower_bound = -np.infty
        self.converged_ = False
        self.is_fitted_ = False
        
        # Summon the Random Number Gods
        rng = RandomState(self.random_seed)
        
        t, n, d, X, y = self._check_data(X, y)
        self.n_targets_ = t
        self.n_input_dims_ = d
        
        for init in range(self.n_init):            
            lower_bound = -np.infty
            self._initialise(X, y, rng)
            bound_curve = []
            smooth_bound_curve = []

            for n_iter in range(1, self.max_iter + 1):
                if self.smoothing:
                    # TODO: This should be done in a less cludgy way
                    if n_iter == 1:
                        prev_lower_bound = lower_bound
                    else:    
                        prev_lower_bound = smooth_bound_curve[-1]
                else:
                    prev_lower_bound = lower_bound

                # E-Step and M-Step
                log_prob_norm, log_resp, _, _, _ = self._e_step(X, y)
                self._m_step(X, y, log_resp)

                # Update lower bound
                lower_bound = self._compute_lower_bound(log_prob_norm)
                bound_curve.append(lower_bound)

                # Smoothen the bound curve
                if n_iter <= self.smooth_window:
                    # Compute mean of n_iter previous values
                    smooth_bound = np.mean(bound_curve)
                    smooth_bound_curve.append(smooth_bound)
                else:
                    smooth_bound = np.mean(bound_curve[-self.smooth_window:])
                    smooth_bound_curve.append(smooth_bound)

                # Compute lower bound gain and check convergence
                if self.smoothing:
                    change = smooth_bound - prev_lower_bound
                else:    
                    change = lower_bound - prev_lower_bound

                if abs(change) < self.tol:
                    self.converged_ = True
                    break
            
            if self.plot:
                # Plot lower bound evolution curve                
                plt.plot(range(1, n_iter+1), bound_curve, label='Standard')
                plt.plot(range(1, n_iter+1), smooth_bound_curve, label='Smoothed')
                plt.xlabel('Iteration')
                plt.ylabel('Lower bound')
                plt.title('Lower bound evolution for init {}'.format(init + 1))
                plt.legend(loc=4)
                plt.show()

            # Keep best initialisation
            if lower_bound > max_lower_bound:
                max_lower_bound = lower_bound
                best_params = self._get_parameters()
                best_n_iter = n_iter
                best_curves = {'standard': bound_curve,
                               'smooth': smooth_bound_curve}


        # Always do a final e-step to guarantee that the labels returned by
        # fit_pred(X, y) are always consistent with fit(X, y).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp, labels_tr, labels_X, labels_y = self._e_step(X, y)

        if not self.converged_:
            warnings.warn('Initialization %d did not converge. '
                          'Try different init parameters, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.'
                          % (init + 1), ConvergenceWarning)        

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound
        self.resp_tr_ = np.exp(log_resp).squeeze()
        self.labels_tr_ =labels_tr.squeeze()  
        self.labels_X_ = labels_X
        self.labels_y_ = labels_y.squeeze()
        self.low_bound_curves_ = best_curves
        self.is_fitted_ = True

    def _e_step(self, X, y):
        """Expectation step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples, n_targets)

        Returns
        -------
        log_prob_norm : array, shape (n_samples, n_targets) 
            Mean of the logarithms of the probabilities of each input-output 
            pair in X & y.
        log_responsibility : array, shape (n_samples, n_targets, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each input-output pair in X & y.
        """
        n, _ = X.shape
        reg_weights = self.reg_weights_
        reg_precisions = self.reg_precisions_
        if self.n_targets_ == 1:
            reg_weights = reg_weights[np.newaxis, :, :]
            reg_precisions = reg_precisions[np.newaxis, :]

        # Compute all the log-factors for the responsibility expression
        log_weights = np.log(self.weights_)
        log_prob_X = _estimate_log_prob_X(X, self.means_, self.precisions_cholesky_)
        log_prob_y = np.empty((n, self.n_targets_, self.n_components))
        weighted_log_prob = np.empty_like(log_prob_y)
        for k in range(self.n_components):
            log_prob_y[:, :, k] = _estimate_log_prob_y_k(X, y, 
                                                         reg_weights[:, :, k], 
                                                         reg_precisions[:, k])
            weighted_log_prob[:, :, k] = (log_weights[k] + 
                                          log_prob_X[:, k][:, np.newaxis] + 
                                          log_prob_y[:, :, k])
        
        # Compute the log-denominator of the responsibility expression
        log_prob_norm = logsumexp(weighted_log_prob, axis=2)

        log_resp = np.empty_like(weighted_log_prob)
        for k in range(self.n_components):
            with np.errstate(under='ignore'):
                # Ignore underflow
                log_resp[:, :, k] = weighted_log_prob[:, :, k] - log_prob_norm
        
        # Compute labels from all viewpoints
        labels_X = log_prob_X.argmax(axis=1)
        labels_y = log_prob_y.argmax(axis=2)
        labels = log_resp.argmax(axis=2)

        return log_prob_norm, log_resp, labels, labels_X, labels_y

    def _m_step(self, X, y, log_resp):
        """Maximization step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples, n_targets)
        log_resp : array-like, shape (n_samples, n_targets, n_components)

        log_resp : array-like, shape (n_samples, n_targets, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n, d = X.shape
        resp = np.exp(log_resp)
        resp_task = np.exp(log_resp.sum(axis=1))
        eps = 10 * np.finfo(resp.dtype).eps

        # Regularization term 
        # Equivalent to Gaussian prior on the regression weights
        reg_term = self.eta / (self.reg_precisions_ + eps)
        # Make sure we can iterate when n_targets = 1
        if self.n_targets_ == 1:
            reg_term = reg_term[np.newaxis, :]
        
        # Update the mixture weights
        weights = (resp_task.sum(axis=0) + eps)/n

        # Update input space mixture parameters
        (_, 
        means, 
        covariances) = _estimate_gaussian_parameters(X, resp_task, self.reg_covar)      
        precisions_cholesky = _compute_precision_cholesky(self.covariances_)

        # Update the output space regression parameters
        reg_weights = np.empty((self.n_targets_, 
                                self.n_input_dims_+1, 
                                self.n_components))
        reg_precisions = np.zeros((self.n_targets_, self.n_components))
        for k in range(self.n_components):
            (reg_weights[:, :, k], 
             reg_precisions[:, k]) = _estimate_regression_params_k(X, y,
                                                                   resp[:, :, k],
                                                                   reg_term[:, k],
                                                                   weights[k])
        
        self.weights_ = weights
        self.means_
        self.covariances_
        self.precisions_cholesky_
        self.reg_weights_ = reg_weights.squeeze()
        self.reg_precisions_ = reg_precisions.squeeze()

    def predict(self, X):
        """Estimate the values of the outputs for a new set of inputs.

        Compute the expected value of y given the trained model and a set
        X of new observations.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        targets : array, shape (n_samples, 1)
        """
        if not self.is_fitted_: 
            print("Model isn't fitted.")
            return

        eps = 10 * np.finfo(self.resp_tr_.dtype).eps
        n, d = X.shape
        if d != self.n_input_dims_:
            print('Incorrect dimensions for input data.')
            sys.exit(0)
        
        reg_weights = self.reg_weights_
        reg_precisions = self.reg_precisions_
        if self.n_targets_ == 1:
            reg_weights = reg_weights[np.newaxis, :, :]
            reg_precisions = reg_precisions[np.newaxis, :]

        X_ext = np.concatenate((np.ones((n, 1)), X), axis=1)
        targets = np.zeros((n, self.n_targets_))

        # Compute all the log-factors for the responsibility expression
        log_weights = np.log(self.weights_)
        log_prob_X = _estimate_log_prob_X(X, self.means_, self.precisions_cholesky_)
        
        # Compute log-responsibilities
        weighted_log_prob = log_weights + log_prob_X
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under='ignore'):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        resp_tst = np.exp(log_resp)
        labels_tst = log_resp.argmax(axis=1)
    
        # Compute the expected value of the predictive posterior.
        for k in range(self.n_components):
            dot_prod = np.dot(X_ext, reg_weights[:, :, k].T)            
            targets += np.multiply((resp_tst[:, k] + eps)[:, np.newaxis], dot_prod)

        self.resp_tst_ = resp_tst
        self.labels_tst_ = labels_tst

        return targets

    def predict_score(self, X, y, metric='R2'):
        """Estimate and score the values of the outputs for a new set of inputs

        Compute the expected value of y given the trained model and a set X of 
        new observations. Calculate the specified error metric for the predicted 
        values against the real values.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y_real : array-like, shape (n_samples, n_targets)

        Returns
        -------
        y : array, shape (n_samples, n_targets)
        score : int
        """
        y_est = self.predict(X)
        
        if metric == 'MSE':
            score = mean_squared_error(y, y_est)
        elif metric == 'R2': 
            score = r2_score(y, y_est)
        elif metric == 'MAE':
            score = mean_absolute_error(y, y_est)    
        elif metric == 'MAPE':
            score = mean_absolute_percentage_error(y, y_est)
        elif metric == 'all': 
            score = [r2_score(y, y_est), 
                     mean_squared_error(y, y_est), 
                     mean_absolute_error(y, y_est), 
                     mean_absolute_percentage_error(y, y_est)]
        else:
            print("""Wrong score metric specified. Must be either 
                  'MSE', 'MAE', 'R2', 'MAPE' or 'all'.""")
            return

        return y_est, score

    def score(self, X, y, metric='R2'):
        """Score the values of the outputs for a new set of inputs

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y_real : array-like, shape (n_samples, n_targets)


        Returns
        -------
        score : int
        """
        _, score = self.predict_score(X, y, metric)

    def _compute_lower_bound(self, log_prob_norm):
        """Compute the model's complete data log likelihood.

        Parameters
        ----------
        log_prob_norm : array, shape (n_samples, n_targets)

        Returns
        -------
        lower_bound : float
        """
        return log_prob_norm.sum()

    def _get_parameters(self):
        return (self.weights_, self.means_, self.covariances_,
                self.precisions_cholesky_, self.reg_weights_, 
                self.reg_precisions_)

    def _set_parameters(self, params):
        (self.weights_, self.means_, self.covariances_,
         self.precisions_cholesky_, self.reg_weights_, 
         self.reg_precisions_) = params
