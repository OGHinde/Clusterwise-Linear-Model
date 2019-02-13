"""CLUSTERWISE LINEAR REGRESSION MODEL

    Author: Óscar García Hinde <oghinde@tsc.uc3m.es>
    Python Version: 3.6

TODO:
    - Multioutput.
    - Implement parallelization with MPI.
    - Implement other input covariances.
    - Update Attributes docstring.

ISSUES:
    - Weirdness in the lower bound results indicates that something's not
      quite right.

NEXT STEPS:
    - Pruning of weak clusters.
"""

import numpy as np
from numpy.random import RandomState
from scipy import linalg
from scipy.stats import norm
from scipy.misc import logsumexp
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from cwlm.kmeans_regressor import KMeansRegressor
from cwlm.gmm_regressor import GMMRegressor

from mpi4py import MPI

import warnings
from sklearn.exceptions import ConvergenceWarning

import matplotlib.pyplot  as plt


###############################################################################
# USED IN THE E STEP 

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

def _estimate_log_prob_y(X, y, reg_weights, reg_precisions):
    """Estimate the log Gaussian probability of the output space,
    i.e. the log probability factor for each sample in y.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    means : array-like, shape (n_components, n_features)

    precisions_chol : array-like
        Cholesky decompositions of the precision matrices.

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n, d = X.shape
    # Extend X with a column of ones 
    X_ext = np.concatenate((np.ones((n, 1)), X), axis=1)

    means = np.dot(X_ext, reg_weights)
    std_devs = np.sqrt(reg_precisions ** -1)
    
    return norm.logpdf(y, loc=means, scale=std_devs)


###############################################################################
# USED IN THE M STEP 

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
        precisions_chol[k] = linalg.solve_triangular(cov_chol, np.eye(n_features), lower=True).T
    
    return precisions_chol

def _estimate_regression_weights(X, y, resp_k, reg_term_k):
    """Estimate the regression weights for the output space for component k.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    y : array-like, shape (n_samples, )

    resp_k : array-like, shape (n_samples, )

    reg_term_k : float

    Returns
    -------
    reg_weights : array, shape (n_features, )
        The regression weights for component k.
    """
    _, d = X.shape
    eps = 10 * np.finfo(resp_k.dtype).eps
    reg_weights_k = np.zeros((d+1,))
    
    solver = Ridge(alpha=reg_term_k)
    solver.fit(X, y, sample_weight=resp_k + eps)
    reg_weights_k[0] = solver.intercept_
    reg_weights_k[1:] = solver.coef_

    return reg_weights_k


###############################################################################
# MAIN CLASS

class ClusterwiseLinModel():
    """Clusterwise Linear Regressor Mixture.

    Representation of a coupled Gaussian and linear regression
    mixture probability distribution.
    
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
        
    reg_weights_init : array-like, shape (n_components, n_features + 1), optional
        The user-provided initial means, defaults to None,
        If it None, means are initialized using the `init_params` method.
    
    reg_precisions_init : array-like, shape (n_components, ), optional.
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
                 covariances_init=None, reg_weights_init=None, reg_precisions_init=None, 
                 random_seed=None, plot=False):
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

    def _initialise(self, X, y, RandomState):
        """Initialization of the Clusterwise Linear Model parameters.

        In this version we'll implement all options: 'kmeans', 'gmm' and 'random'.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        resp : array-like, shape (n_samples, n_components)
        """
        
        n, d = X.shape

        if self.init_params == 'kmeans':
            initializer = KMeansRegressor(n_components=self.n_components, alpha=self.eta)
            initializer.fit(X, y)
            resp = np.zeros((n, self.n_components))
            resp[np.arange(n), initializer.labels_] = 1
            reg_weights = initializer.reg_weights_
            reg_precisions = initializer.reg_precisions_

        elif self.init_params == 'gmm':
            initializer = GMMRegressor(n_components=self.n_components, alpha=self.eta, covariance_type='full')
            initializer.fit(X, y)
            resp = initializer.resp_
            reg_weights = initializer.reg_weights_
            reg_precisions = initializer.reg_precisions_
        
        elif self.init_params == 'random':
            # This tends to work like crap
            resp = RandomState.rand(n, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
            reg_weights = RandomState.randn(d + 1, self.n_components)
            reg_precisions = np.zeros((self.n_components, )) + 1 / np.var(y)

        else:
            raise ValueError("Unimplemented initialization method '%s'"
                             % self.init_params)

        weights, means, covariances = _estimate_gaussian_parameters(X, resp, self.reg_covar)
        weights /= n

        self.weights_ = weights if self.weights_init is None else self.weights_init
        self.means_ = means if self.means_init is None else self.means_init

        if self.covariances_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(covariances)
        else:
            self.covariances_ = self.covariances_init
            self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_)

        self.reg_weights_ = reg_weights if self.reg_weights_init is None else self.reg_weights_init
        self.reg_precisions_ = reg_precisions if self.reg_precisions_init is None else self.reg_precisions_init

    def fit(self, X, y):
        """Fit the clustered linear regressor model for a training 
        data set using the EM algorithm.

        It does n_init instances of the algorithm and keeps the one with the
        highest complete log-likelyhood.
        
        Each initialization of the algorithm runs until convergence or max_iter
        times.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        y : array, shape (n_samples, 1)

        Returns
        -------
        self
        """
        n, d = X.shape
        max_lower_bound = -np.infty
        self.converged_ = False
        # Check shape of y and fix if needed
        if y.shape != (n, 1):
            y.shape = (n, 1)
        rng = RandomState(self.random_seed)

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
                    # compute mean of n_iter previous values
                    smooth_bound = np.mean(bound_curve)
                    smooth_bound_curve.append(smooth_bound)
                else:
                    # compute mean of smooth_window previous values
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
                fig = plt.figure()
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
        _, log_resp, self.labels_, self.labels_X_, self.labels_y_ = self._e_step(X, y)
        self.resp_ = np.exp(log_resp)

        if not self.converged_:
            warnings.warn('Initialization %d did not converge. '
                          'Try different init parameters, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.'
                          % (init + 1), ConvergenceWarning)
            #print('Model did not converge after %d initializations.'%(self.n_init))

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound
        self.low_bound_curves_ = best_curves

    def _e_step(self, X, y):
        """E step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples, 1)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each input-output 
            pair in X & y.

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each input-output pair in X & y.
        """
        # Compute all the log-factors for the responsibility expression
        log_weights = np.log(self.weights_)
        log_prob_X = _estimate_log_prob_X(X, self.means_, self.precisions_cholesky_)
        log_prob_y = _estimate_log_prob_y(X, y, self.reg_weights_, self.reg_precisions_)

        # Compute the log-numerator of the responsibility expression
        weighted_log_prob = log_weights + log_prob_X + log_prob_y
        
        # Compute the log-denominator of the responsibility expression
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)

        with np.errstate(under='ignore'):
            # Ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]

        # Compute labels from all viewpoints
        labels = log_resp.argmax(axis=1)
        labels_X = log_prob_X.argmax(axis=1)
        labels_y = log_prob_y.argmax(axis=1)

        return log_prob_norm, log_resp, labels, labels_X, labels_y

    def _m_step(self, X, y, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples, 1)

        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n, d = X.shape
        _, K = log_resp.shape
        resp = np.exp(log_resp)
        eps = 10 * np.finfo(resp.dtype).eps
        X_ext = np.concatenate((np.ones((n, 1)), X), axis=1)

        # Regularization term (equivalent to Gaussian prior on the regression weights)
        reg_term = self.eta / (self.reg_precisions_ + eps)
        
        # Update the mixture weights
        weights = resp.sum(axis=0) + eps
        self.weights_ = weights/n

        # Update input space mixture parameters
        (_, 
        self.means_, 
        self.covariances_) = _estimate_gaussian_parameters(X, resp, self.reg_covar)
        self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_)

        # Update the output space regression weights
        reg_weights = np.zeros_like(self.reg_weights_)
        for k in range(K):
            reg_weights[:, k] = _estimate_regression_weights(X, 
                y, resp_k=resp[:, k], reg_term_k=reg_term[k])
        self.reg_weights_ = reg_weights
        
        # Update the output space precision terms
        means = np.dot(X_ext, self.reg_weights_)
        err = (np.tile(y, (1, K)) - means) ** 2
        reg_precisions = n * self.weights_ / np.sum(resp * err)
        self.reg_precisions_ = reg_precisions

    def predict(self, X, labels=False):
        """Estimate the values of the outputs for a new set of inputs.

        Compute the expected value of y given the trained model and a set
        X of new observations.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        y_ : array, shape (n_samples, 1)
        """
        n, d = X.shape
        X_ext = np.concatenate((np.ones((n, 1)), X), axis=1)
        y_ = np.zeros((n, 1))

        # Compute all the log-factors for the responsibility expression
        log_weights = np.log(self.weights_)
        log_prob_X = _estimate_log_prob_X(X, self.means_, self.precisions_cholesky_)
        
        # Compute log-responsibilities
        weighted_log_prob = log_weights + log_prob_X
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under='ignore'):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        resp = np.exp(log_resp)
        labels_ = log_resp.argmax(axis=1)
        
        # Compute the expected value of the predictive posterior.
        eps = 10 * np.finfo(resp.dtype).eps
        dot_prod = np.dot(X_ext, self.reg_weights_)
        y_ = np.sum((resp + eps) * dot_prod, axis=1)

        if labels:
            return labels_, y_
        else:
            return y_

    def predict_score(self, X, y, metric='R2', labels=False):
        """Estimate and score the values of the outputs for a new set of inputs

        Compute the expected value of y given the trained model and a set X of 
        new observations. Calculate the mean square error between the predicted 
        values against the real values.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y_real : array-like, shape (n_samples, 1)

        Returns
        -------
        y : array, shape (n_samples, 1)
        score : int
        """
        if labels:
            labels_, y_ = self.predict(X, labels=labels)
        else:
            y_ = self.predict(X, labels=labels)
        
        if metric == 'MSE':
            score = mean_squared_error(y, y_)
        elif metric == 'R2': 
            score = r2_score(y, y_)
        elif metric == 'MAE':
            score = mean_absolute_error(y, y_)    
        elif metric == 'all': 
            score = [r2_score(y, y_), mean_squared_error(y, y_), mean_absolute_error(y, y_)]
        else:
            print("Wrong score metric specified. Must be either 'MSE', 'MAE', 'R2' or 'all'.")
            return

        if labels:
            return labels_, y_, score
        else:
            return y_, score

    def _compute_lower_bound(self, log_prob_norm):
        """We'll do it like this for now.
        It looks right but it's clearly telling us that something's wrong.
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
