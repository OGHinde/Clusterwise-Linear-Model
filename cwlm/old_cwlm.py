"""Linear Regressor Mixture Model.

THIS DOESN'T WORK. USE clusterwise_linear_model.py INSTEAD

"""

# Author: Oscar Garcia Hinde <oghinde@tsc.uc3m.es>

import numpy as np

import sys
from scipy import linalg
from scipy.stats import norm
from scipy.misc import logsumexp
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from cwlm.kmeans_regressor import KMeansRegressor
from cwlm.gmm_regressor import GMMRegressor

import warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.mixture.base import BaseMixture, _check_shape, _check_X
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import row_norms


###############################################################################
# Shape checkers used by the LinRegMixture class

def _check_weights(weights, n_components):
    """Check the user provided 'weights'.

    Parameters
    ----------
    weights : array-like, shape (n_components,)
        The proportions of components of each mixture.

    n_components : int
        Number of components.

    Returns
    -------
    weights : array, shape (n_components,)
    """
    weights = check_array(weights, dtype=[np.float64, np.float32],
                          ensure_2d=False)
    _check_shape(weights, (n_components,), 'weights')

    # check range
    if (any(np.less(weights, 0.)) or
            any(np.greater(weights, 1.))):
        raise ValueError("The parameter 'weights' should be in the range "
                         "[0, 1], but got max value %.5f, min value %.5f"
                         % (np.min(weights), np.max(weights)))

    # check normalization
    if not np.allclose(np.abs(1. - np.sum(weights)), 0.):
        raise ValueError("The parameter 'weights' should be normalized, "
                         "but got sum(weights) = %.5f" % np.sum(weights))
    return weights

def _check_means(means, n_components, n_features):
    """Validate the provided 'means'.

    Parameters
    ----------
    means : array-like, shape (n_components, n_features)
        The centers of the current components.

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    means : array, (n_components, n_features)
    """
    means = check_array(means, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(means, (n_components, n_features), 'means')
    return means

def _check_precision_positivity(precision, covariance_type):
    """Check that a precision vector is positive-definite."""
    if np.any(np.less_equal(precision, 0.0)):
        raise ValueError("'%s precision' should be "
                         "positive" % covariance_type)


def _check_precision_matrix(precision, covariance_type):
    """Check that a precision matrix is symmetric and positive-definite."""
    if not (np.allclose(precision, precision.T) and
            np.all(linalg.eigvalsh(precision) > 0.)):
        raise ValueError("'%s precision' should be symmetric, "
                         "positive-definite" % covariance_type)


def _check_precisions_full(precisions, covariance_type):
    """Check that the precision matrices are symmetric and positive-definite."""
    for k, prec in enumerate(precisions):
        prec = _check_precision_matrix(prec, covariance_type)


def _check_precisions(precisions, covariance_type, n_components, n_features):
    """Validate the user provided precisions.

    Parameters
    ----------
    precisions : array-like,
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : string

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    precisions : array
    """
    precisions = check_array(precisions, dtype=[np.float64, np.float32],
                             ensure_2d=False,
                             allow_nd=covariance_type == 'full')

    precisions_shape = {'full': (n_components, n_features, n_features),
                        'tied': (n_features, n_features),
                        'diag': (n_components, n_features),
                        'spherical': (n_components,)}
    _check_shape(precisions, precisions_shape[covariance_type],
                 '%s precision' % covariance_type)

    _check_precisions = {'full': _check_precisions_full,
                         'tied': _check_precision_matrix,
                         'diag': _check_precision_positivity,
                         'spherical': _check_precision_positivity}
    _check_precisions[covariance_type](precisions, covariance_type)
    return precisions

def _check_reg_weights(reg_weights, n_components, n_features):
    """Validate the user provided 'reg_weights'.

    Parameters
    ----------
    means : array-like, shape (n_components, n_features + 1)
        The regression weights of the current components.

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    means : array, (n_components, n_features)
    """
    reg_weights = check_array(reg_weights,
                              dtype=[np.float64, np.float32], 
                              ensure_2d=False)
    _check_shape(reg_weights, (n_components, n_features + 1), 'regression weights')
    return reg_weights


def _check_reg_precisions(reg_precisions, n_components):
    """Check that all regression precisions are positive."""
    if np.any(np.less_equal(reg_precisions, 0.0)):
        raise ValueError("Precision should be "
                         "positive")


###############################################################################
# Gaussian mixture parameters estimators (used by the M-Step)

def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    """Estimate the full covariance matrices.

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


def _estimate_gaussian_covariances_tied(resp, X, nk, means, reg_covar):
    """Estimate the tied covariance matrix.

    Parameters
    ----------
    resp : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    means : array-like, shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariance : array, shape (n_features, n_features)
        The tied covariance matrix of the components.
    """
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot(nk * means.T, means)
    covariance = avg_X2 - avg_means2
    covariance /= nk.sum()
    covariance.flat[::len(covariance) + 1] += reg_covar
    return covariance


def _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar):
    """Estimate the diagonal covariance vectors.

    Parameters
    ----------
    responsibilities : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    means : array-like, shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features)
        The covariance vector of the current components.
    """
    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means ** 2
    avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
    return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar


def _estimate_gaussian_covariances_spherical(resp, X, nk, means, reg_covar):
    """Estimate the spherical variance values.

    Parameters
    ----------
    responsibilities : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    means : array-like, shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    variances : array, shape (n_components,)
        The variance values of each components.
    """
    return _estimate_gaussian_covariances_diag(resp, X, nk,
                                               means, reg_covar).mean(1)


def _estimate_gaussian_parameters(X, resp, reg_covar, covariance_type):
    """Estimate the Gaussian distribution parameters.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data array.

    resp : array-like, shape (n_samples, n_features)
        The responsibilities for each data sample in X.

    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    nk : array-like, shape (n_components,)
        The numbers of data samples in the current components.

    means : array-like, shape (n_components, n_features)
        The centers of the current components.

    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = {"full": _estimate_gaussian_covariances_full,
                   "tied": _estimate_gaussian_covariances_tied,
                   "diag": _estimate_gaussian_covariances_diag,
                   "spherical": _estimate_gaussian_covariances_spherical
                   }[covariance_type](resp, X, nk, means, reg_covar)
    return nk, means, covariances


def _compute_precision_cholesky(covariances, covariance_type):
    """Compute the Cholesky decomposition of the precisions.

    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

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

    if covariance_type in 'full':
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
    elif covariance_type == 'tied':
        _, n_features = covariances.shape
        try:
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = linalg.solve_triangular(cov_chol, np.eye(n_features),
                                                  lower=True).T
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1. / np.sqrt(covariances)
    
    return precisions_chol


###############################################################################
# Gaussian mixture probability estimators (Used by the E-Step)

def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.

    Parameters
    ----------
    matrix_chol : array-like,
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    n_features : int
        Number of features.

    Returns
    -------
    log_det_precision_chol : array-like, shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    if covariance_type == 'full':
        n_components, _, _ = matrix_chol.shape
        log_det_chol = (np.sum(np.log(
            matrix_chol.reshape(
                n_components, -1)[:, ::n_features + 1]), 1))
    elif covariance_type == 'tied':
        log_det_chol = (np.sum(np.log(np.diag(matrix_chol))))
    elif covariance_type == 'diag':
        log_det_chol = (np.sum(np.log(matrix_chol), axis=1))
    else:
        log_det_chol = n_features * (np.log(matrix_chol))

    return log_det_chol

def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type):
    """Estimate the log Gaussian probability.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    means : array-like, shape (n_components, n_features)

    precisions_chol : array-like,
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    # det(precision_chol) is half of det(precision)
    log_det = _compute_log_det_cholesky(
        precisions_chol, covariance_type, n_features)

    if covariance_type == 'full':
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)
    elif covariance_type == 'tied':
        log_prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)
    elif covariance_type == 'diag':
        precisions = precisions_chol ** 2
        log_prob = (np.sum((means ** 2 * precisions), 1) -
                    2. * np.dot(X, (means * precisions).T) +
                    np.dot(X ** 2, precisions.T))
    elif covariance_type == 'spherical':
        precisions = precisions_chol ** 2
        log_prob = (np.sum(means ** 2, 1) * precisions -
                    2 * np.dot(X, means.T * precisions) +
                    np.outer(row_norms(X, squared=True), precisions))
    
    return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det


###############################################################################
# Main linear regressor mixture model class.

class ClusterwiseLinModel(BaseMixture):
    """Clusterwise Linear Regressor Mixture.

    Representation of a coupled Gaussian mixture and linear regressor mixture 
    model probability distribution.
    This class estimates the parameters of said mixture distribution using
    the EM algorithm.

    Parameters
    ----------
    n_components : int,  defaults to 1.
        The number of mixture components.

    tol : float, defaults to 1e-10.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, defaults to 100.
        The number of EM iterations to perform.

    n_init : int, defaults to 1.
        The number of EM initializations to perform. The best results are kept.

    covariance_type : str, defaults to 'diag'
        The type of covariance matrix for the Gaussian mixture model fitted to the
        input variables.

    init_params : str, defaults to 'gmm'.
        The method used to initialize the Gaussian mixture weights, means and
        precisions; and the linear regressor weights and precisions.
            'gmm' : a Gaussian mixture + Ridge Regression model is used.
            'kmeans' : a K-Means + Ridge Regression model is used.
            'random' : weights are initialized randomly.

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

    random_state : RandomState or an int seed, defaults to None.
        A random number generator instance.

    warm_start : bool, default to False.
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several time on similar problems.

    verbose : int, default to 0.
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default to 10.
        Number of iteration done before the next print.

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

    def __init__(self, n_components=1, tol=1e-10, eta=1e-1, reg_covar=1e-6,
                 max_iter=100, n_init=20, covariance_type='diag',
                 init_params='gmm', weights_init=None,
                 means_init=None, precisions_init=None,
                 reg_weights_init=None, reg_precisions_init=None, random_state=None,
                 warm_start=False, verbose=0, verbose_interval=10):
    
        super(ClusterwiseLinModel, self).__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)

        self.weights_init = weights_init
        self.covariance_type = covariance_type
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.reg_weights_init = reg_weights_init
        self.reg_precisions_init = reg_precisions_init
        self.eta = eta
        self.init_params = init_params

    def _check_y(self, y, n_samples):
        if y.shape != (n_samples, 1):
            y.shape = (n_samples, 1)

        return y

    def _check_parameters(self, X):
        """Check the Gaussian mixture parameters are well defined."""
        _, n_features = X.shape
        
        if self.covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError("Invalid value for 'covariance_type': %s "
                             "'covariance_type' should be in "
                             "['spherical', 'tied', 'diag', 'full']"
                             % self.covariance_type)

        if self.weights_init is not None:
            self.weights_init = _check_weights(self.weights_init,
                                               self.n_components)

        if self.means_init is not None:
            self.means_init = _check_means(self.means_init,
                                           self.n_components, 
                                           n_features)

        if self.precisions_init is not None:
            self.precisions_init = _check_precisions(self.precisions_init,
                                                     self.covariance_type,
                                                     self.n_components,
                                                     n_features)

        if self.reg_weights_init is not None:
            self.reg_weights_init = _check_reg_weights(self.reg_weights_init,
                                                       self.n_components, 
                                                       n_features)

        if self.reg_precisions_init is not None:
            self.reg_precisions_init = _check_reg_precisions(self.reg_precisions_init,
                                                             self.n_components)

    def _initialize(self, X, X_ext, y, random_state):
        """General parameter initialisation function. It calls the initialisation
        functions for the Gaussian mixture parameters and the regression parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        X_ext : array-like, shape (n_samples, n_features + 1)

        y : array-like, shape (n_samples, 1)

        random_state: RandomState
            A random number generator instance.
        """
        n_samples, n_features = X_ext.shape
        if self.init_params == 'kmeans':
            resp = np.zeros((n_samples, self.n_components))
            initializer = KMeansRegressor(n_components=self.n_components, alpha=self.eta)
            initializer.fit(X, y)
            resp[np.arange(n_samples), initializer.labels_] = 1
            reg_weights = initializer.reg_weights_
            reg_precisions = initializer.reg_precisions_
        elif self.init_params == 'gmm':
            initializer = GMMRegressor(n_components=self.n_components, alpha=self.eta)
            initializer.fit(X, y)
            resp = initializer.resp_
            reg_weights = initializer.reg_weights_
            reg_precisions = initializer.reg_precisions_
        elif self.init_params == 'random':
            resp = random_state.rand(n_samples, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
            reg_weights = random_state.rand(n_features, self.n_components)
            reg_precisions = np.zeros((self.n_components, )) + 1 / np.var(y)
        else:
            raise ValueError("Unimplemented initialization method '%s'"
                             % self.init_params)

        self._initialize_gauss(X, resp)
        
        weights = np.zeros(self.n_components) + 1/float(self.n_components)

        self.reg_weights_ = (reg_weights if self.reg_weights_init is None
                            else self.reg_weights_init)
        self.reg_precisions_ = (reg_precisions if self.reg_precisions_init is None
                            else self.precisions_init)
        self.weights_ = (weights if self.weights_init is None
                            else self.weights_init)
        
    def _initialize_gauss(self, X, resp):
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        resp : array-like, shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        weights, means, covariances = _estimate_gaussian_parameters(X, 
            resp, self.reg_covar, self.covariance_type)
        weights /= n_samples

        self.weights_ = (weights if self.weights_init is None
                         else self.weights_init)
        self.means_ = (means if self.means_init is None else self.means_init)

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type)
        elif self.covariance_type == 'full':
            self.precisions_cholesky_ = np.array(
                [linalg.cholesky(prec_init, lower=True)
                 for prec_init in self.precisions_init])
        elif self.covariance_type == 'tied':
            self.precisions_cholesky_ = linalg.cholesky(self.precisions_init,
                                                        lower=True)
        else:
            self.precisions_cholesky_ = self.precisions_init

    def fit(self, X, y=None):
        """Fit the clustered linear regressor model for a training 
        data set using the EM algorithm.
        It does n_init instances of the algorithm and keeps the one with the
        highest complete log-likelyhood.
        Each initialization of the algorithm runs until convergence or max_iter
        times.
        If we enable warm_start, we will have a unique initialisation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        y : array, shape (n_samples, 1)

        Returns
        -------
        self
        """
        # Check that all the data is well conditioned
        print('ESTÁ A 0')
        X = _check_X(X, self.n_components)
        n_samples, n_features = X.shape
        y = self._check_y(y, n_samples)
        self._check_initial_parameters(X)
        random_state = check_random_state(self.random_state)
        
        # Extended X with a column of ones for the bias terms:
        X_ext = np.concatenate((np.ones((n_samples, 1)), X), axis=1)
        
        # If we enable warm_start, we will have a unique initialisation
        do_init = not(self.warm_start and hasattr(self, 'converged_'))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.infty
        self.converged_ = False
        
        init = 0
        while init < n_init:
            self._print_verbose_msg_init_beg(init)
            if do_init:
                try:
                    self._initialize(X, X_ext, y, random_state)
                except (ValueError, linalg.LinAlgError) as error:
                    print("Bad conditions at init. Error type:")
                    print(error)
                    print("Please try a different initialisation strategy.")
                    sys.exit
                self.lower_bound_ = -np.infty
            init += 1
            
            for n_iter in range(self.max_iter):                
                prev_lower_bound = self.lower_bound_
                
                # EM steps
                (log_sum_gamma, 
                log_resp,
                log_mix_probabilities,
                log_reg_probabilities) = self._log_e_step_supervised(X, X_ext, y)            
                self.log_resp_ = log_resp
                self.log_mix_probabilities_ = log_mix_probabilities
                self.log_reg_probabilities_ = log_reg_probabilities
                try:
                    self._m_step_supervised(X, X_ext, y, np.exp(log_resp))
                except (ValueError, linalg.LinAlgError) as error:
                    print("Bad conditions at execution. Error type:")
                    print(error)
                    print("Resetting initialisation {}.".format(init))
                    init -= 1
                    break
                
                # Compute log likelyhood
                self.lower_bound_ = self._compute_log_lower_bound(log_sum_gamma)
            
                # Check convergence
                change = abs(self.lower_bound_ - prev_lower_bound)
                self._print_verbose_msg_iter_end(n_iter, change)
    
                if change < self.tol:
                    self.converged_ = True
                    break
    
            self._print_verbose_msg_init_end(self.lower_bound_)
            
            # If there is an improvement over the last best initialization, save data
            if self.lower_bound_ > max_lower_bound:
                max_lower_bound = self.lower_bound_
                best_params = self._get_parameters()
                best_n_iter = n_iter
                self.labels_ = self.log_resp_.argmax(axis=1)
                self.X_labels_ = self.log_mix_probabilities_.argmax(axis=1)
                self.y_labels_ = self.log_reg_probabilities_.argmax(axis=1)
            
        if not self.converged_:
            warnings.warn('Initialization %d did not converge. '
                          'Try different init parameters, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.'
                          % (init + 1), ConvergenceWarning)
    
        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
    
        return self

    def _m_step(self, X, log_resp):
        """Unsipervised M step.

        Unused in this model 
        """
        pass
    
    def _log_e_step_supervised(self, X, X_ext, y):
        """Compute the log-responsibilities, i.e. the posterior probabilities
        of the latent indicator variables Z.
        
        Parameters
        ----------
        X_ext : array-like, shape (n_samples, n_features + 1)
        
        y : array-like, shape (n_samples, )
        
        Returns
        -------
        log_resp : array-like, shape (n_samples, n_components)
        log_sum_gamma : array-like, shape (n_samples, 1)
        """
        n_samples, _ = X_ext.shape
        eps = np.finfo(float).eps
        overflow = np.log(sys.float_info.max) + 100
        log_weights = np.log(self.weights_)
        
        # Gaussian mixture probabilities
        log_mix_probabilities = _estimate_log_gaussian_prob(X,
                                                            self.means_,
                                                            self.precisions_cholesky_,
                                                            self.covariance_type)
        # Linear regressor mixture probabilities
        means = np.dot(X_ext, self.reg_weights_)
        std_dev = np.sqrt(self.reg_precisions_ ** -1)
        log_reg_probabilities = norm.logpdf(y, loc=means, scale=std_dev)
        # Check for numerical instabilities
        log_reg_probabilities[np.isinf(log_reg_probabilities)] = np.log(eps)
                              
        # Compute log-responsibilities        
        log_gamma = log_weights + log_mix_probabilities + log_reg_probabilities

        # Threshold low resp samples to improve performance
        max_vals = np.max(log_gamma, axis=0)
        log_gamma[log_gamma < max_vals - 4] = -overflow

        log_sum_gamma = logsumexp(log_gamma, axis=1)[:, np.newaxis]
        with np.errstate(under='ignore'):
            # Ignore underflow
            log_resp = log_gamma - log_sum_gamma
        
        #return log_sum_gamma, log_resp
        return log_sum_gamma, log_resp, log_mix_probabilities, log_reg_probabilities

    def _m_step_supervised(self, X, X_ext, y, resp):
        """Maximize the complete log-likelyhood by updating the mixture weights, 
        the input distribution parameters and the regression parameters using the 
        responsibilities calculated in the E-Step. 
        ----------
        X : array-like, shape (n_samples, n_features)

        X_ext : array-like, shape (n_samples, n_features + 1)
                The input data extended with a column of ones.

        y : array-like, shape (n_samples, )
            
        resp : array-like, shape (n_samples, n_components)
               The posterior probabilities (or responsibilities).
        """
        n_samples, n_features = X_ext.shape
        _, n_components = resp.shape
        eps = 7./3 - 4./3 -1
        
        # Ensure that responsabilities are normalized
        for k in range(n_components):
            resp[:, k] = resp[:, k] / resp.sum(axis=1)

        # Regularization term (equivalent to Gaussian prior on the regression weights)
        reg_term = float(self.eta) / (self.reg_precisions_ + eps)
        
        # Max with respect to the mixture weights
        weights = np.mean(resp, axis=0)
        
        self.weights_, self.means_, self.covariances_ = (
            _estimate_gaussian_parameters(X, resp, self.reg_covar,
                                          self.covariance_type))
        self.weights_ /= n_samples

        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)

        # Max with respect to the regression weights
        reg_weights = np.zeros_like(self.reg_weights_)
        for k in range(n_components):
            # Don't work more than you need to, bruh.
            solver = Ridge(alpha=reg_term[k])
            solver.fit(X, y, sample_weight=resp[:, k] + eps)
            reg_weights[0, k] = solver.intercept_
            reg_weights[1:, k] = solver.coef_
        
        # Max with respect to the precision term
        means = np.dot(X_ext, self.reg_weights_)
        err = (np.tile(y, (1, n_components)) - means) ** 2
        reg_precisions = n_samples * weights / np.sum(resp * err)
        
        self.weights_ = weights
        self.reg_weights_ = reg_weights
        self.reg_precisions_ = reg_precisions
    
    def _compute_lower_bound(self, gamma):
        complete_log_likelihood = float(np.sum(np.log(np.sum(gamma, axis=1))))
        
        return complete_log_likelihood
        
    def _compute_log_lower_bound(self, log_sum_gamma):
        complete_log_likelihood = float(np.mean(log_sum_gamma))
        
        return complete_log_likelihood

    def _estimate_log_prob(self, X):
        """Estimate the log-probabilities log P(X | Z).

        Compute the log-probabilities per each component for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_prob : array, shape (n_samples, n_component)
        """
        pass

    def _estimate_log_weights(self):
        return np.log(self.weights_.T)

    def _check_is_fitted(self):
        """Check if the model is fitted"""
        check_is_fitted(self, ['weights_', 'means_', 'precisions_cholesky_'])

    def _get_parameters(self):
        """Return the current parameter values of the model"""
        return (self.weights_, 
                self.means_, 
                self.covariances_, 
                self.reg_weights_, 
                self.reg_precisions_, 
                self.log_resp_,
                self.log_mix_probabilities_,
                self.log_reg_probabilities_)
        #return (self.weights_, self.means_, self.covariances_, self.reg_weights_, self.reg_precisions_)

    def _set_parameters(self, params):
        """Define the current parameter values of the model"""
        (self.weights_, 
         self.means_, 
         self.covariances_, 
         self.reg_weights_, 
         self.reg_precisions_, 
         self.log_resp_,
         self.log_mix_probabilities_,
         self.log_reg_probabilities_) = params
        #(self.weights_, self.means_, self.precisions_, self.reg_weights_, self.reg_precisions_) = params

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        _, n_features_extend = self.reg_weights_.shape
        prec_params = self.reg_precisions.shape
        reg_weights_params = n_features_extend * self.n_components

        return int(prec_params + reg_weights_params + self.n_components - 1)

    def predict(self, X, y=None):
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
        n_samples, n_features = X.shape
        X_ext = np.concatenate((np.ones((n_samples, 1)), X), axis=1)
        y_ = np.zeros((n_samples, 1))

        # Gaussian mixture probabilities.
        log_mix_probabilities = _estimate_log_gaussian_prob(X,
                                                            self.means_,
                                                            self.precisions_cholesky_,
                                                            self.covariance_type)
        
        # Compute log-responsibilities
        log_weights = np.log(self.weights_)
        log_gamma = log_weights + log_mix_probabilities
        log_sum_gamma = logsumexp(log_gamma, axis=1)[:, np.newaxis]
        with np.errstate(under='ignore'):
            # ignore underflow
            log_resp = log_gamma - log_sum_gamma
            resp = np.exp(log_resp)
        
        # Compute the expected value of the predictive posterior.
        dot_prod = np.dot(X_ext, self.reg_weights_)
        y_ = np.sum(resp * dot_prod, axis=1)

        return y_
    
    def score(self, X, y, metric='R2'):
        """Score the values of the outputs for a new set of inputs

        Compute the expected value of y given the trained model and a set X of 
        new observations. Calculate the mean square error between the predicted 
        values against the real values.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y_real : array-like, shape (n_samples, 1)

        Returns
        -------
        score : int
        """
        y_ = self.predict(X)
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

        return score

    def predict_score(self, X, y, metric='R2'):
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
        y_ = self.predict(X)
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

        return y_, score
