"""Module :mod:`sklearn.kernel_ridge` implements kernel ridge regression."""

# This code was adapted from the 
# Original Authors: Mathieu Blondel <mathieu@mblondel.org>
#          Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD 3 clause

import numpy as np
import os
import time
import math
from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.linear_model._ridge import _solve_cholesky_kernel
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.utils.validation import _deprecate_positional_args
from scipy.stats import norm
import scipy
from sklearn import datasets, linear_model, metrics 

def confidence(data):
    c = 0.95 # we want 95 percent confidence
    n = len(data)
    alpha = 1.0 - c
    x_mean = data.mean(axis=0)
    sig = data.std()
    z_critical = scipy.stats.norm.ppf(q=0.975)
    #print("z critical value = ")
    #print(z_critical)
    z_interval = scipy.stats.norm.interval(alpha=c)
    stderror = sig / math.sqrt(n)
    upper = x_mean + z_critical * stderror
    lower = x_mean - z_critical * stderror
    # we are 95% sure the >>>
    #print("upper value = ")
    #print(upper)
    #print("lower = ")
    #print(lower)
    return upper, lower

class KernelRidge(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Kernel ridge regression.
    Kernel ridge regression (KRR) combines ridge regression (linear least
    squares with l2-norm regularization) with the kernel trick. It thus
    learns a linear function in the space induced by the respective kernel and
    the data. For non-linear kernels, this corresponds to a non-linear
    function in the original space.
    The form of the model learned by KRR is identical to support vector
    regression (SVR). However, different loss functions are used: KRR uses
    squared error loss while support vector regression uses epsilon-insensitive
    loss, both combined with l2 regularization. In contrast to SVR, fitting a
    KRR model can be done in closed-form and is typically faster for
    medium-sized datasets. On the other hand, the learned model is non-sparse
    and thus slower than SVR, which learns a sparse model for epsilon > 0, at
    prediction-time.
    This estimator has built-in support for multi-variate regression
    (i.e., when y is a 2d-array of shape [n_samples, n_targets]).
    Read more in the :ref:`User Guide <kernel_ridge>`.
    Parameters
    ----------
    alpha : float or array-like of shape (n_targets,)
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
        Alpha corresponds to ``1 / (2C)`` in other linear models such as
        :class:`~sklearn.linear_model.LogisticRegression` or
        :class:`sklearn.svm.LinearSVC`. If an array is passed, penalties are
        assumed to be specific to the targets. Hence they must correspond in
        number. See :ref:`ridge_regression` for formula.
    kernel : string or callable, default="linear"
        Kernel mapping used internally. This parameter is directly passed to
        :class:`sklearn.metrics.pairwise.pairwise_kernel`.
        If `kernel` is a string, it must be one of the metrics
        in `pairwise.PAIRWISE_KERNEL_FUNCTIONS`.
        If `kernel` is "precomputed", X is assumed to be a kernel matrix.
        Alternatively, if `kernel` is a callable function, it is called on
        each pair of instances (rows) and the resulting value recorded. The
        callable should take two rows from X as input and return the
        corresponding kernel value as a single number. This means that
        callables from :mod:`sklearn.metrics.pairwise` are not allowed, as
        they operate on matrices, not single samples. Use the string
        identifying the kernel instead.
    gamma : float, default=None
        Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
        and sigmoid kernels. Interpretation of the default value is left to
        the kernel; see the documentation for sklearn.metrics.pairwise.
        Ignored by other kernels.
    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.
    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.
    kernel_params : mapping of string to any, optional
        Additional parameters (keyword arguments) for kernel function passed
        as callable object.
    Attributes
    ----------
    dual_coef_ : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Representation of weight vector(s) in kernel space
    X_fit_ : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Training data, which is also required for prediction. If
        kernel == "precomputed" this is instead the precomputed
        training matrix, of shape (n_samples, n_samples).
    References
    ----------
    * Kevin P. Murphy
      "Machine Learning: A Probabilistic Perspective", The MIT Press
      chapter 14.4.3, pp. 492-493
    See also
    --------
    sklearn.linear_model.Ridge:
        Linear ridge regression.
    sklearn.svm.SVR:
        Support Vector Regression implemented using libsvm.
    Examples
    --------
    >>> from sklearn.kernel_ridge import KernelRidge
    >>> import numpy as np
    >>> n_samples, n_features = 10, 5
    >>> rng = np.random.RandomState(0)
    >>> y = rng.randn(n_samples)
    >>> X = rng.randn(n_samples, n_features)
    >>> clf = KernelRidge(alpha=1.0)
    >>> clf.fit(X, y)
    KernelRidge(alpha=1.0)
    """
    @_deprecate_positional_args
    def __init__(self, alpha=1, *, kernel="linear", gamma=None, degree=3,
                 coef0=1, kernel_params=None):
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def fit(self, X, y=None, sample_weight=None):
        """Fit Kernel Ridge regression model
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data. If kernel == "precomputed" this is instead
            a precomputed kernel matrix, of shape (n_samples, n_samples).
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values
        sample_weight : float or array-like of shape [n_samples]
            Individual weights for each sample, ignored if None is passed.
        Returns
        -------
        self : returns an instance of self.
        """
        # Convert data
        t0 = time.time()
        #X, y = self._validate_data(X, y, accept_sparse=("csr", "csc"), multi_output=True, y_numeric=True)
        if sample_weight is not None and not isinstance(sample_weight, float):
            sample_weight = _check_sample_weight(sample_weight, X)

        K = self._get_kernel(X)
        alpha = np.atleast_1d(self.alpha)

        ravel = False
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            ravel = True

        copy = self.kernel == "precomputed"
        self.dual_coef_ = _solve_cholesky_kernel(K, y, alpha,
                                                 sample_weight,
                                                 copy)
        if ravel:
            self.dual_coef_ = self.dual_coef_.ravel()

        self.X_fit_ = X
        t1 = time.time() - t0
        #print("KRR fitted in %.3f s" % t1)
        return self

    def predict(self, X):
        """Predict using the kernel ridge model
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples. If kernel == "precomputed" this is instead a
            precomputed kernel matrix, shape = [n_samples,
            n_samples_fitted], where n_samples_fitted is the number of
            samples used in the fitting for this estimator.
        Returns
        -------
        C : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        """
        check_is_fitted(self)
        K = self._get_kernel(X, self.X_fit_)
        return np.dot(K, self.dual_coef_)

def get_test_train(fname,seed,datatype):
    '''
    Returns a test/train split of the data in fname shuffled with
    the given seed


    Args:
        fname:      A str/file object that points to the CSV file to load, passed to 
                    numpy.genfromtxt()
        seed:       The seed passed to numpy.random.seed(). Typically an int or long
        datatype:   The datatype to pass to genfromtxt(), usually int, float, or str


    Returns:
        train_X:    A NxD numpy array of training data (row-vectors), 80% of all data
        train_Y:    A Nx1 numpy array of class labels for the training data
        test_X:     A MxD numpy array of testing data, same format as train_X, 20% of all data
        test_Y:     A Mx1 numpy array of class labels for the testing data
    '''
    data = np.genfromtxt(fname,delimiter=',',dtype=datatype)
    data = np.nan_to_num(data)
    #print("asdakjshjdasd")
    #print(data)
    np.random.seed(seed)
    shuffled_idx = np.random.permutation(data.shape[0])
    cutoff = int(data.shape[0]*0.7)
    train_data = data[shuffled_idx[:cutoff]]
    test_data = data[shuffled_idx[cutoff:]]
    # Ensure there is no undefined values or numbers exceeding float capacity
    #train_data = numpy.nan_to_num(train_data)
    #test_data = numpy.nan_to_num(test_data)
    
    train_X = train_data[:,:-1].astype(float)
    train_Y = train_data[:,-1].reshape(-1,1)

    # Now set apart 50 % of the test data for Validation testing
    shuffled_idx = np.random.permutation(test_data.shape[0])
    cutoff = int(test_data.shape[0]*0.5)
    val_data = test_data[shuffled_idx[:cutoff]]
    test_data = test_data[shuffled_idx[cutoff:]]

    test_X = test_data[:,:-1].astype(float)
    test_Y = test_data[:,-1].reshape(-1,1)
    val_X = val_data[:,:-1].astype(float)
    val_Y = val_data[:,-1].reshape(-1,1)
    return train_X, train_Y, test_X, test_Y, val_X, val_Y, train_data, test_data

def load_data(path=''):
    #return get_test_train(os.path.join(path,'carbon_nanotubes.csv'),seed=1567708903,datatype=float)
    #return get_test_train(os.path.join(path,'hour.csv'),seed=1567708903,datatype=int)
    return get_test_train(os.path.join(path,'Concrete_Data.csv'),seed=1567708903,datatype=float)

def main():
    data = load_data() #in this order: train_X, train_Y, test_X, test_Y
    X_train = data[0]
    X_test = data[2]
    y_train = data[1]
    y_test = data[3]
    X_val = data[4]
    y_val = data[5]

    #++++++++++++#
    #  test KRR  # uncomment this block to run the KRR hyperparameter test
    #++++++++++++#
    """
    kernels = ['additive_chi2', 'chi2', 'poly', 'rbf', 'laplacian', 'cosine']

    # a way to keep track of the best so far
    # alpha, gamma, kernel, mse, confidence (upper, lower)
    t0 = time.time()
    errors = (1, 1, 'linear', 100000, (0,0))

    for alpha in np.linspace(0.00001, 1.0):
        for gamma in np.linspace(0.00001, 1.0):
            for kernel in kernels:
                krr = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma)
                krr.fit(X_train, y_train)
                temp = krr.predict(X_test)
                #print(temp)
                #print(y_test)
                #confidence(temp)
                mse = metrics.mean_squared_error(temp, y_test)
                
                if mse < errors[3]:
                    print("yesterdays best was ")
                    print(errors)
                    print("but the best of the BEST is:")
                    print(" mse = ")
                    print(mse)
                    print(" confidence interval (upper, lower) = ")
                    c = confidence(temp)
                    print(c)
                    errors = (alpha, gamma, kernel, mse, c)
                    print(errors)

        print(alpha)

    t1 = time.time() - t0
    print("KRR fitted in %.3f s" % t1)
    print(errors)
    """

    # Uncomment this to run on the validation set
    alpha = .00001
    gamma = .00001
    krr = KernelRidge(alpha=alpha, kernel='laplacian', gamma=gamma)
    krr.fit(X_train, y_train)
    temp = krr.predict(X_val)
    mse = metrics.mean_squared_error(temp, y_val)
    print(mse)


if __name__ == '__main__':
    main()
