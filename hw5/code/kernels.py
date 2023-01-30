import numpy as np
from utils import euclidean_dist_squared


class Kernel:
    def __call__(self, X1, X2):
        """
        Evaluate Gram's kernel matrix based on the two input matrices.
        Shape of X1 is (n1, d) and shape of X2 is (n2, d).
        That is, both matrices should have the same number of columns.
        Will return a n1-by-n2 matrix, e.g. X1 @ X2.T
        """
        # This is the "magic method" that happens when you call kernel(X1, X2)
        raise NotImplementedError("this is a base class, don't call this")


class LinearKernel(Kernel):
    def __call__(self, X1, X2):
        return X1 @ X2.T


class PolynomialKernel(Kernel):
    def __init__(self, p):
        """
        p is the degree of the polynomial
        """
        self.p = p

    def __call__(self, X1, X2):
        """
        Evaluate the polynomial kernel.
        A naive implementation will use change of basis.
        A "kernel trick" implementation bypasses change of basis.
        """
        return (1 + X1 @ X2.T) ** self.p


class GaussianRBFKernel(Kernel):
    def __init__(self, sigma):
        """
        sigma is the curve width hyperparameter.
        """
        self.sigma = sigma

    def __call__(self, X1, X2):
        """
        Evaluate Gaussian RBF basis kernel.
        """
        n1, d = X1.shape
        n2, d = X2.shape

        return np.exp((2 * X1 @ X2.T -
                       np.tile(np.sum(X1 ** 2, axis=1), (n2, 1)).T -
                       np.tile(np.sum(X2 ** 2, axis=1), (n1, 1))) / (2 * self.sigma ** 2))
