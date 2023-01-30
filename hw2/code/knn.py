"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
import utils


class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        n = len(X_hat)
        y_hat = np.zeros(n, dtype=np.int8)
        dist = utils.euclidean_dist_squared(self.X, X_hat)
        for i in range(n):
            top_k = np.argsort(dist[:, i])[:self.k]
            y_hat[i] = utils.mode(self.y[top_k])
        return y_hat
