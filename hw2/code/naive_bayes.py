import numpy as np


class NaiveBayes:
    """
    Naive Bayes implementation.
    Assumes the feature are binary.
    Also assumes the labels go from 0,1,...k-1
    """

    p_y = None
    p_xy = None
    not_p_xy = None

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def fit(self, X, y):
        n, d = X.shape

        # Compute the number of class labels
        k = self.num_classes

        # Compute the probability of each class i.e p(y==c), aka "baseline -ness"
        counts = np.bincount(y)
        p_y = counts / n

        p_xy = np.array([np.sum(X[np.where(y == i)], axis=0) for i in range(len(counts))])
        p_xy = np.divide(p_xy.T, counts)  # divide by number of posts in each group

        self.p_y = p_y
        self.p_xy = p_xy
        self.not_p_xy = 1 - p_xy

    def predict(self, X):
        n, d = X.shape
        k = self.num_classes
        p_xy = self.p_xy
        not_p_xy = self.not_p_xy
        p_y = self.p_y

        y_pred = np.zeros(n)
        for i in range(n):

            probs = p_y.copy()  # initialize with the p(y) terms
            for j in range(d):
                if X[i, j] != 0:
                    probs *= p_xy[j, :]
                else:
                    probs *= not_p_xy[j, :]

            y_pred[i] = np.argmax(probs)

        return y_pred


class NaiveBayesLaplace(NaiveBayes):
    def __init__(self, num_classes, beta=0):
        super().__init__(num_classes)
        self.beta = beta

    def fit(self, X, y):
        d, k = len(X[0]), len(np.unique(y))
        X = np.concatenate((X, np.zeros((self.beta * k, d)), np.ones((self.beta * k, d))), axis=0)
        y_extra = [np.arange(k) for _ in range(2 * self.beta)]
        y = np.concatenate((y, *y_extra))
        super().fit(X, y)
