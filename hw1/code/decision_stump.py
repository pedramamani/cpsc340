import numpy as np
import utils


class DecisionStumpEquality:
    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    def fit(self, X, y):
        n, d = X.shape

        # Get an array with the number of 0's, number of one's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)

        # Loop over features looking for the best split
        X = np.round(X)

        for j in range(d):
            for i in range(n):
                # Choose value to equate to
                t = X[i, j]

                # Find most likely class for each split
                y_yes_mode = utils.mode(y[X[:, j] == t])
                y_no_mode = utils.mode(y[X[:, j] != t])

                # Make predictions
                y_pred = y_yes_mode * np.ones(n)
                y_pred[X[:, j] != t] = y_no_mode

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode

    def predict(self, X):

        n, d = X.shape
        X = np.round(X)

        if self.j_best is None:
            return self.y_hat_yes * np.ones(n)

        y_hat = np.zeros(n)

        for i in range(n):
            if X[i, self.j_best] == self.t_best:
                y_hat[i] = self.y_hat_yes
            else:
                y_hat[i] = self.y_hat_no

        return y_hat


class DecisionStumpErrorRate:
    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    def fit(self, X, y):
        n, d = X.shape
        count = np.bincount(y, minlength=2)
        y_mode = np.argmax(count)
        if np.unique(y).size <= 1:
            self.y_hat_yes = y_mode
            self.y_hat_no = None
            self.j_best = None
            self.t_best = None
            return

        minError = np.sum(y != y_mode)
        X = np.round(X)

        for j in range(d):
            for i in range(n):
                t = X[i, j]

                y_yes_mode = utils.mode(y[X[:, j] < t])
                y_no_mode = utils.mode(y[X[:, j] >= t])
                y_pred = y_yes_mode * np.ones(n)
                y_pred[X[:, j] >= t] = y_no_mode

                errors = np.sum(y_pred != y)
                if errors < minError:
                    minError = errors
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode

    def predict(self, X):
        n, d = X.shape
        X = np.round(X)
        if self.j_best is None:
            return np.full(n, self.y_hat_yes)

        y_hat = np.zeros(n)
        for i in range(n):
            if X[i, self.j_best] < self.t_best:
                y_hat[i] = self.y_hat_yes
            else:
                y_hat[i] = self.y_hat_no

        return y_hat


"""
A helper function that computes the entropy of the
discrete distribution p (stored in a 1D numpy array).
The elements of p should add up to one.
This function ensures lim p-->0 of p log(p) = 0
which is mathematically true, but numerically results in NaN
because log(0) returns -Inf.
"""


def entropy(p):
    plogp = 0 * p  # initialize full of zeros
    plogp[p > 0] = p[p > 0] * np.log(p[p > 0])  # only do the computation when p>0
    return -np.sum(plogp)


"""
This is not required, but one way to simplify the code is
to have this class inherit from DecisionStumpErrorRate.
Which methods (init, fit, predict) do you need to overwrite?
"""


class DecisionStumpInfoGain(DecisionStumpErrorRate):
    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    def fit(self, X, y):
        n, d = X.shape
        count = np.bincount(y, minlength=2)
        y_mode = np.argmax(count)

        if np.unique(y).size <= 1:
            self.y_hat_yes = y_mode
            self.y_hat_no = None
            self.j_best = None
            self.t_best = None
            return

        maxInfoGain = 0
        X = np.round(X)

        for j in range(d):
            for i in range(n):
                t = X[i, j]

                y_yes_mode = utils.mode(y[X[:, j] < t])
                y_no_mode = utils.mode(y[X[:, j] >= t])
                y_pred = y_yes_mode * np.ones(n)
                y_pred[X[:, j] >= t] = y_no_mode

                y_yes = y[X[:, j] < t]
                y_no = y[X[:, j] >= t]
                n_yes, n_no = len(y_yes), len(y_no)
                infoGain = entropy(np.bincount(y) / n) \
                           - n_yes / n * entropy(np.bincount(y_yes) / n_yes) \
                           - n_no / n * entropy(np.bincount(y_no) / n_no)

                if maxInfoGain < infoGain:
                    maxInfoGain = infoGain
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode
