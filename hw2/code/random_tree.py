from random_stump import RandomStumpInfoGain
from decision_tree import DecisionTree
import numpy as np
import scipy.stats

import utils


class RandomTree(DecisionTree):
    def __init__(self, max_depth):
        DecisionTree.__init__(self, max_depth=max_depth, stump_class=RandomStumpInfoGain)

    def fit(self, X, y):
        n = X.shape[0]
        boostrap_inds = np.random.choice(n, n, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]

        DecisionTree.fit(self, bootstrap_X, bootstrap_y)


class RandomForest:
    def __init__(self, max_depth, num_trees):
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.trees = []

    def fit(self, X, y):
        self.trees = [RandomTree(self.max_depth) for _ in range(self.num_trees)]

        for tree in self.trees:
            tree.fit(X, y)

    def predict(self, X_hat):
        y_hats = np.array([tree.predict(X_hat) for tree in self.trees])
        modes, _ = scipy.stats.mode(y_hats)
        return modes
