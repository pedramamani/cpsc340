import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime


_funcs = {}


def handle(number):
    def register(func):
        _funcs[number] = func
        return func

    return register


def run(question):
    if question not in _funcs:
        raise ValueError(f"unknown question {question}")
    return _funcs[question]()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q", "--question", required=True, choices=sorted(_funcs.keys()) + ["all"]
    )
    args = parser.parse_args()
    if args.question == "all":
        for q in sorted(_funcs.keys()):
            start = f"== {q} "
            print("\n" + start + "=" * (80 - len(start)))
            run(q)
    else:
        return run(args.question)


def savefig(fname, fig=None, verbose=True):
    path = Path("..", "figs", fname)
    (plt if fig is None else fig).savefig(path, bbox_inches="tight", pad_inches=0)
    if verbose:
        print(f"Figure saved as '{path}'")


def check_gradient(model, X, y, dimensionality, verbose=True, epsilon=1e-6):
    # This checks that the gradient implementation is correct
    w = np.random.randn(dimensionality)
    f, g = model.funObj(w, X, y)

    # Check the gradient
    estimated_gradient = approx_fprime(
        w, lambda w: model.funObj(w, X, y)[0], epsilon=epsilon
    )

    implemented_gradient = model.funObj(w, X, y)[1]

    if (
        np.max(np.abs(estimated_gradient - implemented_gradient))
        / np.linalg.norm(estimated_gradient)
        > 1e-6
    ):
        raise Exception(
            "User and numerical derivatives differ:\n%s\n%s"
            % (estimated_gradient[:5], implemented_gradient[:5])
        )
    else:
        if verbose:
            print("User and numerical derivatives agree.")


def create_rating_matrix(
    ratings, n, d, user_key="user", item_key="item", valid_frac=0.2
):
    # shuffle the order
    ratings = ratings.sample(frac=1, random_state=123)

    user_mapper = dict(zip(np.unique(ratings[user_key]), list(range(n))))
    item_mapper = dict(zip(np.unique(ratings[item_key]), list(range(d))))

    # user_inverse_mapper = dict(zip(list(range(n)), np.unique(ratings[user_key])))
    # item_inverse_mapper = dict(zip(list(range(d)), np.unique(ratings[item_key])))


    n_train = int(len(ratings) * (1 - valid_frac))
    ratings_train = ratings[:n_train]
    ratings_valid = ratings[n_train:]

    user_ind_train = [user_mapper[i] for i in ratings_train[user_key]]
    item_ind_train = [item_mapper[i] for i in ratings_train[item_key]]

    user_ind_valid = [user_mapper[i] for i in ratings_valid[user_key]]
    item_ind_valid = [item_mapper[i] for i in ratings_valid[item_key]]

    Y_train = np.full((n, d), np.nan)
    Y_train[user_ind_train, item_ind_train] = ratings_train["rating"]

    Y_valid = np.full((n, d), np.nan)
    Y_valid[user_ind_valid, item_ind_valid] = ratings_valid["rating"]

    return Y_train, Y_valid
