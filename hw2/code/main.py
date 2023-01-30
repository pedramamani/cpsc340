#!/usr/bin/env python
import argparse
from functools import partial
import os
import pickle
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
import myplot
import utils
from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from kmeans import Kmeans
from knn import KNN
from naive_bayes import NaiveBayes, NaiveBayesLaplace
from random_tree import RandomForest, RandomTree


def load_dataset(filename):
    with open(Path("..", "data", filename), "rb") as f:
        return pickle.load(f)


# this just some Python scaffolding to conveniently run the functions below;
# don't worry about figuring out how it works if it's not obvious to you
func_registry = {}


def handle(number):
    def register(func):
        func_registry[number] = func
        return func

    return register


def run(question):
    if question not in func_registry:
        raise ValueError(f"unknown question {question}")
    return func_registry[question]()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--question", required=True, choices=func_registry.keys())
    args = parser.parse_args()
    return run(args.question)


@handle("1")
def q1():
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    for k in [10, 3, 1]:
        model = KNN(k)
        model.fit(X, y)
        y_hat_train = model.predict(X)
        e_train = np.mean(y_hat_train != y) * 100
        y_hat_test = model.predict(X_test)
        e_test = np.mean(y_hat_test != y_test) * 100
        print(f'k={k}: E_{{train}} = {e_train}\\%,\\: E_{{test}} = {e_test}\\%')

    utils.plot_classifier(model, X, y)


@handle("2")
def q2():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    ks = list(range(1, 30, 4))
    cv_accs = np.zeros_like(ks, dtype=np.float16)
    n = len(X)
    n_fold = 10

    for i, k in enumerate(ks):
        model = KNN(k)

        for j in range(n_fold):
            mask = np.zeros(n, dtype=np.bool8)
            mask[j * n // n_fold: (j+1) * n // n_fold] = 1
            i_validate, = np.nonzero(mask)
            i_train, = np.nonzero(~mask)
            X_train, y_train = X[i_train], y[i_train]
            X_validate, y_validate = X[i_validate], y[i_validate]

            model.fit(X_train, y_train)
            cv_accs[i] += np.mean(model.predict(X_validate) != y_validate)
        cv_accs[i] *= 100. / n_fold

    e_test = np.zeros_like(ks, dtype=np.float16)
    for i, k in enumerate(ks):
        model = KNN(k)
        model.fit(X, y)
        y_hat_test = model.predict(X_test)
        e_test[i] = np.mean(y_hat_test != y_test) * 100

    e_train = np.zeros_like(ks, dtype=np.float16)
    for i, k in enumerate(ks):
        model = KNN(k)
        model.fit(X, y)
        y_hat_train = model.predict(X)
        e_train[i] = np.mean(y_hat_train != y) * 100

    g = myplot.Plot()
    g.line(ks, e_test)
    g.line(ks, cv_accs)
    g.show(xlabel='k', ylabel='Percent Error', legend=['Test', 'Cross-validation'], grid=True)

    g = myplot.Plot()
    g.line(ks, e_train)
    g.show(xlabel='k', ylabel='Training Error', grid=True)


@handle("3.2")
def q3_2():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"].astype(bool)
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]
    groupnames = dataset["groupnames"]
    wordlist = dataset["wordlist"]

    print(f'1. {wordlist[72]}')
    print(f'2. {[wordlist[i] for i in np.nonzero(X[803])]}')
    print(f'3. {groupnames[y[803]]}')


@handle("3.3")
def q3_3():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)
    print(model.p_xy)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")


@handle("3.4")
def q3_4():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    """YOUR CODE HERE FOR Q3.4"""
    raise NotImplementedError()


@handle("4")
def q4():
    dataset = load_dataset("vowel.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    print(f"n = {X.shape[0]}, d = {X.shape[1]}")

    def evaluate_model(model):
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print(f"    Training error: {tr_error:.3f}")
        print(f"    Testing error: {te_error:.3f}")

    print("Decision tree")
    evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))
    print("Random tree")
    evaluate_model(RandomTree(max_depth=np.inf))
    print("Random forest")
    evaluate_model(RandomForest(max_depth=np.inf, num_trees=50))


@handle("5")
def q5():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "kmeans_basic_rerun.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("5.1")
def q5_1():
    X = load_dataset("clusterData.pkl")["X"]
    best_error = np.inf

    for _ in range(50):
        model = Kmeans(k=4)
        error = model.fit(X)
        print(error)
        if error < best_error:
            best_model = model
            best_error = error

    y = best_model.predict(X)
    print(best_error)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "kmeans_best.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("5.2")
def q5_2():
    X = load_dataset("clusterData.pkl")["X"]
    best_errors = []
    ks = range(1, 11)

    for k in ks:
        best_error = np.inf

        for _ in range(50):
            model = Kmeans(k=k)
            error = model.fit(X)
            if error < best_error:
                best_error = error
        best_errors.append(best_error)

    p = myplot.Plot()
    p.line(np.array(ks), np.array(best_errors))
    p.show(xlabel='k', ylabel='Error (a.u.)', grid=True)


if __name__ == "__main__":
    main()
