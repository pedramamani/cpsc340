#!/usr/bin/env python
import os
from pathlib import Path
import numpy as np
import pandas as pd

os.chdir(Path(__file__).parent.resolve())

import utils  # noqa: E402
from utils import handle, main  # noqa: E402

np.random.seed(1) # leave this here, for reproducibility during grading

def loss_fun(Z, W, avg_rating, Y_train, lammyZ, lammyW):
    Rs = (Z @ W + avg_rating) - Y_train
    Rs[np.isnan(Y_train)] = 0  # mask out the missing ratings
    loss = np.sum(Rs ** 2) / 2
    reg = lammyZ / 2 * np.sum(Z ** 2) + lammyW / 2 * np.sum(W ** 2)
    return loss + reg


def rmse(Z, W, avg_rating, Y):
    Y_hat = Z @ W + avg_rating
    return np.sqrt(np.nanmean((Y_hat - Y) ** 2))


@handle("1")
def q1():
    ratings = pd.read_csv("../data/ml-latest-small/ratings.csv")
    n = len(set(ratings["userId"]))
    d = len(set(ratings["movieId"]))

    Y_train, Y_valid = utils.create_rating_matrix(ratings, n, d, "userId", "movieId")

    avg_rating = np.nanmean(Y_train)

    k = 10
    lammyZ = 10
    lammyW = 10
    alpha = 0.01
    n_iters = 20

    # parameter initialization
    Z = np.random.randn(n, k) / np.sqrt(k)
    W = np.random.randn(k, d) / np.sqrt(k)
    ratings_per_user_i = np.sum(~np.isnan(Y_train), axis=1)
    ratings_per_movie_j = np.sum(~np.isnan(Y_train), axis=0)
    Z[ratings_per_user_i == 0] = 0  # if there are no ratings, set to 0
    W[:, ratings_per_movie_j == 0] = 0  # if there are no ratings, set to 0

    # print the initial loss
    print(
        f"Initial:   "
        f"f(Z,W): {loss_fun(Z, W, avg_rating, Y_train, lammyZ, lammyW):8.1f}   "
        f"Train RMSE: {rmse(Z, W, avg_rating, Y_train):5.3f}   "
        f"Valid RMSE: {rmse(Z, W, avg_rating, Y_valid):5.3f}   "
        f"||W||^2: {np.sum(W**2):6.1f}   "
        f"||Z||^2: {np.sum(Z**2):6.1f}"
    )
    for iteration in range(n_iters):
        # compute Rs
        Y_hat = Z @ W + avg_rating
        Rs = Y_hat - Y_train
        Rs[np.isnan(Y_train)] = 0

        # compute the gradients
        Z_grad = Rs @ W.T + lammyZ * Z
        W_grad = Z.T @ Rs + lammyW * W

        # update Z and W
        Z -= alpha * Z_grad
        W -= alpha * W_grad

        # compute and print the loss/RMSE
        print(
            f"Iter {iteration + 1:>2}:   "
            f"f(Z,W): {loss_fun(Z, W, avg_rating, Y_train, lammyZ, lammyW):8.1f}   "
            f"Train RMSE: {rmse(Z, W, avg_rating, Y_train):5.3f}   "
            f"Valid RMSE: {rmse(Z, W, avg_rating, Y_valid):5.3f}   "
            f"||W||^2: {np.sum(W**2):6.1f}   "
            f"||Z||^2: {np.sum(Z**2):6.1f}"
        )


@handle("1.1")
def q1_1():
    ratings = pd.read_csv("../data/ml-latest-small/ratings.csv")
    n = len(set(ratings["userId"]))
    d = len(set(ratings["movieId"]))

    Y_train, Y_valid = utils.create_rating_matrix(ratings, n, d, "userId", "movieId")

    avg_rating = np.nanmean(Y_train)

    k = 10
    lammyZ = 10
    lammyW = 10
    alpha = 0.1
    n_epoch = 10

    # parameter initialization
    Z = np.random.randn(n, k) / np.sqrt(k)
    W = np.random.randn(k, d) / np.sqrt(k)
    ratings_per_user_i = np.sum(~np.isnan(Y_train), axis=1)
    ratings_per_movie_j = np.sum(~np.isnan(Y_train), axis=0)
    Z[ratings_per_user_i == 0] = 0  # if there are no ratings, set to 0
    W[:, ratings_per_movie_j == 0] = 0  # if there are no ratings, set to 0

    ratings_inds_train = np.array(np.where(~np.isnan(Y_train)))
    num_ratings_train = ratings_inds_train.shape[1]

    # print the initial loss
    print(
        f"Initial:   "
        f"f(Z,W): {loss_fun(Z, W, avg_rating, Y_train, lammyZ, lammyW):8.1f}   "
        f"Train RMSE: {rmse(Z, W, avg_rating, Y_train):5.3f}   "
        f"Valid RMSE: {rmse(Z, W, avg_rating, Y_valid):5.3f}   "
        f"||W||^2: {np.sum(W**2):6.1f}   "
        f"||Z||^2: {np.sum(Z**2):6.1f}"
    )
    for epoch in range(n_epoch):

        for iteration in range(num_ratings_train):


            i, j = ratings_inds_train[:, np.random.choice(num_ratings_train)]

            # compute Rs
            Y_hat = Z @ W + avg_rating
            Rs = Y_hat - Y_train
            Rs[np.isnan(Y_train)] = 0

            Rsij = Rs[i, j]
            Rs[:, :] = 0
            Rs[i, j] = Rsij


            # compute the gradients
            # note that we divide lammy by |S| which is num_ratings_train in the code
            Z_grad = Rs @ W.T + lammyZ * Z / num_ratings_train
            W_grad = Z.T @ Rs + lammyW * W / num_ratings_train

            # update Z and W
            Z -= alpha / (np.sqrt(epoch + 1)) * Z_grad
            W -= alpha / (np.sqrt(epoch + 1)) * W_grad

        # compute the loss/RMSE and print it out
        print(
            f"Iter {epoch + 1:>2}   "
            f"f(Z,W): {loss_fun(Z, W, avg_rating, Y_train, lammyZ, lammyW):8.1f}   "
            f"Train RMSE: {rmse(Z, W, avg_rating, Y_train):5.3f}   "
            f"Valid RMSE: {rmse(Z, W, avg_rating, Y_valid):5.3f}   "
            f"||W||^2: {np.sum(W**2):6.1f}   "
            f"||Z||^2: {np.sum(Z**2):6.1f}"
        )


@handle("1.3")
def q1_3():
    ratings = pd.read_csv("../data/ml-latest-small/ratings.csv")
    n = len(set(ratings["userId"]))
    d = len(set(ratings["movieId"]))

    Y_train, Y_valid = utils.create_rating_matrix(ratings, n, d, "userId", "movieId")

    avg_rating = np.nanmean(Y_train)

    k = 10
    lammyZ = 10
    lammyW = 10
    alpha = 0.1
    n_epoch = 10

    # parameter initialization
    Z = np.random.randn(n, k) / np.sqrt(k)
    W = np.random.randn(k, d) / np.sqrt(k)
    ratings_per_user_i = np.sum(~np.isnan(Y_train), axis=1)
    ratings_per_movie_j = np.sum(~np.isnan(Y_train), axis=0)
    Z[ratings_per_user_i == 0] = 0  # if there are no ratings, set to 0
    W[:, ratings_per_movie_j == 0] = 0  # if there are no ratings, set to 0

    ratings_inds_train = np.array(np.where(~np.isnan(Y_train)))
    num_ratings_train = ratings_inds_train.shape[1]

    # print the initial loss
    print(
        f"Initial    "
        f"f(Z,W): {loss_fun(Z, W, avg_rating, Y_train, lammyZ, lammyW):8.1f}   "
        f"Train RMSE: {rmse(Z, W, avg_rating, Y_train):5.3f}   "
        f"Valid RMSE: {rmse(Z, W, avg_rating, Y_valid):5.3f}   "
        f"||W||^2: {np.sum(W**2):6.1f}   "
        f"||Z||^2: {np.sum(Z**2):6.1f}"
    )

    for epoch in range(n_epoch):

        for iter in range(num_ratings_train):

            # sample a random rating (i,j)
            i, j = ratings_inds_train[:, np.random.choice(num_ratings_train)]
            
            Y_hat_ij = Z[i] @ W[:, j] + avg_rating
            Rs_ij = Y_hat_ij - Y_train[i, j]

            Z_i_grad = Rs_ij * W[:,j] + lammyZ * Z[i] / ratings_per_user_i[i]
            W_j_grad = Rs_ij * Z[i] + lammyW * W[:,j] / ratings_per_movie_j[j]

            Z[i] -= alpha / (np.sqrt(epoch + 1)) * Z_i_grad
            W[:, j] -= alpha / (np.sqrt(epoch + 1)) * W_j_grad

        print(
            f"Epoch {epoch + 1:>2}   "
            f"f(Z,W): {loss_fun(Z, W, avg_rating, Y_train, lammyZ, lammyW):8.1f}   "
            f"Train RMSE: {rmse(Z, W, avg_rating, Y_train):5.3f}   "
            f"Valid RMSE: {rmse(Z, W, avg_rating, Y_valid):5.3f}   "
            f"||W||^2: {np.sum(W**2):6.1f}   "
            f"||Z||^2: {np.sum(Z**2):6.1f}"
        )


if __name__ == "__main__":
    main()
