import numpy as np
import pandas as pd
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

from movie_lens_100k.movie_100k import train_test_split, get_ratings, ExplicitMF

sns.set()


def preprocess_rating(ratings, n_users, n_items):
    x = (ratings >= 3) * np.ones((n_users, n_items))  # X, split [0, 1, 2] -> 0, [3, 4, 5] -> 1
    a = np.count_nonzero(x == 1)
    return x


def main():
    np.random.seed(444)
    path = '../ml_100k/u.data'
    ratings, n_users, n_items = get_ratings(path)
    trainset, testset = train_test_split(ratings)

    x_train = preprocess_rating(trainset, n_users, n_items)
    x_test = preprocess_rating(testset, n_users, n_items)

    s = np.dot(x_train.T, x_train)
    # print(s)

    mf_als = ExplicitMF(s, n_factors=40, learning='als', item_fact_reg=1e-2, user_fact_reg=1e-2)
    mf_als.train(n_iter=100)  # MF
    s_hat = np.dot(mf_als.user_vecs, mf_als.item_vecs.T)  # matrix completion
    scores = np.dot(x_test, s_hat)
    a = np.max(scores)
    b = np.min(scores)
    print(scores[0])


if __name__ == '__main__':
    main()
