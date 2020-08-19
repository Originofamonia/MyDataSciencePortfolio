# https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea

import numpy as np
import pandas as pd
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


def plot_learning_curve(iter_array, model):
    plt.plot(iter_array, model.train_mse, label='Training', linewidth=5)
    plt.plot(iter_array, model.test_mse, label='Test', linewidth=5)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('iterations', fontsize=30)
    plt.ylabel('MSE', fontsize=30)
    plt.legend(loc='best', fontsize=20)
    plt.show()


class ExplicitMF:
    def __init__(self,
                 ratings,
                 n_factors=40,
                 learning='als',
                 item_fact_reg=0.0,
                 user_fact_reg=0.0,
                 item_bias_reg=0.0,
                 user_bias_reg=0.0,
                 verbose=False):
        """
        Train a matrix factorization model to predict empty
        entries in a matrix. The terminology assumes a
        ratings matrix which is ~ user x item

        Params
        ======
        ratings : (ndarray)
            User x Item matrix with corresponding ratings

        n_factors : (int)
            Number of latent factors to use in matrix
            factorization model
        learning : (str)
            Method of optimization. Options include
            'sgd' or 'als'.

        item_fact_reg : (float)
            Regularization term for item latent factors

        user_fact_reg : (float)
            Regularization term for user latent factors

        item_bias_reg : (float)
            Regularization term for item biases

        user_bias_reg : (float)
            Regularization term for user biases

        verbose : (bool)
            Whether or not to printout training progress
        """
        self.test_mse = []
        self.train_mse = []
        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.item_fact_reg = item_fact_reg
        self.user_fact_reg = user_fact_reg
        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg
        self.learning = learning
        if self.learning == 'sgd':
            self.sample_row, self.sample_col = self.ratings.nonzero()
            self.n_samples = len(self.sample_row)
        self._v = verbose
        self.item_vecs = np.random.normal(scale=1. / self.n_factors,
                                          size=(self.n_items, self.n_factors))
        self.user_vecs = np.random.normal(scale=1. / self.n_factors,
                                          size=(self.n_users, self.n_factors))

    def als_step(self,
                 latent_vectors,
                 fixed_vecs,
                 ratings,
                 _lambda,
                 type='user'):
        """
        One of the two ALS steps. Solve for the latent vectors
        specified by type.
        """
        if type == 'user':
            # Precompute
            YTY = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(YTY.shape[0]) * _lambda

            for u in range(latent_vectors.shape[0]):
                latent_vectors[u, :] = solve((YTY + lambdaI),
                                             ratings[u, :].dot(fixed_vecs))
        elif type == 'item':
            # Precompute
            XTX = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(XTX.shape[0]) * _lambda

            for i in range(latent_vectors.shape[0]):
                latent_vectors[i, :] = solve((XTX + lambdaI),
                                             ratings[:, i].T.dot(fixed_vecs))
        return latent_vectors

    def train(self, n_iter=10, learning_rate=0.1):
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vectors

        if self.learning == 'als':
            self.partial_train(n_iter)
        elif self.learning == 'sgd':
            self.learning_rate = learning_rate
            self.user_bias = np.zeros(self.n_users)
            self.item_bias = np.zeros(self.n_items)
            self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])
            self.partial_train(n_iter)

    def partial_train(self, n_iter):
        """
        Train model for n_iter iterations. Can be
        called multiple times for further training.
        """
        ctr = 1
        while ctr <= n_iter:
            if ctr % 10 == 0 and self._v:
                print('\tcurrent iteration: {}'.format(ctr))
            if self.learning == 'als':
                self.user_vecs = self.als_step(self.user_vecs,
                                               self.item_vecs,
                                               self.ratings,
                                               self.user_fact_reg,
                                               type='user')
                self.item_vecs = self.als_step(self.item_vecs,
                                               self.user_vecs,
                                               self.ratings,
                                               self.item_fact_reg,
                                               type='item')
            elif self.learning == 'sgd':
                self.training_indices = np.arange(self.n_samples)
                np.random.shuffle(self.training_indices)
                self.sgd()
            ctr += 1

    def sgd(self):
        for idx in self.training_indices:
            u = self.sample_row[idx]
            i = self.sample_col[idx]
            prediction = self.predict(u, i)
            e = (self.ratings[u, i] - prediction)  # error

            # Update biases
            self.user_bias[u] += self.learning_rate * \
                                 (e - self.user_bias_reg * self.user_bias[u])
            self.item_bias[i] += self.learning_rate * \
                                 (e - self.item_bias_reg * self.item_bias[i])

            # Update latent factors
            self.user_vecs[u, :] += self.learning_rate * \
                                    (e * self.item_vecs[i, :] -
                                     self.user_fact_reg * self.user_vecs[u, :])
            self.item_vecs[i, :] += self.learning_rate * \
                                    (e * self.user_vecs[u, :] -
                                     self.item_fact_reg * self.item_vecs[i, :])

    def predict(self, u, i):
        """ Single user and item prediction."""
        if self.learning == 'als':
            return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        elif self.learning == 'sgd':
            prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
            prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
            return prediction

    def predict_all(self):
        """ Predict ratings for every user and item."""
        predictions = np.zeros((self.user_vecs.shape[0],
                                self.item_vecs.shape[0]))
        for u in range(self.user_vecs.shape[0]):
            for i in range(self.item_vecs.shape[0]):
                predictions[u, i] = self.predict(u, i)

        return predictions

    def calculate_learning_curve(self, iter_array, test, learning_rate=0.1):
        """
        Keep track of MSE as a function of training iterations.

        Params
        ======
        iter_array : (list)
            List of numbers of iterations to train for each step of
            the learning curve. e.g. [1, 5, 10, 20]
        test : (2D ndarray)
            Testing dataset (assumed to be user x item).

        The function creates two new class attributes:

        train_mse : (list)
            Training data MSE values for each value of iter_array
        test_mse : (list)
            Test data MSE values for each value of iter_array
        """
        iter_array.sort()
        iter_diff = 0
        for (i, n_iter) in enumerate(iter_array):
            if self._v:
                print('Iteration: {}'.format(n_iter))
            if i == 0:
                self.train(n_iter - iter_diff, learning_rate)
            else:
                self.partial_train(n_iter - iter_diff)

            predictions = self.predict_all()

            self.train_mse += [get_mse(predictions, self.ratings)]
            self.test_mse += [get_mse(predictions, test)]
            if self._v:
                print('Train mse: ' + str(self.train_mse[-1]))
                print('Test mse: ' + str(self.test_mse[-1]))
            iter_diff = n_iter


def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                        size=10,
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]

    # Test and training are truly disjoint
    assert (np.all((train * test) == 0))
    return train, test


def get_ratings(path):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(path, sep='\t', names=names)
    df.head()

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    ratings = np.zeros((n_users, n_items))
    for row in df.itertuples():
        ratings[row[1] - 1, row[2] - 1] = row[3]

    return ratings, n_users, n_items


def main():
    np.random.seed(444)
    path = '../ml_100k/u.data'
    ratings, n_users, n_items = get_ratings(path)

    print(ratings)

    print(str(n_users) + ' users')
    print(str(n_items) + ' items')
    sparsity = float(len(ratings.nonzero()[0]))
    sparsity /= (ratings.shape[0] * ratings.shape[1])
    sparsity *= 100
    print('Sparsity: {:4.2f}%'.format(sparsity))

    train, test = train_test_split(ratings)
    # print(train, test)

    MF_ALS = ExplicitMF(train, n_factors=40)
    iter_array = [1, 2, 5, 10, 25, 50, 100]
    MF_ALS.calculate_learning_curve(iter_array, test)
    plot_learning_curve(iter_array, MF_ALS)

    MF_ALS = ExplicitMF(train, n_factors=40)

    iter_array = [1, 2, 5, 10, 25, 50, 100]
    MF_ALS.calculate_learning_curve(iter_array, test)
    plot_learning_curve(iter_array, MF_ALS)

    latent_factors = [5, 10, 20, 40, 80]
    regularizations = [0.1, 1., 10., 100.]
    regularizations.sort()
    iter_array = [1, 2, 5, 10, 25, 50, 100]

    best_params = {'n_factors': latent_factors[0], 'reg': regularizations[0], 'n_iter': 0, 'train_mse': np.inf,
                   'test_mse': np.inf, 'model': None}

    for fact in latent_factors:
        print('Factors: {}'.format(fact))
        for reg in regularizations:
            print('Regularization: {}'.format(reg))
            MF_ALS = ExplicitMF(train, n_factors=fact, user_reg=reg, item_reg=reg)
            MF_ALS.calculate_learning_curve(iter_array, test)
            min_idx = np.argmin(MF_ALS.test_mse)
            if MF_ALS.test_mse[min_idx] < best_params['test_mse']:
                best_params['n_factors'] = fact
                best_params['reg'] = reg
                best_params['n_iter'] = iter_array[min_idx]
                best_params['train_mse'] = MF_ALS.train_mse[min_idx]
                best_params['test_mse'] = MF_ALS.test_mse[min_idx]
                best_params['model'] = MF_ALS
                print('New optimal hyperparameters')
                print(pd.Series(best_params))

    best_als_model = best_params['model']
    plot_learning_curve(iter_array, best_als_model)

    MF_SGD = ExplicitMF(train, 40, learning='sgd', verbose=True)
    iter_array = [1, 2, 5, 10, 25, 50, 100, 200]
    MF_SGD.calculate_learning_curve(iter_array, test, learning_rate=0.001)
    plot_learning_curve(iter_array, MF_SGD)

    iter_array = [1, 2, 5, 10, 25, 50, 100, 200]
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]

    best_params = {'learning_rate': None, 'n_iter': 0, 'train_mse': np.inf, 'test_mse': np.inf, 'model': None}

    for rate in learning_rates:
        print('Rate: {}'.format(rate))
        MF_SGD = ExplicitMF(train, n_factors=40, learning='sgd')
        MF_SGD.calculate_learning_curve(iter_array, test, learning_rate=rate)
        min_idx = np.argmin(MF_SGD.test_mse)
        if MF_SGD.test_mse[min_idx] < best_params['test_mse']:
            best_params['n_iter'] = iter_array[min_idx]
            best_params['learning_rate'] = rate
            best_params['train_mse'] = MF_SGD.train_mse[min_idx]
            best_params['test_mse'] = MF_SGD.test_mse[min_idx]
            best_params['model'] = MF_SGD
            print('New optimal hyperparameters')
            print(pd.Series(best_params))

    iter_array = [1, 2, 5, 10, 25, 50, 100, 200]
    latent_factors = [5, 10, 20, 40, 80]
    regularizations = [0.001, 0.01, 0.1, 1.]
    regularizations.sort()

    best_params = {'n_factors': latent_factors[0], 'reg': regularizations[0], 'n_iter': 0, 'train_mse': np.inf,
                   'test_mse': np.inf, 'model': None}

    for fact in latent_factors:
        print('Factors: {}'.format(fact))
        for reg in regularizations:
            print('Regularization: {}'.format(reg))
            MF_SGD = ExplicitMF(train, n_factors=fact, learning='sgd',
                                user_fact_reg=reg, item_fact_reg=reg,
                                user_bias_reg=reg, item_bias_reg=reg)
            MF_SGD.calculate_learning_curve(iter_array, test, learning_rate=0.001)
            min_idx = np.argmin(MF_SGD.test_mse)
            if MF_SGD.test_mse[min_idx] < best_params['test_mse']:
                best_params['n_factors'] = fact
                best_params['reg'] = reg
                best_params['n_iter'] = iter_array[min_idx]
                best_params['train_mse'] = MF_SGD.train_mse[min_idx]
                best_params['test_mse'] = MF_SGD.test_mse[min_idx]
                best_params['model'] = MF_SGD
                print('New optimal hyperparameters')
                print(pd.Series(best_params))

    plot_learning_curve(iter_array, best_params['model'])
    print('Best regularization: {}'.format(best_params['reg']))
    print('Best latent factors: {}'.format(best_params['n_factors']))
    print('Best iterations: {}'.format(best_params['n_iter']))


if __name__ == '__main__':
    main()
