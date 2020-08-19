import sys
import os
import time

# spark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import UserDefinedFunction, explode, desc
from pyspark.sql.types import StringType, ArrayType
from pyspark.mllib.recommendation import ALS

# data science imports
import math
import numpy as np
import pandas as pd

# visualization imports
import seaborn as sns
import matplotlib.pyplot as plt


def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)


def train_ALS(train_data, validation_data, num_iters, reg_param, ranks):
    """
    Grid Search Function to select the best model based on RMSE of hold-out data
    """
    # initial
    min_error = float('inf')
    best_rank = -1
    best_regularization = 0
    best_model = None
    for rank in ranks:
        for reg in reg_param:
            # train ALS model
            model = ALS.train(
                ratings=train_data,    # (userID, productID, rating) tuple
                iterations=num_iters,
                rank=rank,
                lambda_=reg,           # regularization param
                seed=99)
            # make prediction
            valid_data = validation_data.map(lambda p: (p[0], p[1]))
            predictions = model.predictAll(valid_data).map(lambda r: ((r[0], r[1]), r[2]))
            # get the rating result
            ratesAndPreds = validation_data.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
            # get the RMSE
            MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
            error = math.sqrt(MSE)
            print('{} latent factors and regularization = {}: validation RMSE is {}'.format(rank, reg, error))
            if error < min_error:
                min_error = error
                best_rank = rank
                best_regularization = reg
                best_model = model
    print('\nThe best model has {} latent factors and regularization = {}'.format(best_rank, best_regularization))
    return best_model


def plot_learning_curve(arr_iters, train_data, validation_data, reg, rank):
    """
    Plot function to show learning curve of ALS
    """
    errors = []
    for num_iters in arr_iters:
        # train ALS model
        model = ALS.train(
            ratings=train_data,    # (userID, productID, rating) tuple
            iterations=num_iters,
            rank=rank,
            lambda_=reg,           # regularization param
            seed=99)
        # make prediction
        valid_data = validation_data.map(lambda p: (p[0], p[1]))
        predictions = model.predictAll(valid_data).map(lambda r: ((r[0], r[1]), r[2]))
        # get the rating result
        ratesAndPreds = validation_data.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
        # get the RMSE
        MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
        error = math.sqrt(MSE)
        # add to errors
        errors.append(error)

    # plot
    plt.figure(figsize=(12, 6))
    plt.plot(arr_iters, errors)
    plt.xlabel('number of iterations')
    plt.ylabel('RMSE')
    plt.title('ALS Learning Curve')
    plt.grid(True)
    plt.show()


def get_movieId(df_movies, fav_movie_list, movies):
    """
    return all movieId(s) of user's favorite movies

    Parameters
    ----------
    df_movies: spark Dataframe, movies data

    fav_movie_list: list, user's list of favorite movies

    Return
    ------
    movieId_list: list of movieId(s)
    """
    movieId_list = []
    for movie in fav_movie_list:
        movieIds = df_movies \
            .filter(movies.title.like('%{}%'.format(movie))) \
            .select('movieId') \
            .rdd \
            .map(lambda r: r[0]) \
            .collect()
        movieId_list.extend(movieIds)
    return list(set(movieId_list))


def add_new_user_to_data(train_data, movieId_list, spark_context):
    """
    add new rows with new user, user's movie and ratings to
    existing train data

    Parameters
    ----------
    train_data: spark RDD, ratings data

    movieId_list: list, list of movieId(s)

    spark_context: Spark Context object

    Return
    ------
    new train data with the new user's rows
    """
    # get new user id
    new_id = train_data.map(lambda r: r[0]).max() + 1
    # get max rating
    max_rating = train_data.map(lambda r: r[2]).max()
    # create new user rdd
    user_rows = [(new_id, movieId, max_rating) for movieId in movieId_list]
    new_rdd = spark_context.parallelize(user_rows)
    # return new train data
    return train_data.union(new_rdd)


def get_inference_data(train_data, df_movies, movieId_list):
    """
    return a rdd with the userid and all movies (except ones in movieId_list)

    Parameters
    ----------
    train_data: spark RDD, ratings data

    df_movies: spark Dataframe, movies data

    movieId_list: list, list of movieId(s)

    Return
    ------
    inference data: Spark RDD
    """
    # get new user id
    new_id = train_data.map(lambda r: r[0]).max() + 1
    # return inference rdd
    return df_movies.rdd \
        .map(lambda r: r[0]) \
        .distinct() \
        .filter(lambda x: x not in movieId_list) \
        .map(lambda x: (new_id, x))


def make_recommendation(best_model_params, ratings_data, df_movies,
                        fav_movie_list, n_recommendations, spark_context, movies):
    """
    return top n movie recommendation based on user's input list of favorite movies


    Parameters
    ----------
    best_model_params: dict, {'iterations': iter, 'rank': rank, 'lambda_': reg}

    ratings_data: spark RDD, ratings data

    df_movies: spark Dataframe, movies data

    fav_movie_list: list, user's list of favorite movies

    n_recommendations: int, top n recommendations

    spark_context: Spark Context object

    Return
    ------
    list of top n movie recommendations
    """
    # modify train data by adding new user's rows
    movieId_list = get_movieId(df_movies, fav_movie_list, movies)
    train_data = add_new_user_to_data(ratings_data, movieId_list, spark_context)

    # train best ALS
    model = ALS.train(
        ratings=train_data,
        iterations=best_model_params.get('iterations', None),
        rank=best_model_params.get('rank', None),
        lambda_=best_model_params.get('lambda_', None),
        seed=99)

    # get inference rdd
    inference_rdd = get_inference_data(ratings_data, df_movies, movieId_list)

    # inference
    predictions = model.predictAll(inference_rdd).map(lambda r: (r[1], r[2]))

    # get top n movieId
    topn_rows = predictions.sortBy(lambda r: r[1], ascending=False).take(n_recommendations)
    topn_ids = [r[0] for r in topn_rows]

    # return movie titles
    return df_movies.filter(movies.movieId.isin(topn_ids)) \
        .select('title') \
        .rdd \
        .map(lambda r: r[0]) \
        .collect()


def main():
    # spark config
    spark = SparkSession \
        .builder \
        .appName("movie recommendation") \
        .config("spark.driver.maxResultSize", "16g") \
        .config("spark.driver.memory", "16g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.master", "local[12]") \
        .getOrCreate()
    # get spark context
    sc = spark.sparkContext

    # path config
    data_dir = '../ml_latest'  # data_dir should be relative path
    movies = spark.read.load(os.path.join(data_dir, 'movies.csv'), format='csv', header=True, inferSchema=True)
    ratings = spark.read.load(os.path.join(data_dir, 'ratings.csv'), format='csv', header=True, inferSchema=True)
    links = spark.read.load(os.path.join(data_dir, 'links.csv'), format='csv', header=True, inferSchema=True)
    tags = spark.read.load(os.path.join(data_dir, 'tags.csv'), format='csv', header=True, inferSchema=True)
    movies.show(3)
    ratings.show(3)
    links.show(3)
    tags.show(3)

    # print('Distinct values of ratings:')
    # print(sorted(ratings.select('rating').distinct().rdd.map(lambda r: r[0]).collect()))  # very slow

    # tmp1 = ratings.groupBy("userID").count().toPandas()['count'].min()
    # tmp2 = ratings.groupBy("movieId").count().toPandas()['count'].min()
    # print('For the users that rated movies and the movies that were rated:')
    # print('Minimum number of ratings per user is {}'.format(tmp1))
    # print('Minimum number of ratings per movie is {}'.format(tmp2))

    tmp1 = sum(ratings.groupBy("movieId").count().toPandas()['count'] == 1)
    tmp2 = ratings.select('movieId').distinct().count()
    print('{} out of {} movies are rated by only one user'.format(tmp1, tmp2))

    tmp = ratings.select('userID').distinct().count()
    print('We have a total of {} distinct users in the data sets'.format(tmp))

    tmp = movies.select('movieID').distinct().count()
    print('We have a total of {} distinct movies in the data sets'.format(tmp))

    tmp1 = movies.select('movieID').distinct().count()
    tmp2 = ratings.select('movieID').distinct().count()
    print('We have a total of {} distinct movies that are rated by users in ratings table'.format(tmp2))
    print('We have {} movies that are not rated yet'.format(tmp1 - tmp2))

    # create a temp SQL table view for easier query
    movies.createOrReplaceTempView("movies")
    ratings.createOrReplaceTempView("ratings")
    print('List movies that are not rated yet: ')
    # SQL query (NOTE: WHERE ... NOT IN ... == ... LEFT JOIN ... WHERE ... IS NULL)
    # Approach 1
    spark.sql(
        "SELECT movieId, title "
        "FROM movies "
        "WHERE movieId NOT IN (SELECT distinct(movieId) FROM ratings)"
    ).show(10)
    # Approach 2
    # spark.sql(
    #     "SELECT m.movieId, m.title "
    #     "FROM movies m LEFT JOIN ratings r ON m.movieId=r.movieId "
    #     "WHERE r.movieId IS NULL"
    # ).show(10)

    # define a udf for splitting the genres string
    splitter = UserDefinedFunction(lambda x: x.split('|'), ArrayType(StringType()))
    # query
    print('All distinct genres: ')
    movies.select(explode(splitter("genres")).alias("genres")).distinct().show()

    print('Counts of movies per genre')
    movies.select('movieID', explode(splitter("genres")).alias("genres")) \
        .groupby('genres') \
        .count() \
        .sort(desc('count')) \
        .show()

    # load data
    movie_rating = sc.textFile(os.path.join(data_dir, 'ratings.csv'))
    # preprocess data -- only need ["userId", "movieId", "rating"]
    header = movie_rating.take(1)[0]
    rating_data = movie_rating \
        .filter(lambda line: line != header) \
        .map(lambda line: line.split(",")) \
        .map(lambda tokens: (int(tokens[0]), int(tokens[1]), float(tokens[2]))) \
        .cache()
    # check three rows
    rating_data.take(3)

    train, validation, test = rating_data.randomSplit([6, 2, 2], seed=99)
    # cache data
    train.cache()
    validation.cache()
    test.cache()

    # hyper-param config
    num_iterations = 10
    ranks = [8, 10, 12, 14, 16, 18, 20]
    reg_params = [0.001, 0.01, 0.05, 0.1, 0.2]

    # grid search and select best model
    start_time = time.time()
    final_model = train_ALS(train, validation, num_iterations, reg_params, ranks)

    print('Total Runtime: {:.2f} seconds'.format(time.time() - start_time))

    # create an array of num_iters
    iter_array = list(range(1, 11))
    # create learning curve plot
    plot_learning_curve(iter_array, train, validation, 0.05, 20)

    # make prediction using test data
    test_data = test.map(lambda p: (p[0], p[1]))
    predictions = final_model.predictAll(test_data).map(lambda r: ((r[0], r[1]), r[2]))
    # get the rating result
    ratesAndPreds = test.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    # get the RMSE
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    error = math.sqrt(MSE)
    print('The out-of-sample RMSE of rating predictions is', round(error, 4))

    # my favorite movies
    my_favorite_movies = ['Iron Man']

    # get recommends
    recommends = make_recommendation(
        best_model_params={'iterations': 10, 'rank': 20, 'lambda_': 0.05},
        ratings_data=rating_data,
        df_movies=movies,
        fav_movie_list=my_favorite_movies,
        n_recommendations=10,
        spark_context=sc)

    print('Recommendations for {}:'.format(my_favorite_movies[0]))
    for i, title in enumerate(recommends):
        print('{0}: {1}'.format(i + 1, title))


if __name__ == '__main__':
    main()
