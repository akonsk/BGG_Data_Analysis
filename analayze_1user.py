from get_games_ids import request
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
import time
import timeit
import pickle
import scipy.io
from sparse_pearson_cor import nan_sparse_pearson_correlation

connex = sqlite3.connect("bgg_ratings_recommender_deduplicated_toy.db")  # Opens file if exists, else creates file
sql = "SELECT * FROM data"
df = pd.read_sql_query(sql, connex)
connex.close()

### analyze how well the alg predicts all games for 1 user

def predict_ratings(pt,chosen_friends_inds, i_user):
    """
    make predictions for the rating of all games rated
    by the user.
    """
    user = pt[i_user]
    df_ratings = pt[chosen_friends_inds]
    df_ratings = pd.DataFrame(df_ratings)

    # Center the ratings
    user_means = df_ratings.mean(axis=1)
    user_stds = df_ratings.std(axis=1) + 1e-100
    assert ~(np.isnan(user_stds.values).any()), 'users with only 1 rating'
    df_ratings = df_ratings.subtract(user_means, axis=0)
    df_ratings = df_ratings.divide(user_stds, axis=0)

    # Make a prediction based on unweighted average of k nearest neighbors (and "uncenter")
    predicted = (np.nanmean(user) + np.nanstd(user) * df_ratings.mean(0)).values
    num_voters = df_ratings.count(0).values

    return predicted, num_voters

def analyze_errors(pt, predicted, num_voters, i_user, silent=False):
    """
    Analyze the errors in predictions for games by user,
    and plot a comparison to the errors from the simplest
    naive estimator of user rating - the games average rating.
    """

    df_ratings = pd.DataFrame(pt)
    i_rated_by_user = ~np.isnan(pt[i_user]) & (num_voters != 0)
    user = pt[i_user, i_rated_by_user]

    # Compute naive estimator - the rounded mean rating of the game
    naive = df_ratings.mean(0).values
  
    # Compute MSE for our algorithm and for simple estimator
    mse_algo = mean_squared_error(user, predicted[i_rated_by_user])
    mse_naive = mean_squared_error(user, naive[i_rated_by_user])

    # If running silently just return the mean squared errors of algorithm and simple estimator
    if silent:
        return mse_algo, mse_naive, \
               num_voters[i_rated_by_user], user, predicted[i_rated_by_user]

    # If not running silently, plot errors
    else:
        fig, ax = plt.subplots()
        plt.title("Prediction Errors\n"
                  " MSE={:.2f} \n"
                  " MSE_BM={:.2f}".format(mse_algo, mse_naive))
        plt.ylabel("Counts")
        plt.xlabel("Distribution of Prediction Errors")
        bins = np.arange(-4, 4, 1) - 0.5
        trash = ax.hist(user - predicted[i_rated_by_user], bins=bins, color="red", label="algorithm")
        trash = ax.hist(user - naive[i_rated_by_user], bins=bins, color="blue", label="benchmark", alpha=0.3)
        ax.legend(loc="upper right")

        plt.figure()
        plt.scatter(num_voters[i_rated_by_user],user - predicted[i_rated_by_user])
        print("Mean Squared Error of algorithm is %.2f" % (mse_algo,))
        print("Mean Squared Error of naive mean game rating estimator is %.2f" % (mse_naive,))
        return ax, mse_algo, mse_naive


pt = df.pivot_table('rating', 'username', 'gameid').values
# filter users that rated less than 4 games
pt = pt[((~np.isnan(pt)).sum(1) >= 10)]
print(pt.shape)
# define hyperparameters
N_friends=100
friends_threshold=0.9

L=[]
for i_user in range(1000):
    # i_user=1
    user = pt[i_user]
    cor = nan_sparse_pearson_correlation(pt[np.array(range(len(pt))) != i_user], user, min_periods=4)
    cor[np.isnan(cor)]=-1
    # indices of nearest friends, from best to worst
    knn_inds=np.argsort(cor)[::-1]

    # take up to N friends, with similarity>threshold

    chosen_friends_inds=knn_inds[cor[knn_inds]>=friends_threshold][:N_friends]

    predicted, num_voters = predict_ratings(pt,chosen_friends_inds, i_user)
    mse_algo, mse_naive, num_voters, user, predicted = \
        analyze_errors(pt, predicted, num_voters, i_user, silent=True)

    L.append({'num_voters': num_voters,
              'user': user,
              'predicted': predicted})
