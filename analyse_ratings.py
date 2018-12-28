
from get_games_ids import request
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
import timeit
import pickle
import scipy.io

connex = sqlite3.connect("bgg_ratings_recommender_deduplicated_toy.db")  # Opens file if exists, else creates file
sql = "SELECT * FROM data"
df = pd.read_sql_query(sql, connex)
connex.close()

AN=0
if AN:
    gameid = 21892
    users = df.loc[df["gameid"] == gameid, "username"].values  # list of all users who rated this game
    df_sample = df[df["username"].isin(users)]  # all ratings for all games from users who rated this game
    df_pivot = df_sample.pivot_table(index="username", columns="gameid", values="rating")  # pivot to matrix (user x game) matrix

    # compute pearson correlation
    temp = df_pivot.transpose()
    corrs = temp.corr()  # Default for min_periods is 1

    # look at the histogram of the values of the cor matrix
    # Look at just the upper triangular submatrix (excluding the diagonal)since matrix is symmetric.
    Xcorr = corrs.as_matrix()
    np.fill_diagonal(Xcorr, np.nan)

    fig, ax = plt.subplots()
    plt.ylabel("2*Counts")
    plt.xlabel("Correlation")
    trash = ax.hist(Xcorr[~np.isnan(Xcorr)], bins=40)

    # Number of Rated Games in Common
    X = df_pivot.as_matrix()  # get values as a numpy matrix
    # Make a matrix holding number of common games rated between users (i, j)
    m1 = (~np.isnan(X)).astype(np.float64)  # zeroes for missing ratings, one elsewhere
    shared = np.dot(m1, m1.T)
    np.fill_diagonal(shared, np.nan)  # diagonals are matching a user with themselves so don't include!
    # Plot the distribution of number of rated games in common between users
    fig, ax = plt.subplots()
    plt.title("Log10 Number of Rated Games in Common")
    plt.ylabel("2*Counts")
    plt.xlabel("Log10 Number of Rated Games in Common")
    trash = ax.hist(np.log10(shared[~np.isnan(shared)]), bins=2500)

    # Relationship Between Correlation and Number of Games in Common
    # Grab just the upper triangle (above the diagonal) to avoid duplicating scatter points
    trngl_count = np.triu(shared, k=1)
    trngl_corr = np.triu(Xcorr, k=1)

    mask = trngl_count > 1
    trngl_cnt = trngl_count[mask]
    trngl_cor = trngl_corr[mask]
    fig, ax = plt.subplots()
    plt.xlabel("# Shared Games")
    plt.ylabel("Correlation")
    plt.title("# Shared Games vs. Correlations")
    ax.scatter(trngl_cnt, trngl_cor, alpha=0.1)
    ax.set_xlim([-1, 2000])
    ax.axhline(y=0, color="red")

    # Bin the data points by number of shared games and compute mean/var in each bin
    binned_means = stats.binned_statistic(trngl_cnt[trngl_cnt < 150], trngl_cor[trngl_cnt < 150], statistic='mean', bins=100)
    binned_stds = stats.binned_statistic(trngl_cnt[trngl_cnt < 150], trngl_cor[trngl_cnt < 150], statistic='std', bins=100)
    fig, ax = plt.subplots()
    plt.xlabel("# Binned Shared Games")
    plt.ylabel("Mean Correlation")
    plt.title("# Shared Games vs. Correlations")
    ax.plot(binned_means[1][:-1], binned_means[0], color="r", label="$\mu$ in bin")
    upper = binned_means[0] + binned_stds[0]
    lower = binned_means[0] - binned_stds[0]
    plt.fill_between(binned_means[1][:-1], lower, upper, color='b', alpha=0.2, label="$\sigma$ in bin")
    ax.set_ylim([-0.1, 0.5])
    ax.legend(loc="lower right")

##########################
# recommendation system
##########################
# pick all users that voted for a game

def compute_similarity(gameid, simtype="pearson",
                       beta=7, min_shared_votes=4):
    """
    Compute the similarity between every pair of users from the set of all users who rated the game 'gameid'.
    """
    # Restrict to all users who rated this game and all games rated by this set of users
    users = df.loc[df["gameid"] == gameid, "username"].values
    df_sample = df[df["username"].isin(users)]

    # Pivot to matrix (user x game) and then get a correlation marix
    df_ratings = df_sample.pivot_table(index="username", columns="gameid", values="rating")

    if simtype == "pearson":
        sims = df_ratings.transpose().corr(min_periods=min_shared_votes)
    elif simtype == "uncertainty_discounted_pearson":
        # First compute standard pearson correlation
        sims = df_ratings.transpose().corr(min_periods=4)

        # Make a matrix holding number of common games rated between users (i, j)
        X = df_ratings.as_matrix()  # get values as a numpy matrix
        m1 = (~np.isnan(X)).astype(np.float64)  # zeroes for missing ratings, one elsewhere
        shared = np.dot(m1, m1.T)

        # Turn the number of shared games for each user pair into an uncertainty
        uncertainties = np.minimum(shared, beta) / beta

        # Multiply uncertainties by pearson correlation
        sims = sims * uncertainties
    else:
        print("We didn't recognize that similarity metric.")
        return 0

    return sims, df_ratings


def predict_ratings(sims, df_ratings, gameid, k=5):
    """
    Find the k nearest neighbors of a user in terms of the similarities 'sims' and use those users ratings to
    make a prediction for the rating of the game 'gameid' by each user.
    """

    # Identify k nearest neighbors based on similarity
    X = sims.fillna(-1).as_matrix()
    np.fill_diagonal(X, -1)
    k_nearest_indcs = np.argsort(X, axis=1)[:,-k:]  # Return matrix where each row holds index order that would sort that row

    # Center the ratings
    #TODO: try changing center subtraction to mean of KNN,
    #TODO: then adding the user mean. std as well.
    user_means = df_ratings.mean(axis=1)
    df_ratings = df_ratings.subtract(user_means, axis=0)

    # Get ratings from the k nearest neighbors of each user
    ratings = df_ratings.as_matrix()
    neighbor_ratings = np.zeros(
        k_nearest_indcs.shape)  # Each row holds the ratings from the k nearest neighbors for that user
    this_movie_indx = df_ratings.columns.get_loc(gameid)  # Column int pos'n of movie we are predicting ratings for
    for col, neighbor_indxs in enumerate(k_nearest_indcs.T):
        neighbor_ratings[:, col] = ratings[neighbor_indxs, this_movie_indx]

    # Make a prediction based on unweighted average of k nearest neighbors (and "uncenter")
    predicted = user_means + neighbor_ratings.mean(axis=1)  # Average the neighbors ratings and "uncenter"

    return predicted


def analyze_errors(df_ratings, predicted, gameid, silent=False):
    """
    Analyze the errors in predictions for game 'gameid', and plot a comparison to the errors from the simplest
    naive estimator of user rating - the games average rating.
    """

    # Compute naive estimator - the rounded mean rating of the game
    naive = round(df_ratings.loc[:, gameid].values.mean())

    # Compute MSE for our algorithm and for simple estimator
    mse_algo = mean_squared_error(df_ratings.loc[:, gameid].values, predicted)
    mse_naive = mean_squared_error(df_ratings.loc[:, gameid].values, [naive] * len(df_ratings))

    # If running silently just return the mean squared errors of algorithm and simple estimator
    if silent:
        return mse_algo, mse_naive

    # If not running silently, plot errors
    else:
        fig, ax = plt.subplots()
        plt.title("Prediction Errors for Movie ID {}\n MSE={:.2f} \n MSE_BM={:.2f}".format(gameid, mse_algo, mse_naive))
        plt.ylabel("Counts")
        plt.xlabel("Distribution of Prediction Errors")
        bins = np.arange(-4, 4, 1) - 0.5
        trash = ax.hist(df_ratings.loc[:, gameid].values - predicted.round(), bins=bins, color="red", label="algorithm")
        trash = ax.hist(df_ratings.loc[:, gameid].values - naive, bins=bins, color="blue", label="benchmark", alpha=0.3)
        ax.legend(loc="upper right")

        print("Mean Squared Error of algorithm is %.2f" % (mse_algo,))
        print("Mean Squared Error of naive mean game rating estimator is %.2f" % (mse_naive,))
        return ax, mse_algo, mse_naive

gameid = 21892

sims, df_ratings = compute_similarity(gameid)
predicted = predict_ratings(sims, df_ratings, gameid, k=5)
analyze_errors(df_ratings, predicted, gameid)

# analyze k effect
#TODO: try on larger dataset, and sum all games
ks = np.arange(1, 300, 1)
mses_basic = [0]*len(ks)
for idx, k in enumerate(ks):
    predicted_basic = predict_ratings(sims, df_ratings, gameid, k=k)
    mses_basic[idx] = analyze_errors(df_ratings, predicted_basic, gameid, silent=True)[0]

fig, ax = plt.subplots()
plt.title("k neighbors")
plt.ylabel("MSE")
plt.xlabel("Performance vs k for PC")
ax.plot(ks, mses_basic, color="red", label="algorithm")
plt.axhline(analyze_errors(df_ratings, predicted_basic, gameid, silent=True)[1], color="blue", label="game mean rating estimator")
ax.legend(loc="center right")


# analyze beta effect
fig, ax = plt.subplots()
plt.title("k neighbors")
plt.ylabel("MSE")
plt.xlabel("Performance vs k for Weighted PC")

# Plot the weighted PCs vs. k for different values of beta
for beta in np.arange(3, 12, 2):
    sims, df_ratings = compute_similarity(gameid, simtype="uncertainty_discounted_pearson", beta=beta)
    ks = np.arange(5, 25, 1)
    mses = [0]*len(ks)
    for idx, k in enumerate(ks):
        predicted = predict_ratings(sims, df_ratings, gameid, k=k)
        mses[idx] = analyze_errors(df_ratings, predicted, gameid, silent=True)[0]
    ax.plot(ks, mses, label="Discounted PC beta = %i" % (beta,))

# Plot the unweighted PC and naive mean rating estimator vs. k
ax.plot(np.arange(1, 300, 1), mses_basic, color="red", label="Undiscounted PC")
ax.set_xlim([5, 40])
ax.set_ylim([0.47, 0.6])
ax.legend(loc="lower right")