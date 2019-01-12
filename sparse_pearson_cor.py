
import numpy as np
from scipy import sparse

import time

def nan_sparse_pearson_correlation(x, y=None, min_periods=None):
    '''
    calculate pearson correlation on a NAN sparse matrix
    (a matrix with many NAN values) using scipy.sparse
    :param x: a numpy NAN sparse matrix, rows are variables, columns are observations
    :param y: a numpy NAN sparse vector of a variable
    :param min_periods: Minimum number of observations required per pair of rows to have a valid result
    :return:
    '''
    t = time.time()
    x = x.astype('float32')
    # calculate sparse centered x
    x_submean = x - np.expand_dims(np.nanmean(x, 1), 1)
    x_sp = sparse.csr_matrix(np.nan_to_num(x_submean))

    if y is None:
        # calculate std matrix for normalization
        notnan = sparse.csr_matrix((~np.isnan(x) * 1).astype('uint16'))
        # num_common_games = notnan.dot(notnan.T).toarray() + 1e-100 # reducted

        # conditional std (elements are taken from Vi only from places Vj!=nan)
        # cond_std_mat = np.sqrt((x_sp.power(2)).dot(notnan.T).toarray()/num_common_games)
        cond_std_mat = np.sqrt((x_sp.power(2)).dot(notnan.T).toarray(),dtype='float32')
        var_mat = cond_std_mat * cond_std_mat.T + 1e-38

        del x_submean,cond_std_mat
        # cov = x_sp.dot(x_sp.T).toarray()/num_common_games
        cov = x_sp.dot(x_sp.T).toarray()
    else:
        y_submean = y - np.nanmean(y)
        y_sp = sparse.csr_matrix(np.nan_to_num(y_submean))
        cov = x_sp.dot(y_sp.T).toarray().squeeze()

        y_std = np.sqrt((y_sp.power(2)).sum())
        x_std = np.sqrt((x_sp.power(2).toarray()).sum(1)).squeeze()
        var_mat = x_std * y_std + 1e-38

    PC = cov/var_mat
    if min_periods:
        del var_mat,x_sp,cov
        if y is None:
            num_common_games = notnan.dot(notnan.T).toarray()
        else:
            num_common_games = np.dot(
                (~np.isnan(x) * 1).astype('uint16'),
                (~np.isnan(y) * 1).astype('uint16').T
            )
        PC[num_common_games < min_periods] = np.nan
    return PC

if __name__=='__main__':
    import pandas as pd
    import sqlite3
    import pylab as plt

    connex = sqlite3.connect("bgg_ratings_recommender_deduplicated_toy.db")  # Opens file if exists, else creates file
    sql = "SELECT * FROM data"
    df = pd.read_sql_query(sql, connex)
    connex.close()

    pt = df.pivot_table('rating', 'username', 'gameid').values
    # filter users that rated less than 4 games
    pt=pt[((~np.isnan(pt)).sum(1)>=4)]

    n_users=1000
    n_games_rated = (~np.isnan(pt[:n_users])).sum(1)
    print(n_games_rated)
    CORS=[]
    MAX_COR=[]
    for i in range(n_users):
        user=pt[i]
        cor=nan_sparse_pearson_correlation(pt[np.array(range(len(pt)))!=i],user,min_periods=4)
        CORS.append(cor)
        MAX_COR.append(np.nanmax(cor))
        if i%10==0:
            print(i)

