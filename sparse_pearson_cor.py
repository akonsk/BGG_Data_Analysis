
import numpy as np
from scipy import sparse

import time

def nan_sparse_pearson_correlation(x, min_periods=None):
    '''
    calculate pearson correlation on a NAN sparse matrix
    (a matrix with many NAN values) using scipy.sparse
    :param x: a numpy NAN sparse matrix, rows are variables, columns are observations
    :param min_periods: Minimum number of observations required per pair of rows to have a valid result
    :return:
    '''
    t = time.time()
    x = x.astype('float32')
    # calculate sparse centered x
    x_submean = x - np.expand_dims(np.nanmean(x, 1), 1)
    x_sp = sparse.csr_matrix(np.nan_to_num(x_submean))

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

    PC = cov/var_mat
    if min_periods:
        t=time.time()
        del var_mat,x_sp,cov
        num_common_games = notnan.dot(notnan.T).toarray()
        PC[num_common_games < min_periods] = np.nan
    return PC

if __name__=='__main__':
    x=np.array([[1,2,np.nan,np.nan,2],[np.nan,2,np.nan,3,np.nan]])
    nan_sparse_pearson_correlation(x)