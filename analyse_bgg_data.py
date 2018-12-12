import os
import pandas as pd
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import mplcursors
import matplotlib as mpl
from adjustText import adjust_text

from get_games_info import mine_games_info



MINE_INFO_FROM_BGG = 0
path_gameIDs = os.path.join(os.getcwd(), 'bgg_gamelist.csv')
# path_gameIDs = ''
if MINE_INFO_FROM_BGG:
    data_path = mine_games_info(path_gameIDs)
else:
    data_path = os.path.join(os.getcwd(),'games.csv')

df = pd.read_csv(data_path,index_col=0)

# add percent to players num poll
for i in range(2,11):
    voters = df.as_matrix(['poll_{}p_B'.format(i),
                           'poll_{}p_R'.format(i),
                           'poll_{}p_NR'.format(i)])
    df['{}p_b_percent'.format(i)] = voters[:, 0] / (voters.sum(1))
    df['{}p_r_percent'.format(i)] = voters[:, 1] / (voters.sum(1))
    df['{}p_nr_percent'.format(i)] = voters[:, 2] / (voters.sum(1))
    df['{}p_num_voters'.format(i)] = voters.sum(1)

# add Language dependency average
mat = df.loc[:, ['LD_num_votes_%i' % i for i in range(5)]].as_matrix()
df['LD_average'] = (mat * np.array([np.arange(5) + 1])).sum(1) / mat.sum(1)

# turn publishers back to lists
df['publishers'] = df['publishers'].map(eval)

# n_v=df1.as_matrix(['num_voters'])
# h,l=np.histogram(n_v,50)
# plt.figure()
# plt.bar(l[:-1],h,l[1]-l[0])
numeric_cats = ['yearpublished','maxplaytime','users_rated',\
                'average_rating','total_wanters','total_weights',
                 'average_weight'] + \
               ['poll_{}p_{}'.format(n, cat) for n in range(1, 11) for cat in ['B', 'R', 'NR']]\
             + ['totalvotes_numPlayers'] \
             + ['LD_num_votes_{}'.format(i) for i in range(5)]

# #### N players plots ######
def n_players_plot(df, num_p, metric='geo', Nvoters_threshold = 100):
    '''
    calculate scores for each game according to players best and average rating.
    if a single number of players - plots a graph of games' votes for
    "best/nr [geo,euc] for n players"  Vs. bayes_average_rating, with
    score map as background.
    :param df: dataframe
    :param num_p: number of players
    :return:
    '''
    if not isinstance(num_p,list):
        num_p = [num_p]

    df2 = df[
            np.array([(df['%ip_num_voters' % i] > Nvoters_threshold).tolist()
                      for i in num_p]
                     ).all(0)
            ].reset_index(drop=True)

    if metric == 'euc':
        mat = df2.as_matrix(['%ip_nr_percent' % i for i in num_p])
        d = np.sqrt((mat**2).sum(1))
        ind = np.argsort(d)
        print(df2.loc[ind[:10],['name','bayes_average_rating']])
    elif metric == 'geo':
        # include rating
        mat = df2.as_matrix(['%ip_b_percent' % i for i in num_p])
        # mat = 1 - mat # needed if using nr
        rating = df2['bayes_average_rating'].values
        if len(mat.shape) == 2:
            rating = rating.reshape([-1, 1])

        mat = np.concatenate([mat, rating], 1)
        geo_mean = np.prod(mat, 1) ** (1 / mat.shape[1])
        ind = np.argsort(geo_mean)
        n2print = 100
        df3 = df2.iloc[ind[-n2print:]].reset_index(drop=True)
        print(df3.loc[:, ['name', 'bayes_average_rating']])

        if mat.shape[1]==2:  # plot with score map
            df3.plot.scatter('bayes_average_rating', '%ip_b_percent' % num_p[0],
                             c='k', marker='o')
                             #c=geo_mean[ind[-n2print:]], cmap='jet')
            labels = df3['name']
            mplcursors.cursor(hover=True).connect(
                "add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))

            xmin = df3['bayes_average_rating'].min()
            xmax = df3['bayes_average_rating'].max()
            ymin = df3['%ip_b_percent' % num_p[0]].min()
            ymax = df3['%ip_b_percent' % num_p[0]].max()
            x = np.linspace(xmin, xmax, 100)
            y = np.linspace(ymax, ymin, 100)
            X, Y = np.meshgrid(x, y)
            z3 = np.sqrt(X * Y)
            plt.imshow(z3, cmap='jet',
                       extent=[xmin, xmax, ymin, ymax],
                       aspect=(xmax - xmin) / (ymax - ymin))
            plt.title('%ip games' % num_p[0])


for n in []:
    n_players_plot(df, n)

PLOT=0
if PLOT:
    ### 2/4 players
    # filter below Nvoters (and Nans)
    Nvoters_threshold=100
    df2 = df[(df['2p_num_voters'] > Nvoters_threshold)
             & (df['4p_num_voters'] > Nvoters_threshold)
             ].reset_index(drop=True)
    df2.plot.scatter('2p_nr_percent', '4p_nr_percent',
                     marker='x', s=10)
    labels = df2['name']
    mplcursors.cursor(hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))

    plt.xlabel('2p_nr')
    plt.ylabel('4p_nr')
    plt.title('2/4 players games')

if PLOT:
    # 2-4 players 3D
    Nvoters_threshold=100
    df2 = df[(df['2p_num_voters'] > Nvoters_threshold)
             & (df['3p_num_voters'] > Nvoters_threshold)
             & (df['4p_num_voters'] > Nvoters_threshold)
             ].reset_index(drop=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df2['2p_nr_percent'].astype(float),
               df2['3p_nr_percent'].astype(float),
               df2['4p_nr_percent'].astype(float),
               marker='x', s=10
               )
    labels = df2['name']
    mplcursors.cursor(hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))

    plt.xlabel('2p_nr')
    plt.ylabel('3p_nr')
    ax.set_zlabel('4p_nr')
    plt.title('2-4 players games')

if PLOT:
    # average_weight Vs average_rating
    df.plot.scatter('average_rating', 'average_weight',
                    marker='x', s=10)
    labels = df['name']
    mplcursors.cursor(hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))
    plt.title('average_weight Vs average_rating')

if PLOT:
    # Language dependence Vs average_rating
    mat = df.loc[:, ['LD_num_votes_%i' % i for i in range(5)]].as_matrix()
    df['LD_average'] = (mat * np.array([np.arange(5) + 1])).sum(1) / mat.sum(1)
    df.plot.scatter('average_rating', 'LD_average',
                    marker='x', s=10, c=mat.sum(1), cmap='jet')
    labels = df['name']
    mplcursors.cursor(hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))

# correlation
CORR=0
if CORR:
    cats=['yearpublished','maxplaytime','users_rated', 'stddev',\
          'average_rating','average_weight', 'LD_average'] + \
         ['{}p_{}_percent'.format(n, cat) for n in range(2, 7) for cat in ['nr']]
    cor=df.loc[:,cats].corr(min_periods=30)
    f,ax=plt.subplots()
    plt.imshow(np.abs(cor),interpolation='none',cmap='jet')
    plt.colorbar()
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats,rotation=90)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats,rotation=0)

LIST=0
if LIST:
    # list of games good for N players
    df1=df[(df['6p_b_percent'] > 0.4)&
           (df['maxplaytime'] < 60) &
           (df['yearpublished'] > 201)
    ]
    df1.reset_index(drop=True)
    print(
        df1.sort_values('average_rating')\
              [['name', 'average_rating']]\
              [-30:]
    )
if PLOT:
    # plot related games
    name = 'dominion'
    df1 = df[[name in gamename.lower() for gamename in df['name'].values]
           ].reset_index(drop=True)
    df1.plot.scatter('average_rating','average_weight',
                     c=df1['users_rated'],cmap='jet',
                     norm=mpl.colors.LogNorm())
    labels = df1['name'].values

    PUBLISH=1
    if PUBLISH:
        fig=plt.gcf()
        fig.set_size_inches(15.36,8)
        _ = fig.canvas.manager.window.wm_geometry('+-10+0')
        texts=[plt.text(df1['average_rating'][i],df1['average_weight'][i],txt,
                     fontsize=8) for i, txt in enumerate(labels)]
        adjust_text(texts,arrowprops=dict(arrowstyle='-', color='gray'),
                    autoalign='xy')
    else:
        mplcursors.cursor(hover=True).connect(
            "add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))

if PLOT:
    # print games of publisher
    publishers = df['publishers'].tolist()
    publishers = [[st.lower() for st in l] for l in publishers]
    inds = [any(['rio grande' in st for st in l]) for l in publishers]
    df2 = df[inds].reset_index(drop=True)
    df2 = df2.sort_values('average_rating').reset_index(drop=True)
    for a in df2.loc[-20:,['name','average_rating']].values:
        print (a)