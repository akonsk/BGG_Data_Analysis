import pandas as pd
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import mplcursors
from get_games_info import mine_games_info
import os


MINE_INFO_FROM_BGG = 0
path_gameIDs = os.path.join(os.getcwd(), 'bgg_gamelist.csv')
if MINE_INFO_FROM_BGG:
    data_path = mine_games_info(path_gameIDs)
else:
    data_path = 'C:/Users/kfir/PycharmProjects/BGG_Data_Analysis/games.csv'

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

# n_v=df1.as_matrix(['num_voters'])
# h,l=np.histogram(n_v,50)
# plt.figure()
# plt.bar(l[:-1],h,l[1]-l[0])

##### N players plots ######
def n_players_plot(N,df):
    hover = True  # label show on hovering
    Nvoters_threshold=100
    cat='b'

    # filter below Nvoters (and Nans)
    games = df.loc[df['{}p_num_voters'.format(N)] > Nvoters_threshold]
    games = games.reset_index(drop=True)

    # i_nr = np.argsort(games['{}p_nr_percent'.format(N)].tolist())
    # print(games.loc[i_nr[:10], ['name']])

    games.plot.scatter('bayes_average_rating','{}p_{}_percent'.format(N,cat),
                       marker='x', s=10
                       )
    labels = games['name'].tolist()
    mplcursors.cursor(hover=hover).connect(
        "add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))

    plt.xlabel('bayes_average_rating'.format(N))
    plt.ylabel('{}p_{}_percent'.format(N,cat))
    plt.title('{} players games'.format(N))

for n in [4]:
    n_players_plot(n, df)

### 2/4 players
PLOT=0
if PLOT:
    # filter below Nvoters (and Nans)
    Nvoters_threshold=100
    df2 = df[(df['2p_num_voters'] > Nvoters_threshold) & (df['4p_num_voters'] > Nvoters_threshold)]
    df2 = df2.reset_index(drop=True)
    plt.figure()
    # mat=df.as_matrix(['2p_nr_percent','4p_nr_percent'])
    # df2=df.loc[(~np.isnan(mat)).all(1),['name','2p_nr_percent','4p_nr_percent']]
    # df2=df.reset_index(drop=True)
    plt.scatter(df2['2p_nr_percent'].astype(float), df2['4p_nr_percent'].astype(float),
                marker='x', s=10
                )
    labels = df2['name']
    mplcursors.cursor(hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))

    plt.xlabel('2p_nr')
    plt.ylabel('4p_nr')
    plt.title('2/4 players games')

    # 2-4 players 3D
    Nvoters_threshold=100
    df2 = df[(df['2p_num_voters'] > Nvoters_threshold) & (df['3p_num_voters'] > Nvoters_threshold) & (df['4p_num_voters'] > Nvoters_threshold)]
    df2 = df2.reset_index(drop=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df2['2p_nr_percent'].astype(float), df2['3p_nr_percent'].astype(float), df2['4p_nr_percent'].astype(float),
                marker='x', s=10
                )
    labels = df2['name']
    mplcursors.cursor(hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))

    plt.xlabel('2p_nr')
    plt.ylabel('3p_nr')
    ax.set_zlabel('4p_nr')
    plt.title('2-4 players games')

    ### list of games that are most suitable to 2-5 players
    num_p=[3,4]
    Nvoters_threshold=100
    df2 = df[
            np.array([(df['%ip_num_voters' % i] > Nvoters_threshold).tolist() for i in num_p]).all(0)
    ]
    df2 = df2.reset_index(drop=True)

    mat = df2.as_matrix(['%ip_nr_percent' % i for i in num_p])
    d = np.sqrt((mat**2).sum(1))
    ind = np.argsort(d)
    print(df2.loc[ind[:10],['name','bayes_average_rating']])

### print list
df1=df[(df['2p_b_percent']>0.4 )&
       (df['maxplaytime']<90) &
       (df['yearpublished']>2012)
]
df1.reset_index(drop=True)
print(
    df1.sort_values('average_rating')\
          [['name','average_rating']]\
          [-15:]
)


