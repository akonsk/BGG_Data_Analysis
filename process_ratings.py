# 27.12.18
# taken from: http://sdsawtelle.github.io/blog/output/boardgamegeek-data-scraping.html
# with slight changes by me

from get_games_ids import request
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

df=pd.read_csv('bgg_gamelist.csv')
df_toy=df.loc[(1000 < df["nrate"]) * (df["nrate"] < 1200), ].copy()

connex = sqlite3.connect("bgg_ratings.db")  # Opens file if exists, else creates file
cur = connex.cursor()  # This object lets us actually send messages to our DB and receive results
sql="SELECT * FROM data;"
df_ratings=pd.read_sql_query(sql, connex)

def check_sanity(df_toy,df_ratings):
    fig, axs = plt.subplots(figsize=[16, 4], nrows=1, ncols=3)
    axs = axs.ravel()
    for idx, game in enumerate(df_toy["id"].sample(n=3, random_state=999)):
        df = df_ratings[df_ratings["gameid"] == game]
        nm = df_toy.loc[df_toy["id"] == game, "name"].values
        __ = axs[idx].hist(df["rating"], bins=10, normed=True)
        axs[idx].set_title("%s (%i)" % (nm, game))
        print("%s Our Data: Mean = %.2f. StdDev = %.2f" % (nm, df["rating"].mean(), df["rating"].std()))

        # Request actual stats from the server to compare with scraped data
        r = request("http://www.boardgamegeek.com/xmlapi2/thing?id=%i&stats=1" % (game,))
        soup = BeautifulSoup(r.text, "xml")
        std = float(soup("stddev")[0]["value"])
        mn = float(soup("average")[0]["value"])
        print("%s Server Stats: Mean = %.2f. StdDev = %.2f" % (nm, mn, std))
        sleep(1.5)

# check_sanity(df_toy,df_ratings)

# Retrieve usernames for all users with < 20 ratings in the DB
thr_ratings = 20
sql = " ".join((
    "SELECT username",
    "FROM (SELECT username, count(*) as freqn FROM data GROUP BY username)",
    "AS tbl WHERE freqn < %i" % thr_ratings,
))
users = pd.read_sql_query(sql, connex)

# Drop all the rows for the above list of users
usrs = ["'" + usr + "'" for usr in users["username"].values]
str_matching = "(" + ",".join(usrs) + ")"  # Construct the string of SQL language
sql = "DELETE FROM data WHERE username IN " + str_matching + ";"
cur.execute(sql)

# Load full data set into RAM
sql = "SELECT * FROM data"
df = pd.read_sql_query(sql, connex)

print('len before drop: %i' % len(df))
df.drop_duplicates(inplace=True)
print('len after drop: %i' % len(df))

# Calculate sparsity of ratings data
max_n_ratings = len(df["gameid"].unique())*len(df["username"].unique())
actual_n_ratings = len(df)
print("Sparsity of Ratings Data is %.2f%%" % (100*actual_n_ratings/max_n_ratings))

try:
    sql = "CREATE INDEX usr_idx ON data (username);"
    cur.execute(sql)
    connex.commit()
    connex.close()
    print('index added successfully')
except Exception as e:
    print(e)

# users that rated more than once - small analysis
gameid = 30
cnts = df.loc[df["gameid"]==gameid,
              ["username", "rating"]].groupby("username").count()  # Num. of entries for this game for each user
multi_users = cnts[cnts["rating"] > 1].index  # list of usernames who rated the game more than once
multis = df[(df["gameid"]==gameid) & (df["username"].isin(multi_users))].sort_values(by="username")
stds = multis.groupby("username")["rating"].std()  # standard deviation of ratings for users with multiple entries
print(stds)
multis = df.loc[df.duplicated(subset=["gameid", "username"], keep=False)]  # rows where (game, user) is duplicate
df_no_multis = df.loc[~df.duplicated(subset=["gameid", "username"], keep=False)] # rows where (game, user) is NOT duplicate
print('how often do users re-rate a game?')
print(len(multis))

# replace each set of duplicates with avg
means = multis.groupby(["gameid", "username"])["rating"].mean().reset_index()
df = pd.concat([df_no_multis, means])  # Add the de-duplicated average rows back in
any(df[["username", "gameid"]].duplicated())  # Verify that we successfully completely deduplicated

# Write the updated values to a new database
connex = sqlite3.connect("bgg_ratings_recommender_deduplicated.db")
df.to_sql("data", connex, index=False)
connex.commit()
connex.close()