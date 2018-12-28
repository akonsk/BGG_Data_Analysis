# 27.12.18
# taken from: http://sdsawtelle.github.io/blog/output/boardgamegeek-data-scraping.html
# with slight changes by me

from get_games_ids import request
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from time import sleep

import sqlite3

# Create the database and make a cursor to talk to it.
connex = sqlite3.connect("bgg_ratings.db")  # Opens file if exists, else creates file
cur = connex.cursor()  # This object lets us actually send messages to our DB and receive results

# read gamelist file
df=pd.read_csv('bgg_gamelist.csv')
df_toy=df.loc[(200 < df["nrate"]) & (df["nrate"] < 1e10), ].copy()
# Prepare a "# of FULL pages of ratings" column to track # API calls needed
df_toy["nfullpages"] = (df_toy["nrate"]-50).apply(round, ndigits=-2)/100  # Round DOWN to nearest 100

#############################################################
# Gathering all ratings from all games in toy data set
#############################################################

# Get ratings page-by-page for all games, but do it in chunks of 150 games
for nm, grp in df_toy.groupby(np.arange(len(df_toy)) // 150):
    # Initialize a DF to hold all the responses for this chunk of games
    df_ratings = pd.DataFrame(columns=["gameid", "username", "rating"], index=range(grp["nrate"].sum() + 100000))

    # Initialize indices for writing to the ratings dataframe
    dfidx_start = 0
    dfidx = 0

    # For this group of games, make calls until all FULL pages of every game have been pulled
    pagenum = 1
    while len(grp[grp["nfullpages"] > 0]) > 0:
        # Get a restricted DF with only still-active games (have ratings pages left)
        active_games = grp[grp["nfullpages"] > 0]

        # Set the next chunk of the DF "gameid" column using the list of game IDs
        id_list = []
        for game in active_games["id"]:
            id_list += [game] * 100
        dfidx_end = dfidx_start + len(active_games) * 100
        df_ratings.iloc[dfidx_start:dfidx_end, df_ratings.columns.get_loc("gameid")] = id_list

        # Make the request with the list of all game IDs that have ratings left
        id_strs = [str(gid) for gid in active_games["id"]]
        gameids = ",".join(id_strs)
        sleep(1.5)  # Keep the server happy
        r = request("http://www.boardgamegeek.com/xmlapi2/thing?id=%s&ratingcomments=1&page=%i" % (gameids, pagenum))
        soup = BeautifulSoup(r.text, "xml")
        comments = soup("comment")
        #         print("Response status was %i - number of ratings retrieved was %i" % (r.status_code, len(comments)))

        # Parse the response and assign it into the dataframe
        l1 = [0] * len(active_games) * 100
        l2 = [0] * len(active_games) * 100
        j = 0
        for comm in comments:
            l1[j] = comm["username"]
            l2[j] = float(comm["rating"])
            j += 1
        df_ratings.iloc[dfidx_start:dfidx_end, df_ratings.columns.get_loc("username")] = l1
        df_ratings.iloc[dfidx_start:dfidx_end, df_ratings.columns.get_loc("rating")] = l2

        grp["nfullpages"] -= 1  # Decrement the number of FULL pages of each game id
        dfidx_start = dfidx_end
        pagenum += 1
        print("pagenum updated to %i" % (pagenum,))

    # Strip off the empty rows
    df_ratings = df_ratings.dropna(how="all")
    # Write this batch of all FULL pages of ratings for this chunk of games to the DB
    df_ratings.to_sql(name="data", con=connex, if_exists="append", index=False)
    print("Processed ratings for batch #%i of games." % (nm))

#############################################################
# Request the final partial page of ratings for each game
#############################################################
# Restore the correct number of FULL pages
df_toy["nfullpages"] = (df_toy["nrate"] - 50).apply(round,
                                                    ndigits=-2) / 100  # Round DOWN to nearest 100, then divide by 100

# Initialize a DF to hold all the responses over all the chunks of games
df_ratings = pd.DataFrame(columns=["gameid", "username", "rating"], index=range(len(df_toy) * 100))

# Initialize indices for writing to the ratings dataframe
dfidx_start = 0
dfidx = 0

# Loop through game-by-game and request the final page of ratings for each game
for idx, row in df_toy.iterrows():
    # Get the game ID and the last page number to request
    pagenum = row["nfullpages"] + 1
    gameid = row["id"]

    # Make the request for just the last page of ratings of this game
    sleep(1.5)  # Keep the server happy
    r = request("http://www.boardgamegeek.com/xmlapi2/thing?id=%i&ratingcomments=1&page=%i" % (gameid, pagenum))
    soup = BeautifulSoup(r.text, "xml")
    comments = soup("comment")
    #         print("Response status was %i - length of comments is %i" % (r.status_code, len(comments)))

    # Set the next chunk of the DF "gameids" column with this gameid
    id_list = [gameid] * len(comments)
    dfidx_end = dfidx_start + len(comments)
    df_ratings.iloc[dfidx_start:dfidx_end, df_ratings.columns.get_loc("gameid")] = id_list

    # Parse the response and assign it into the dataframe
    l1 = [0] * len(comments)
    l2 = [0] * len(comments)
    j = 0
    for comm in comments:
        l1[j] = comm["username"]
        l2[j] = float(comm["rating"])
        j += 1
    df_ratings.iloc[dfidx_start:dfidx_end, df_ratings.columns.get_loc("username")] = l1
    df_ratings.iloc[dfidx_start:dfidx_end, df_ratings.columns.get_loc("rating")] = l2

    dfidx_start = dfidx_end  # Increment the starting index for next round

    if idx % 100 == 0:
        print("Finished %i games" % idx)

# Strip off the empty rows
df_ratings = df_ratings.dropna(how="all")

# Write this final batch of all partial pages of ratings for this chunk of games to the DB
df_ratings.to_sql(name="data", con=connex, if_exists="append", index=False)

# Save our changes and close the connection
connex.commit()
connex.close()

