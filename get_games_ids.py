import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
import time
import os

def get_games_ids():
    def request(msg, slp=1):
        '''A wrapper to make robust https requests.'''
        status_code = 500  # Want to get a status-code of 200
        while status_code != 200:
            sleep(slp)  # Don't ping the server too often
            try:
                r = requests.get(msg)
                status_code = r.status_code
                if status_code != 200:
                    print("Server Error! Response Code %i. Retrying..." % (r.status_code))
            except:
                print("An exception has occurred, probably a momentory loss of connection. Waiting one seconds...")
                sleep(1)
        return r


    # Initialize a DF to hold all our scraped game info
    df_all = pd.DataFrame(columns=["id", "name", "nrate", "pic_url"])
    min_nrate = 1e5
    npage = 1

    # Scrap successful pages in the results until we get down to games with < 1000 ratings each
    while min_nrate > 1000:
        # Get full HTML for a specific page in the full listing of boardgames sorted by
        r = request("https://boardgamegeek.com/browse/boardgame/page/%i?sort=numvoters&sortdir=desc" % (npage,))
        t=time.time()
        soup = BeautifulSoup(r.text, "html.parser")

        # Get rows for the table listing all the games on this page
        table = soup.find_all("tr",
                              attrs={"id": "row_"})  # Get list of all the rows (tags) in the list of games on this page
        df = pd.DataFrame(columns=["id", "name", "nrate", "pic_url"],
                          index=range(len(table)))  # DF to hold this pages results

        # Loop through each row and pull out the info for that game
        for idx, row in enumerate(table):
            # Row may or may not start with a "boardgame rank" link, if YES then strip it
            links = row.find_all("a")
            if "name" in links[0].attrs.keys():
                del links[0]
            gamelink = links[1]  # Get the relative URL for the specific game
            gameid = int(gamelink["href"].split("/")[2])  # Get the game ID by parsing the relative URL
            gamename = gamelink.contents[0]  # Get the actual name of the game as the link contents
            imlink = links[0]  # Get the URL for the game thumbnail
            thumbnail = imlink.contents[0]["src"]

            ratings_str = row.find_all("td", attrs={"class": "collection_bggrating"})[2].contents[0]
            nratings = int("".join(ratings_str.split()))

            df.iloc[idx, :] = [gameid, gamename, nratings, thumbnail]

        # Concatenate the results of this page to the master dataframe
        min_nrate = df["nrate"].min()  # The smallest number of ratings of any game on the page
        print("Page %i scraped, minimum number of ratings was %i" % (npage, min_nrate))
        df_all = pd.concat([df_all, df], axis=0)
        npage += 1

        sleep(max(0,(1-(time.time()-t))))  # Keep the BGG server happy.

    df = df_all.copy()
    # Reset the index since we concatenated a bunch of DFs with the same index into one DF
    df.reset_index(inplace=True, drop=True)
    # Write the DF to .csv for future use
    path=os.path.join(os.getcwd(),"bgg_gamelist.csv")
    df.to_csv(path, index=False, encoding="utf-8")
    df.head()

    print("Number of games with > 1000 ratings is approximately %i" % (len(df),))
    print("Total number of ratings from all these games is %i" % (df["nrate"].sum(),))

    return path


if __name__ == '__main__':
    p = get_games_ids()
