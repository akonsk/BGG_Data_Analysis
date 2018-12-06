import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from get_games_ids import get_games_ids
import os

def isnum(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def get_val(tag, term):
    try:
        val = tag.find(term)['value']
        if isnum(val) and not term == 'name':
            if '.' in val:
                val=float(val)
            else:
                val=int(val)
    except:
        val = 'NaN'
    return val


def get_numPlayers_poll_results(item):
    # returns a list of number of voters for each players numbers up to 10,
    # each ordered in Best,Recommended,Not Recommended
    poll_numPlayers_item = item('poll')[0]
    poll_results = []
    if poll_numPlayers_item['name'] == 'suggested_numplayers':
        result_items = poll_numPlayers_item('result')
        poll_results = [it['numvotes'] for it in result_items]
        assert(len(poll_results)%3==0,'poll_results is not a multiple of 3')

    poll_results += ['NaN']*(10*3-len(poll_results))
    poll_results = poll_results[:10*3]

    poll_results += [poll_numPlayers_item('totalvotes')]

    return poll_results


def get_langDep_poll_results(item):
    # returns a list of number of voters for each category,
    # each ordered in Best,Recommended,Not Recommended
    poll = item('poll')[2]
    poll_results = []
    if poll['name'] == 'language_dependence':
        result_items = poll('result')
        poll_results = [it['numvotes'] for it in result_items]
        assert(len(poll_results) == 5,'poll_results length is less than 5')

    poll_results += ['NaN']*(5-len(poll_results))

    return poll_results


def mine_games_info(path_ids=''):

    base = 'http://www.boardgamegeek.com/xmlapi2/thing?id={}&stats=1'
    if not path_ids:
        path_ids = get_games_ids()

    bgg_gamelist = pd.read_csv(path_ids)
    ids = bgg_gamelist['id'].tolist()

    split = 100
    fname = os.path.join(os.getcwd(), 'games.csv')
    df = pd.DataFrame(columns=
      ['id', 'type', 'name', 'yearpublished', 'minplayers', 'maxplayers', 'playingtime',
       'minplaytime', 'maxplaytime', 'minage', 'users_rated', 'average_rating',
       'bayes_average_rating', 'total_owners', 'total_traders', 'total_wanters',
       'total_wishers', 'total_comments', 'total_weights', 'average_weight','publishers']
    + ['poll_{}p_{}'.format(n, cat) for n in range(1, 11) for cat in ['B', 'R', 'NR']]
    + ['totalvotes_numPlayers']
    + ['LD_num_votes_{}'.format(i) for i in range(5)]
    )
    for i in range(0, len(ids), split):
        url = base.format(','.join([str(id) for id in ids[i:i+split]]))
        print('Requesting {}'.format(url))
        t = time.time()
        req = requests.get(url)
        print('request time: {:.2f}s'.format(time.time()-t))
        soup = BeautifulSoup(req.content, 'xml')
        items = soup.find_all('item')
        for i1,item in enumerate(items):
            gid = item['id']
            gtype = item['type']
            gname = get_val(item, 'name')
            gyear = get_val(item, 'yearpublished')
            gmin = get_val(item, 'minplayers')
            gmax = get_val(item, 'maxplayers')
            gplay = get_val(item, 'playingtime')
            gminplay = get_val(item, 'minplaytime')
            gmaxplay = get_val(item, 'maxplaytime')
            gminage = get_val(item, 'minage')
            usersrated = get_val(item.statistics.ratings, 'usersrated')
            avg = get_val(item.statistics.ratings, 'average')
            bayesavg = get_val(item.statistics.ratings, 'bayesaverage')
            owners = get_val(item.statistics.ratings, 'owned')
            traders = get_val(item.statistics.ratings, 'trading')
            wanters = get_val(item.statistics.ratings, 'wanting')
            wishers = get_val(item.statistics.ratings, 'wishing')
            numcomments = get_val(item.statistics.ratings, 'numcomments')
            numweights = get_val(item.statistics.ratings, 'numweights')
            avgweight = get_val(item.statistics.ratings, 'averageweight')
            numPlayers_poll_results = get_numPlayers_poll_results(item)
            langDep_poll_results = get_langDep_poll_results(item)
            publishers = [a['value'] for a in item('link',type='boardgamepublisher')]

            df.loc[i+i1]=[gid, gtype, gname, gyear, gmin, gmax, gplay, gminplay, gmaxplay, gminage,
                             usersrated, avg, bayesavg, owners, traders, wanters, wishers, numcomments,
                             numweights, avgweight, publishers] + numPlayers_poll_results + langDep_poll_results \

            time.sleep(max(0, (1 - (time.time() - t))))

    df.to_csv(fname)

    return fname


if __name__ == '__main__':
    path_gameIDs = os.path.join(os.getcwd(), 'bgg_gamelist.csv')
    p = mine_games_info(path_gameIDs)
