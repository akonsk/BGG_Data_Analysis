import time
import pandas as pd
import requests
from bs4 import BeautifulSoup


def isnum(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def get_val(tag, term):
    try:
        val = tag.find(term)['value'].encode('ascii', 'ignore')
        if isnum(val) and not term == 'name':
            if '.' in val.decode('utf-8'):
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

    return poll_results

base = 'http://www.boardgamegeek.com/xmlapi2/thing?id={}&stats=1'
path_ids='C:/Users/kfir/PycharmProjects/untitled/bgg_gamelist.csv'
bgg_gamelist = pd.read_csv(path_ids)
ids = bgg_gamelist['id'].tolist()

split = 100
fname = 'games.csv'
df = pd.DataFrame(columns=('id', 'type', 'name', 'yearpublished', 'minplayers', 'maxplayers', 'playingtime',
                           'minplaytime', 'maxplaytime', 'minage', 'users_rated', 'average_rating',
                           'bayes_average_rating', 'total_owners', 'total_traders', 'total_wanters',
                           'total_wishers', 'total_comments', 'total_weights', 'average_weight',
                           'poll_1p_B','poll_1p_R','poll_1p_NR',
                           'poll_2p_B', 'poll_2p_R', 'poll_2p_NR',
                           'poll_3p_B', 'poll_3p_R', 'poll_3p_NR',
                           'poll_4p_B', 'poll_4p_R', 'poll_4p_NR',
                           'poll_5p_B', 'poll_5p_R', 'poll_5p_NR',
                           'poll_6p_B', 'poll_6p_R', 'poll_6p_NR',
                           'poll_7p_B', 'poll_7p_R', 'poll_7p_NR',
                           'poll_8p_B', 'poll_8p_R', 'poll_8p_NR',
                           'poll_9p_B', 'poll_9p_R', 'poll_9p_NR',
                           'poll_10p_B', 'poll_10p_R', 'poll_10p_NR',
                           )
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


        df.loc[i+i1]=[gid, gtype, gname, gyear, gmin, gmax, gplay, gminplay, gmaxplay, gminage,
                         usersrated, avg, bayesavg, owners, traders, wanters, wishers, numcomments,
                         numweights, avgweight] + numPlayers_poll_results

        time.sleep(max(0, (1 - (time.time() - t))))

df.to_csv(fname)