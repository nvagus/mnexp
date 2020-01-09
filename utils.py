import json
import logging
import os
import pickle
import threading

import keras
import numpy as np
import scipy.sparse as ss
import tensorflow as tf

import models

open = tf.gfile.GFile


def logging_history(history: keras.callbacks.History):
    try:
        history = sorted(history.history.items(), key=lambda x: x[0])
        logs = ['{}: {:.4f}'.format(k, np.mean(v) if k.startswith('val') else v[-1]) for k, v in history]
        logging.info('[*] {}'.format('\t'.join(logs)))
    except:
        pass


def logging_evaluation(evaluations):
    try:
        logs = ['{}: {:.4f}'.format(k, v) for k, v in sorted(evaluations.items())]
        logging.info('[*] {}'.format('\t'.join(logs)))
    except:
        pass


def load_textual_embedding(path, dimension):
    logging.info('[+] loading embedding data from {}'.format(os.path.split(path)[-1]))

    data = {
        int(r[-2]): np.array([float(x) for x in r[-1].split(' ')], dtype=np.float32)
        for r in [s.strip().split('\t') for s in open(path)]
        if r[-1].count(' ') == dimension - 1
    }

    embedding_matrix = np.array(
        [
            data[i] if i in data else
            np.zeros(dimension, dtype=np.float32)
            for i in range(max(data.keys()) + 1)
        ]
    )

    logging.info('[-] found {} vectors from {}'.format(len(data), os.path.split(path)[-1]))
    return embedding_matrix


def load_sparse_matrix(path):
    logging.info('[+] loading sparse matrix from {}'.format(os.path.split(path)[-1]))
    with open(path, 'rb') as file:
        zip = dict(np.load(file))
    result = {k: v for k, v in zip.items() if k not in ['data', 'indices', 'indptr']}
    result['matrix'] = ss.csr_matrix((zip['data'], zip['indices'], zip['indptr']))
    logging.info('[+] found sparse matrix from {}'.format(os.path.split(path)[-1]))
    return result


def load_model(paths) -> keras.Model:
    json_path, weight_path = paths
    with open(json_path, 'r') as file:
        model = keras.models.model_from_json(json.load(file), models.__dict__)
    with open(weight_path, 'rb') as file:
        model.set_weights(pickle.load(file))
    return model


def save_model(paths, model: keras.Model):
    json_path, weight_path = paths
    with open(json_path, 'w') as file:
        json.dump(model.to_json(), file)
    with open(weight_path, 'wb') as file:
        pickle.dump(model.get_weights(), file, protocol=pickle.HIGHEST_PROTOCOL)


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_true, y_pred)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


class ThreadSafeIter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)

    def next(self):
        with self.lock:
            return next(self.it)


def interactive_console(local):
    import code
    code.interact(local=local)


verticals = {'N/A': 0, 'autos': 1, 'entertainment': 2, 'finance': 3, 'foodanddrink': 4, 'health': 5, 'kids': 6,
             'lifestyle': 7, 'movies': 8, 'music': 9, 'news': 10, 'sports': 11, 'travel': 12, 'tv': 13, 'video': 14,
             'weather': 15}

subverticals = {'N/A': 0, 'ads-emmyawards': 1, 'ads-foodforhealth': 2, 'ads-latingrammys': 3, 'ads-lung-health': 4,
                'ads-tourism-australia': 5, 'adventure sports': 6, 'adventure-sports': 7, 'animal': 8, 'animals': 9,
                'autos-autoinsurance': 10, 'autos-videos': 11, 'autosbdc': 12, 'autosbuyersguide': 13,
                'autosbuying': 14, 'autoscartech': 15, 'autosclassics': 16, 'autoscompact': 17,
                'autosconvertibles': 18, 'autosdetroit': 19, 'autosenthusiasts': 20, 'autosfueleconomy': 21,
                'autosgeneva': 22, 'autosgreen': 23, 'autoshows': 24, 'autoshybrids': 25, 'autoslarge': 26,
                'autoslosangeles': 27, 'autosluxury': 28, 'autosmidsize': 29, 'autosmotorbikes': 30,
                'autosmotorcycles': 31, 'autosnews': 32, 'autosnewyork': 33, 'autosownership': 34, 'autosparis': 35,
                'autospassenger': 36, 'autospebblebeach': 37, 'autosracing': 38, 'autosresearch': 39,
                'autosresearchguides': 40, 'autosreview': 41, 'autossema': 42, 'autossports': 43, 'autossuvs': 44,
                'autostrucks': 45, 'autosusedcars': 46, 'autosvans': 47, 'autosvideos': 48, 'awards': 49,
                'awards-video': 50, 'baseball_mlb': 51, 'baseball_mlb_videos': 52, 'basketball': 53,
                'basketball_nba': 54, 'basketball_nba_videos': 55, 'basketball_ncaa': 56, 'basketball_ncaa_videos': 57,
                'beerandcider': 58, 'beverages': 59, 'casual': 60, 'causes-poverty': 61, 'celebhub': 62,
                'celebrity': 63, 'celebritynews': 64, 'cma-awards': 65, 'cocktails': 66, 'columnists': 67,
                'comedy': 68, 'cookingschool': 69, 'diet': 70, 'downtime': 71, 'entertainment-celebrity': 72,
                'entertainment-gallery': 73, 'esports': 74, 'fantasy': 75, 'finance-auto-insurance': 76,
                'finance-billstopay': 77, 'finance-career-education': 78, 'finance-companies': 79,
                'finance-credit': 80, 'finance-healthcare': 81, 'finance-home-loans': 82,
                'finance-insidetheticker': 83, 'finance-insurance': 84, 'finance-mutual-funds': 85,
                'finance-real-estate': 86, 'finance-retirement': 87, 'finance-savemoney': 88,
                'finance-saving-investing': 89, 'finance-small-business': 90, 'finance-spending-borrowing': 91,
                'finance-startinvesting': 92, 'finance-taxes': 93, 'finance-technology': 94, 'finance-top-stocks': 95,
                'finance-top-stories': 96, 'finance-video': 97, 'financenews': 98, 'fitness': 99, 'foodchristmas': 100,
                'foodculture': 101, 'foodnews': 102, 'foodrecipes': 103, 'foodtips': 104, 'football_cfl': 105,
                'football_ncaa': 106, 'football_ncaa_videos': 107, 'football_nfl': 108, 'football_nfl_videos': 109,
                'fun': 110, 'games': 111, 'gaming': 112, 'geneva-autoshow2018': 113, 'goldenglobes-video': 114,
                'golf': 115, 'grab-bag': 116, 'halloween': 117, 'health-cancer': 118, 'health-news': 119,
                'health-osteoporosis': 120, 'health-pain': 121, 'healthagingwell': 122, 'healthy-heart': 123,
                'healthyliving': 124, 'humor': 125, 'humorpolitics': 126, 'icehockey_nhl': 127,
                'icehockey_nhl_videos': 128, 'lifestyle': 129, 'lifestyle-news-feature': 130, 'lifestyle-wedding': 131,
                'lifestyleanimals': 132, 'lifestylebeauty': 133, 'lifestylebuzz': 134, 'lifestylecareer': 135,
                'lifestylecelebstyle': 136, 'lifestylecleaningandorganizing': 137, 'lifestyledeathday': 138,
                'lifestyledecor': 139, 'lifestyledesign': 140, 'lifestyledidyouknow': 141, 'lifestylediy': 142,
                'lifestylefamily': 143, 'lifestylefamilyandrelationships': 144, 'lifestylefamilyfun': 145,
                'lifestylefamilyrelationships': 146, 'lifestylefashion': 147, 'lifestyleges': 148,
                'lifestyleholidayshoppingdeals': 149, 'lifestylehomeandgarden': 150, 'lifestylehoroscope': 151,
                'lifestylehoroscopefish': 152, 'lifestylelovesex': 153, 'lifestylemindandsoul': 154,
                'lifestyleparenting': 155, 'lifestylepets': 156, 'lifestylepetsanimals': 157,
                'lifestylerelationships': 158, 'lifestyleroyals': 159, 'lifestyleshoppinghomegarden': 160,
                'lifestylesmartliving': 161, 'lifestylestyle': 162, 'lifestyletravel': 163, 'lifestylevideo': 164,
                'lifestyleweddings': 165, 'lifestylewhatshot': 166, 'lifestylewhatstrending': 167,
                'lifetsylepets': 168, 'markets': 169, 'medical': 170, 'mentalhealth': 171, 'mma': 172, 'mmaufc': 173,
                'more_sports': 174, 'movienews': 175, 'movies': 176, 'movies-awards': 177, 'movies-celebrity': 178,
                'movies-gallery': 179, 'movies-golden-globes': 180, 'movies-oscars': 181, 'movies-virals': 182,
                'movievideo': 183, 'music-awards': 184, 'music-celebrity': 185, 'music-gallery': 186,
                'music-grammys': 187, 'musicnews': 188, 'musicvideos': 189, 'nerdcore': 190, 'news': 191,
                'news-feature': 192, 'news-featured': 193, 'news-videos': 194, 'newsafrica': 195, 'newsbusiness': 196,
                'newscrime': 197, 'newsfactcheck': 198, 'newsgoodnews': 199, 'newslocal': 200, 'newsnational': 201,
                'newsoffbeat': 202, 'newsopinion': 203, 'newsphotos': 204, 'newspolitics': 205, 'newsscience': 206,
                'newsscienceandtechnology': 207, 'newstrends': 208, 'newsuk': 209, 'newsus': 210, 'newsvideo': 211,
                'newsvideos': 212, 'newsweather': 213, 'newsworld': 214, 'nutrition': 215, 'other': 216,
                'outdoors': 217, 'partiesandentertaining': 218, 'people-places': 219, 'peopleandplaces': 220,
                'personalfinance': 221, 'pets-search': 222, 'photos': 223, 'politicsvideo': 224, 'popculture': 225,
                'pregnancyparenting': 226, 'quickandeasy': 227, 'racing': 228, 'recipes': 229,
                'restaurantsandnews': 230, 'retirement': 231, 'reviews': 232, 'reviewsandreservations': 233,
                'rvs-trailers': 234, 'science': 235, 'seasonal': 236, 'seasonalvideos': 237, 'sexualhealth': 238,
                'shop-all': 239, 'shop-apparel': 240, 'shop-books-movies-tv': 241, 'shop-computers-electronics': 242,
                'shop-holidays': 243, 'shop-home-goods': 244, 'shop-toys': 245, 'shopping': 246, 'soccer': 247,
                'soccer_bund': 248, 'soccer_chlg': 249, 'soccer_epl': 250, 'soccer_fifa_wc': 251,
                'soccer_fifa_wwc': 252, 'soccer_fran_ligue_one': 253, 'soccer_lib': 254, 'soccer_liga': 255,
                'soccer_mls': 256, 'soccer_seria': 257, 'soccer_uefa_europa': 258, 'soccer_uefa_nl': 259,
                'soccer_videos': 260, 'spendingandborrowing': 261, 'sports': 262, 'sports_news': 263, 'spotlight': 264,
                'strength': 265, 'technologyinvesting': 266, 'tennis': 267, 'tennis_intl': 268, 'tipsandtricks': 269,
                'topnews': 270, 'travel': 271, 'travel-accessible': 272, 'travel-adventure-travel': 273,
                'travel-points-rewards': 274, 'travelarticle': 275, 'travelnews': 276, 'traveltips': 277,
                'traveltripideas': 278, 'travelvideos': 279, 'tunedin': 280, 'tv': 281, 'tv-awards': 282,
                'tv-celebrity': 283, 'tv-emmys': 284, 'tv-gallery': 285, 'tv-golden-globes': 286, 'tv-oscars': 287,
                'tv-recaps': 288, 'tv-reviews': 289, 'tvnews': 290, 'tvvideos': 291, 'veggie': 292, 'video': 293,
                'video-be-prepared': 294, 'videos': 295, 'viral': 296, 'voices': 297, 'watch': 298,
                'weathertopstories': 299, 'weight-loss': 300, 'weightloss': 301, 'wellness': 302, 'wine': 303,
                'wines': 304, 'wonder': 305, 'yearinoffbeatgoodnews': 306}


def get_vertical(name):
    if name in verticals:
        return verticals[name]
    else:
        return 0


def get_subvertical(name):
    if name in subverticals:
        return subverticals[name]
    else:
        return 0
