import argparse
import datetime

import requests

from data_preparation.loader import MongoLoader
from data_preparation.models import RedditPost

FLAIRS = ['NO A-HOLES HERE', 'ASSHOLE', 'NOT THE A-HOLE', 'EVERYONE SUCKS']

PUSHSHIFT_URL = 'https://api.pushshift.io/reddit/search/submission/'
REDDIT_URL = 'https://www.reddit.com/api/info.json'


def get_pshift_params(before_epoch):
    return {
        'subreddit': 'amitheasshole',
        'sort': 'desc',
        'size': 100,
        'before': int(before_epoch),
        'score': '>10'
    }


def fetch_pshift(before_epoch):
    r_pshift = requests.get(
        PUSHSHIFT_URL, params=get_pshift_params(before_epoch))
    pshift_response = r_pshift.json()
    return pshift_response


def get_reddit_params(id):
    # id was concatenated with t3_ because of https://www.reddit.com/dev/api/#fullnames
    return {
        'id': 't3_{}'.format(id)
    }


def fetch_reddit_info(post_id):
    """[summary]

    Arguments:
        post_id {string} -- the reddit post's identifier

    Returns:
        2d tuple -- (label, title)
    """
    r_reddit = requests.get(
        REDDIT_URL, params=get_reddit_params(post_id), headers={'User-agent': 'AITA bot'})
    reddit_response = r_reddit.json()
    label = reddit_response['data']['children'][0]['data'].get(
        'link_flair_text', None)
    title = reddit_response['data']['children'][0]['data'].get(
        'title', None)
    return (label, title)


class DataStore:
    def __init__(self):
        self.label_dict = {flair: [] for flair in FLAIRS}
        self.saved_counts = {flair: 0 for flair in FLAIRS}
        self.loader = MongoLoader()

    def is_enough_data(self, min_count):
        for label in self.label_dict:
            if len(self.label_dict[label]) + self.saved_counts[label] < min_count:
                return False

        return True

    def add_data_point(self, label, self_text):
        self.label_dict[label].append(self_text)

    def print_counts(self):
        count_list = [(label, len(selftexts) + self.saved_counts[label])
                      for label, selftexts in self.label_dict.items()]
        for label_count in count_list:
            print(label_count)
    
    def save_posts(self):
        self.loader.save_data(self.label_dict)
        for label in self.label_dict:
            self.saved_counts[label] += len(self.label_dict[label])
            self.label_dict[label] = []


def extract(min_count, before_epoch, end_epoch):
    """ Fetches and stores reddit data

    Arguments:
        min_count {integer} -- This function keeps fetching until all labels have more 
        than <min_count> number of posts associated with it

        before_epoch {integer} -- Starts fetching the data from this specified timestamp and goes backwards
        end_epoch {integer} -- Stops fetching once this timestamp is reached
    Returns:
        a dict with the following format
        {label: [post1, post2], ...}
    """
    ds = DataStore()
    n_batch = 1
    while not ds.is_enough_data(min_count):
        pshift_dict = fetch_pshift(before_epoch)
        for post in pshift_dict['data']:
            self_text = post['selftext']
            label, title = fetch_reddit_info(post['id'])
            created_at = post['created_utc']

            if created_at < end_epoch:
                # Preemptively breaks when posts are older than end_epoch
                ds.save_posts()
                ds.print_counts()
                break

            reddit_post = RedditPost(title, self_text, created_at)
            if label is not None and label.upper() in FLAIRS:
                ds.add_data_point(label.upper(), reddit_post)
            before_epoch = created_at
        else:
            # Continues if inner loop wasn't broken
            print('At Batch #{}'.format(n_batch))
            n_batch += 1
            ds.save_posts()
            ds.print_counts()
            continue
        
        break


    return ds.label_dict
