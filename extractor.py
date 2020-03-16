import argparse
import datetime
import sys

import requests

from loader import MongoLoader
from models import RedditPost

FLAIRS = ['No A-holes here', 'Asshole', 'Not the A-hole', 'Everyone Sucks']
PUSHSHIFT_URL = 'https://api.pushshift.io/reddit/search/submission/'
REDDIT_URL = 'https://www.reddit.com/api/info.json'


def get_pshift_params(after_epoch):
    return {
        'subreddit': 'amitheasshole',
        'sort': 'asc',
        'size': 100,
        'after': int(after_epoch),
        'score': '>10'
    }


def fetch_pshift(after_epoch):
    r_pshift = requests.get(
        PUSHSHIFT_URL, params=get_pshift_params(after_epoch))
    pshift_response = r_pshift.json()
    return pshift_response


def get_reddit_params(id):
    # id was concatenated with t3_ because of https://www.reddit.com/dev/api/#fullnames
    return {
        'id': f't3_{id}'
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
            if len(self.label_dict[label]) < min_count:
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


def get_data(min_count, after_epoch):
    """ Fetches and stores reddit data

    Arguments:
        min_count {integer} -- This function keeps fetching until all labels have more 
        than <min_count> number of posts associated with it

    Keyword Arguments:
        after_epoch {integer} -- Starts fetching the data from this specified timestamp and onwards
        (default: {datetime.datetime(2019, 1, 1).timestamp()})
    Returns:
        a dict with the following format
        {label: [post1, post2], ...}
    """
    ds = DataStore()
    n_batch = 1
    while not ds.is_enough_data(min_count):
        pshift_dict = fetch_pshift(after_epoch)
        for post in pshift_dict['data']:
            self_text = post['selftext']
            label, title = fetch_reddit_info(post['id'])
            created_at = post['created_utc']
            reddit_post = RedditPost(title, self_text, created_at)
            if label in FLAIRS:
                ds.add_data_point(label, reddit_post)
            after_epoch = created_at
        print(f'At Batch #{n_batch}:')
        n_batch += 1
        ds.save_posts()
        ds.print_counts()

    return ds.label_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts reddit data')
    parser.add_argument(
        '--min_count', help='Data will be fetched in batches until all classes have a <min_count> number of posts associated with it.(Default=100)',
        default=100)
    parser.add_argument(
        '--start_epoch', help='Posts will be fetched started from that epoch. (Default=1546300800)', default=datetime.datetime(2019, 1, 1).timestamp())
    args = parser.parse_args()

    start_epoch = args.start_epoch
    min_count = args.min_count
    get_data(min_count, after_epoch=start_epoch)
