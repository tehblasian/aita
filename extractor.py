import datetime

import requests

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


def fetch_label(post_id):
    r_reddit = requests.get(
        REDDIT_URL, params=get_reddit_params(post_id), headers={'User-agent': 'AITA bot'})
    reddit_response = r_reddit.json()
    label = reddit_response['data']['children'][0]['data'].get(
        'link_flair_text', None)
    return label


class DataStore:
    def __init__(self):
        self.label_dict = {flair: [] for flair in FLAIRS}

    def is_enough_data(self, min_count):
        for label in self.label_dict:
            if len(self.label_dict[label]) < min_count:
                return False

        return True

    def add_data_point(self, label, self_text):
        self.label_dict[label].append(self_text)

    def print_counts(self):
        count_list = [(label, len(selftexts))
                      for label, selftexts in self.label_dict.items()]
        for label_count in count_list:
            print(label_count)


def get_data(min_count, after_epoch=datetime.datetime(2019, 1, 1).timestamp()):
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
    while not ds.is_enough_data(min_count):
        pshift_dict = fetch_pshift(after_epoch)
        for post in pshift_dict['data']:
            self_text = post['selftext']
            label = fetch_label(post['id'])
            if label in FLAIRS:
                ds.add_data_point(label, self_text)
            after_epoch = post['created_utc']

    ds.print_counts()
    return ds.label_dict


if __name__ == '__main__':
    get_data(1)
