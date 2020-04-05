from pymongo import MongoClient


class MongoLoader:
    def __init__(self):
        client = MongoClient('localhost', 27017)
        db = client['aitaDB']
        self.collection = db['extractedinfo']

    def save_data(self, label_dict):
        """Saves data to mongo db
        
        Arguments:
            label_dict {dict} -- dictionnary mapping a label to a list of RedditPosts
        """
        db_dict = [self._convert_to_db_dict(label, post) for label in label_dict for post in label_dict[label]]
        self.collection.insert_many(db_dict)
    
    def _convert_to_db_dict(self, label, reddit_post):
        """Used to convert data into a more db-friendly format
        
        Arguments:
            label {string} -- The flair of the post
            reddit_post {RedditPost} -- The reddit post itself
        """
        return {
            'label': label,
            'header': reddit_post.header,
            'content': reddit_post.content,
            'created_at': reddit_post.created_at
        }
