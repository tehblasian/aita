class RedditPost:
    """Internal representation of a reddit post
    """

    def __init__(self, header, content, created_at):
        self.header = header
        self.content = content
        self.created_at = created_at
