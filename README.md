# AITA - SOEN 499 Term Project

## Team
- Chirac Manoukian (40028500)
- Jeremiah-David Wreh (40028325)
- Justin Cotarla (40027609)
- Karen Bie (40018058)

## ABSTRACT

Am I the A-hole (AITA) is an online community on the popular social news website [Reddit](https://www.reddit.com). Users post descriptions of personal experiences asking the community to vote and determine whether or not they were the a-hole from the situation.


We will collect a large amount of posts from the [Am I the A-hole](https://www.reddit.com/r/AmItheAsshole/) subreddit and analyze the dataset in order to predict what label will be assigned to a particular scenario as to find out who is in the wrong. We will discuss the performance of our model and the methods we used to pre-process the data.


## INTRODUCTION

### Context

Am I the A-hole is a subreddit (i.e. a community on Reddit) where users to submit posts describing a personal experience to then have the community decide on whether or not they were right under their given situation. When a post is submitted to the subreddit, community members may leave comments on it with their reactions and opinions on the described scenario from the post. Members also include a judgement label with their comment which they deem most appropriate for the situation, chosen from one of the following:

- You're the A-hole (YTA)
- You're Not the A-hole (NTA)
- Everyone Sucks Here (ESH)
- No A-holes here (NAH)
- Not Enough Info (INFO) 


Members may as well upvote comments from the threads to signify that they agree with the commenter’s view. The judgement label from the comment with the most upvotes is the final label attributed to the original post as a way to indicate the consensus.


### Objectives

The objective of our project is to design a classification model to accurately predict which final judgement label would most likely reflect the general agreement of the AITA community in response to a given scenario. In the current state of the subreddit, a winning verdict is determined after 18 hours. With our project, a verdict would be more quickly brought to the table for conflicted users to find out if they had be_en in the wrong, and to be sooner rid of their worries. 

## Related works

_Liang, H., Sun, X., Sun, Y. et al. J Wireless Com Network (2017) 2017: 211. https://doi.org/10.1186/s13638-017-0993-1_

## Materials and Methods

### The dataset

Pushshift is a big data storage project that copies Reddit submissions and comments. Its goal is to provide an API so that anyone can access this data on a large scale. Using this, we will extract all the posts submitted to AITA including their labels. Each item in the resultant dataset will have the following attributes:

- Title
- Content
- Label


### Technologies and Algorithms
The core of our project will be written in Python. We will use natural language processing techniques to first clean up our data. Because our data will consist of written texts, it should be filtered and processed with normalization to enhance our results. Among others, we will apply lemmatization and stop words removal. We will also explore the usage of word embeddings (e.g. word2vec, fastText) to understand the sentiment of any given post, as this information may help increase the performance of our model. We will use Apache Spark to parallelize computations on the data. 