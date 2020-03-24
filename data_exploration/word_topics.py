from pyspark.sql import Row
from pyspark.ml.feature import CountVectorizer
from pyspark.mllib.clustering import LDA
from pyspark.mllib.linalg import Vectors


import sys
sys.path.append('../')
from spark.init_spark import init_spark
from config import AITA_CLEANED_COLLECTION

NUM_TOPICS = 15
NUM_WORDS_PER_TOPICS = 10


def word_topics(num_topics, num_words_per_topics):
    """Generates topics from word clusters.

    Arguments:
        num_topics {integer} -- Number of topics to infer
        num_words_per_topics {integer} -- Number of terms to collect for each topic

    Returns:
        None
    """
    spark = init_spark(AITA_CLEANED_COLLECTION)
    data_rdd = spark.read.format('mongo').load().rdd

    preprocessed_rdd = data_rdd\
        .flatMap(lambda row: [row['header'].lower().split(' ') + row['content'].lower().split(' ')]) \
        .zipWithIndex() \
        .map(lambda x: Row(index=x[1], words=x[0]))

    preprocessed_df = spark.createDataFrame(preprocessed_rdd)

    cv = CountVectorizer(inputCol='words', outputCol='vectors')
    model = cv.fit(preprocessed_df)
    vector_df = model.transform(preprocessed_df)

    corpus = vector_df.select('index', 'vectors').rdd.map(lambda x: [x[0], Vectors.fromML(x[1])]).cache()

    lda_model = LDA.train(corpus, k=num_topics, maxIterations=100, optimizer='online')
    vocab_array = model.vocabulary

    topic_indices = spark.sparkContext.parallelize(lda_model.describeTopics(maxTermsPerTopic=num_words_per_topics))

    def vector_id_to_word(topic):
        terms = topic[0]
        weights = topic[1]
        result = []
        for i in range(num_words_per_topics):
            result.append((vocab_array[terms[i]], weights[i]))
        return result

    topics = topic_indices.map(lambda topic: vector_id_to_word(topic)).collect()

    for i in range(len(topics)):
        print('Topic {}:'.format(i))
        for item in topics[i]:
            print(item)
        print('\n')


if __name__ == '__main__':
    word_topics(NUM_TOPICS, NUM_WORDS_PER_TOPICS)
