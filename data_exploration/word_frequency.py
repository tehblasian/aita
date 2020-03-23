import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../')
from spark.init_spark import init_spark
from config import AITA_CLEANED_COLLECTION


N = 30
words_to_ignore = ['i', 'u', 'aita', 'like', 'would', 'this', 'it', 'get', 'got', 'the']


def word_frequency(n):
    spark = init_spark(AITA_CLEANED_COLLECTION)
    data_rdd = spark.read.format('mongo').load().rdd

    def filter(row):
        words = []
        list = [word.lower() for sublist in row['header'] for word in sublist] + \
               [word.lower() for sublist in row['content'] for word in sublist]

        for word in list:
            if word not in words_to_ignore:
                words.append(word)

        return words

    flattened_rdd = data_rdd \
        .flatMap(lambda row: filter(row))

    result_rdd = flattened_rdd \
        .map(lambda x: (x, 1)) \
        .reduceByKey(lambda x, y: x+y) \
        .sortBy(lambda x: -x[1])

    result = result_rdd.take(n)
    plot_count(result, n)


def plot_count(data, n):
    words = [row[0] for row in data]
    counts = [row[1] for row in data]

    y_pos = np.arange(len(words))
    plt.figure(figsize=(20, 5))
    plt.bar(y_pos, counts)
    plt.xticks(y_pos, words)

    plt.title('Distribution of the top {} words for posts in r/AITA'.format(n))
    plt.ylabel('Count')
    plt.xlabel('Words')

    plt.show()


if __name__ == '__main__':
    word_frequency(N)
