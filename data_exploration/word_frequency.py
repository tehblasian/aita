import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql.types import Row


import sys
sys.path.append('../')
from spark.init_spark import init_spark
from config import AITA_CLEANED_COLLECTION
from constants import NAH, AH, NTAH, ES, label_colors


N = 30
words_to_ignore = ['i', 'u', 'aita', 'like', 'would', 'this', 'it', 'get', 'got', 'the']


def word_frequency(n):
    """Generates four plots for each AITA label that display the distribution of their top n words.

    Arguments:
         n {integer} -- Number of top words to display

    Returns:
        None
    """
    spark = init_spark(AITA_CLEANED_COLLECTION)
    data_rdd = spark.read.format('mongo').load().rdd

    def filter_words(row):
        words = []
        tokens = row['header'].lower().split(' ') + row['content'].lower().split(' ')

        for word in tokens:
            if word not in words_to_ignore:
                words.append(Row(label=row['label'], word=word))

        return words

    flattened_rdd = data_rdd.flatMap(lambda row: filter_words(row)).cache()

    plt.figure(figsize=(18, 8))
    for index, label in enumerate([NTAH, AH, NAH, ES]):
        result_rdd = generate_result_rdd(flattened_rdd, label)
        generate_plot(index+1, result_rdd.take(n), n, label)

    plt.subplots_adjust(hspace=1)
    plt.xlabel('Words')
    plt.show()


def generate_result_rdd(rdd, label):
    return rdd.filter(lambda row: row['label'] == label) \
        .map(lambda row: (row['word'], 1)) \
        .reduceByKey(lambda x, y: x+y) \
        .sortBy(lambda x: -x[1])


def generate_plot(index, data, n, label):
    words = [row[0] for row in data]
    counts = [row[1] for row in data]

    plt.subplot(4, 1, index)
    plt.title('{}: Top {} Words Distribution'.format(label, n))
    plt.ylabel('Count')

    y_pos = np.arange(len(words))
    plt.bar(y_pos, counts, color=label_colors.get(label))
    plt.xticks(y_pos, words)


if __name__ == '__main__':
    word_frequency(N)
