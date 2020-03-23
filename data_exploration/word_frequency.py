import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import Row
import collections

import nltk
from nltk import pos_tag

import sys
sys.path.append('../')
from spark.init_spark import init_spark
from config import AITA_CLEANED_COLLECTION

nltk.download('averaged_perceptron_tagger')

adjective_tags = ['JJ', 'JJR', 'JJS']
words_to_ignore = ['i', 'u', 'aita', 'like', 'would', 'this', 'it', 'get', 'got', 'the']


labels = ['NO A-HOLES HERE', 'ASSHOLE', 'NOT THE A-HOLE', 'EVERYONE SUCKS']


def fill_0_values(max_label_len, list):
    list_len = len(list)
    if list_len < max_label_len:
        list.extend([0] * (max_label_len - list_len))

    return list


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
        .flatMap(lambda row: [Row(label=row['label'], word=word) for word in filter(row)])

    result_rdd = flattened_rdd \
        .map(lambda row: ((row['label'], row['word']), 1)) \
        .reduceByKey(lambda x, y: x+y) \
        .sortBy(lambda x: -x[1])

    result = result_rdd.take(n)
    print(result)
    labels = spark.sparkContext.parallelize(result).map(lambda x: x[0][1]).distinct().collect()

    r_data = labels
    x = range(len(r_data))

    d = collections.defaultdict(list)

    for val in result:
        print(val)
        label = val[0][0]
        d[label].append(val[1])

    print(d[labels[0]])
    print(d[labels[1]])
    print(d[labels[2]])
    print(d[labels[3]])

    max_label_len = len(d[[k for k in d.keys() if len(d.get(k)) == max([len(n) for n in d.values()])][0]])

    nah = fill_0_values(max_label_len, d[labels[0]])
    ah = fill_0_values(max_label_len, d[labels[1]])
    ntah = fill_0_values(max_label_len, d[labels[2]])
    es = fill_0_values(max_label_len, d[labels[3]])
    print(nah)
    print(ah)
    print(ntah)
    print(es)

    bottom_ntah = np.add(nah, ah).tolist()
    bottom_es = np.add(bottom_ntah, ntah).tolist()

    plt.figure(figsize=(20, 5))

    p1 = plt.bar(x, nah, color='grey')
    p2 = plt.bar(x, ah, bottom=nah, color='red')
    p3 = plt.bar(x, ntah, bottom=bottom_ntah, color='green')
    p4 = plt.bar(x, es, bottom=bottom_es, color='black')

    plt.title('Distribution of the top {} {} for posts in r/AITA'.format(n, 'words'))
    plt.xticks(x, r_data)
    plt.xlabel('Words')
    plt.ylabel('Count')

    plt.legend((p1[0], p2[0], p3[0], p4[0]), labels)

    plt.show()


def adjective_frequency(n):
    spark = init_spark(AITA_CLEANED_COLLECTION)
    data_rdd = spark.read.format('mongo').load().rdd

    def filter(row):
        words = []
        list = pos_tag([word.lower() for sublist in row['header'] for word in sublist] + \
                         [word.lower() for sublist in row['content'] for word in sublist])
        for word in list:
            if word[1] in adjective_tags and word[0] not in words_to_ignore:
                words.append(word[0])

        return words

    flattened_rdd = data_rdd.flatMap(lambda row: [Row(label=row['label'], word=word) for word in filter(row)])

    result_rdd = flattened_rdd\
        .map(lambda row: ((row['label'], row['word']), 1)) \
        .reduceByKey(lambda x, y: x+y) \
        .sortBy(lambda x: -x[1])

    result = result_rdd.take(n)
    labels = spark.sparkContext.parallelize(result).map(lambda x: x[0][1]).distinct().collect()

    plot_count_by_label('adjectives', result, labels, n)


def plot_count_by_label(pos, data, labels, n):
    r_data = labels
    x = range(len(r_data))

    d = collections.defaultdict(list)

    for val in data:
        label = val[0][0]
        d[label].append(val[1])

    max_label_len = len(d[[k for k in d.keys() if len(d.get(k)) == max([len(n) for n in d.values()])][0]])

    nah = fill_0_values(max_label_len, d[labels[0]])
    ah = fill_0_values(max_label_len, d[labels[1]])
    ntah = fill_0_values(max_label_len, d[labels[2]])
    es = fill_0_values(max_label_len, d[labels[3]])

    bottom_ntah = np.add(nah, ah).tolist()
    bottom_es = np.add(bottom_ntah, ntah).tolist()

    plt.figure(figsize=(20, 5))

    p1 = plt.bar(x, nah, color='grey')
    p2 = plt.bar(x, ah, bottom=nah, color='red')
    p3 = plt.bar(x, ntah, bottom=bottom_ntah, color='green')
    p4 = plt.bar(x, es, bottom=bottom_es, color='black')

    plt.title('Distribution of the top {} {} for posts in r/AITA'.format(n, pos))
    plt.xticks(x, r_data)
    plt.xlabel('Words')
    plt.ylabel('Count')

    plt.legend((p1[0], p2[0], p3[0], p4[0]), labels)

    plt.show()


if __name__ == '__main__':
    word_frequency(30)
    # adjective_frequency(30)
