import collections
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_utc_timestamp

AITA_DB_NAME = 'aitaDB'
AITA_POSTS_COLLECTION = 'extractedinfo'
MONGODB_URI = 'mongodb://127.0.0.1/{}.{}'.format(AITA_DB_NAME, AITA_POSTS_COLLECTION)

def init_spark():
    return SparkSession \
        .builder \
        .appName('AITA Class Distribution') \
        .master('local') \
        .config('spark.driver.host','127.0.0.1') \
        .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.11:2.4.1') \
        .config('spark.mongodb.input.uri', MONGODB_URI) \
        .config('spark.mongodb.output.uri', MONGODB_URI) \
        .getOrCreate()

def days_of_the_week_count():
    spark = init_spark()
    posts = spark.read.format('mongo').load().rdd

    def get_weekday(timestamp):
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return datetime.datetime.fromtimestamp(timestamp).weekday()

    label_day = posts \
        .map(lambda row: (row['label'], get_weekday(row['created_at']), 1)) \
        .groupBy(lambda p: (p[0], p[1])) \
        .mapValues(len) \
        .collect()

    label_day = sorted(label_day, key=lambda t: t[0][1])

    labels = ['NO A-HOLES HERE', 'ASSHOLE', 'NOT THE A-HOLE', 'EVERYONE SUCKS']

    d = collections.defaultdict(list)

    for val in label_day:
        label = val[0][0]
        d[label].append(val[1])

    nah = d[labels[0]]
    ah = d[labels[1]]
    ntah = d[labels[2]]
    es = d[labels[3]]

    bottom_ntah = np.add(nah, ah).tolist()
    bottom_es = np.add(bottom_ntah, ntah).tolist()

    r = [0,1,2,3,4,5,6]

    plt.figure(figsize=(10,5))

    p1 = plt.bar(r, nah, color='grey')
    p2 = plt.bar(r, ah, bottom=nah, color='red')
    p3 = plt.bar(r, ntah, bottom=bottom_ntah, color='green')
    p4 = plt.bar(r, es, bottom=bottom_es, color='black')

    plt.title('Distribution of days when posts were submitted to r/AITA, per label')
    plt.xticks(r, ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.xlabel('Day of the week')
    plt.ylabel('Number of posts')

    plt.legend((p1[0], p2[0], p3[0], p4[0]), labels)

    plt.show()


if __name__ == '__main__':
    days_of_the_week_count()