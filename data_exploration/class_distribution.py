import matplotlib.pyplot as plt
import numpy as np
import pyspark
from pyspark.sql import SparkSession

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

def class_distribution():
    spark = init_spark()
    df = spark.read.format('mongo').load()
    
    distribution = df.groupBy('label').count().collect()
    show_bar_plot(distribution)


def show_bar_plot(distribution):
    counts = [row['count'] for row in distribution]
    labels = [row['label'] for row in distribution]

    y_pos = np.arange(len(labels))
    plt.figure(figsize=(10,5))
    plt.bar(y_pos, counts, color=['grey', 'black', 'red', 'green'])
    plt.xticks(y_pos, labels)

    plt.title('Distribution of labels for posts in r/AITA')
    plt.ylabel('Count')
    plt.xlabel('Label')

    plt.show()


if __name__ == '__main__':
    class_distribution()