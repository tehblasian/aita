import matplotlib.pyplot as plt
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
    df.printSchema()

if __name__ == '__main__':
    class_distribution()