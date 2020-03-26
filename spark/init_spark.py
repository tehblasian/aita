from pyspark.sql import SparkSession
from config import MONGO_CONNECTION_STRING, AITA_DB_NAME


def init_spark(collection_name):
    mongodb_uri = '{}/{}.{}'.format(MONGO_CONNECTION_STRING, AITA_DB_NAME, collection_name)

    return SparkSession \
        .builder \
        .appName('AITA') \
        .master('local') \
        .config('spark.driver.host', '127.0.0.1') \
        .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.11:2.4.1') \
        .config('spark.mongodb.input.uri', mongodb_uri) \
        .config('spark.mongodb.output.uri', mongodb_uri) \
        .getOrCreate()
