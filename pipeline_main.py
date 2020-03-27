from pyspark.ml.linalg import Vectors

from ml.classifiers import NaiveBayesClassifier
from ml.transformers import FrequencyTransformer, TfIdfTransformer
from ml.pipeline import AITAPipeline
from spark.init_spark import init_spark
from config import AITA_CLEANED_COLLECTION

if __name__ == '__main__':
    spark = init_spark(AITA_CLEANED_COLLECTION)
    dataset = spark.read.format('mongo').load()

    pipeline = AITAPipeline(spark)
    pipeline.dataset(dataset).transformer(TfIdfTransformer).classifier(NaiveBayesClassifier).run()
