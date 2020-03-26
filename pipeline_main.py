from pyspark.ml.linalg import Vectors

from ml.classifiers import NaiveBayesClassifier
from ml.pipeline import AITAPipeline
from spark.init_spark import init_spark

if __name__ == '__main__':
    spark = init_spark()
    training = spark.createDataFrame([
        (Vectors.dense([1,1, 1]), 1.0),
        (Vectors.dense([1, 2 ,5]), 1.0),
        (Vectors.dense([0, 1, 10]), 0.0),
        (Vectors.dense([0, 0, 0]), 0.0),
        (Vectors.dense([23, 1, 10]), 1.0),
        (Vectors.dense([46, 0, 19]), 0.0),
        (Vectors.dense([46, 69, 19]), 1.0),
        (Vectors.dense([46, 69, 0]), 0.0),
        (Vectors.dense([46, 6969, 119]), 1.0),
        (Vectors.dense([46, 130, 0]), 0.0),
        (Vectors.dense([12,11,31]), 1.0),
        (Vectors.dense([0, 69, 111]), 0.0)
    ], ["features", "label"])
    pipeline = AITAPipeline()
    pipeline.dataset(training).classifier(NaiveBayesClassifier).run()
