from abc import ABC
from pyspark.ml.feature import CountVectorizer, Word2Vec, IDF
from pyspark.sql import Row
from constants import label_ids

class AbstractTransformer(ABC):
    def __init__(self, spark, input_data):
        self._spark = spark
        self._mapped_data = self.prepare_data(input_data)

    def prepare_data(self, data):
        """Prepares an input dataframe by mapping labels to numbers and by appending header and content into a list of tokens

        Keyword Arguments:
            data {DataFrame} -- Raw text-based dataset with the following schema: [_id: struct<oid:string>, content: string, created_at: bigint, header: string, label: string]

        Returns:
            DataFrame -- A dataframe with the following schema: [content: array<string>, label: bigint]
        """
        return (self._spark.createDataFrame(
            data
            .rdd
            .map(lambda r: Row(content=r.header.split(' ') + r.content.split(' '), label=label_ids[r.label]))
        ))

    def transform(self):
        """Transforms a text-based dataframe to a vector-based dataframe

        Returns:
            DataFrame -- A dataframe with the following schema: [label: bigint, features: DenseVector<float>]
        """
        pass

class FrequencyTransformer(AbstractTransformer):
    def __init__(self, spark, input_data, min_tf=1.0, min_df=1.0):
        self._min_tf = min_tf
        self._min_df = min_df
        super().__init__(spark, input_data)

    def transform(self):
        cv = CountVectorizer(minTF=self._min_tf, minDF=self._min_df, inputCol='content', outputCol='features')
        cv_model = cv.fit(self._mapped_data)
        cv_df = cv_model.transform(self._mapped_data)

        return cv_df.drop('content')

class Doc2VecTransformer(AbstractTransformer):
    def __init__(self, spark, input_data, vector_size=100):
        self._vector_size = vector_size
        super().__init__(spark, input_data)

    def transform(self):
        word2Vec = Word2Vec(vectorSize=self._vector_size, inputCol="content", outputCol="features")
        model = word2Vec.fit(self._mapped_data)

        w2v_data = model.transform(self._mapped_data)

        return w2v_data.drop('content')

class TfIdfTransformer(AbstractTransformer):
    def __init__(self, spark, input_data, min_doc_freq=0):
        self._min_doc_freq = min_doc_freq
        super().__init__(spark, input_data)

    def transform(self):
        cv = CountVectorizer(inputCol='content', outputCol='raw_features')
        cv_model = cv.fit(self._mapped_data)
        tf_df = cv_model.transform(self._mapped_data)

        idf = IDF(minDocFreq=self._min_doc_freq, inputCol='raw_features', outputCol='features')
        tfidf_model = idf.fit(tf_df)
        tfidf_df = tfidf_model.transform(tf_df)

        return tfidf_df.drop('content').drop('raw_features')