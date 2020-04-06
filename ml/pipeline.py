import inspect

from pyspark.ml.feature import MinMaxScaler

class AITAPipeline:
    def __init__(self, spark, fix_imbalance=True):
        self._spark = spark
        self.fix_imbalance = fix_imbalance

    def dataset(self, dataset):
        """Sets the data that will be transformed, and trained/tested on
        
        Arguments:
            dataset {Dataframe} -- DataFrame representing a monogDB record
        
        Returns:
            AITAPipeline -- The instance of the pipeline
        """
        self._dataset = dataset
        return self

    def transformer(self, transformer):
        """Sets the transformation that will be applied to the data
        
        Arguments:
            transformer {TextTransformer} -- The transformation that will be applied to
            the data
        
        Returns:
            AITAPipeline -- The instance of the pipeline
        """
        self._transformer = transformer
        return self

    def classifier(self, classifier):
        """Sets the classification algorithm that will be used to train a model
        
        Arguments:
            classifier {AbstractClassifier} -- The classification algorithm that will be used to train a model
        
        Returns:
            AITAPipeline -- The instance of the pipeline
        """
        self._classifier = classifier
        return self


    def undersample_data(self):
        """Fixes data imbalances by undersampling overrepresented classes
        
        Returns:
            DataFrame -- Dataframe representation of an article
        """
        class_distribution = self._dataset.groupBy(self._dataset.label).count()
        distribution_dict = class_distribution.rdd.map(lambda x: (x['label'], x['count'])).collectAsMap()
        lowest_count = min(distribution_dict.values())
        sampling_fractions = {label: lowest_count/distribution_dict[label] for label in distribution_dict}
        return self._dataset.sampleBy('label', fractions=sampling_fractions)
    
    def scale_data(self, data):
        """Scales feature data by remapping the range to [0, 1]

        Returns:
            DataFrame -- Dataframe representation of an article
        """
        scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
        scaler_model = scaler.fit(data)
        scaled_data = scaler_model.transform(data)

        return scaled_data.drop("features").withColumnRenamed("scaledFeatures", "features")

    def run(self):
        """Runs the pipeline
        
        Raises:
            AttributeError: Raised if one of dataset, classifier or transformer are not set
        """
        
        if not hasattr(self, '_dataset') or not hasattr(self, '_transformer') or not hasattr(self, '_classifier'):
            raise AttributeError('Please specify a dataset, a data transformer and a classifier')

        dataset = self.undersample_data() if self.fix_imbalance else self._dataset

        self._transformer = self._transformer(self._spark, dataset)
        data = self._transformer.transform()
        
        scaled_data = self.scale_data(data)

        self._classifier = self._classifier(scaled_data)
        self._classifier.train()

        return self._classifier.evaluate()
