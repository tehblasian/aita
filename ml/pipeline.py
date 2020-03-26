
class AITAPipeline:
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

    def run(self):
        """Runs the pipeline
        
        Raises:
            AttributeError: Raised if one of dataset, classifier or transformer are not set
        """
        
        # if not hasattr(self, '_dataset') or not hasattr(self, '_transformer') or not hasattr(self, '_classifier'):
        #     raise AttributeError('Please specify a dataset, a data transformer and a classifier')
        
        # data = self._transformer.transform(self._dataset)
        data = self._dataset
        self._classifier = self._classifier(data)
        self._classifier.train()
        f1, accuracy, precision, recall = self._classifier.evaluate()

        print('Accuracy', accuracy)  
        print('Precision:', precision)
        print('Recall:', recall)
        print('F1-Score:', f1)
