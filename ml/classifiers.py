from abc import ABC

from pyspark.ml.classification import LinearSVC, NaiveBayes, OneVsRest
from pyspark.ml.classification import RandomForestClassifier as RandomForest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


class AbstractClassifier(ABC):
    def train(self, k_folds=7):
        """Trains a classification model using k-fold cross validation
        
        Keyword Arguments:
            k_folds {int} -- Number of folds (default: {7})
        """
        param_grid = self._get_param_grid()
        cross_validator = CrossValidator(estimator=self.classifier, estimatorParamMaps=param_grid, evaluator=self.evaluator, numFolds=k_folds)
        self.model = cross_validator.fit(self.train_set)

    def evaluate(self):
        """Evaluates the trained model
        
        Raises:
            AttributeError: Raises exception if model is not trained before evaluating
        
        Returns:
            Tuple -- A tuple containing the following metrics: f1-score, accuracy, precision, recall
        """
        if not hasattr(self, 'model'):
            raise AttributeError('Please train the model before evaluation')
        predictions = self.model.transform(self.test_set)
        
        f1 = self.evaluator.evaluate(predictions)
        accuracy = self.evaluator.evaluate(predictions, {self.evaluator.metricName: 'accuracy'})
        precision = self.evaluator.evaluate(predictions, {self.evaluator.metricName: 'weightedPrecision'})
        recall = self.evaluator.evaluate(predictions, {self.evaluator.metricName: 'weightedRecall'})

        return f1, accuracy, precision, recall

    def _get_param_grid(self):
        pass

class NaiveBayesClassifier(AbstractClassifier):
    def __init__(self, dataset, train_ratio=0.8):
        self.dataset = dataset
        self.classifier = NaiveBayes(labelCol='label', featuresCol='features')
        self.evaluator = MulticlassClassificationEvaluator()
        self.train_set, self.test_set = dataset.randomSplit([train_ratio, 1-train_ratio])
    
    def _get_param_grid(self):
        return ParamGridBuilder() \
               .addGrid(self.classifier.smoothing, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) \
               .build()

class RandomForestClassifier(AbstractClassifier):
    def __init__(self, dataset, train_ratio=0.8):
        self.dataset = dataset
        self.classifier = RandomForest(labelCol='label', featuresCol='features', impurity='gini',
                                              maxBins=31)
        self.evaluator = MulticlassClassificationEvaluator()
        self.train_set, self.test_set = dataset.randomSplit([train_ratio, 1-train_ratio])

    def _get_param_grid(self):
        return ParamGridBuilder() \
            .addGrid(self.classifier.maxDepth, [5, 7, 11, 13]) \
            .addGrid(self.classifier.numTrees, [5, 7, 11, 13, 17]) \
            .build()

class SVMClassifier(AbstractClassifier):
    def __init__(self, dataset, train_ratio=0.8):
        self.dataset = dataset
        self.classifier = OneVsRest(classifier=LinearSVC(maxIter=10))
        self.evaluator = MulticlassClassificationEvaluator()
        self.train_set, self.test_set = dataset.randomSplit([train_ratio, 1-train_ratio])
    
    def _get_param_grid(self):
        self.classifier.getClassifier
        return ParamGridBuilder() \
            .addGrid(self.classifier.getClassifier().regParam, [0.1, 0.2, 0.4, 0.6, 0.8, 1]) \
            .build()
