import argparse
import datetime

from spark.init_spark import init_spark

from data_preparation.extractor import extract
from data_preparation.cleaner import clean

from data_exploration.time_analysis import time_analysis
from data_exploration.class_distribution import class_distribution
from data_exploration.word_frequency import word_frequency
from data_exploration.word_topics import word_topics

from ml.classifiers import NaiveBayesClassifier, RandomForestClassifier, SVMClassifier
from ml.transformers import FrequencyTransformer, TfIdfTransformer, Doc2VecTransformer
from ml.pipeline import AITAPipeline

from config import AITA_CLEANED_COLLECTION

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts reddit data')

    subparsers = parser.add_subparsers(dest='action')

    extract_parser = subparsers.add_parser('extract', help='Extract data')
    extract_parser.add_argument(
        '--min_count', help='Data will be fetched in batches until all classes have a <min_count> number of posts associated with it.(Default=100)',
        default=100, type=int)
    extract_parser.add_argument(
        '--start_epoch', help='Posts will be fetched started from that epoch and will go backwards in time (Default=Current epoch)', default=datetime.datetime.now().timestamp(),
        type=int)
    extract_parser.add_argument(
        '--end_epoch', help='Extractor will stop fetching if it sees that the post\'s timestamp is lower than this value (Default=0)', default=0,
        type=int)

    clean_parser = subparsers.add_parser('clean', help='Clean data')

    explore_parser = subparsers.add_parser('explore', help='Explore data')
    explore_parser.add_argument('type', choices=['class-distribution', 'time-analysis', 'word-frequency', 'word-topics'])

    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('transformer', choices=['frequency', 'doc2vec', 'tfidf'])
    train_parser.add_argument('classifier', choices=['naive-bayes', 'random-forest', 'svm'])
    train_parser.add_argument('--normalize', action='store_const', const=True, default=False, help='Normalize transformed data before classifying')
    train_parser.add_argument('--undersample', action='store_const', const=True, default=False, help='Undersample overrepresented data before classifying')
    train_parser.add_argument('--weights', action='store_const', const=True, default=False, help='Use weights to balance data before classifying')
    train_parser.add_argument('--log', help='Log results to file')

    args = parser.parse_args()

    if args.action == 'extract':
        start_epoch = args.start_epoch
        min_count = args.min_count
        end_epoch = args.end_epoch
        extract(min_count, start_epoch, end_epoch)

    elif args.action == 'clean':
        clean()

    elif args.action == 'explore':
        if args.type == 'class-distribution':
            class_distribution()
        elif args.type == 'time-analysis':
            time_analysis()
        elif args.type == 'word-frequency':
            word_frequency()
        elif args.type == 'word-topics':
            word_topics()

    elif args.action == 'train':
        transformer = None
        classifier = None

        if args.transformer == 'frequency':
            transformer = FrequencyTransformer
        elif args.transformer == 'doc2vec':
            transformer = Doc2VecTransformer
        elif args.transformer == 'tfidf':
            transformer = TfIdfTransformer

        if args.classifier == 'naive-bayes':
            classifier = NaiveBayesClassifier
        if args.classifier == 'random-forest':
            classifier = RandomForestClassifier
        if args.classifier == 'svm':
            classifier = SVMClassifier

        spark = init_spark(AITA_CLEANED_COLLECTION)
        dataset = spark.read.format('mongo').load()

        pipeline = AITAPipeline(spark, normalize=args.normalize, undersample=args.undersample, use_weights=args.weights)
        f1, accuracy, precision, recall = (pipeline.dataset(dataset)
            .transformer(transformer)
            .classifier(classifier)
            .run())

        if args.log is not None:
            log = open(args.log, 'a')
            log.write("[{} - {}], f1 {}, accuracy {}, precision {}, recall {}\n"
                .format(args.classifier, args.transformer, f1, accuracy, precision, recall))
        else:
            print('Accuracy', accuracy)  
            print('Precision:', precision)
            print('Recall:', recall)
            print('F1-Score:', f1)