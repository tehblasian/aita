import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from pyspark.sql import Row

from spark.init_spark import init_spark

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
STOPWORDS_SET = set(stopwords.words('english'))


def main():
    spark = init_spark()
    rdd = spark.read.format('mongo').load().rdd
    cleaned = clean_data(rdd)
    cleaned.toDF().write.format("mongo").mode("overwrite").option("database", "aitaDB").option("collection", "cleaneddata").save()


def clean_data(rdd):
    lemmatizer = WordNetLemmatizer()

    def filter_word(word, pos):
        """Filters out stop words, punctuation, numbers, and proper nouns
        Arguments:
            word {string} -- The word being checked
            pos {string} -- The word's tag
        
        Returns:
            boolean -- true if the input is not a stop word, punctuation, number or a proper noun
        """
        is_stopword = word in STOPWORDS_SET
        is_word = word.isalnum()
        is_number = word.replace('.', '', 1).isdigit()
        # str.isdigit returns false with numbers that have a decimal point
        # so replacing it with a number makes it work as intended
        is_proper_noun = pos == 'NNP'
        return not is_stopword and is_word and not is_number and not is_proper_noun

    def process_record(record):
        """Cleans the content in the record by tokenizing, removing stop words,
        and lemmatizing
        
        Arguments:
            record {Row} -- Row representation of a record stored in mongo
        
        Returns:
            Row -- A row that has been processed
        """

        content = ' '.join([lemmatizer.lemmatize(word)\
                    for sentence in sent_tokenize(record['content'])\
                        for word, pos in pos_tag(word_tokenize(sentence)) if filter_word(word, pos)])
        header = ' '.join([lemmatizer.lemmatize(word)\
                    for sentence in sent_tokenize(record['header'])\
                        for word, pos in pos_tag(word_tokenize(sentence)) if filter_word(word, pos)])
        return Row(content=content, header=header, label=record['label'], created_at=record['created_at'])

    return rdd.map(process_record)


if __name__ == '__main__':
    main()
