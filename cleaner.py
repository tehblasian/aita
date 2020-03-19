import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from pyspark.sql import Row

from spark.init_spark import init_spark

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
STOPWORDS_SET = set(stopwords.words('english'))
  
def main():
    spark = init_spark()
    rdd = spark.read.format('mongo').load().rdd
    cleaned = clean_data(rdd)
    cleaned.toDF().write.format("mongo").mode("overwrite").option("database", "aitaDB").option("collection", "cleaneddata").save()


def clean_data(rdd):
    lemmatizer = WordNetLemmatizer()
    def filter_word(word):
        """Filters out stop words, punctuation, and numbers
        
        Returns:
            boolean -- true if the input is not a stop word, punctuation or number
        """
        is_stopword = word in STOPWORDS_SET
        is_word = word.isalnum()
        is_number = word.replace('.','',1).isdigit()
        # str.isdigit returns false with numbers that have a decimal point
        # so replacing it with a number makes it work as intended
        return not is_stopword and is_word and not is_number

    def process_record(record):
        """Cleans the content in the record by tokenizing, removing stop words,
        and lemmatizing
        
        Arguments:
            record {Row} -- Row representation of a record stored in mongo
        
        Returns:
            Row -- A row that has been processed
        """
        content = [[lemmatizer.lemmatize(word)\
                    for word in word_tokenize(sentence) if filter_word(word)]\
                        for sentence in sent_tokenize(record['content'])]
        header = [[lemmatizer.lemmatize(word)\
                    for word in word_tokenize(sentence) if filter_word(word)]\
                        for sentence in sent_tokenize(record['header'])]
        return Row(content=content, header=header, label=record['label'], created_at=record['created_at'])

    return rdd.map(process_record)

if __name__ == '__main__':
    main()
