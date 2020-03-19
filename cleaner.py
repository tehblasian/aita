import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from pyspark.sql import Row

from spark.init_spark import init_spark

nltk.download('punkt')
nltk.download('stopwords')
STOPWORDS_SET = set(stopwords.words('english'))
  
def main():
    spark = init_spark()
    rdd = spark.read.format('mongo').load().rdd
    cleaned = clean_data(rdd)
    cleaned.toDF().write.format("mongo").mode("append").option("database", "aitaDB").option("collection", "cleanedinfo").save()


def clean_data(rdd):
    lemmatizer = WordNetLemmatizer()

    def process_record(record):
        """Cleans the content in the record by tokenizing, removing stop words,
        and lemmatizing
        
        Arguments:
            record {Row} -- Row representation of a record stored in mongo
        
        Returns:
            Row -- A row that has been processed
        """
        content = [[lemmatizer.lemmatize(word)\
                    for word in word_tokenize(sentence) if word not in STOPWORDS_SET]\
                        for sentence in sent_tokenize(record['content'])]
        return Row(content=content, header=record['header'], label=record['label'], created_at=record['created_at'])

    return rdd.map(process_record)

if __name__ == '__main__':
    main()
