import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../')
from spark.init_spark import init_spark

def class_distribution():
    spark = init_spark()
    df = spark.read.format('mongo').load()
    
    distribution = df.groupBy('label').count().collect()
    show_bar_plot(distribution)


def show_bar_plot(distribution):
    counts = [row['count'] for row in distribution]
    labels = [row['label'] for row in distribution]

    y_pos = np.arange(len(labels))
    plt.figure(figsize=(10,5))
    plt.bar(y_pos, counts, color=['grey', 'black', 'red', 'green'])
    plt.xticks(y_pos, labels)

    plt.title('Distribution of labels for posts in r/AITA')
    plt.ylabel('Count')
    plt.xlabel('Label')

    plt.show()


if __name__ == '__main__':
    class_distribution()