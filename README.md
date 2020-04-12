# SOEN 499 Project: Big Data for AITA

The following is a term project for the [SOEN 499 (Big Data Analytics)](https://users.encs.concordia.ca/~tglatard/teaching/big-data/) course in Winter 2020, taught by Dr. Tristan Glatard at Concordia University.

### Team Members

| Username         | Name                | Student ID |
| :--------------- | :------------------ | :--------- |
| `chimano`        | Chirac Manoukian    | 40028500   |
| `justin-cotarla` | Justin Cotarla      | 40027609   |
| `kwreen`         | Karen Bie           | 40018058   |
| `tehblasian`     | Jeremiah-David Wreh | 40028325   |

---

### Abstract

[Am I the A-hole (AITA)](https://www.reddit.com/r/AmItheAsshole/) is an online community on the popular social news website [Reddit](https://www.reddit.com). Users post descriptions of personal experiences asking the community to vote and determine whether or not they were the a-hole from the situation. In this report, we collect a large amount of posts from the AITA subreddit and subsequently analyze the dataset in order to predict what label will be assigned to a particular scenario as to find out who is in the wrong. The report aims to discuss the methods we used to pre-process the data and the performance of our model.

---

## I. Introduction

### I.a. Context

Am I the A-hole is a subreddit (i.e. a community on Reddit) where users submit posts describing a personal experience to then have the community decide on whether or not they were right under their given situation. When a post is submitted to the subreddit, community members may leave comments on it with their reactions and opinions on the described scenario from the post. Members also include a distinctive judgement label with their comment which they deem most appropriate for the situation, chosen from one of the following:

- You're the A-hole (YTA)
- You're Not the A-hole (NTA)
- Everyone Sucks Here (ESH)
- No A-holes here (NAH)
- Not Enough Info (INFO)

Members may as well upvote comments from the threads to signify that they agree with a commenter’s view. The judgement label from the comment with the most upvotes is the final label attributed to the original post as a way to indicate the consensus.

### I.b. Objectives

The objective of our project is to design a classification model to accurately predict which final judgement label would most likely reflect the general agreement of the AITA community in response to a given scenario. In the current state of the subreddit, a winning verdict is determined after 18 hours. With our project, a verdict would be more quickly brought to the table for conflicted users to find out if they had been in the wrong, and to be sooner rid of their worries.

### I.c. Related Works

_Liang, H., Sun, X., Sun, Y. et al. J Wireless Com Network (2017) 2017: 211. https://doi.org/10.1186/s13638-017-0993-1_

In this article, different methods are discussed to extract features from text, and to classify texts using these features. The authors of the article have essentially summarized the techniques used in text classification. We will clean, process, and classify the textual data using the methods discussed in the article, such as lemmatization and stop word removal, and Random Forest. This article’s summarizations will be used as a source of information to guide us through the experiment.

## II. Materials and Methods

### II.a. Dataset

[Pushshift](https://pushshift.io/) is a big data storage project that copies Reddit submissions and comments. Its goal is to provide a common API for anyone to access this data on a large scale. Using this, we will extract all the posts submitted to AITA including their labels. Each item in the resultant dataset will have the following attributes that we deemed to be useful for classification:

- Title - the title of the reddit post
- Content - the content of the reddit post
- Label - the concluded label associated to the post

The data extracted from Pushshift dates from Monday, February 24, 2014 9:50:51 PM to Saturday, April 4, 2020 8:41:28 PM. The number of posts retrieved total to 37679 items - these are the number of posts with labels YTA, NTA, ESH and NAH. We decided to drop posts that have been judged to be lacking information (INFO) as it is not a relevant label to classify with.

### II.b. Technologies and Algorithms

The core of the project is written in Python and requires version 3.5+. The project has been run with version 3.7.5. Apache Spark is used for this project in parallelizing computations on the data.

#### Text Pre-processing

As with most text classification problems, it is important to pre-process the data in order to remove noise (Srividhya & Anitha, 2010). Consequently, natural language processing techniques and other methods were used to clean up the AITA submissions extracted from Pushshift.

More specifically, the nltk package was used to remove stopwords and proper nouns, and to perform lemmatization. Stopword removal and lemmatization are important because they reduce the text’s dimensions. We decided to remove singular proper nouns (tagged as NNP by nltk) as well. Although many names are used by posters in telling their stories, we have assumed that they would not be pertinent for classification. Digits were also removed from the data, and all text was converted to lowercase in this process. Once the data was cleaned, it was stored in a MongoDB instance on a server to be used for training in the following machine learning algorithms.

Code for this portion can be found under `/data_preparation`.

#### Feature Extraction

Since machine learning algorithms operate on a numerical feature space, the input text must be converted from its raw string representation to a fixed-length vector representation. To do so, we tried three different approaches.

##### Bag-of-Words (BoW)

In this approach, each submission is represented by a vector which describes the frequency of the words that appear within it.

##### Term Frequency-Inverse Document Frequency (TF-IDF)

Tf-idf is a numerical statistic used in information theory that aims to reflect how important a word is to a given text document. Words that appear frequently in a document but sparingly in others are given a higher score. Intuitively, tf-idf should yield greater performance than the simpler BoW approach because it reduces the weight of words that are common to all documents. In this approach, the output vector contains the tf-idf scores corresponding to each word contained in the submission.

##### Doc2Vec

Doc2Vec is an algorithm based on the word2vec algorithm, which uses a neural network to generate word embeddings. In this manner, relations between the words in the documents can be created.

Word2vec creates embeddings which encode the semantic meaning of words by taking into account the surrounding words, or the context of the word. Doc2vec goes a step forward by representing documents as word embeddings. While several approaches exist for generating document vectors, in this project we decided to leverage vector averaging. In this approach, word2vec embeddings are created for every word in a post, and the corresponding vectors are averaged to create a final document embedding.

#### Model Training and Evaluation

In this project, we decided to train three different machine learning models: Naive-Bayes, Random Forest and Support Vector Machines.

##### Naive-Bayes

Naive-Bayes is a simple multiclass classification algorithm that is based on Bayes’ theorem. Briefly, it computes the prior probabilities of each class given the input dataset. Then, using Bayes’ theorem, it computes the conditional probability distribution of each label given a sample and selects as a prediction the label with the highest probability.

##### Random Forest

A Random Forest is an ensemble of decision trees. In this model, the decision trees are built using a random sampling of training points, and by considering random subsets of features when splitting nodes. Predictions are obtained by either taking the mode or average of the different tree outputs.

##### Support Vector Machines (SVM)

Support vector machines aim to find the best separation of data points in high-dimensional space by constructing a hyperplane that maximizes the distance to the nearest training example in any class. Since SVMs are traditionally used for binary classification, we combined them with the One-vs-Rest classifier so that multiclass classification could be performed. The One-vs-Rest classifier trains an ensemble of binary classifiers and obtains predictions by selecting the output of the one with the most confident result.

To evaluate the trained models, we used k-fold cross validation and obtained the F1 score, accuracy, precision and recall for each.

The code for this portion can be found under `/ml`.

More information on these classification algorithms can be found [here](https://spark.apache.org/docs/2.4.5/ml-classification-regression.html).

## III. Results

### III.a. A First Look at the Data

Before diving into the results obtained from applying machine learning classification techniques, let's take a first look at the data. Scripts for the generation of this section's graphs can be found under `/data_exploration`.

<p align="center">
    <img src="data_exploration/imgs/class_distribution.png" width=375> </br>
    Figure 1: Distribution of Labels for AITA Posts
</p>

From figure 1, the distribution tells us that there is a strong imbalance in the dataset because the majority of the submissions are labelled as NTA. This observation pushed us to randomly undersample the NTA labelled submissions as to remove bias towards the majority class.

<p align="center">
    <img src="data_exploration/imgs/time_analysis.png" width=375> </br>
    Figure 2: Distribution of AITA Posts by Week Days
</p>

<p align="center">
    <img src="data_exploration/imgs/word_frequency.png" width=500> </br>
    Figure 3: Distributions of the Top 30 Words by Label
</p>

We also attempted to extract topics from the content of the posts to see if there might be any common themes. To do this, we used Latent Dirichlet Allocation (LDA). Similar to k-means clustering, we needed to select the number of topics the algorithm should try to discover. Selecting a correct number of topics was difficult, and overall, modelling for the topics was tedious due to the sheer amount of words involved. In the end, we were able to obtain some interesting results. Below are the topics we discovered, with labels that we have termed appropriately and tuples representing (word, weight_of_word_for_topic):

```
Topic 0: "Family"
('i', 0.058737517748202545)
('wedding', 0.013851375462612508)
('name', 0.012834816141956749)
('my', 0.011579272101059657)
('family', 0.010909367820150517)
('husband', 0.00810355276764783)
('would', 0.00773857722996206)
('want', 0.007291003499384249)
('child', 0.0068284651474224065)
('like', 0.0068162963353990405)

Topic 1: "Food"
('food', 0.048583658839314425)
('eat', 0.04744095305274857)
('eating', 0.021835719736446414)
('meat', 0.01929569588324601)
('cook', 0.01828335146548378)
('vegan', 0.0179028588735922)
('meal', 0.01737827811989212)
('i', 0.015973424706517296)
('dish', 0.012921412983558695)
('cooking', 0.011299819640166252)

Topic 2:  "Beliefs"
('temple', 0.009472794798211203)
('i', 0.004546260400681341)
('hindu', 0.001498963433516801)
('unicorn', 0.0010895702311568025)
('disorganised', 0.0005537448308678305)
('imovie', 0.0005525654754604156)
('he', 0.0005318494761121402)
('mormon', 0.00048748062270736824)
('sacredness', 0.0004510896817122591)
('ramadan', 0.0004500217435091548)
```

In the end, only results from the distribution of labels from Figure 1 were factored into the classification pipeline. Results from the label distribution indicated for us a flaw in our dataset whereas we failed to find value in the rest of the analysis. However, it remains interesting for us to have explored the data in an attempt to observe possible trends.

### III.b. Model Performance

The following tables show the results for the three different feature extraction techniques used to train the different models.

<center>

|         | F1    | Accuracy | Precision | Recall |
| :------ | :---- | :------- | :-------- | :----- |
| BoW     | 0.357 | 0.313    | 0.491     | 0.313  |
| TF-IDF  | 0.358 | 0.309    | 0.502     | 0.309  |
| Doc2Vec | 0.202 | 0.197    | 0.504     | 0.197  |

Table 1: Naive-Bayes Model Performance

|         | F1    | Accuracy | Precision | Recall |
| :------ | :---- | :------- | :-------- | :----- |
| BoW     | 0.283 | 0.250    | 0.491     | 0.250  |
| TF-IDF  | 0.303 | 0.261    | 0.493     | 0.261  |
| Doc2Vec | 0.307 | 0.267    | 0.492     | 0.267  |

Table 2: Random Forest Model Performance

|         | F1    | Accuracy | Precision | Recall |
| :------ | :---- | :------- | :-------- | :----- |
| BoW     | 0.357 | 0.309    | 0.504     | 0.309  |
| TF-IDF  | 0.337 | 0.297    | 0.493     | 0.297  |
| Doc2Vec | 0.307 | 0.267    | 0.492     | 0.267  |

Table 3: SVM Model Performance

</center>

## IV. Discussion

As shown in the tables in the previous section, the overall performance of the models trained with the various feature extraction techniques is quite poor - all performance metrics result in at most ~35%, with the exception of precision.

To our surprise, the use of TF-IDF over BoW did not show a strong increase in performance for all the models. For instance, it led to performance gains across all metrics for the Random Forest classifier, but only resulted in a 0.001 increase in the F1 score for the Naive-Bayes classifier, and even led to a 5.76% decrease in the F1 score for the SVM classifier. Nevertheless, the performance of all classifiers when paired with Doc2Vec was noticeably worse. One reason for this could be that the output vector size we selected (dim=100) was either too small or too large. To find the optimal size, more tuning would be required. In the end, the best combinations were TF-IDF with Naive-Bayes, TF-IDF with Random Forest, and TF-IDF with BoW.

This experiment, however, was performed with undersampling since the dataset collected for this project was heavily skewed as shown in figure 1 of the results. More specifically, we sampled each individual class so as to match the data point count of the most under-represented class. In doing so, we greatly reduced the amount of data points used for training the models. Though a total of nearly 40k posts were retrieved from Pushshift, undersampling in favor of the minority class (ESH) reduced the size of the training set to around 8k documents. Due to this, we believe that the performance of the models was severely hindered.

In future work, it could be interesting to divide the data such that the NTA and NAH labels are grouped as one class, while the YTA and ESH labels are grouped as another. We suspect that this would lead to stronger performance gains since it would reduce the problem from multi-class to binary classification, and would help mitigate the imbalance in the dataset. Another possible improvement to this experiment would be to use deep learning models, as they have been frequently used for text classification with successful results (Liang, et al. 2017). Nonetheless, deep learning models require larger amounts of data compared to other types of classifiers. Assuming an increase of AITA submissions in following years, we believe this experiment could be redone with a deep learning model to potentially yield a better performing classifier.
