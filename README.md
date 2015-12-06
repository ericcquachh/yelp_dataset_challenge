#yelp_dataset_challenge

One of probably the best entries to the dataset challenge, this project challenges YOU.

Just kidding. Nothing is implemented.

# Implemented Features

1) Filtered out the stop words.

2) Sort by highest TF-IDF for each review type.

# Implemented Classifiers

1) Naive Bayes - The current only feature is sort by highest TF-IDF value

Positive and Negative Classification: 84%
Neutral: 66%
Star Classification: 30%

2) Decision Tree - The current only feature is sort by highest word frequencies

Positive and Negative Classification: 80%
Neutral: NA
Star Classification: NA

3) Support Vector Machine - Using the library doc2vec, we've created a vector for each document and classified using SVM's Vector Library

Positive and Negative Classification: 86%
Neutral: 73%
Star Classification: 46%

4) K Nearest Neighbors - Using doc2vec again, we tried running kNN over the same data

Positive and Negative Classification: 84%
Neutral: 69%
Star Classification: 36%

5) Logistic Regression - Using doc2vec for our representation, we used sklearn's library for vector representation.

Positive and Negative Classification: 86%
Neutral: 73%
Star Classification: 46%

# In Progress Featurization

1) Sentiment Analysis with TextBlob

2) Using Word Stemming

3) Using N-grams with Textblob

4) Random Forests 