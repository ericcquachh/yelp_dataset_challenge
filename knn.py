""" kNN classifier """

""" word 2 vec """
""" want doc2vec instead"""

import nltk
import json
import time
import operator
import math
import numpy as np

from textblob import TextBlob as tb
from collections import defaultdict
from os import listdir
from os.path import isfile, join

from gensim import corpora, models, similarities
import gensim
from gensim.models import doc2vec
from gensim.models.doc2vec import LabeledSentence

from nltk import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

""" Retrieve the stars of each review in their corresponding array. """
def star_array(n):
	array_of_stars = [0 for i in range(1569264)] # just gonna hardcode this lol
	with open(file_name) as json_file:
		for i, line in enumerate(json_file):
			if i >= n:
				break
			parsed_json = json.loads(line)
			stars = parsed_json['stars']
			array_of_stars[i] = stars
	return array_of_stars

""" Put the text as elements of an array. """
def text_array(n):
	array_of_text = [0 for i in range(1569264)]
	with open(file_name) as json_file:
		for i, line in enumerate(json_file):
			if i >= n:
				break
			parsed_json = json.loads(line)
			text = parsed_json['text']
			array_of_text[i] = text
	return array_of_text

""" Have the positive or negative array for classification. """
def positive_negative_array(array):
	array_of_pos_neg = [0 for i in range(1569264)]
	for i, elem in enumerate(array):
		if elem >= 3:
			array_of_pos_neg[i] = "positive"
		else:
			array_of_pos_neg[i] = "negative"
	return array_of_pos_neg

""" Here we provide the document labels to feed our doc2vec """
def doc_label_array(n):
	string = "DOC_"
	array_of_doc = [0 for i in range(1569264)]
	for i in range(n):
		array_of_doc[i] = string + str(i)
	return array_of_doc


DATA_PATH = "/Users/colin.garcia/Desktop/yelp_dataset_challenge_academic_dataset/"
file_name = DATA_PATH  + "yelp_academic_dataset_review.json"
iterations = int(raw_input("What number of reviews would you like to train on? : "))
# iterations = 10000

print "Starting Classification for " + str(iterations) + " reviews."

start_time = time.time()
stars_array = star_array(iterations)[0:iterations]
pos_neg_array = positive_negative_array(stars_array)[0:iterations]
review_text = text_array(iterations)[0:iterations] # possibly train on more data
document_labels = doc_label_array(iterations)[0:iterations]

class LabeledLineSentence(object):

	def __init__(self, doc_list, labels_list):
		self.labels_list = labels_list
		self.doc_list = doc_list

	def __iter__(self):
		for idx, doc in enumerate(self.doc_list):
			yield LabeledSentence(words=doc.split(), tags=[self.labels_list[idx]])


it = LabeledLineSentence(review_text, document_labels)
# it = LabeledLineSentence(review_text, pos_neg_array)

model = gensim.models.Doc2Vec(size=300, window=10, min_count=5,
 workers=11, alpha=0.025, min_alpha=0.025)

model.build_vocab(it)

for epoch in range(10):
	print "Iteration: " + str(epoch)
	model.train(it)
	model.alpha -= 0.002
	model.min_alpha = model.alpha
	model.train(it)

result_list = []
testing_list = []
training_y = []
testing_y = []
star_training_y = []
star_testing_y = []
for i in range(iterations):
	if i >= int(float(iterations)*3/4):
		testing_list.append(model.docvecs["DOC_" + str(i)])
		star_testing_y.append(stars_array[i])
		if (stars_array[i] >= 3):
			testing_y.append(1)
		else:
			testing_y.append(0)
	else:
		result_list.append(model.docvecs["DOC_" + str(i)])
		star_training_y.append(stars_array[i])
		if (stars_array[i] >= 3):
			training_y.append(1)
		else:
			training_y.append(0)

result_list = np.array(result_list)
testing_list = np.array(testing_list)
training_y = np.array(training_y)
testing_y = np.array(testing_y)
star_training_y = np.array(star_training_y)
star_testing_y = np.array(star_testing_y)

i_values = [1, 2, 3, 5, 10, 20, 30, 50, 100]

best_i = 0
top_accuracy = 0
# for i in range(1, 31):
for i in i_values:
	knn = KNeighborsClassifier(n_neighbors=i, weights='uniform', algorithm='auto', 
		leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)

	knn.fit(result_list, training_y)
	predicted = knn.predict(testing_list)
	count = 0
	for j in range(len(predicted)):
		if predicted[j] == testing_y[j]:
			count += 1

	curr_accuracy = float(count) / len(predicted)
	print i, curr_accuracy
	if (curr_accuracy > top_accuracy):
		top_accuracy = curr_accuracy
		best_i = i

		

best_i2 = 0
top_accuracy2 = 0
# for i in range(1, 31):
for i in i_values:
	knn = KNeighborsClassifier(n_neighbors=i, weights='uniform', algorithm='auto', 
		leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)

	knn.fit(result_list, star_training_y)
	predicted = knn.predict(testing_list)
	count = 0
	for j in range(len(predicted)):
		if predicted[j] == star_testing_y[j]:
			count += 1

	curr_accuracy = float(count) / len(predicted)
	print i, curr_accuracy
	if (curr_accuracy > top_accuracy2):
		top_accuracy2 = curr_accuracy
		best_i2 = i

"""
knn = KNeighborsClassifier(n_neighbors=4, weights='uniform', algorithm='auto', 
	leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)

knn.fit(result_list, training_y)

predicted = knn.predict(testing_list)

count = 0
for i in range(len(predicted)):
	if predicted[i] == testing_y[i]:
		count += 1

print "accuracy " + str(float(count) / len(predicted))

print("Classification report for classifier %s:\n%s\n"
      % (knn, metrics.classification_report(testing_y, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(testing_y, predicted))

# knn2 = KNeighborsClassifier()
# knn2.fit(result_list, star_training_y)

# predicted2 = knn2.predict(testing_list)

# print("Classification report for classifier %s:\n%s\n"
#       % (knn2, metrics.classification_report(star_testing_y, predicted2)))
# print("Confusion matrix:\n%s" % metrics.confusion_matrix(star_testing_y, predicted2)) """

