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

from sklearn import svm, metrics

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


DATA_PATH = "/Users/colin.garcia/Desktop/yelp_dataset_challenge_academic_dataset/"
file_name = DATA_PATH  + "yelp_academic_dataset_review.json"
iterations = 5000

print "Starting Classification for " + str(iterations) + " reviews."

start_time = time.time()
stars_array = star_array(iterations)[0:iterations]
pos_neg_array = positive_negative_array(stars_array)[0:iterations]
review_text = text_array(iterations)[0:iterations] # possibly train on more data

class LabeledLineSentence(object):

	def __init__(self, doc_list, labels_list):
		self.labels_list = labels_list
		self.doc_list = doc_list

	def __iter__(self):
		for idx, doc in enumerate(self.doc_list):
			yield LabeledSentence(words=doc.split(), tags=[self.labels_list[idx]])


it = LabeledLineSentence(review_text, pos_neg_array)

model = gensim.models.Doc2Vec(size=100, window=10, min_count=5,
 workers=11, alpha=0.025, min_alpha=0.025)

model.build_vocab(it)

for epoch in range(10):
	print "Iteration: " + str(epoch)
	model.train(it)
	model.alpha -= 0.002
	model.min_alpha = model.alpha
	model.train(it)

#result_list = model['positive'][0:int(float(len(model['positive']))*3/4)] + model['negative'][0:int(float(len(model['positive']))*3/4)]
# for i in range(int(float(iterations)*3/4)):
# 	result_list.append(model())
#test_list = model['positive'][int(float(iterations)*3/4):len(model['positive'])] + model['negative'][int(float(iterations)*3/4):len(model['negative'])]

result_list = []
testing_list = []
training_y = []
testing_y = []
for i in range(len(model['positive'])):
	if i >= int(float(iterations)*3/4):
		testing_list.append(model['positive'][i])
		testing_y.append(1)
	else:
		result_list.append(model['positive'][i])
		training_y.append(1)

for i in range(len(model['negative'])):
	if i >= int(float(iterations)*3/4):
		testing_list.append(model['negative'][i])
		testing_y.append(0)
	else:
		result_list.append(model['negative'][i])
		training_y.append(0)

result_list = np.array(result_list)
testing_list = np.array(testing_list)
training_y = np.array(training_y)
testing_y = np.array(testing_y)

#training_y = [1 for i in range(int(float(iterations)*3/4))]
#testing_y = [1 for i in range(int(float(iterations)*3/4), len(model['positive']))]

svc = svm.SVC(kernel='linear')
svc.fit(result_list.reshape(-1, 1), training_y)

predicted = svc.predict(testing_list.reshape(-1, 1))

print metrics.classification_report(testing_y, predicted)









