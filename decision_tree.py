""" DECISION TREE """

# For reference
# http://www.nltk.org/_modules/nltk/classify/decisiontree.html

import nltk
import json
import time
import operator
import math

from textblob import TextBlob as tb

from collections import defaultdict

from nltk import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist, MLEProbDist, entropy
from nltk.classify.api import ClassifierI
from nltk.compat import python_2_unicode_compatible
from nltk.classify.decisiontree import DecisionTreeClassifier

# included as a test to understand decision tree example
from nltk.classify.util import names_demo, binary_names_demo_features

# Exists soley for the purpose of me understanding what's happening.
# Since binary is set to True as a default in this example, it can only classify feature/value pairs
# rather than an n-way branch.
def f(x):
	return DecisionTreeClassifier.train(x, binary=True, verbose=True)

def demo():
	classifier = names_demo(f, binary_names_demo_features)
	#print (classifier.pp(depth=7))
	print (classifier.pseudocode(depth=7))


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

#def freq(word, ):

# def n_containing(word, bloblist):
# 	return sum(1 for blob in bloblist if word in blob)

def tf(word, blob):
	#value = float(blob.words.count(word)) / len(blob.words)
	#print value
	return float(blob.words.count(word)) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
	#value = float(math.log(len(bloblist)) / (1 + n_containing(word, bloblist)))
	#print value
   	return float(math.log(len(bloblist)) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


def word_feat(words):
    return dict([(word, True) for word in words])

DATA_PATH = "/Users/colin.garcia/Desktop/yelp_dataset_challenge_academic_dataset/"
file_name = DATA_PATH  + "yelp_academic_dataset_review.json"
iterations = 5000

print "Starting Classification for " + str(iterations) + " reviews."

start_time = time.time()
stars_array = star_array(iterations)
review_text = text_array(iterations)

print "Finished setting up the Star Array and Text Array. It took " + str(time.time() - start_time) + " to run."
start_time = time.time()

total_text = []
for i in xrange(iterations): 
	if (i % (iterations/10) == 0): 
		print "We have finished " + str(int((float(i) / iterations) * 100)) + " percent of STOP_WORDS iterations"
	words = RegexpTokenizer(r'\w+').tokenize(review_text[i])

	""" FILTERED OUT THE STOPWORDS. """
	filtered_words = []
	for word in words:
		word = word.lower()
		if word not in set(stopwords.words('english')):
			if len(word) != 1:
				filtered_words.append(word)

	total_text.append(filtered_words)

	""" Format of total text is array of arrays, with each first index as a list
		of all the non stop words contained within """

print "Finished filtering the Stop Words. It took " + str(time.time() - start_time) + " to run."
start_time = time.time()

bloblist = []
for i in total_text:

    string = ' '.join(i)
    bloblist.append(tb(string))

positive_stars = {}
negative_stars = {}
for i, blob in enumerate(bloblist):

	if (i % (iterations/10) == 0): 
		print "We have finished " + str(int((float(i) / iterations) * 100)) + " percent of BLOBLIST iterations"

	scores = {word: n_containing(word, bloblist) for word in blob.words} # Number of documents a word occurs.
	#scores = {word: tfidf(word, blob, bloblist) for word in blob.words} # TFIDF
   	sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
   	for word, score in sorted_words:
   		w = word
   		if stars_array[i] >= 3:
   			if w not in positive_stars:
   				positive_stars[w] = [score]
   			else:
   				positive_stars[w] += [score]
   		else:
   			if w not in negative_stars:
   				negative_stars[w] = [score]
   			else:
   				negative_stars[w] += [score]

print "Finished Blob List iterations. It took " + str(time.time() - start_time) + " to run."
start_time = time.time()

for key, value in positive_stars.iteritems():
	positive_stars[key] = round(sum(value)/len(value), 5)

for key, value in negative_stars.iteritems():
	negative_stars[key] = round(sum(value)/len(value), 5)

print len(positive_stars), len(negative_stars)

pos = sorted(positive_stars.items(), key=operator.itemgetter(1), reverse=True)[0:2000] # should I change this number?
neg = sorted(negative_stars.items(), key=operator.itemgetter(1), reverse=True)[0:2000]

combined = pos + neg
feature_set = []
for i, elem in enumerate(total_text):
	# elem is the words in a document
	features = {}
	for word, number in combined:
		features['contains({})'.format(word)] = (word in elem)
	pos_or_neg = "positive" if (stars_array[i] >= 3) else "negative"
	feature_set.append((features, pos_or_neg))

print "Finished Featurization. It took " + str(time.time() - start_time) + " to run."

cutoff = int(len(feature_set) * 3/4)
training = feature_set[:cutoff]
testing = feature_set[cutoff:]

classifier = DecisionTreeClassifier.train(training, binary=True, verbose=True)

count = 0
for elem in testing:
	val = classifier.classify(elem[0])
	if val == elem[1]:
		count += 1

print float(count)/len(testing)


