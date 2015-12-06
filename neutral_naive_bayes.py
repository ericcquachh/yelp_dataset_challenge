""" the kids are alright """ 

import nltk
import re
import json
import codecs
import os
import time
import nltk.classify.util
import operator
import math

# Need to run these download lines once:
# nltk.download("stopwords") 
# nltk.download("punkt")
from textblob import TextBlob as tb
from nltk import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier

#DATA_PATH = "/Users/ericquach/Github/yelp_data/"
DATA_PATH = "/Users/colin.garcia/Desktop/yelp_dataset_challenge_academic_dataset/"
# DATA_PATH = "/Users/Janet/Downloads/CS194/yelp_dataset_challenge_academic_dataset/"
file_name = DATA_PATH  + "yelp_academic_dataset_review.json"

NUM_REVIEWS = 1569264
iterations = 3000 			# CHANGE NUMBER FOR NUMBER OF DOCUMENTS YOU WISH TO TEST
NUM_TRAIN = 4000			# Change number for number to train on for each rating/bucket

# For colin's personal pleasure:
# """ Retrieve the number of lines in a json file. 
# 	For this project we have 1,569,264 different reviews. """
# def num_lines():
# 	counter = 0
# 	with open(file_name) as json_file:
# 		for i, line in enumerate(json_file):
# 			counter += 1
# 	return counter

""" Extract the stars of each review. """
def star_array(n):
	array_of_stars = [0 for i in range(NUM_REVIEWS)]
	with open(file_name) as json_file:
		for i, line in enumerate(json_file):
			if i >= n:
				break
			parsed_json = json.loads(line)
			stars = parsed_json['stars']
			array_of_stars[i] = stars
	return array_of_stars

""" Extract the text of each review. """
def text_array(n):
	array_of_text = [0 for i in range(NUM_REVIEWS)]
	with open(file_name) as json_file:
		for i, line in enumerate(json_file):
			if i >= n:
				break
			parsed_json = json.loads(line)
			text = parsed_json['text']
			array_of_text[i] = text
	return array_of_text

def tf(word, blob):
	return float(blob.words.count(word)) / len(blob.words)

""" Number of documents that contain word in corpus, which is bloblist. """
def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
   	return float(math.log(len(bloblist)) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


if __name__ == "__main__":
	stars_array = star_array(iterations)
	review_text = text_array(iterations)
	positive_reviews = []
	negative_reviews = []
	print "We have set up our stars and text."

	""" FILTERING OUT STOPWORDS. """
	total_text = []
	for i in xrange(iterations): 
		if (i % (iterations/10) == 0): 
			print "We have finished " + str(int((float(i) / iterations) * 100)) + " percent of STOP_WORDS iterations"
		words = RegexpTokenizer(r'\w+').tokenize(review_text[i])

		filtered_text = []
		for word in words:
			word = word.lower()
			if word not in set(stopwords.words('english')):
				if len(word) != 1:
					filtered_text.append(word)

		""" Format of total text is array of arrays, with each entry being the text of a document with all stop words
			filtered out. """
		total_text.append(filtered_text)

	print "Stop words done."


	""" CALCULATING TFIDF. """

	""" Creating corpus (called bloblist) to be used for TFIDF calculations. """
	bloblist = []
	for i in total_text:
	    string = ' '.join(i)
	    """ Bloblist is a list of strings (text of a document separated by whitespace). """
	    bloblist.append(tb(string))

	negative_stars = {}
	positive_stars = {}
	neutral_stars = {}

	for i, blob in enumerate(bloblist):
		if (i % (iterations/10) == 0): 
			print "We have finished " + str(int((float(i) / iterations) * 100)) + " percent of BLOBLIST iterations"
		# diff tfidf for same word in diff documents

		""" Scores is a dictionary of word to its TFIDF. """
		scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
		
		""" Sorted_words is scores sorted by decreasing TFIDF. """
	   	sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
	   	for w, score in sorted_words:
	   		if stars_array[i] < 3:
	   			if w not in negative_stars:
	   				negative_stars[w] = [score]
	   			else: # can contain repeats becaues the same word can have multiple TFIDFs for different documents
	   				negative_stars[w] += [score]
	   		elif stars_array[i] == 3:
	   			if w not in neutral_stars:
	   				neutral_stars[w] = [score]
	   			else:
	   				neutral_stars[w] += [score]
	   		elif stars_array[i] > 3:
	   			if w not in positive_stars:
	   				positive_stars[w] = [score]
	   			else:
	   				positive_stars[w] += [score]

	print "Blob list iterations are done."
	for key, value in negative_stars.iteritems():
		negative_stars[key] = round(sum(value)/len(value), 5)

	for key, value in neutral_stars.iteritems():
		neutral_stars[key] = round(sum(value)/len(value), 5)

	for key, value in positive_stars.iteritems():
		positive_stars[key] = round(sum(value)/len(value), 5)

	neg = sorted(negative_stars.items(), key=operator.itemgetter(1), reverse=True)[:NUM_TRAIN]
	neu = sorted(neutral_stars.items(), key=operator.itemgetter(1), reverse=True)[:NUM_TRAIN]
	pos = sorted(positive_stars.items(), key=operator.itemgetter(1), reverse=True)[:NUM_TRAIN]

	combined = neg + neu + pos
	feature_set = []
	for i, text in enumerate(total_text):
		if (i % (iterations/10) == 0): 
			print "We have finished " + str(int((float(i) / iterations) * 100)) + " percent of FEATURIZATION"

		features = {}
		for word, number in combined:
			features['contains({})'.format(word)] = (word in text)

		pos_neg_neu = ""
		if stars_array[i] > 3:
			pos_neg_neu = "positive"
		elif stars_array[i] == 3:
			pos_neg_neu = "neutral"
		else:
			pos_neg_neu = "negative"

		feature_set.append((features, pos_neg_neu)) 
		#print feature_set

	cutoff = int(len(feature_set) * 3/4)
	training = feature_set[:cutoff]
	testing = feature_set[cutoff:]
	first_classifier = NaiveBayesClassifier.train(training)
	print 'Our accuracy is', nltk.classify.util.accuracy(first_classifier, testing)