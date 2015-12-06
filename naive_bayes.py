""" alright NAIVE BAYES CLASSIFIER """ 

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
file_name = DATA_PATH  + "yelp_academic_dataset_review.json"

""" Retrieve the number of lines in a json file. 
	For this project we have 1,569,264 different reviews. """
def num_lines():
	counter = 0
	with open(file_name) as json_file:
		for i, line in enumerate(json_file):
			counter += 1

	return counter


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

def tf(word, blob):
	#value = float(blob.words.count(word)) / len(blob.words)
	#print value
	return float(blob.words.count(word)) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
	#value = float(math.log(len(bloblist)) / (1 + n_containing(word, bloblist)))
	#print value
   	return float(math.log(float(len(bloblist)) / (1 + n_containing(word, bloblist))))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


def main():
	start_time = time.time()
	stars_array = star_array()
	review_text = text_array()
	print "That shit ran in: ", time.time() - start_time 
	return review_text

def word_feat(words):
    return dict([(word, True) for word in words])

# CHANGE NUMBER FOR NUMBER OF DOCUMENTS YOU WISH TO TEST.
iterations = 10000

stars_array = star_array(iterations)
review_text = text_array(iterations)
positive_reviews = []
negative_reviews = []

print "We have set up our stars and text."

""" Thus far the features we have implemented are:
	1) Taking the top 2K words in TFIDF and contains(word): True
	2) filtered_words : True 

	Possible Extensions:
	2) Take the top 1000 words returned from TFIDF?
	3) Previous Words (probably can't filter in this instance.)
	4) Punctuation?
	5) 

"""
total_text = []
for i in xrange(iterations): 
	if (i % (iterations/10) == 0): 
		print "We have finished " + str(int((float(i) / iterations) * 100)) + " percent of STOP_WORDS iterations"
	# padding = 2 - len(str(i)) 
	# f = open('reviews_txt/review_text' + ('0' * padding) + str(i) + '.txt')
	# text = f.read()
	# words = word_tokenize(review_text[i])
	# print words
	words = RegexpTokenizer(r'\w+').tokenize(review_text[i])

	#filtered_words = [word.lower() for word in words if (word not in set(stopwords.words('english')) or len(word) != 1)]

	""" FILTERED OUT THE STOPWORDS. """
	filtered_words = []
	for word in words:
		word = word.lower()
		if word not in set(stopwords.words('english')):
			if len(word) != 1:
				filtered_words.append(word)

	if stars_array[i] > 3:
		positive_reviews.append(filtered_words)
	#elif (stars_array[i] == 3):
	#	continue
	else:
		negative_reviews.append(filtered_words)
	total_text.append(filtered_words)
	
	""" Format of total text is array of arrays, with each first index as a list
		of all the non stop words contained within """

print "Stop words done."

#print total_text

bloblist = []
for i in total_text:

    string = ' '.join(i)
    bloblist.append(tb(string))

# print bloblist

positive_stars = {}
negative_stars = {}
for i, blob in enumerate(bloblist):

	if (i % (iterations/10) == 0): 
		print "We have finished " + str(int((float(i) / iterations) * 100)) + " percent of BLOBLIST iterations"
	# padding = 2 - len(str(i)) 	#print blob.words
	#print blob.words.count()

	scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
	#print scores
   	sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
   	for word, score in sorted_words:
   		w = word
   		#print word, score
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

print "Blob list iterations are done."
#print positive_stars
#print negative_stars
for key, value in positive_stars.iteritems():
	positive_stars[key] = round(sum(value)/len(value), 5)
	#print positive_stars[key]

for key, value in negative_stars.iteritems():
	negative_stars[key] = round(sum(value)/len(value), 5)

print len(positive_stars), len(negative_stars)

pos = sorted(positive_stars.items(), key=operator.itemgetter(1), reverse=True)[:2000] # should I change this number?
neg = sorted(negative_stars.items(), key=operator.itemgetter(1), reverse=True)[:2000]

combined = pos + neg
feature_set = []
for i, elem in enumerate(total_text):
	# elem is the words in a document
	features = {}
	for word, number in combined:
		features['contains({})'.format(word)] = (word in elem)
	#print features
	#raw_input()
	pos_or_neg = "positive" if (stars_array[i] >= 3) else "negative"
	# if stars_array[i] >= 3:
	# 	pos_or_neg = 'pos'
	# else:
	# 	pos_or_neg = 'neg'
	feature_set.append((features, pos_or_neg))
	#print feature_set

cutoff = int(len(feature_set) * 3/4)
training = feature_set[:cutoff]
testing = feature_set[cutoff:]
first_classifier = NaiveBayesClassifier.train(training)
print 'Our accuracy is', nltk.classify.util.accuracy(first_classifier, testing)


#print pos, len(pos)
#print neg, len(neg)

#for i, star in enumerate(stars_array):
    #padding = '0' * (4 - len(str(i)))
    #file_name = 'review_text' + padding + str(i) + '.txt'
    #if star >= 3:
     #   positive_reviews.append(file_name)
    #else:
     #   negative_reviews.append(file_name)

negative_features = []
positive_features = []

""" Length of review, weight specific words, sentiments... """

for elem, number in pos:
#for elem in positive_reviews: # switch this out yolo
    #p_file = open(elem)
    #loc_positive_features = []
    #for line in p_file:
    #    line_array = re.split('\W+', line)
    #    for elem in line_array:
    #        if elem != '':
    #            loc_positive_features.append(elem)
    dictionary = word_feat(elem)
    positive_features.append((dictionary, 'positive'))
    
for elem, number in neg:
#for elem in negative_reviews:
    #n_file = open(elem)
    #loc_negative_features = []
    #for line in n_file:
    #    line_array = re.split('\W+', line)
    #    for elem in line_array:
    #        if elem != '':
    #            loc_negative_features.append(elem)
    dictionary = word_feat(elem)
    negative_features.append((dictionary, 'negative'))    
                
neg_cutoff = int(len(negative_features) * 3/4)
pos_cutoff = int(len(positive_features) * 3/4)

training = negative_features[:neg_cutoff] + positive_features[:pos_cutoff]
testing = negative_features[neg_cutoff:] + positive_features[pos_cutoff:]

classifier = NaiveBayesClassifier.train(training)
print 'Our accuracy is', nltk.classify.util.accuracy(classifier, testing)

