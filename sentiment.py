""" alright """ 
import random
import nltk
import re
import json
import codecs
import os
import nltk
import time
import nltk.classify.util

# Need to run these download lines once:
# nltk.download("stopwords") 
# nltk.download("punkt")
from nltk import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier

#DATA_PATH = "/Users/ericquach/Github/yelp_data/"
#DATA_PATH = "/Users/colin.garcia/Desktop/yelp_dataset_challenge_academic_dataset/"
DATA_PATH = "/Users/User/Desktop/yelp_dataset_challenge_academic_dataset/"
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


def main():
	start_time = time.time()
	stars_array = star_array()
	review_text = text_array()
	print "That shit ran in: ", time.time() - start_time 
	return review_text

def word_feat(words):
    return dict([(word, True) for word in words])

iterations = 1000

stars_array = star_array(iterations)
review_text = text_array(iterations)
positive_reviews = []
negative_reviews = []

print "We have set up our stars and text."

""" """
total_text = []
for i in xrange(iterations): 
	if (i % (iterations/10) == 0):
		print "We have finished " + str(int((float(i) / iterations) * 100)) + " percent of iterations"
	# padding = 2 - len(str(i)) 
	# f = open('reviews_txt/review_text' + ('0' * padding) + str(i) + '.txt')
	# text = f.read()
	# words = word_tokenize(review_text[i])
	# print words
	words = RegexpTokenizer(r'\w+').tokenize(review_text[i])

	#filtered_words = [word.lower() for word in words if (word not in set(stopwords.words('english')) or len(word) != 1)]
	filtered_words = []
	for word in words:
		word = word.lower()
		if word not in set(stopwords.words('english')):
			if len(word) != 1:
				filtered_words.append(word)

	if stars_array[i] > 3:
		positive_reviews.append(filtered_words)
	else:
		negative_reviews.append(filtered_words)
	total_text.append(filtered_words)
	
#print total_text

print "Stop words done."


print len(negative_reviews)
def get_words_in_reviews(positive_reviews, negative_reviews):
	all_words = []
	for sentence in positive_reviews:
		all_words.extend(sentence)
	for sentence in negative_reviews:
		all_words.extend(sentence)
	return all_words


def get_word_features(all_words):
	wordlist = nltk.FreqDist(all_words)
	word_features = wordlist.keys()
	return word_features

word_features = get_word_features(get_words_in_reviews(positive_reviews, negative_reviews))

def extract_features(document):
	document_words = set(document)
	features = {}
	for word in word_features:
		features['contains(%s)' % word] = (word in document_words)
	return features


reviews = [(word, 'positive') for word in positive_reviews]
reviews += [(word, 'negative') for word in negative_reviews]
random.shuffle(reviews)


training_set = nltk.classify.util.apply_features(extract_features, reviews)

classifier = nltk.NaiveBayesClassifier.train(training_set)



