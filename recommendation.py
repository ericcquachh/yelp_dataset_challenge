
import nltk
import re
import json
import codecs
import os
import time
import nltk.classify.util
import operator
import math
import io
import nltk
from nltk.collocations import *
from nltk.tokenize import TweetTokenizer


DATA_PATH = "/Users/User/Desktop/yelp_dataset_challenge_academic_dataset"
file_name = DATA_PATH  + "/yelp_academic_dataset_review.json"
location_name = DATA_PATH + "/yelp_academic_dataset_business.json"


def dictionary_maker(n):
	array_of_stars = [0 for i in range(1569264)] # just gonna hardcode this lol
	total_dictionary = {}
	with open(file_name) as json_file:
		for i, line in enumerate(json_file):
			if i >= n:
				break
			parsed_json = json.loads(line)
			data_key = parsed_json['business_id']
			data_dic = parsed_json['text']
			if data_key not in total_dictionary:
				total_dictionary[data_key] =[]
				total_dictionary[data_key].append((data_dic, data_key))
			else:
				total_dictionary[data_key].append((data_dic, data_key))

	return total_dictionary
	 
def places_maker():
	place_dictionary = {}
	with open(location_name) as location_file:
		for i, line in enumerate(location_file):
			parsed_json= json.loads(line)
			key = parsed_json['business_id']
			name = parsed_json['name']
			review_count = parsed_json['review_count']
			city = parsed_json['city']
			categories = parsed_json['categories']
			stars = parsed_json['stars']

			place_dictionary[key] = (name, city, categories, review_count, stars)

		return place_dictionary
	        	



d = dictionary_maker(100000)
p = places_maker()
test = sorted(d.values(), key=len)[-2]

key = sorted(d.values(), key=len)[-2][0][1]

"""
tknzr = TweetTokenizer()
for sentence in test:
	new_list.append(tknzr.tokenize(sentence))
"""

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

a = ' '.join(word[0] for word in test)

text_file = open("words.txt", "w+")
text_file.write(a.encode('utf8') + '\n')

text_file.close()
# change this to read in your data
finder3 = TrigramCollocationFinder.from_words(nltk.corpus.genesis.words("words.txt"))
finder2= BigramCollocationFinder.from_words(nltk.corpus.genesis.words("words.txt"))


# only bigrams that appear 3+ times         
finder3.apply_freq_filter(3) 
finder2.apply_freq_filter(3)
# return the 10 n-grams with the highest PMI
print finder3.nbest(trigram_measures.pmi, 15) 
print finder2.nbest(bigram_measures.pmi, 15)
print key
print p[key]

