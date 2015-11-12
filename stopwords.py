### REMOVING STOP WORDS ###

import nltk
import re

# Need to run these download lines once:
# nltk.download("stopwords") 
# nltk.download("punkt")
from nltk import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords

total_text = [] 
for i in xrange(100): 
	padding = 2 - len(str(i)) 
	f = open('reviews_txt/review_text' + ('0' * padding) + str(i) + '.txt') 
	text = f.read()
	words = word_tokenize(text)
	words = RegexpTokenizer(r'\w+').tokenize(text) # [^A-Za-z0-9]+
	filtered_words = [word.lower() for word in words if (word not in stopwords.words('english') and len(word) != 1)]
	total_text.append(filtered_words)
#     print filtered_words
# print total_text

### EXTRACING ROOT WORDS ###
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

stemmed_total_text = []
for text in total_text:
    text = [stemmer.stem(word).encode('utf-8') for word in text]
    stemmed_total_text.append(text)
    
# print stemmed_total_text

