import os, json
import numpy as np 
import pandas as pd 
from nltk.corpus import stopwords
try:
    import statsmodels.api as sm
except ImportError:
    import scikits.statsmodels.api as sm

from sklearn.feature_extraction.text import CountVectorizer

DATA_PATH = "/Users/Janet/Downloads/CS194/yelp_dataset_challenge_academic_dataset/"
file_name = DATA_PATH  + "yelp_academic_dataset_review.json"

MAX_FILES = 1569264

data_dir = 'reviews_txt/'
filenames = np.array(os.listdir(data_dir))

filenames_with_path = [os.path.join(data_dir, fn) for fn in filenames]
vectorizer = CountVectorizer(input='filename', min_df=1, max_df=.95, stop_words=stopwords.words('english'), max_features=3000)
dtm = vectorizer.fit_transform(filenames_with_path)
dtm = dtm.toarray()
vocab = np.array(vectorizer.get_feature_names())

# dtm is matrix (document-term matrix)
# 	row is review_text file #
# 	column is word count for a word (vocab tells us which word based on index)
# for row in dtm:
# 	print row[0]

""" Retrieve the stars of each review in their corresponding array. """
def star_array(n):
	array_of_stars = [0 for i in range(100)] # just gonna hardcode this lol
	with open(file_name) as json_file:
		for i, line in enumerate(json_file):
			if i >= n:
				break
			parsed_json = json.loads(line)
			stars = parsed_json['stars']
			array_of_stars[i] = stars
	return array_of_stars

stars = star_array(100)
pn = [] # need to determine which rows (files) are pos (> 3) and neg
for elem in stars:
	if elem > 2:
		pn.append(1)
	else:
		pn.append(0)

num_pos = np.count_nonzero(pn)
num_neg = 100 - num_pos

count_of_first_word = dtm[:, 0]
X = sm.add_constand(count_of_first_word)
model = sm.GLM(X)


pos_rows = []
neg_rows = []
for i in xrange(len(dtm)):
	row = dtm[i]
	print row 
	if pn[i]: # pos 
		pos_rows.append(row)
	else:
		neg_rows.append(row)

for i in xrange(len(vocab)):
	# for positive
	word = vocab[i].encode('ascii', 'ignore')

	count = np.count_nonzero(pos_rows[:,i])
	ppercent = count / float(num_pos)
	print word + ', ' + str(ppercent) # vocab word + percent 
	
	count = np.count_nonzero(neg_rows[:,i])
	npercent = count / float(num_neg) 
	print word + ', ' + str(npercent) # vocab word + 

for i in xrange(len(dtm)):
	row = dtm[i]
	rating = stars[i]
	feel = pn[i]


# print np.count_nonzero(pn) # 83 pos

# print dtm[:, 2]

for i in xrange(len(vocab)):
	word = vocab[i]
