"""TF-IDF is the number intended to reflect how important a word is to a document in a collection. In this case, if a word appears in many documents, it in not important, and it would have a lower score. However, if i a word appears frequently in a document, then it is important, therefore having a higher score. We were able to calculate the TF-IDF by multiplying the term frequency, the number of times a word appears in a document, with the inverse document frequency, the number of times the word appears among all documents. We were able to verify for the most part that our calculation correlated to the TF-IDF numbers. For example, for five star reviews, we received words like "awesome" and "good". While for one star reviews, we received words like "waste".

Note: To run the code included below you must install TextBlob with the following commands:

sudo pip install -U textblob

python -m textblob.download_corpora
"""

import math
from textblob import TextBlob as tb
from __future__ import division, unicode_literals

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


bloblist = []
import operator


for num in range(100):
    review = "review_text" + str(num).zfill(4) + ".txt"
    file = open(review)
    t = file.read() 
    t = t.decode("utf8")
        
    bloblist.append(tb(t))
    
     
five_stars = {}
four_stars = {}
three_stars = {}
two_stars = {}
one_star = {}
for i, blob in enumerate(bloblist):
    #print("Top words in document {}".format(i + 1))
    
    review = "review_text" + str(i).zfill(4) + ".txt"
    
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words:
        #print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
        w = word
        if text_to_stars[review] == 5:
            
            if w not in five_stars:
                five_stars[w] = [score]
            else:
                five_stars[w] += [score]
        elif text_to_stars[review] == 4:
            if w not in four_stars:
                four_stars[w] = [score]
            else:
                four_stars[w] += [score]
        elif text_to_stars[review] == 3:
            if w not in three_stars:
                three_stars[w] = [score]
            else:
                three_stars[w] += [score]
        elif text_to_stars[review] == 2:
            if str(word) not in two_stars:
                two_stars[w] = [score]
            else:
                two_stars[w] += [score]
        elif text_to_stars[review] == 1:
            if w not in one_star:
                one_star[w] = [score]
            else:
                one_star[w] += [score]

for key, value in five_stars.iteritems():
    five_stars[key] = round(sum(value)/len(value),5)
                
for key, value in four_stars.iteritems():
    four_stars[key] = round(sum(value)/len(value),5)              

for key, value in three_stars.iteritems():
    three_stars[key] = round(sum(value)/len(value),5)
    
for key, value in two_stars.iteritems():
    two_stars[key] = round(sum(value)/len(value),5)
    
for key, value in one_star.iteritems():
    one_star[key] = round(sum(value)/len(value),5)

five = sorted(five_stars.items(), key=operator.itemgetter(1), reverse = True)[:10]
four = sorted(four_stars.items(), key=operator.itemgetter(1), reverse=True)[:10]
three = sorted(three_stars.items(), key=operator.itemgetter(1), reverse=True)[:10]
two = sorted(two_stars.items(), key=operator.itemgetter(1), reverse=True)[:10]
one = sorted(one_star.items(), key=operator.itemgetter(1), reverse=True)[:10]

print "Top Words with Highest tf-dfs"
print " "
temp = []
for i in five:
    key = str(i[0])
    val = i[1]
    temp.append((key,val))
five = temp
temp = []
print "Five Stars"
print five

for i in four:
    key = str(i[0])
    val = i[1]
    temp.append((key,val))
four = temp
temp = []
print "Four Stars"
print four

for i in three:
    key = str(i[0])
    val = i[1]
    temp.append((key,val))
three = temp
temp = []
print "Three Stars"
print three

for i in two:
    key = str(i[0])
    val = i[1]
    temp.append((key,val))
two = temp
temp = []
print "Two Stars"
print two

for i in one:
    key = str(i[0])
    val = i[1]
    temp.append((key,val))
one = temp
temp = []
print "One Star"
print one


# FULL TF-IDFS
# TF-IDF for five-stars
full_five = sorted(five_stars.items(), key=operator.itemgetter(1), reverse = True)
full_four = sorted(four_stars.items(), key=operator.itemgetter(1), reverse=True)
full_three = sorted(three_stars.items(), key=operator.itemgetter(1), reverse=True)
full_two = sorted(two_stars.items(), key=operator.itemgetter(1), reverse=True)
full_one = sorted(one_star.items(), key=operator.itemgetter(1), reverse=True)

temp = []
for i in full_five:
    key = i[0].encode('utf-8')
    val = i[1]
    temp.append((key,val))
full_five = temp
temp = []

for i in full_four:
    key = i[0].encode('utf-8')
    val = i[1]
    temp.append((key,val))
full_four = temp
temp = []

for i in full_three:
    key = i[0].encode('utf-8')
    val = i[1]
    temp.append((key,val))
full_three = temp
temp = []

for i in full_two:
    key = i[0].encode('utf-8')
    val = i[1]
    temp.append((key,val))
full_two = temp
temp = []

for i in full_one:
    key = i[0].encode('utf-8')
    val = i[1]
    temp.append((key,val))
full_one = temp
temp = []
