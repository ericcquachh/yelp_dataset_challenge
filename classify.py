# Time to build a very very basic Naive Bayes classifer that only focuses on positive and negative reivews
# and a second classifier based on each rating.
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier

def word_feat(words):
    return dict([(word, True) for word in words])

positive_reviews = []
negative_reviews = []
one_star = []
two_star = []
three_star = []
four_star = []
five_star = []
for i, star in enumerate(stars_array):
    padding = '0' * (4 - len(str(i)))
    file_name = 'review_text' + padding + str(i) + '.txt'
    if star == 5:
        five_star.append(file_name)
    if star == 4:
        four_star.append(file_name)
    if star == 3:
        three_star.append(file_name)
    if star == 2:
        two_star.append(file_name)
    if star == 1:
        one_star.append(file_name)
    if star > 3:
        positive_reviews.append(file_name)
    else:
        negative_reviews.append(file_name)
   
print len(one_star)
print len(two_star)
print len(three_star)
print len(four_star)
print len(five_star)

negative_features = []
positive_features = []

for elem in positive_reviews:
    p_file = open(elem)
    loc_positive_features = []
    for line in p_file:
        line_array = re.split('\W+', line)
        for elem in line_array:
            if elem != '':
                loc_positive_features.append(elem)
    dictionary = word_feat(loc_positive_features)
    positive_features.append((dictionary, 'positive'))
    
for elem in negative_reviews:
    n_file = open(elem)
    loc_negative_features = []
    for line in n_file:
        line_array = re.split('\W+', line)
        for elem in line_array:
            if elem != '':
                loc_negative_features.append(elem)
    dictionary = word_feat(loc_negative_features)
    negative_features.append((dictionary, 'negative'))    
                
neg_cutoff = int(len(negative_features) * 3/4)
pos_cutoff = int(len(positive_features) * 3/4)

training = negative_features[:neg_cutoff] + positive_features[:pos_cutoff]
testing = negative_features[neg_cutoff:] + positive_features[pos_cutoff:]

classifier = NaiveBayesClassifier.train(training)
print 'Our accuracy is', nltk.classify.util.accuracy(classifier, testing)

five_features = []
four_features = []
three_features = []
two_features = []
one_feature = []
for elem in five_star:
    n_file = open(elem)
    loc_five = []
    for line in n_file:
        line_array = re.split('\W+', line)
        for elem in line_array:
            if elem != '':
                loc_five.append(elem)
                five_features.append(({elem:True}, 'five'))
    #dictionary = word_feat(loc_five)
    #five_features.append((dictionary, 'five')) 

for elem in four_star:
    n_file = open(elem)
    loc_four = []
    for line in n_file:
        line_array = re.split('\W+', line)
        for elem in line_array:
            if elem != '':
                loc_four.append(elem)
                four_features.append(({elem:True}, 'four'))
    #dictionary = word_feat(loc_four)
    #four_features.append((dictionary, 'four'))
    
for elem in three_star:
    n_file = open(elem)
    loc_three = []
    for line in n_file:
        line_array = re.split('\W+', line)
        for elem in line_array:
            if elem != '':
                loc_three.append(elem)
                three_features.append(({elem:True}, 'three'))
    #dictionary = word_feat(loc_three)
    #three_features.append((dictionary, 'three'))
    
for elem in two_star:
    n_file = open(elem)
    loc_two = []
    for line in n_file:
        line_array = re.split('\W+', line)
        for elem in line_array:
            if elem != '':
                loc_two.append(elem)
                two_features.append(({elem:True}, 'two'))
    #dictionary = word_feat(loc_two)
    #two_features.append((dictionary, 'two'))
    
for elem in one_star:
    n_file = open(elem)
    loc_one = []
    for line in n_file:
        line_array = re.split('\W+', line)
        for elem in line_array:
            if elem != '':
                loc_one.append(elem)
                one_feature.append(({elem:True}, 'one'))
    #dictionary = word_feat(loc_one)
    #one_feature.append((dictionary, 'one'))
    
one_cutoff = int(len(one_feature) * 3/4)
two_cutoff = int(len(two_features) * 3/4)
three_cutoff = int(len(three_features) * 3/4)
four_cutoff = int(len(four_features) * 3/4)
five_cutoff = int(len(five_features) * 3/4)

training = one_feature[:one_cutoff] + two_features[:two_cutoff] + three_features[:three_cutoff] + four_features[:four_cutoff] + five_features[:five_cutoff]
testing = one_feature[one_cutoff:] + two_features[two_cutoff:] + three_features[three_cutoff:] + four_features[four_cutoff:] + five_features[five_cutoff:]    

classifier = NaiveBayesClassifier.train(training)
print 'Our accuracy is', nltk.classify.util.accuracy(classifier, testing)