import json
from bokeh.charts import Bar, output_file, show
from pandas import DataFrame as df
from bokeh.models.ranges import FactorRange
from bokeh.plotting import *

DATA_PATH = "/Users/User/Desktop/yelp_dataset_challenge_academic_dataset"
file_name = DATA_PATH  + "/yelp_academic_dataset_review.json"

"""s
def total_review_stars():
	array_of_stars = [0 for i in range(6)]
	with open(file_name) as json_file:
		for i, line in enumerate(json_file):
			parsed_json = json.loads(line)
			stars = int(parsed_json['stars'])
			if stars > 0 and stars <= 5:
				array_of_stars[stars] += 1
			else:
				array_of_stars[0] += 1

	print array_of_stars	
	return array_of_stars

#data = total_review_stars()[1:]


"""

"""
stars = [1,2,3,4,5]
data = [159811, 140608, 222719, 466599, 579527]
d = {'Stars': stars, 'Number of Stars': data}
df1 = df(data = d, columns =['Stars', 'Number of Stars'])
print df1

output_file("swag.html")

p = Bar(df1, 'Stars', values = 'Number of Stars', title = "Distribution of Reviews", ylabel = 'Frequency')
p.left[0].formatter.use_scientific = False


show(p)
"""

def parse_elements(list_of_elems):
	list_of_words = []
	list_of_num = []
	for elem in list_of_elems:
		first = elem[0]
		second = elem[1]
		list_of_words.append(first)
		list_of_num.append(second)
	return list_of_words, list_of_num


a = [(u'colada', 1.06465), (u'mmmmmmmmmmmmmmmmmmmmm', 0.85172), (u'brunched', 0.85172), (u'combinacion', 0.85172), (u'altar', 0.85172), (u'heckled', 0.70977), (u'guitarist', 0.70977), (u'benched', 0.60837), (u'suprisingly', 0.60837), (u'jombot', 0.60837), (u'cpt', 0.53232), (u'pubgrub', 0.53232), (u'prepaired', 0.53232), (u'bestpapersonaltrainercom', 0.53232), (u'touristic', 0.47318), (u'delucas', 0.47318), (u'stickler', 0.47318), (u'parfaits', 0.47318), (u'mochas', 0.45543), (u'bearclaw', 0.42586), (u'isntead', 0.42586), (u'portapottys', 0.42586), (u'centerpieces', 0.42586), (u'newcastle', 0.42586), (u'anf', 0.42586), (u'hollywoodlanes', 0.42586), (u'hideaway', 0.38715), (u'haiku', 0.38715), (u'muffaletta', 0.38715), (u'anazing', 0.38715),(u'balboa', 0.06347), (u'novak', 0.06342), (u'aamco', 0.06337), (u'faded', 0.06317), (u'jojo', 0.06312), (u'unos', 0.06309), (u'proves', 0.06309), (u'refurbish', 0.06309), (u'fleet', 0.06309), (u'koh', 0.06309)]

list_of_words, list_of_num = parse_elements(a)

model = {'TF-IDF': list_of_num, 'Words': list_of_words}
df1 = df(data = model, columns = ['TF-IDF','Words'])

graph1 = Bar(df1, 'Words', values = 'TF-IDF', title = "TF-IDF of top positive words", ylabel = 'TF-IDF  scores', width = 800, height = 400, color = 'blue')
graph1.left[0].formatter.use_scientific= False
graph1.x_range = FactorRange(factors=df1['Words'].tolist())



c = [(u'hoofah', 4.2586), (u'ewwww', 4.2586), (u'disconnected', 0.70977), (u'precooked', 0.70977), (u'crimes', 0.60837), (u'greasiest', 0.60837), (u'redevelopment', 0.60837), (u'sloooooow', 0.53232), (u'yucky', 0.53232), (u'horrifying', 0.53232), (u'rancho', 0.47318), (u'crisco', 0.47318), (u'grabn', 0.42586), (u'gnocci', 0.42586), (u'strictest', 0.38715), (u'clout', 0.38715), (u'unenjoyable', 0.38715), (u't3', 0.36502), (u'terriable', 0.32758), (u'foe', 0.32758), (u'pararell', 0.32758), (u'thrifters', 0.32758), (u'society', 0.30419), (u'cremation', 0.28391), (u'fabricated', 0.28391), (u'yiengling', 0.28391), (u'delish', 0.27475), (u'uncover', 0.26616), (u'm8n', 0.26616), (u'pepparoni', 0.26616),(u'drops', 0.03097), (u'scheduling', 0.03097), (u'towed', 0.03089), (u'stark', 0.03086), (u'bjork', 0.03086), (u'arrogance', 0.03086), (u'maitre', 0.03086), (u'electronica', 0.03086), (u'basing', 0.03086), (u'caters', 0.03086)]
list_of_words2, list_of_num2 = parse_elements(c)
model2 = {'TF-IDF': list_of_num2, 'Words': list_of_words2}
df2 = df(data = model2, columns = ['TF-IDF','Words'])

graph2 = Bar(df2, 'Words', values = 'TF-IDF', title = "TF-IDF of top negative words", ylabel = 'TF-IDF  scores', width = 800, height = 400, color = 'red')
graph2.left[0].formatter.use_scientific= False
graph2.x_range = FactorRange(factors=df2['Words'].tolist())

output_file("swag1.html")
total = vplot(graph1, graph2)
show(total)



