import json
import codecs
import os

DATA_PATH = "/Users/ericquach/Github/yelp_data/"
#DATA_PATH = "/Users/Janet/Downloads/CS194/yelp_dataset_challenge_academic_dataset/"
#DATA_PATH = "/Users/colin.garcia/Desktop/yelp_dataset_challenge_academic_dataset/"
file_name = DATA_PATH + "yelp_academic_dataset_review.json"

text_to_stars = {}

def generate_stars():
	with open(file_name) as json_file:
		counter = 0
		for i, line in enumerate(json_file):
			# Since we're just exploring, we're only counting the first 100 files
			if counter >= 100:
				break

			# Returns a dictionary for the each line
			parsed_json = json.loads(line) 
			# Important fields are accumulated below
			stars = parsed_json['stars']
			text = parsed_json['text']

			# Wrtiting to a file
			padding = 2 - len(str(i))
			written_file_name = 'review_text' + ('0' * padding) + str(i) + '.txt'
	#		written_file = codecs.open(written_file_name, "w", "utf-8")
	#		written_file.write(text)
	#		written_file.close()

			# Adding to a dictionary
			text_to_stars[written_file_name] = stars
			counter += 1

#for i in range(100):
#    zeroes = 2 - len(str(i))
#    zeroes_str = zeroes * '0'
#    comm = "dependencyparser.sh review_text" + zeroes_str + str(i) + ".txt > review_parsed" + zeroes_str + str(i) + ".xml"
#    os.system(comm)