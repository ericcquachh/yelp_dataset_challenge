import json
import codecs
import os
import nltk

DATA_PATH = "/Users/colin.garcia/Desktop/yelp_dataset_challenge_academic_dataset/"
#DATA_PATH = "/Users/colin.garcia/Desktop/yelp_dataset_challenge/"
file_name = "yelp_academic_dataset_review.json"

""" Retrieve the number of lines in a json file. 
	For this project we have 1,569,264 different reviews. """
def num_lines():
	counter = 0
	with open(file_name) as json_file:
		for i, line in enumerate(json_file):
			counter += 1

	return counter


""" Retrieve the stars of each review in their corresponding array. """
def star_array():
	array_of_stars = [0 for i in range(1569264)] # just gonna hardcode this lol
	with open(file_name) as json_file:
		for i, line in enumerate(json_file):
			parsed_json = json.loads(line)
			stars = parsed_json['stars']
			array_of_stars[i] = stars
	return array_of_stars

""" Put the text as elements of an array. """
def text_array(array):
	array_of_text = [0 for i in range(1569264)]
	with open(file_name) as json_file:
		for i, line in enumerate(json_file):
			parsed_json = json.loads(line)
			text = parsed_json['text']
			array_of_text[i] = text
	return array_of_text


def main():
	stars_array = star_array()
	review_text = text_array(stars_array)
