from markov_models import HMMTagger
from sys import argv
from torch import nn
import pickle, time, re

hmmt0 = pickle.load(open('hmmt_smooth0.mdlobj', 'rb'))

regex_patterns = [
	r'^[A-Z]',
	r'^[^A-Za-z]*[0-9]+[^A-Za-z]*',
	r'^[A-Za-z]{1,2}$',
	r'^[A-Za-z]{1,3}$',
	r'^[A-Za-z]{1,4}$',
	r'^[A-Za-z]{1,5}$',
	r'^[A-Za-z]{1,6}$',
	r'^[A-Za-z]{1,7}$',
	r'^[A-Za-z]{1,8}$',
	r'^[A-Za-z]+ity$',
	r'^[A-Za-z]+al$',
	r'^[A-Za-z]+s$',
	r'^[A-Za-z]+ed$',
	r'^[A-Za-z]+er$',
	r'^[A-Za-z]+est$',
	r'^[Ii]n[A-Za-z]+$',
	r'^[Ii]r[A-Za-z]+$',
	r'^[Ii]m[A-Za-z]+$',
	r'^[Uu]n[A-Za-z]+$',
	r'^[D]e[A-Za-z]+$',
	r'^[D]is[A-Za-z]+$',
	r'^[A-Za-z]+-[A-Za-z]+$',
	r'^[A-Za-z]+-(one|two|three|four|five|six|seven|eight|nine)$',
	r'^[.!?]$',
	r'^[,]$',
	r'^[Ww][Hh]',
	r'^[:;-]$',
	r'^\'[Ss]$',
	r'^[\'\"]$',
	r'^[A-Za-z.]+\.$'
]

feature_names = [
	'f01',
	'f02',
	'f03',
	'f04',
	'f05',
	'f06',
	'f07',
	'f08',
	'f09',
	'f10',
	'f11',
	'f12',
	'f13',
	'f14',
	'f15',
	'f16',
	'f17',
	'f18',
	'f19',
	'f20',
	'f21',
	'f22',
	'f23',
	'f24',
	'f25',
	'f26',
	'f27',
	'f28',
	'f29',
	'f30',
]

word_frequencies = {}
word_in_doc_frequencies = {}
tag_set = set()

def extract_morphology_feature(word):
	feature_list = []

	for re_pattern in regex_patterns:
		if re.match(re_pattern, word):
			feature_list.append('1')
		else:
			feature_list.append('0')

	return feature_list

def refined(tag):
	if tag == ',': return 'COMMA'
	return tag


for word, tag in hmmt0.word2tag_frequencies:
	tag_set.add(tag)

	if (word, tag) not in word_frequencies: word_frequencies[(word, tag)] = 0
	if (word, tag) not in word_in_doc_frequencies: word_in_doc_frequencies[(word, tag)] = 0
	word_frequencies[(word, tag)] += hmmt0.word2tag_frequencies[(word, tag)]
	word_in_doc_frequencies[(word, tag)] += hmmt0.word2tag_document_frequencies[(word, tag)]

word_list = []
for word, tag in word_frequencies:
	word_list.append((word, tag, word_frequencies[(word, tag)], word_in_doc_frequencies[(word, tag)]))

word_list.sort(key=lambda token: (token[3], token[2], token[1], token[0]), reverse=True)

# for tag in tag_set:
# 	print(tag)

# for .csv
print('{},tag'.format(','.join(feature_names)))
for word, tag, freq, docfreq in word_list:
	if docfreq >= 25:
		print('{},"{}"'.format(','.join(extract_morphology_feature(word)), refined(tag)))
	else: break

# for .arff
# print('word,{},tag'.format(','.join(feature_names)))
# for word, tag, freq, docfreq in word_list:
# 	if docfreq >= 25:
# 		print('{},"{}"'.format(','.join(extract_morphology_feature(word)), tag))
# 	else: break