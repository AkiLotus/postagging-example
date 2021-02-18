# Used for extracting .tei.xml file from Royal Society Corpus, provided by Clarin-D
# Link: http://fedora.clarin-d.uni-saarland.de/rsc_v4/access.html#download

from bs4 import BeautifulSoup
from sys import argv
import os

root_dir = '../Royal_Society_Corpus_v4.0.1_texts_tei'

file_count = 0
for filename in os.listdir(root_dir):
	file_count += 1
	print('#{:5d} '.format(file_count) + 'Accessing file ' + filename + '...')
	xml_file = open(root_dir + "/" + filename, "r")
	xml_content = BeautifulSoup(xml_file.read())

	output_file = open('corpus/' + filename.replace('Royal_Society_Corpus_v4.0.1_', '').replace('.tei.xml', '.txt'), "w")

	for s in xml_content.findAll("s"):
		words_found = []
		for w in s.findAll("w"):
			words_found.append('{}/{}'.format(w.get_text(), w["pos"]))
		output_file.write(' '.join(words_found))
		output_file.write('\n')