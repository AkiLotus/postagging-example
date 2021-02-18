from markov_models import HMMTagger
from sys import argv
import pickle, time

hmmt0 = pickle.load(open('hmmt_smooth0.mdlobj', 'rb'))

word_frequencies = {}
word_in_doc_frequencies = {}

for word, tag in hmmt0.word2tag_frequencies:
	if (word, tag) not in word_frequencies: word_frequencies[(word, tag)] = 0
	if (word, tag) not in word_in_doc_frequencies: word_in_doc_frequencies[(word, tag)] = 0
	word_frequencies[(word, tag)] += hmmt0.word2tag_frequencies[(word, tag)]
	word_in_doc_frequencies[(word, tag)] += hmmt0.word2tag_document_frequencies[(word, tag)]

word_list = []
for word, tag in word_frequencies:
	word_list.append((word, tag, word_frequencies[(word, tag)], word_in_doc_frequencies[(word, tag)]))

word_list.sort(key=lambda token: (token[3], token[2], token[1], token[0]), reverse=True)

for word, tag, freq, docfreq in word_list:
	print('{:64s} - {:8s}: {:10d} occurences: {:6d} documents'.format(word, tag, freq, docfreq))