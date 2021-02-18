from markov_models import HMMTagger
from sys import argv
import pickle
import os

root_dir = 'corpus'
hmmt0 = HMMTagger(smoothing=False)
hmmt1 = HMMTagger(smoothing=True)

file_count = 0
for filename in os.listdir(root_dir):
	file_count += 1
	print('#{:5d} '.format(file_count) + 'Accessing file ' + filename + '...')
	hmmt0.insert_corpus(root_dir + "/" + filename)
	hmmt1.insert_corpus(root_dir + "/" + filename)

hmmt0.finalize_corpus()
hmmt1.finalize_corpus()

pickle0_file = open("hmmt_smooth0.mdlobj", "wb")
model_object = pickle.dump(hmmt0, pickle0_file)
pickle1_file = open("hmmt_smooth1.mdlobj", "wb")
model_object = pickle.dump(hmmt1, pickle1_file)