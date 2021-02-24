from markov_models import HMMTagger
from sys import argv
import pickle
import os
import vlsp_reader as vlspr

train_data = vlspr.read_train()


hmmt0 = HMMTagger(smoothing=False)
hmmt1 = HMMTagger(smoothing=True)

hmmt0.insert_corpus(train_data)
hmmt1.insert_corpus(train_data, logging=True)

hmmt0.finalize_corpus()
hmmt1.finalize_corpus(logging=True)

pickle0_file = open("hmmt_smooth0.mdlobj", "wb")
model_object = pickle.dump(hmmt0, pickle0_file)
pickle1_file = open("hmmt_smooth1.mdlobj", "wb")
model_object = pickle.dump(hmmt1, pickle1_file)