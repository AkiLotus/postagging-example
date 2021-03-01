from markov_models import HMMTagger
from sys import argv
import pickle
import vlsp_reader as vlspr

train_data = vlspr.read_train()


hmmt = HMMTagger(smoothing=True)

hmmt.insert_corpus(train_data, logging=True)

hmmt.finalize_corpus(logging=True)

pickle_file = open("models/hmmt_smooth.mdlobj", "wb")
model_object = pickle.dump(hmmt, pickle_file)