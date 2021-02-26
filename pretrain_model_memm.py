from markov_models import MEMMTagger

from sys import argv
import pickle
import os
import vlsp_reader as vlspr

train_data = vlspr.read_train()


memmt = MEMMTagger(learning_rate = 4 * 1e-3)

memmt.insert_corpus(train_data, logging=True)

pickle0_file = open("memmt.mdlobj", "wb")
model_object = pickle.dump(memmt, pickle0_file)