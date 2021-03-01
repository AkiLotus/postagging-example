from sys import argv
import pickle, time
import vlsp_reader as vlspr
import numpy as np
import utilities

hmmt = pickle.load(open('models/hmmt_smooth.mdlobj', 'rb'))

valid_data = vlspr.read_valid()

result = utilities.evaluate_markov_model(hmmt, valid_data)

pickle_output = open("results/hmm-validset.rstobj", "wb")
model_object = pickle.dump(result, pickle_output)