from sys import argv
import pickle
import vlsp_reader as vlspr
import utilities

pipeline_id = int(argv[1]) if len(argv) > 1 else 0

memmt = pickle.load(open("models/memmt-pipeline{}.mdlobj".format(('0' if pipeline_id < 10 else '') + str(pipeline_id)), "rb"))

valid_data = vlspr.read_valid()

result = utilities.evaluate_markov_model(memmt, valid_data)

pickle_output = open("results/memmt-pipeline{}-validset.rstobj".format(('0' if pipeline_id < 10 else '') + str(pipeline_id)), "wb")
model_object = pickle.dump(result, pickle_output)