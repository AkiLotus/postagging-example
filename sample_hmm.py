from sys import argv
import joblib
import vlsp_reader as vlspr
import utilities

hmmt = joblib.load('models/hmmt_smooth.mdlobj')

valid_data = vlspr.read_valid()

result = utilities.evaluate_markov_model(hmmt, valid_data)

output_file = "results/hmm-validset.rstobj"
model_result = joblib.dump(result, output_file)