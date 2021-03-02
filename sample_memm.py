from sys import argv
import joblib
import vlsp_reader as vlspr
import utilities

pipeline_id = int(argv[1]) if len(argv) > 1 else 0

memmt = joblib.load("models/memmt-pipeline{}.mdlobj".format(('0' if pipeline_id < 10 else '') + str(pipeline_id)))

valid_data = vlspr.read_valid()

result = utilities.evaluate_markov_model(memmt, valid_data)

output_file = "results/memmt-pipeline{}-validset.rstobj".format(('0' if pipeline_id < 10 else '') + str(pipeline_id))
model_result = joblib.dump(result, output_file)