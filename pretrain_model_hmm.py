from markov_models import HMMTagger
from sys import argv
import joblib
import vlsp_reader as vlspr

train_data = vlspr.read_train()


hmmt = HMMTagger(smoothing=True)

hmmt.insert_corpus(train_data, logging=True)

hmmt.finalize_corpus(logging=True)

model_file = "models/hmmt_smooth.mdlobj"
model_object = joblib.dump(hmmt, model_file)