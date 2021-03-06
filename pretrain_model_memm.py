from markov_models import MEMMTagger

from sys import argv
import joblib
import vlsp_reader as vlspr
import pipeline.memm

train_data = vlspr.read_train()
pipeline_id = int(argv[1]) if len(argv) > 1 else 0

memmt = MEMMTagger()

if pipeline_id == 1:
	memmt = pipeline.memm.PipelineTagger_01()
if pipeline_id == 2:
	memmt = pipeline.memm.PipelineTagger_02()
if pipeline_id == 3:
	memmt = pipeline.memm.PipelineTagger_03()
if pipeline_id == 4:
	memmt = pipeline.memm.PipelineTagger_04()
if pipeline_id == 5:
	memmt = pipeline.memm.PipelineTagger_05()
if pipeline_id == 6:
	memmt = pipeline.memm.PipelineTagger_06()
if pipeline_id == 7:
	memmt = pipeline.memm.PipelineTagger_07()
if pipeline_id == 8:
	memmt = pipeline.memm.PipelineTagger_08()
if pipeline_id == 9:
	memmt = pipeline.memm.PipelineTagger_09()
if pipeline_id == 10:
	memmt = pipeline.memm.PipelineTagger_10()
if pipeline_id == 11:
	memmt = pipeline.memm.PipelineTagger_11()

trainres = memmt.insert_corpus(train_data, logging=True)

model_file = "models/memmt-pipeline{}.mdlobj".format(('0' if pipeline_id < 10 else '') + str(pipeline_id))
model_object = joblib.dump(memmt, model_file)

log_file = "results/memmt-pipeline{}-trainlogs.logobj".format(('0' if pipeline_id < 10 else '') + str(pipeline_id))
logs_object = joblib.dump(trainres, log_file)