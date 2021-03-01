import matplotlib.pyplot as plt
import pickle

pipeline_id = 1

trainlogs_memmt_01 = pickle.load(open("results/memmt-pipeline{}-trainlogs.logobj".format(('0' if pipeline_id < 10 else '') + str(pipeline_id)), "rb"))

plt.title('Error-through-iteration plot: MEMMTagger of Pipeline {} / Stochastic GD'.format(('0' if pipeline_id < 10 else '') + str(pipeline_id)))
plt.xlabel('Iteration timestamp')
plt.ylabel('Average error per record')
plt.plot(trainlogs_memmt_01.timestamps, trainlogs_memmt_01.costs, color='red', label = 'Pipeline #{}'.format(('0' if pipeline_id < 10 else '') + str(pipeline_id)))
plt.legend()
plt.show()