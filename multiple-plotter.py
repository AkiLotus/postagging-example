import matplotlib.pyplot as plt
import pickle

pipeline_ids = [1, 8, 11]
colors = ['red', 'blue', 'green']
trainlogs = []

pipeline_ids = list(map(lambda x: ('0' if x < 10 else '') + str(x), pipeline_ids))

for index in range(len(pipeline_ids)):
	trainlogs.append(pickle.load(open("results/memmt-pipeline{}-trainlogs.logobj".format(pipeline_ids[index]), "rb")))\

plt.title('Error-through-iteration plot: MEMMTagger of Pipeline ({}) / Stochastic GD'.format(' + '.join(pipeline_ids)))
plt.xlabel('Iteration timestamp')
plt.ylabel('Average error per record')
for index in range(len(pipeline_ids)):
	plt.plot(trainlogs[index].timestamps, trainlogs[index].costs, color=colors[index], label = 'Pipeline #{}'.format(pipeline_ids[index]))
plt.legend()
plt.show()