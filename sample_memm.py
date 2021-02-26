from markov_models import MEMMTagger
from sys import argv
import pickle, time
import vlsp_reader as vlspr
import numpy as np

memmt = pickle.load(open('memmt.mdlobj', 'rb'))

pos_tags = list(memmt.pos_tags)
pos_tags.sort(key=lambda x: memmt.pos_tags_mapping[x])
confusion_matrix = np.zeros((len(pos_tags), len(pos_tags)))
# print(pos_tags)

valid_data = vlspr.read_valid()

words_found = 0
words_correctly_tagged = 0
sentence_accuracies = []

for sentence in valid_data:
	prompt = [token for token, _ in sentence]
	tags = [tag for _, tag in sentence]

	# print(prompt)
	# print(tags)

	# time0 = time.time()
	# assigned_tags = memmt.assign_tags(prompt)
	# time1 = time.time()
	# time_smoothfull_beamless = time1 - time0

	time0 = time.time()
	assigned_tags = memmt.assign_tags(prompt, beam_search=2)
	time1 = time.time()
	time_smoothfull_beamfull = time1 - time0

	tagged_sentence = ' '.join(['{}/{}'.format(prompt[index], assigned_tags[index]) for index in range(len(prompt))])

	sentence_word = 0
	sentence_correct_tag = 0

	for index in range(len(tags)):
		sentence_word += 1
		if tags[index] == assigned_tags[index]:
			sentence_correct_tag += 1

		confusion_matrix[memmt.pos_tags_mapping[assigned_tags[index]]][memmt.pos_tags_mapping[tags[index]]] += 1
	
	words_found += sentence_word
	words_correctly_tagged += sentence_correct_tag

	# print('Answer w/o beam in', '{:3.5f}'.format(time_smoothfull_beamless), 'seconds:', tagged_sentence)
	print('Answer w/  beam in', '{:3.5f}'.format(time_smoothfull_beamfull), 'seconds:', tagged_sentence)
	print('Sentence accuracy = {:.5f}'.format(sentence_correct_tag / sentence_word))
	sentence_accuracies.append(sentence_correct_tag / sentence_word)

print('{} sentences tagged.'.format(len(valid_data)))
print('Final accuracy by words     = {:.5f}'.format(words_correctly_tagged / words_found))
print('Final accuracy by sentences = {:.5f}'.format(np.average(np.array(sentence_accuracies))))

def print_row(row, spacing=5):
	for item in row:
		formatted_item = item
		if type(item) is int:
			formatted_item = str(item)
		print(formatted_item + ' ' * (spacing - len(formatted_item)), end='')
	print()

print('Confusion matrix:')
print_row([''] + pos_tags)
for row_id in range(confusion_matrix.shape[0]):
	print_row([pos_tags[row_id]] + list(map(int, confusion_matrix[row_id])))


tag_TP = np.zeros(len(pos_tags))
tag_FP = np.zeros(len(pos_tags))
tag_FN = np.zeros(len(pos_tags))

for index in range(len(pos_tags)):
	tag_TP[index] = confusion_matrix[index][index]
	tag_FP[index] = np.sum(confusion_matrix[index, :]) - tag_TP[index]
	tag_FN[index] = np.sum(confusion_matrix[:, index]) - tag_TP[index]
	print('Tag =', pos_tags[index])
	if tag_TP[index] > 0:
		print('Precision = {:.5f}'.format(tag_TP[index] / (tag_TP[index] + tag_FP[index])))
		print('Recall    = {:.5f}'.format(tag_TP[index] / (tag_TP[index] + tag_FN[index])))
		print('F1-score  = {:.5f}'.format(2 * tag_TP[index] / (2 * tag_TP[index] + tag_FP[index] + tag_FN[index])))
	else:
		print('Invalid, due to zero True Positive')