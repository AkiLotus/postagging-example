import numpy as np
import time

class EvaluationResult:
	def __init__(self, pos_tags):
		self.tags = pos_tags
		self.words_found = 0
		self.words_correctly_tagged = [0, 0]
		self.sentence_accuracies = [[], []]
		self.time_spent = [[], []]
		self.confusion_matrix = [np.zeros((len(self.tags), len(self.tags))), np.zeros((len(self.tags), len(self.tags)))]
	
	def accuracy_by_words(self, beam):
		return self.words_correctly_tagged[beam] / self.words_found
	
	def accuracy_by_sentences(self, beam):
		return np.average(np.array(self.sentence_accuracies[beam]))


def print_row(row, spacing=5):
	for item in row:
		formatted_item = item
		if type(item) is int:
			formatted_item = str(item)
		print(formatted_item + ' ' * (spacing - len(formatted_item)), end='')
	print()

def print_confusion_matrix(confusion_matrix, pos_tags):
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

def evaluate_markov_model(model, test_data):
	pos_tags = list(model.pos_tags)
	pos_tags.sort(key=lambda x: model.pos_tags_mapping[x])
	# print(pos_tags)

	result = EvaluationResult(pos_tags)

	for sentence in test_data:
		prompt = [token for token, _ in sentence]
		tags = [tag for _, tag in sentence]

		# print(prompt)
		# print(tags)

		time0 = time.time()
		assigned_tags_beamless = model.assign_tags(prompt)
		time1 = time.time()
		time_smoothfull_beamless = time1 - time0
		result.time_spent[0].append(time_smoothfull_beamless)

		time0 = time.time()
		assigned_tags_beamfull = model.assign_tags(prompt, beam_search=2)
		time1 = time.time()
		time_smoothfull_beamfull = time1 - time0
		result.time_spent[1].append(time_smoothfull_beamfull)

		tagged_sentence_beamless = ' '.join(['{}/{}'.format(prompt[index], assigned_tags_beamless[index]) for index in range(len(prompt))])
		tagged_sentence_beamfull = ' '.join(['{}/{}'.format(prompt[index], assigned_tags_beamfull[index]) for index in range(len(prompt))])

		sentence_word = 0
		sentence_correct_tag_beamless = 0
		sentence_correct_tag_beamfull = 0

		for index in range(len(tags)):
			sentence_word += 1
			if tags[index] == assigned_tags_beamless[index]:
				sentence_correct_tag_beamless += 1
			if tags[index] == assigned_tags_beamfull[index]:
				sentence_correct_tag_beamfull += 1

			result.confusion_matrix[0][model.pos_tags_mapping[assigned_tags_beamless[index]]][model.pos_tags_mapping[tags[index]]] += 1
			result.confusion_matrix[1][model.pos_tags_mapping[assigned_tags_beamfull[index]]][model.pos_tags_mapping[tags[index]]] += 1
		
		result.words_found += sentence_word
		result.words_correctly_tagged[0] += sentence_correct_tag_beamless
		result.words_correctly_tagged[1] += sentence_correct_tag_beamfull

		print('Answer w/o beam in', '{:3.5f}'.format(time_smoothfull_beamless), 'seconds:', tagged_sentence_beamless)
		print('Sentence accuracy (beamless) = {:.5f}'.format(sentence_correct_tag_beamless / sentence_word))
		print('Answer w/  beam in', '{:3.5f}'.format(time_smoothfull_beamfull), 'seconds:', tagged_sentence_beamfull)
		print('Sentence accuracy (beam-ful) = {:.5f}'.format(sentence_correct_tag_beamfull / sentence_word))
		result.sentence_accuracies[0].append(sentence_correct_tag_beamless / sentence_word)
		result.sentence_accuracies[1].append(sentence_correct_tag_beamfull / sentence_word)

	print('{} sentences tagged.\n'.format(len(test_data)))

	print('Beamless:')
	print('Final accuracy by words     = {:.5f}'.format(result.accuracy_by_words(0)))
	print('Final accuracy by sentences = {:.5f}'.format(result.accuracy_by_sentences(0)))

	print('Confusion matrix:')
	print_confusion_matrix(result.confusion_matrix[0], pos_tags)

	print('Beam-ful:')
	print('Final accuracy by words     = {:.5f}'.format(result.accuracy_by_words(1)))
	print('Final accuracy by sentences = {:.5f}'.format(result.accuracy_by_sentences(1)))

	print('Confusion matrix:')
	print_confusion_matrix(result.confusion_matrix[1], pos_tags)

	return result