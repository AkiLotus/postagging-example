from markov_models import HMMTagger
from sys import argv
import pickle, time
import vlsp_reader as vlspr
import numpy as np

hmmt0 = pickle.load(open('hmmt_smooth0.mdlobj', 'rb'))
hmmt1 = pickle.load(open('hmmt_smooth1.mdlobj', 'rb'))

valid_data = vlspr.read_test()

words_found = 0
words_correctly_tagged = 0
sentence_accuracies = []

for sentence in valid_data:
	prompt = [token for token, _ in sentence]
	tags = [tag for _, tag in sentence]

	# print(prompt)
	# print(tags)

	# time0 = time.time()
	# assigned_tags = hmmt0.assign_tags(prompt)
	# time1 = time.time()
	# time_smoothless_beamless = time1 - time0

	# time0 = time.time()
	# assigned_tags = hmmt0.assign_tags(prompt, beam_search=2)
	# time1 = time.time()
	# time_smoothless_beamfull = time1 - time0

	# time0 = time.time()
	# assigned_tags = hmmt1.assign_tags(prompt)
	# time1 = time.time()
	# time_smoothfull_beamless = time1 - time0

	time0 = time.time()
	assigned_tags = hmmt1.assign_tags(prompt, beam_search=2)
	time1 = time.time()
	time_smoothfull_beamfull = time1 - time0

	tagged_sentence = ' '.join(['{}/{}'.format(prompt[index], assigned_tags[index]) for index in range(len(prompt))])

	sentence_word = 0
	sentence_correct_tag = 0

	for index in range(len(tags)):
		sentence_word += 1
		sentence_correct_tag += 1 if tags[index] == assigned_tags[index] else 0
	
	words_found += sentence_word
	words_correctly_tagged += sentence_correct_tag

	# print('Without smoothing:')
	# print('Answer w/o beam in', '{:3.5f}'.format(time_smoothless_beamless), 'seconds:', tagged_sentence)
	# print('Answer w/  beam in', '{:3.5f}'.format(time_smoothless_beamfull), 'seconds:', tagged_sentence)
	# print ('With smoothing:')
	# print('Answer w/o beam in', '{:3.5f}'.format(time_smoothfull_beamless), 'seconds:', tagged_sentence)
	print('Answer w/  beam in', '{:3.5f}'.format(time_smoothfull_beamfull), 'seconds:', tagged_sentence)
	print('Sentence accuracy = {:.5f}'.format(sentence_correct_tag / sentence_word))
	sentence_accuracies.append(sentence_correct_tag / sentence_word)

print('{} sentences tagged.'.format(len(valid_data)))
print('Final accuracy by words     = {:.5f}'.format(words_correctly_tagged / words_found))
print('Final accuracy by sentences = {:.5f}'.format(np.average(np.array(sentence_accuracies))))