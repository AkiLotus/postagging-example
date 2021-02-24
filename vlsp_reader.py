TRAIN_FILE = '06_vlsp_pos/viettreebank_train.pos.conll'
VALID_FILE = '06_vlsp_pos/viettreebank_valid.pos.conll'
TEST_FILE = '06_vlsp_pos/viettreebank_test.pos.conll'

def read_file(filename):
	file = open(filename, mode='r')
	sentences = []
	current_sentence = []
	current_token = ''

	for line in file.readlines():
		line = line.replace('\n', '')
		if line == '':
			if len(current_sentence):
				sentences.append(current_sentence)
				current_sentence = []
			continue

		name, tagset = line.split('\t')
		tag_seg, tag_pos = tagset.split('-')

		if tag_seg == 'S':
			current_sentence.append((name, tag_pos))
		if tag_seg == 'B':
			current_token = name
		if tag_seg == 'I':
			current_token += ' ' + name
		if tag_seg == 'E':
			current_token += ' ' + name
			current_sentence.append((current_token, tag_pos))
	
	return sentences

def read_train(): return read_file(TRAIN_FILE)
def read_valid(): return read_file(VALID_FILE)
def read_test(): return read_file(TEST_FILE)