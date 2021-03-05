import sys, re
sys.path.append("../")
from markov_models import MEMMTagger

__learning_rate__ = 0.015

# Pipeline 1: only current word and previous tag, theoretically equivalent in feature set to HMM
# Pipeline 2: Pipeline 1 + previous word (full bigram features)
# Pipeline 3: Pipeline 2 + 2nd previous word/tag (full trigram features)
# Pipeline 4: Pipeline 3 + check for uppercase/lowercase letters in a word
# Pipeline 5: Pipeline 4 + check for digit character
# Pipeline 6: Pipeline 5 + check for uppercase letter beginning a word
# Pipeline 7: Pipeline 6 + check for hyphens in a word
# Pipeline 8: Pipeline 7 + check for special characters in a word (full model from revision 2021-02-28)
# Pipeline 9: Pipeline 8 + check if the number of syllables in a word is greater than 2 (which is more likely to be an NNP)
# Pipeline 10: Pipeline 9 + check if all syllables of the word are capitalized (which is more likely to be an NNP)
# Pipeline 11: Pipeline 10 + check if containing foreign characters for Vietnamese (F, J, W, Z)

class PipelineTagger_01(MEMMTagger):
	def __init__(self):
		MEMMTagger.__init__(self, learning_rate = __learning_rate__)

	def create_feature_vector(self, word, word_back_1, word_back_2, tag_back_1, tag_back_2):
		# Pipeline 1: only current word and previous tag, theoretically equivalent in feature set to HMM
		glossary_count = len(self.glossary_mapping.keys())
		pos_tags_count = len(self.pos_tags_mapping.keys())
		feature_vector = [0 for _ in range(glossary_count + pos_tags_count)]

		offset = 0

		if word in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word]] = 1
		offset += glossary_count

		feature_vector[offset + self.pos_tags_mapping[tag_back_1]] = 1
		offset += pos_tags_count

		return feature_vector

class PipelineTagger_02(MEMMTagger):
	def __init__(self):
		MEMMTagger.__init__(self, learning_rate = __learning_rate__)

	def create_feature_vector(self, word, word_back_1, word_back_2, tag_back_1, tag_back_2):
		# Pipeline 2: Pipeline 1 + previous word (full bigram features)
		glossary_count = len(self.glossary_mapping.keys())
		pos_tags_count = len(self.pos_tags_mapping.keys())
		feature_vector = [0 for _ in range(glossary_count * 2 + pos_tags_count)]

		offset = 0
		if word in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word]] = 1
		offset += glossary_count

		if word_back_1 in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word_back_1]] = 1
		offset += glossary_count
		
		feature_vector[offset + self.pos_tags_mapping[tag_back_1]] = 1
		offset += pos_tags_count

		return feature_vector

class PipelineTagger_03(MEMMTagger):
	def __init__(self):
		MEMMTagger.__init__(self, learning_rate = __learning_rate__)

	def create_feature_vector(self, word, word_back_1, word_back_2, tag_back_1, tag_back_2):
		# Pipeline 3: Pipeline 2 + 2nd previous word/tag (full trigram features)
		glossary_count = len(self.glossary_mapping.keys())
		pos_tags_count = len(self.pos_tags_mapping.keys())
		feature_vector = [0 for _ in range(glossary_count * 3 + pos_tags_count + pos_tags_count ** 2)]

		offset = 0
		if word in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word]] = 1
		offset += glossary_count

		if word_back_1 in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word_back_1]] = 1
		offset += glossary_count

		if word_back_2 in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word_back_2]] = 1
		offset += glossary_count

		feature_vector[offset + self.pos_tags_mapping[tag_back_1]] = 1
		offset += pos_tags_count

		feature_vector[offset + self.pos_tags_mapping[tag_back_2] * pos_tags_count + self.pos_tags_mapping[tag_back_1]] = 1
		offset += pos_tags_count ** 2

		return feature_vector

class PipelineTagger_04(MEMMTagger):
	def __init__(self):
		MEMMTagger.__init__(self, learning_rate = __learning_rate__)

	def create_feature_vector(self, word, word_back_1, word_back_2, tag_back_1, tag_back_2):
		# Pipeline 4: Pipeline 3 + check for uppercase/lowercase letters in a word
		glossary_count = len(self.glossary_mapping.keys())
		pos_tags_count = len(self.pos_tags_mapping.keys())
		feature_vector = [0 for _ in range(glossary_count * 3 + pos_tags_count + pos_tags_count ** 2 + 2)]

		offset = 0
		if word in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word]] = 1
		offset += glossary_count

		if word_back_1 in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word_back_1]] = 1
		offset += glossary_count

		if word_back_2 in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word_back_2]] = 1
		offset += glossary_count

		feature_vector[offset + self.pos_tags_mapping[tag_back_1]] = 1
		offset += pos_tags_count

		feature_vector[offset + self.pos_tags_mapping[tag_back_2] * pos_tags_count + self.pos_tags_mapping[tag_back_1]] = 1
		offset += pos_tags_count ** 2

		if re.match('^.*[A-Z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*[a-z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		return feature_vector

class PipelineTagger_05(MEMMTagger):
	def __init__(self):
		MEMMTagger.__init__(self, learning_rate = __learning_rate__)

	def create_feature_vector(self, word, word_back_1, word_back_2, tag_back_1, tag_back_2):
		# Pipeline 5: Pipeline 4 + check for digit character
		glossary_count = len(self.glossary_mapping.keys())
		pos_tags_count = len(self.pos_tags_mapping.keys())
		feature_vector = [0 for _ in range(glossary_count * 3 + pos_tags_count + pos_tags_count ** 2 + 3)]

		offset = 0
		if word in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word]] = 1
		offset += glossary_count

		if word_back_1 in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word_back_1]] = 1
		offset += glossary_count

		if word_back_2 in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word_back_2]] = 1
		offset += glossary_count

		feature_vector[offset + self.pos_tags_mapping[tag_back_1]] = 1
		offset += pos_tags_count

		feature_vector[offset + self.pos_tags_mapping[tag_back_2] * pos_tags_count + self.pos_tags_mapping[tag_back_1]] = 1
		offset += pos_tags_count ** 2

		if re.match('^.*[A-Z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*[a-z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*[0-9].*$', word):
			feature_vector[offset] = 1
		offset += 1

		return feature_vector

class PipelineTagger_06(MEMMTagger):
	def __init__(self):
		MEMMTagger.__init__(self, learning_rate = __learning_rate__)

	def create_feature_vector(self, word, word_back_1, word_back_2, tag_back_1, tag_back_2):
		# Pipeline 6: Pipeline 5 + check for uppercase letter beginning a word
		glossary_count = len(self.glossary_mapping.keys())
		pos_tags_count = len(self.pos_tags_mapping.keys())
		feature_vector = [0 for _ in range(glossary_count * 3 + pos_tags_count + pos_tags_count ** 2 + 4)]

		offset = 0
		if word in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word]] = 1
		offset += glossary_count

		if word_back_1 in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word_back_1]] = 1
		offset += glossary_count

		if word_back_2 in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word_back_2]] = 1
		offset += glossary_count

		feature_vector[offset + self.pos_tags_mapping[tag_back_1]] = 1
		offset += pos_tags_count

		feature_vector[offset + self.pos_tags_mapping[tag_back_2] * pos_tags_count + self.pos_tags_mapping[tag_back_1]] = 1
		offset += pos_tags_count ** 2

		if re.match('^.*[A-Z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*[a-z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*[0-9].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^[A-Z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		return feature_vector

class PipelineTagger_07(MEMMTagger):
	def __init__(self):
		MEMMTagger.__init__(self, learning_rate = __learning_rate__)

	def create_feature_vector(self, word, word_back_1, word_back_2, tag_back_1, tag_back_2):
		# Pipeline 7: Pipeline 6 + check for hyphens in a word
		glossary_count = len(self.glossary_mapping.keys())
		pos_tags_count = len(self.pos_tags_mapping.keys())
		feature_vector = [0 for _ in range(glossary_count * 3 + pos_tags_count + pos_tags_count ** 2 + 5)]

		offset = 0
		if word in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word]] = 1
		offset += glossary_count

		if word_back_1 in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word_back_1]] = 1
		offset += glossary_count

		if word_back_2 in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word_back_2]] = 1
		offset += glossary_count

		feature_vector[offset + self.pos_tags_mapping[tag_back_1]] = 1
		offset += pos_tags_count

		feature_vector[offset + self.pos_tags_mapping[tag_back_2] * pos_tags_count + self.pos_tags_mapping[tag_back_1]] = 1
		offset += pos_tags_count ** 2

		if re.match('^.*[A-Z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*[a-z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*[0-9].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^[A-Z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*-.*$', word):
			feature_vector[offset] = 1
		offset += 1

		return feature_vector

class PipelineTagger_08(MEMMTagger):
	def __init__(self):
		MEMMTagger.__init__(self, learning_rate = __learning_rate__)

	def create_feature_vector(self, word, word_back_1, word_back_2, tag_back_1, tag_back_2):
		# Pipeline 8: Pipeline 7 + check for special characters in a word (full model from revision 2021-02-28)
		glossary_count = len(self.glossary_mapping.keys())
		pos_tags_count = len(self.pos_tags_mapping.keys())
		feature_vector = [0 for _ in range(glossary_count * 3 + pos_tags_count + pos_tags_count ** 2 + 6)]

		offset = 0
		if word in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word]] = 1
		offset += glossary_count

		if word_back_1 in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word_back_1]] = 1
		offset += glossary_count

		if word_back_2 in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word_back_2]] = 1
		offset += glossary_count

		feature_vector[offset + self.pos_tags_mapping[tag_back_1]] = 1
		offset += pos_tags_count

		feature_vector[offset + self.pos_tags_mapping[tag_back_2] * pos_tags_count + self.pos_tags_mapping[tag_back_1]] = 1
		offset += pos_tags_count ** 2

		if re.match('^.*[A-Z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*[a-z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*[0-9].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^[A-Z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*-.*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*[^0-9A-Za-z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		return feature_vector

class PipelineTagger_09(MEMMTagger):
	def __init__(self):
		MEMMTagger.__init__(self, learning_rate = __learning_rate__)

	def create_feature_vector(self, word, word_back_1, word_back_2, tag_back_1, tag_back_2):
		# Pipeline 9: Pipeline 8 + check if the number of syllables in a word is greater than 2 (which is more likely to be an NNP)
		glossary_count = len(self.glossary_mapping.keys())
		pos_tags_count = len(self.pos_tags_mapping.keys())
		feature_vector = [0 for _ in range(glossary_count * 3 + pos_tags_count + pos_tags_count ** 2 + 7)]

		offset = 0
		if word in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word]] = 1
		offset += glossary_count

		if word_back_1 in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word_back_1]] = 1
		offset += glossary_count

		if word_back_2 in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word_back_2]] = 1
		offset += glossary_count

		feature_vector[offset + self.pos_tags_mapping[tag_back_1]] = 1
		offset += pos_tags_count

		feature_vector[offset + self.pos_tags_mapping[tag_back_2] * pos_tags_count + self.pos_tags_mapping[tag_back_1]] = 1
		offset += pos_tags_count ** 2

		if re.match('^.*[A-Z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*[a-z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*[0-9].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^[A-Z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*-.*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*[^0-9A-Za-z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		tokens = word.split()
		if len(tokens) > 2:
			feature_vector[offset] = 1
		offset += 1

		return feature_vector

class PipelineTagger_10(MEMMTagger):
	def __init__(self):
		MEMMTagger.__init__(self, learning_rate = __learning_rate__)

	def create_feature_vector(self, word, word_back_1, word_back_2, tag_back_1, tag_back_2):
		# Pipeline 10: Pipeline 9 + check if all syllables of the word are capitalized (which is more likely to be an NNP)
		glossary_count = len(self.glossary_mapping.keys())
		pos_tags_count = len(self.pos_tags_mapping.keys())
		feature_vector = [0 for _ in range(glossary_count * 3 + pos_tags_count + pos_tags_count ** 2 + 8)]

		offset = 0
		if word in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word]] = 1
		offset += glossary_count

		if word_back_1 in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word_back_1]] = 1
		offset += glossary_count

		if word_back_2 in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word_back_2]] = 1
		offset += glossary_count

		feature_vector[offset + self.pos_tags_mapping[tag_back_1]] = 1
		offset += pos_tags_count

		feature_vector[offset + self.pos_tags_mapping[tag_back_2] * pos_tags_count + self.pos_tags_mapping[tag_back_1]] = 1
		offset += pos_tags_count ** 2

		if re.match('^.*[A-Z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*[a-z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*[0-9].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^[A-Z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*-.*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*[^0-9A-Za-z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		tokens = word.split()
		if len(tokens) > 2:
			feature_vector[offset] = 1
		offset += 1

		for token in tokens:
			if not re.match('^[A-Z].*$', token): break
		else:
			feature_vector[offset] = 1
		offset += 1

		return feature_vector

class PipelineTagger_11(MEMMTagger):
	def __init__(self):
		MEMMTagger.__init__(self, learning_rate = __learning_rate__)

	def create_feature_vector(self, word, word_back_1, word_back_2, tag_back_1, tag_back_2):
		# Pipeline 11: Pipeline 10 + check if containing foreign characters for Vietnamese (F, J, W, Z)
		glossary_count = len(self.glossary_mapping.keys())
		pos_tags_count = len(self.pos_tags_mapping.keys())
		feature_vector = [0 for _ in range(glossary_count * 3 + pos_tags_count + pos_tags_count ** 2 + 9)]

		offset = 0
		if word in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word]] = 1
		offset += glossary_count

		if word_back_1 in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word_back_1]] = 1
		offset += glossary_count

		if word_back_2 in self.glossary_mapping:
			feature_vector[offset + self.glossary_mapping[word_back_2]] = 1
		offset += glossary_count

		feature_vector[offset + self.pos_tags_mapping[tag_back_1]] = 1
		offset += pos_tags_count

		feature_vector[offset + self.pos_tags_mapping[tag_back_2] * pos_tags_count + self.pos_tags_mapping[tag_back_1]] = 1
		offset += pos_tags_count ** 2

		if re.match('^.*[A-Z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*[a-z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*[0-9].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^[A-Z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*-.*$', word):
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*[^0-9A-Za-z].*$', word):
			feature_vector[offset] = 1
		offset += 1

		tokens = word.split()
		if len(tokens) > 2:
			feature_vector[offset] = 1
		offset += 1

		for token in tokens:
			if not re.match('^[A-Z].*$', token): break
		else:
			feature_vector[offset] = 1
		offset += 1

		if re.match('^.*[FJWZfjwz].*$', word):
			feature_vector[offset] = 1
		offset += 1

		return feature_vector