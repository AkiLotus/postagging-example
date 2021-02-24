import numpy as np
import scipy.special
import re
import heapq

# sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
sigmoid = lambda x: scipy.special.expit(x)
softmax = lambda x: scipy.special.softmax(x, axis=1)
__default_seed__ = 170299

START_TAG = 'STG'
START_SYMBOL = '^start^'

class LimitedMinheap:
	def __init__(self, max_capacity = np.inf):
		self.max_capacity = max_capacity
		self.__heap_content = []
	
	def clear(self):
		self.__heap_content.clear()
	
	def remove_peak(self):
		return heapq.heappop(self.__heap_content)

	def insert(self, obj):
		heapq.heappush(self.__heap_content, obj)
		if len(self.__heap_content) > self.max_capacity:
			self.remove_peak()
	
	def to_array(self):
		return self.__heap_content.copy()


class HMMTagger:
	def __init__(self, smoothing = False):
		self.smoothing = smoothing

		self.pos_tags = set()
		self.glossary = set()
		self.pos_tags_mapping = None
		self.glossary_mapping = None

		self.word2tag_frequencies = {}
		self.word2tag_document_frequencies = {}
		self.tag2tag_frequencies = {}
		self.tag2tag_count = 0
		self.word2tag_count = 0
	
	def reset_corpus(self):
		self.__init__()
	
	def insert_corpus(self, sentences, logging=False):
		# no insertion after finalized. if willing to, reset corpus
		if type(self.pos_tags) is list: return

		self.pos_tags = set(self.pos_tags)
		self.glossary = set(self.glossary)

		document_wordset = set()

		for sentence in sentences:
			last_tag = None

			for word, tag in sentence:
				self.glossary.add(word)
				self.pos_tags.add(tag)

				if (word, tag) not in self.word2tag_frequencies:
					self.word2tag_frequencies[(word, tag)] = 0
				if last_tag is not None and (last_tag, tag) not in self.tag2tag_frequencies:
					self.tag2tag_frequencies[(last_tag, tag)] = 0
				
				self.word2tag_frequencies[(word, tag)] += 1
				self.word2tag_count += 1
				if last_tag is not None:
					self.tag2tag_frequencies[(last_tag, tag)] += 1
					self.tag2tag_count += 1

				last_tag = tag

				if (word, tag) not in document_wordset:
					document_wordset.add((word, tag))
					if (word, tag) not in self.word2tag_document_frequencies:
						self.word2tag_document_frequencies[(word, tag)] = 0
					self.word2tag_document_frequencies[(word, tag)] += 1
		
		if logging: print('{} sentences added.'.format(len(sentences)))

	def finalize_corpus(self, logging=False):
		self.pos_tags_mapping = {}
		self.glossary_mapping = {}

		self.pos_tags = list(self.pos_tags)
		self.glossary = list(self.glossary)

		for index in range(len(self.pos_tags)):
			self.pos_tags_mapping[self.pos_tags[index]] = index

		for index in range(len(self.glossary)):
			self.glossary_mapping[self.glossary[index]] = index

		if logging:
			print('Glossary size:', len(self.glossary))
			# print('Glossary examples:', self.glossary[:15], '...', self.glossary[-15:])
			print('POS tags size:', len(self.pos_tags))
			print('POS tags:', self.pos_tags)
	
	def __get_log_probability_tag2tag(self, last_tag, tag):
		if (last_tag, tag) in self.tag2tag_frequencies:
			if not self.smoothing:
				return np.log(self.tag2tag_frequencies[(last_tag, tag)] / self.tag2tag_count)
			else:
				return np.log((self.tag2tag_frequencies[(last_tag, tag)] + 1) / (self.tag2tag_count + len(self.pos_tags) * len(self.pos_tags)))
		else:
			if not self.smoothing:
				return -np.inf
			else:
				return np.log(1.0 / (self.tag2tag_count + len(self.pos_tags) * len(self.pos_tags)))
	
	def __get_log_probability_word2tag(self, token, tag):
		if (token, tag) in self.word2tag_frequencies:
			if not self.smoothing:
				return np.log(self.word2tag_frequencies[(token, tag)] / self.word2tag_count)
			else:
				return np.log((self.word2tag_frequencies[(token, tag)] + 1) / (self.word2tag_count + len(self.glossary) * len(self.pos_tags)))
		else:
			if not self.smoothing:
				return -np.inf
			else:
				return np.log(1.0 / (self.word2tag_count + len(self.glossary) * len(self.pos_tags)))
	
	def assign_tags(self, tokens, beam_search = np.inf):
		# DP through Viterbi algorithm, applied beam search
		best_prob = np.zeros((len(tokens), len(self.pos_tags))) - np.inf
		best_path = [[None for _ in range(len(self.pos_tags))] for _ in range(len(tokens))]

		beam_heap = LimitedMinheap(max_capacity=beam_search)

		for tag_index in range(len(self.pos_tags)):
			best_prob[0][tag_index] = self.__get_log_probability_word2tag(tokens[0], self.pos_tags[tag_index])
			if best_prob[0][tag_index] > -np.inf: beam_heap.insert((best_prob[0][tag_index], tag_index))

		for token_index in range(1, len(tokens)):
			best_candidates = beam_heap.to_array()
			beam_heap.clear()

			for tag_index in range(len(self.pos_tags)):
				found_better = False
				for _, last_tag_index in best_candidates:
					new_prob = best_prob[token_index-1][last_tag_index]
					new_prob += self.__get_log_probability_word2tag(tokens[token_index], self.pos_tags[tag_index])
					new_prob += self.__get_log_probability_tag2tag(self.pos_tags[last_tag_index], self.pos_tags[tag_index])

					if new_prob > best_prob[token_index][tag_index]:
						best_prob[token_index][tag_index] = new_prob
						best_path[token_index][tag_index] = (token_index - 1, last_tag_index)
						found_better = True
				
				if found_better: beam_heap.insert((best_prob[token_index][tag_index], tag_index))
		
		assigned_tags = [None for _ in tokens]
		curr_tag_index = np.argmax(best_prob[len(tokens)-1])
		curr_token_index = len(tokens) - 1
		assigned_tags[curr_token_index] = self.pos_tags[curr_tag_index]

		while best_path[curr_token_index][curr_tag_index] is not None:
			curr_token_index, curr_tag_index = best_path[curr_token_index][curr_tag_index]

			assigned_tags[curr_token_index] = self.pos_tags[curr_tag_index]
		
		return assigned_tags
		
		# return ' '.join(['{}/{}'.format(tokens[index], assigned_tags[index]) for index in range(len(tokens))])


def create_feature_vector(word, word_back_1, word_back_2, tag, tag_back_1, tag_back_2, glossary_mapping, pos_tags_mapping):
	glossary_count = len(glossary_mapping.keys())
	pos_tags_count = len(pos_tags_mapping.keys())
	feature_vector = [0 for _ in range(glossary_count * 3 * pos_tags_count + pos_tags_count ** 2 + pos_tags_count ** 3 + 4 * glossary_count * pos_tags_count)]

	offset = 0
	if word in glossary_mapping:
		feature_vector[offset + glossary_mapping[word] * pos_tags_count + pos_tags_mapping[tag]] = 1
	
	offset += glossary_count * pos_tags_count
	if word_back_1 in glossary_mapping:
		feature_vector[offset + glossary_mapping[word_back_1] * pos_tags_count + pos_tags_mapping[tag]] = 1
	
	offset += glossary_count * pos_tags_count
	if word_back_2 in glossary_mapping:
		feature_vector[offset + glossary_mapping[word_back_2] * pos_tags_count + pos_tags_mapping[tag]] = 1
	
	offset += glossary_count * pos_tags_count
	feature_vector[offset + pos_tags_mapping[tag_back_1] * pos_tags_count + pos_tags_mapping[tag]] = 1
	
	offset += pos_tags_count ** 2
	feature_vector[offset + pos_tags_mapping[tag_back_2] * pos_tags_count * pos_tags_count + pos_tags_mapping[tag_back_1] * pos_tags_count + pos_tags_mapping[tag]] = 1

	offset += pos_tags_count ** 3
	if re.match('^.*[0-9].*$', word):
		feature_vector[offset + pos_tags_mapping[tag]] = 1

	offset += pos_tags_count
	if re.match('^.*[A-Z].*$', word):
		feature_vector[offset + pos_tags_mapping[tag]] = 1

	offset += pos_tags_count
	if re.match('^.*[a-z].*$', word):
		feature_vector[offset + pos_tags_mapping[tag]] = 1

	offset += pos_tags_count
	if re.match('^.*-.*$', word):
		feature_vector[offset + pos_tags_mapping[tag]] = 1

	return feature_vector

class MEMMTagger:
	def __init__(self, smoothing = False, learning_rate = 0.2, regularization_lambda = 0.4, seed=__default_seed__):
		self.smoothing = smoothing
		self.learning_rate = learning_rate
		self.regularization_lambda = regularization_lambda
		self._seed = seed
		np.random.seed(seed)

		self.pos_tags = set()
		self.glossary = set()
		self.pos_tags_mapping = None
		self.glossary_mapping = None

		self.parameters = None
	
	def reset_corpus(self):
		self.__init__()
	
	def fit(self, feature_vectors, labels, iterations=1000, regularizing_metric=None):
		feature_vectors = np.array(feature_vectors)
		labels = np.array(labels)
		label_vectors = np.zeros((labels.shape[0], len(self.pos_tags)))
		for row_id in range(labels.shape[0]):
			label_vectors[row_id][labels[row_id]] = 1
		
		self.parameters = np.random.rand(feature_vectors.shape[1], 1)

		for iter in range(iterations):
			parameter_gradient = ...
	
	def insert_corpus(self, sentences, logging=False):
		# no insertion after fit. if willing to, reset corpus
		if type(self.pos_tags) is list: return

		for sentence in sentences:
			for word, tag in sentence:
				self.glossary.add(word)
				self.pos_tags.add(tag)
		
		self.pos_tags_mapping = {}
		self.glossary_mapping = {}

		self.pos_tags = [START_TAG] + list(self.pos_tags)
		self.glossary = [START_SYMBOL] + list(self.glossary)

		for index in range(len(self.pos_tags)):
			self.pos_tags_mapping[self.pos_tags[index]] = index

		for index in range(len(self.glossary)):
			self.glossary_mapping[self.glossary[index]] = index

		if logging:
			print('{} sentences added.'.format(len(sentences)))
			print('Glossary size:', len(self.glossary) - 1)
			# print('Glossary examples:', self.glossary[:15], '...', self.glossary[-15:])
			print('POS tags size:', len(self.pos_tags) - 1)
			print('POS tags:', self.pos_tags)
		
		feature_vectors = []
		tags = []

		for sentence in sentences:
			for index in range(len(sentence)):
				word, tag = sentence[index]
				word_back_1 = sentence[index - 1][0] if index > 0 else START_SYMBOL
				word_back_2 = sentence[index - 2][0] if index > 1 else START_SYMBOL
				tag_back_1 = sentence[index - 1][0] if index > 0 else START_TAG
				tag_back_2 = sentence[index - 2][0] if index > 1 else START_TAG

				feature_vector = create_feature_vector(word, word_back_1, word_back_2, tag, tag_back_1, tag_back_2, self.glossary_mapping, self.pos_tags_mapping)

				feature_vectors.append(feature_vector)
				tags.append(self.pos_tags_mapping[tag])

		self.fit(feature_vectors, tags)
	
	def __get_log_probability_tag2tag(self, last_tag, tag):
		if (last_tag, tag) in self.tag2tag_frequencies:
			if not self.smoothing:
				return np.log(self.tag2tag_frequencies[(last_tag, tag)] / self.tag2tag_count)
			else:
				return np.log((self.tag2tag_frequencies[(last_tag, tag)] + 1) / (self.tag2tag_count + len(self.pos_tags) * len(self.pos_tags)))
		else:
			if not self.smoothing:
				return -np.inf
			else:
				return np.log(1.0 / (self.tag2tag_count + len(self.pos_tags) * len(self.pos_tags)))
	
	def __get_log_probability_word2tag(self, token, tag):
		if (token, tag) in self.word2tag_frequencies:
			if not self.smoothing:
				return np.log(self.word2tag_frequencies[(token, tag)] / self.word2tag_count)
			else:
				return np.log((self.word2tag_frequencies[(token, tag)] + 1) / (self.word2tag_count + len(self.glossary) * len(self.pos_tags)))
		else:
			if not self.smoothing:
				return -np.inf
			else:
				return np.log(1.0 / (self.word2tag_count + len(self.glossary) * len(self.pos_tags)))
	
	def assign_tags(self, tokens, beam_search = np.inf):
		# DP through Viterbi algorithm, applied beam search
		best_prob = np.zeros((len(tokens), len(self.pos_tags))) - np.inf
		best_path = [[None for _ in range(len(self.pos_tags))] for _ in range(len(tokens))]

		beam_heap = LimitedMinheap(max_capacity=beam_search)

		for tag_index in range(len(self.pos_tags)):
			best_prob[0][tag_index] = self.__get_log_probability_word2tag(tokens[0], self.pos_tags[tag_index])
			if best_prob[0][tag_index] > -np.inf: beam_heap.insert((best_prob[0][tag_index], tag_index))

		for token_index in range(1, len(tokens)):
			best_candidates = beam_heap.to_array()
			beam_heap.clear()

			for tag_index in range(len(self.pos_tags)):
				found_better = False
				for _, last_tag_index in best_candidates:
					new_prob = best_prob[token_index-1][last_tag_index]
					new_prob += self.__get_log_probability_word2tag(tokens[token_index], self.pos_tags[tag_index])
					new_prob += self.__get_log_probability_tag2tag(self.pos_tags[last_tag_index], self.pos_tags[tag_index])

					if new_prob > best_prob[token_index][tag_index]:
						best_prob[token_index][tag_index] = new_prob
						best_path[token_index][tag_index] = (token_index - 1, last_tag_index)
						found_better = True
				
				if found_better: beam_heap.insert((best_prob[token_index][tag_index], tag_index))
		
		assigned_tags = [None for _ in tokens]
		curr_tag_index = np.argmax(best_prob[len(tokens)-1])
		curr_token_index = len(tokens) - 1
		assigned_tags[curr_token_index] = self.pos_tags[curr_tag_index]

		while best_path[curr_token_index][curr_tag_index] is not None:
			curr_token_index, curr_tag_index = best_path[curr_token_index][curr_tag_index]

			assigned_tags[curr_token_index] = self.pos_tags[curr_tag_index]
		
		return assigned_tags
		
		# return ' '.join(['{}/{}'.format(tokens[index], assigned_tags[index]) for index in range(len(tokens))])