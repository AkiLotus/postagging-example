from markov_models import HMMTagger
from sys import argv
import pickle, time

hmmt0 = pickle.load(open('hmmt_smooth0.mdlobj', 'rb'))
hmmt1 = pickle.load(open('hmmt_smooth1.mdlobj', 'rb'))

while True:
	prompt = input('Type in a sentence: ')

	time0 = time.time()
	response_smoothless_beamless = hmmt0.assign_tags(prompt)
	time1 = time.time()
	time_smoothless_beamless = time1 - time0

	time0 = time.time()
	response_smoothless_beamfull = hmmt0.assign_tags(prompt, beam_search=2)
	time1 = time.time()
	time_smoothless_beamfull = time1 - time0

	time0 = time.time()
	response_smoothfull_beamless = hmmt1.assign_tags(prompt)
	time1 = time.time()
	time_smoothfull_beamless = time1 - time0

	time0 = time.time()
	response_smoothfull_beamfull = hmmt1.assign_tags(prompt, beam_search=2)
	time1 = time.time()
	time_smoothfull_beamfull = time1 - time0

	print('Without smoothing:')
	print('answer w/o beam in', '{:3.5f}'.format(time_smoothless_beamless), 'seconds:', response_smoothless_beamless)
	print('answer w/  beam in', '{:3.5f}'.format(time_smoothless_beamfull), 'seconds:', response_smoothless_beamfull)
	print ('With smoothing:')
	print('answer w/o beam in', '{:3.5f}'.format(time_smoothfull_beamless), 'seconds:', response_smoothfull_beamless)
	print('answer w/  beam in', '{:3.5f}'.format(time_smoothfull_beamfull), 'seconds:', response_smoothfull_beamfull)