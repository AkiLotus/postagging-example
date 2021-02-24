# postagging-example
Just a random POS tagging repository, solely for educational purposes...

# Data preparation:
- Corpus source: VLSP Dataset

# Usage
1. Hidden Markov Model (HMM):
   - Run `python3 pretrain_model.py` to train the model and save it in pickle dump format. Two model variants will be made, one without Laplace smoothing (`_smooth0`), and one with it (`_smooth1`).
   - Run `python3 sample.py` to load the pre-trained models and test it with the cross-validation file. Beam-search is included in each run, for the sake of comparison.
   - Details on how the dataset is read can be found on the `vlsp_reader.py` file.
2. Maximum Entropy Markov Model (MEMM):
   - Currently working...