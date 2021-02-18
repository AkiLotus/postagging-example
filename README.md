# postagging-example
Just a random POS tagging repository, solely for educational purposes...

# Data preparation:
- Corpus source: [Royal Society Corpus, provided by Clarin-D](http://fedora.clarin-d.uni-saarland.de/rsc_v4/access.html#download)
- Used file: Royal_Society_Corpus_v4.0.1_texts_tei.zip (140 MB) / Checksum (might be changed if this file is updated in the future, this checksum is for my current working version): `045579bc7cb2032641323a4845c317b7`
- Unzip the file, run: `python3 xml_extraction_batch.py` to transform the folder into a non-XML format (each token in a file is a word, then a "/" character, then its POS tag).

# Usage
- Run `python3 pretrain_model.py` to train the model and save it in pickle dump format. Two model variants will be made, one without Laplace smoothing (`_smooth0`), and one with it (`_smooth1`).
- Run `python3 sample.py` to load the pre-trained models and test it with some hand-made example (the example should be tokenized well beforehands). Beam-search is included in each run, for the sake of comparison.