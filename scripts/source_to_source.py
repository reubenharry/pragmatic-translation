import matplotlib
matplotlib.use('Agg')
import re
import requests
import time
import pickle
import numpy as np
import scipy
import scipy.special
from collections import defaultdict
from utils.config import *
from utils.helper_functions import uniform_vector, make_initial_prior, display,byte_pair_encoding,byte_pair_decoding
from translators.translators import Coalgebra, Anamorphism, Factor_To_Character, Constant,Compose
from bayesian_agents.bayesian_pragmatics import Pragmatic
from utils.paraphrase import get_paraphrases


source_target_word_model = Coalgebra(path='wmt14.en-de.fconv-py',source='en',target='de')
target_source_word_model = Coalgebra(path='iwslt14.tokenized.de-en',source='de',target='en')

source_to_source_model = Compose(rightward_model=source_target_word_model,leftward_model=target_source_word_model)
unfolded_source_to_source_model = Anamorphism(underlying_model=source_to_source_model,beam_width=5,diverse=True)


# probs,support = unfolded_source_to_source_model.forward(source_sentence="He might go . ",sequence=[],debug=True)
# print(display(probs,support))
# raise Exception

sentences = ["He might go . ","He could go . "]

# explicit_distractor_source_source = Constant(support=sentences,unfolded_model=unfolded_source_to_source_model,change_direction=False)



# probs,support = explicit_distractor_source_source.forward(sequence=[],source_sentence="der .",debug=True)
# print(display(probs=probs,support=support))

# probs,support = explicit_distractor_source_source.likelihood(sequence=[],source_sentence="der .",source="das")
# print(display(probs=probs,support=support))

# probs,support = explicit_distractor_source_source.likelihood(sequence=[],source_sentence="der .",source="das")
# print(display(probs=probs,support=support))

# raise Exception


rightward_pragmatic = Pragmatic(
	rightward_model=source_to_source_model,
	leftward_model=source_to_source_model, 
	unfolded_leftward_model=None,
	width=10,rat=1.0,
	EXTEND=False)
# leftward_pragmatic = Pragmatic(de_en_word_model,en_de_word_model,width=2,rat=1.0)
# double_pragmatic = Pragmatic(rightward_pragmatic,leftward_pragmatic,width=2,rat=1.0)
# probs,support = speaker.forward(sequence=[],source_sentence=sentence,width=2)
# print(display(probs=probs,support=support))
# probs,support = double_pragmatic.forward(sequence=[],source_sentence=sentence,debug=True)
# print(display(probs=new_probs,support=support))
results_dict = {}
for sentence in sentences:

	unrolled_Pragmatic = Anamorphism(underlying_model=rightward_pragmatic)
	probs,support = rightward_pragmatic.forward(sequence=[],source_sentence=sentence,debug=False)
	print(display(probs=probs,support=support))
	results_dict[sentence]=display(probs=probs,support=support)

print(results_dict)


