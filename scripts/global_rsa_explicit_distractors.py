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
from translators.translators import Coalgebra, Anamorphism, Factor_To_Character, Constant
from bayesian_agents.bayesian_pragmatics import Pragmatic
from utils.paraphrase import get_paraphrases

source_target_word_model = Coalgebra(path='wmt14.en-de.fconv-py',source='en',target='de')
source_target_translation_model = source_target_word_model

target_source_translation_model = Coalgebra(path='iwslt14.tokenized.de-en',source='de',target='en')


# char_level_translator_de_en = Factor_To_Character(de_en_translation_model)
# char_level_translator_en_de = Factor_To_Character(en_de_translation_model)

# unfolded_de_en_model = Anamorphism(underlying_model=char_level_translator_de_en)
# unfolded_en_de_model = Anamorphism(underlying_model=char_level_translator_en_de)

# unfolded_de_en_model = Anamorphism(underlying_model=de_en_translation_model)
unfolded_source_target_model = Anamorphism(underlying_model=source_target_translation_model,beam_width=5)
unfolded_target_source_model = Anamorphism(underlying_model=target_source_translation_model,beam_width=5)


# sentences = ["He might go . ", "He could go . ",]

sentences = ["Some of the chairs are blue . ", "Some chairs are blue . "]

sentences = [byte_pair_encoding(sentence=s.lower(),code_path="iwslt14.tokenized.de-en/code") for s in sentences]

print("sentences:",sentences)

explicit_distractor_target_source = Constant(support=sentences,unfolded_model=unfolded_source_target_model,change_direction=True)



# probs,support = unfolded_source_target_model.forward(sequence=[],source_sentence=sentences[0],debug=True)
# print(display(probs=probs,support=support))

# likelihood = explicit_distractor_target_source.likelihood(sequence=[],source_sentence="Vielleicht könnte er gehen",target='he might go . ')
# print(likelihood)

# likelihood = explicit_distractor_target_source.likelihood(sequence=[],source_sentence="er könnte gehen",target='he might go . ')
# print(likelihood)

# probs,support = explicit_distractor_target_source.likelihood(sequence=[],source_sentence="der .",target="das")
# print(display(probs=probs,support=support))

# raise Exception


rightward_pragmatic = Pragmatic(
	rightward_model=unfolded_source_target_model,
	leftward_model=None, 
	unfolded_leftward_model=unfolded_target_source_model,
	width=10,rat=10.0,
	EXTEND=False)
# leftward_pragmatic = Pragmatic(de_en_translation_model,en_de_translation_model,width=2,rat=1.0)
# double_pragmatic = Pragmatic(rightward_pragmatic,leftward_pragmatic,width=2,rat=1.0)
# probs,support = speaker.forward(sequence=[],source_sentence=sentence,width=2)
# print(display(probs=probs,support=support))
# probs,support = double_pragmatic.forward(sequence=[],source_sentence=sentence,debug=True)
# print(display(probs=new_probs,support=support))
results_dict = {}
for sentence in sentences:

	# unrolled_Pragmatic = Anamorphism(underlying_model=rightward_pragmatic)
	probs,support = rightward_pragmatic.forward(sequence=[],source_sentence=sentence,debug=False)
	print(display(probs=probs,support=support))
	results_dict[sentence]=display(probs=probs,support=support)

print(results_dict)


