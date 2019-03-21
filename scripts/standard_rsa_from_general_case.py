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



de_en_translation_model = Coalgebra(path='iwslt14.tokenized.de-en',source='de',target='en')
en_de_translation_model = Coalgebra(path='wmt14.en-de.fconv-py',source='en',target='de')

# char_level_translator_de_en = Factor_To_Character(de_en_translation_model)
# char_level_translator_en_de = Factor_To_Character(en_de_translation_model)

# unfolded_de_en_model = Anamorphism(underlying_model=char_level_translator_de_en)
# unfolded_en_de_model = Anamorphism(underlying_model=char_level_translator_en_de)

unfolded_de_en_model = Anamorphism(underlying_model=de_en_translation_model)
unfolded_en_de_model = Anamorphism(underlying_model=en_de_translation_model)


# sentences = get_paraphrases(
# 	sentence="Sometimes the most difficult questions have the simplest solutions .",
# 	unfolded_rightward_model=unfolded_en_de_model,
# 	unfolded_leftward_model=unfolded_de_en_model,
# 	rightward_code_path="wmt14.en-de.fconv-py/bpecodes",
# 	leftward_code_path="iwslt14.tokenized.de-en/code")
# sentences = [byte_pair_decoding(x) for x in out]

# sentences = ["people fear the dogs . ","the dogs terri@@ fy people . "]

# sentences = ['sometimes the most difficult questions have the easi@@ est solutions .', 'sometimes the most difficult questions have the simpl@@ est solutions .', 'sometimes the har@@ dest questions have the easi@@ est solutions .', 'sometimes the har@@ dest questions have the simpl@@ est solutions .', 'sometimes the most challen@@ ging questions have the easi@@ est solutions .']
# sentence = sentences[0]
sentences = ["He might go . ", "He could go . ",]
# sentence_2 = "He could go"
# sentence = byte_pair_encoding(sentence=sentence.lower(),code_path="wmt14.en-de.fconv-py/bpecodes")
# sentence_2 = byte_pair_encoding(sentence=sentence_2.lower(),code_path="wmt14.en-de.fconv-py/bpecodes")
# sentences = [byte_pair_encoding(sentence=s.lower(),code_path="wmt14.en-de.fconv-py/bpecodes") for s in sentences]
# sentence = byte_pair_encoding(sentence=byte_pair_decoding(sentences[0].lower()),code_path="wmt14.en-de.fconv-py/bpecodes")
# print("sentence:",sentence)
print("sentences:",sentences)


explicit_distractor_de_en = Constant(support=sentences,unfolded_model=unfolded_de_en_model,change_direction=True)

# probs,support = explicit_distractor_de_en.forward(sequence=[],source_sentence="das .",)
# print(display(probs=probs,support=support))
# raise Exception


rightward_pragmatic = Pragmatic(
	rightward_model=en_de_translation_model,
	leftward_model=de_en_translation_model, 
	unfolded_leftward_model=explicit_distractor_de_en,
	width=2,rat=100.0,
	EXTEND=False)
# leftward_pragmatic = Pragmatic(de_en_translation_model,en_de_translation_model,width=2,rat=1.0)
# double_pragmatic = Pragmatic(rightward_pragmatic,leftward_pragmatic,width=2,rat=1.0)
# probs,support = speaker.forward(sequence=[],source_sentence=sentence,width=2)
# print(display(probs=probs,support=support))
# probs,support = double_pragmatic.forward(sequence=[],source_sentence=sentence,debug=True)
# print(display(probs=new_probs,support=support))
results_dict = {}
for sentence in sentences:

	unrolled_Pragmatic = Anamorphism(underlying_model=rightward_pragmatic)
	probs,support = unrolled_Pragmatic.forward(sequence=[],source_sentence=sentence,beam_width=1)
	print(display(probs=probs,support=support))
	results_dict[sentence]=display(probs=probs,support=support)

print(results_dict)


