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
from utils.helper_functions import uniform_vector, make_initial_prior, display,byte_pair_encoding
from translators.translators import Coalgebra, Anamorphism, Factor_To_Character, Constant
from bayesian_agents.bayesian_pragmatics import Pragmatic


# sentences = ["Sometimes the most difficult questions have the simplest solutions ."]
# sentences = ["He could go . "]
sentences = [
			"Have you got a minute? ",
			# "Do you have a minute? ",
			]
# 
# sentence = "He might go"
# sentence_2 = "He could go"
# sentence = byte_pair_encoding(sentence=sentence.lower(),code_path="wmt14.en-de.fconv-py/bpecodes")
# sentence_2 = byte_pair_encoding(sentence=sentence_2.lower(),code_path="wmt14.en-de.fconv-py/bpecodes")
sentences = [byte_pair_encoding(sentence=s.lower(),code_path="wmt14.en-de.fconv-py/bpecodes") for s in sentences]
sentence = sentences[0]
print("sentence:",sentence)

de_en_translation_model = Coalgebra(path='iwslt14.tokenized.de-en',source='de',target='en')
en_de_translation_model = Coalgebra(path='wmt14.en-de.fconv-py',source='en',target='de')

char_level_translator_de_en = Factor_To_Character(de_en_translation_model)
char_level_translator_en_de = Factor_To_Character(en_de_translation_model)

unfolded_de_en_model = Anamorphism(underlying_model=char_level_translator_de_en)
unfolded_en_de_model = Anamorphism(underlying_model=char_level_translator_en_de)

unfolded_char_de_en_model = Anamorphism(underlying_model=de_en_translation_model)
unfolded_char_en_de_model = Anamorphism(underlying_model=en_de_translation_model)

toc = time.time()
probs,support = unfolded_en_de_model.forward(sequence=[],source_sentence=sentence,beam_width=1,debug=True,stop_on=None)
print(display(probs=probs,support=support))
tic = time.time()
print("TIME:",tic-toc)

# toc = time.time()
# probs,support = unfolded_char_en_de_model.forward(sequence=[],source_sentence=sentence,beam_width=1,debug=True,stop_on=None)
# print(display(probs=probs,support=support))
# tic = time.time()
# print("TIME:",tic-toc)

# explicit_distractor_de_en = Constant(support=sentences,unfolded_model=unfolded_de_en_model)

# rightward_pragmatic = Pragmatic(
# 	rightward_model=en_de_translation_model,
# 	leftward_model=de_en_translation_model, 
# 	unfolded_leftward_model=None,
# 	EXTEND=False,
# 	width=40000,rat=100.0)
# leftward_pragmatic = Pragmatic(de_en_translation_model,en_de_translation_model,width=2,rat=1.0)
# double_pragmatic = Pragmatic(rightward_pragmatic,leftward_pragmatic,width=2,rat=1.0)
# probs,support = speaker.forward(sequence=[],source_sentence=sentence,width=2)
# print(display(probs=probs,support=support))
# probs,support = double_pragmatic.forward(sequence=[],source_sentence=sentence,debug=True)
# print(display(probs=new_probs,support=support))

# unrolled_Pragmatic = Anamorphism(underlying_model=rightward_pragmatic)
# probs,support = unrolled_Pragmatic.forward(sequence=[],source_sentence=sentence,beam_width=1,debug=True,stop_on=None)
# print(display(probs=probs,support=support))

# [('Manchmal haben die schwierig@@ sten Fragen die einfach@@ sten Lösungen .', 0.6775947167917629), ('Manchmal haben die schwierig@@ ste Fragen die einfach@@ sten Lösungen .', 0.31908804600083307), ('Manchmal haben die komplizi@@ er@@ testen Fragen die einfach@@ sten Lösungen .', 0.003317886055249216), ('Manchmal haben die schwierig@@ sten Fragen die einfach@@ sten Lösungen zu haben .', 4.355218427991361e-07), ('Manchmal haben die komplizi@@ bel@@ sten Fragen die einfach@@ sten Lösungen .', 9.054743609888236e-08), ('Manchmal haben die schwierig@@ sten Fragen die einfach@@ sten Lösungen zu haben !', 9.998303854652057e-12)]




# rightward_pragmatic = Pragmatic(en_de_translation_model,de_en_translation_model,width=2,rat=1.0)
# leftward_pragmatic = Pragmatic(de_en_translation_model,en_de_translation_model,width=2,rat=1.0)
# double_pragmatic = Pragmatic(rightward_pragmatic,leftward_pragmatic,width=2,rat=1.0)
# # probs,support = speaker.forward(sequence=[],source_sentence=sentence,width=2)
# # print(display(probs=probs,support=support))
# probs,support = double_pragmatic.forward(sequence=[],source_sentence=sentence,debug=True)
# print(display(probs=new_probs,support=support))

# # unrolled_Pragmatic = Anamorphism(underlying_model=double_pragmatic)
# # probs,support = unrolled_Pragmatic.forward(sequence=[],source_sentence=sentence,beam_width=1)
# # print(display(probs=probs,support=support))
