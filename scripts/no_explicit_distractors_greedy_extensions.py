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
from utils.helper_functions import display,byte_pair_encoding
from translators.translators import Coalgebra, Anamorphism, Factor_To_Character, Constant, Compose
from bayesian_agents.bayesian_pragmatics import Pragmatic


from base_case import source_target_model, target_source_model, target_source_word_model


# unfolded_target_source_model = Anamorphism(underlying_model=char_level_translator_target_source)
# unfolded_source_target_model = Anamorphism(underlying_model=char_level_translator_source_target)

unfolded_target_source_model = Anamorphism(underlying_model=target_source_model)
unfolded_source_target_model = Anamorphism(underlying_model=source_target_model)




# explicit_distractor_target_source = Constant(support=sentences,unfolded_model=unfolded_target_source_model)

rightward_pragmatic = Pragmatic(
	rightward_model=source_target_model,
	leftward_model=target_source_model, 
	unfolded_leftward_model=None,
	EXTEND=True,
	width=2,rat=10.0)

leftward_pragmatic = Pragmatic(
	rightward_model=target_source_model, 
	leftward_model=source_target_model,
	unfolded_leftward_model=None,
	EXTEND=True,
	width=2,rat=10.0)


# source_to_source = Compose(rightward_model=source_target_model,leftward_model=unfolded_target_source_model, unfolded_rightward_model=unfolded_source_target_model)
# unfolded_rightward_pragmatic = Anamorphism(underlying_model=rightward_pragmatic,beam_width=1)


source_to_source_pragmatic = Compose(rightward_model=rightward_pragmatic,leftward_model=unfolded_target_source_model, unfolded_rightward_model=None)
unfolded_source_to_source_pragmatic = Anamorphism(underlying_model=source_to_source_pragmatic,beam_width=2)

# source_to_source = Compose(rightward_model=source_target_model,leftward_model=target_source_word_model, unfolded_rightward_model=None)
# unfolded_source_to_source = Anamorphism(underlying_model=source_to_source,beam_width=2)



# results_dict = {"literal":{},"pragmatic":{}}
# for sentence in sentences[:1]:

# 	probs,support = unfolded_source_to_source.forward(sequence=[],source_sentence=sentence,debug=True)
# 	print(display(probs=probs,support=support))
# 	results_dict["literal"][sentence]=display(probs=probs,support=support)

# 	probs,support = unfolded_source_to_source_pragmatic.forward(sequence=[],source_sentence=sentence,debug=True)
# 	print(display(probs=probs,support=support))
# 	results_dict["pragmatic"][sentence]=display(probs=probs,support=support)

# print(results_dict)
# # [('Manchmal haben die schwierig@@ sten Fragen die einfach@@ sten Lösungen .', 0.6775947167917629), ('Manchmal haben die schwierig@@ ste Fragen die einfach@@ sten Lösungen .', 0.31908804600083307), ('Manchmal haben die komplizi@@ er@@ testen Fragen die einfach@@ sten Lösungen .', 0.003317886055249216), ('Manchmal haben die schwierig@@ sten Fragen die einfach@@ sten Lösungen zu haben .', 4.355218427991361e-07), ('Manchmal haben die komplizi@@ bel@@ sten Fragen die einfach@@ sten Lösungen .', 9.054743609888236e-08), ('Manchmal haben die schwierig@@ sten Fragen die einfach@@ sten Lösungen zu haben !', 9.998303854652057e-12)]


