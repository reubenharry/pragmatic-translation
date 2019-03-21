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
from utils.helper_functions import uniform_vector, make_initial_prior, display,byte_pair_encoding,byte_pair_decoding,kl
from translators.translators import Coalgebra, Anamorphism, Factor_To_Character, Constant
from bayesian_agents.bayesian_pragmatics import Pragmatic
from utils.paraphrase import get_paraphrases



# de_en_translation_model = 
source_target_word_model = Coalgebra(path='wmt14.en-fr.fconv-py',source='en',target='fr')
# Coalgebra(path='iwslt14.tokenized.de-en',source='de',target='en')

# source_target_translation_model = Factor_To_Character(source_target_word_model)
source_target_translation_model = source_target_word_model

# char_level_translator_de_en = Factor_To_Character(de_en_translation_model)
# char_level_translator_en_de = Factor_To_Character(en_de_translation_model)

# unfolded_de_en_model = Anamorphism(underlying_model=char_level_translator_de_en)
# unfolded_en_de_model = Anamorphism(underlying_model=char_level_translator_en_de)

# unfolded_de_en_model = Anamorphism(underlying_model=de_en_translation_model)
unfolded_source_target_model = Anamorphism(underlying_model=source_target_translation_model)



# probs,support = unfolded_source_target_model.forward(sequence=["le"],source_sentence="the boy is . ",)
# print(display(probs=probs,support=support))

# probs,support = unfolded_source_target_model.forward(sequence=["le"],source_sentence="the boy is . ",)
# print(display(probs=probs,support=support))
# raise Exception

# sentences = get_paraphrases(
# 	sentence="Sometimes the most difficult questions have the simplest solutions .",
# 	unfolded_rightward_model=unfolded_en_de_model,
# 	unfolded_leftward_model=unfolded_de_en_model,
# 	rightward_code_path="wmt14.en-de.fconv-py/bpecodes",
# 	leftward_code_path="iwslt14.tokenized.de-en/code")
# sentences = [byte_pair_decoding(x) for x in out]

sentences = ["people fear the dogs . ","the dogs terri@@ fy people . "]

# sentences = [
# 	# "some of the chairs are not blue",
# 	"some of the chairs are blue . ",
# 	"some chairs are blue . ",
# 	]

# sentences = ["He might go . ", "He could go . ",]

sentences = [byte_pair_encoding(sentence=s,code_path="wmt14.en-fr.fconv-py/bpecodes") for s in sentences]

# sentences = ['sometimes the most difficult questions have the easi@@ est solutions .', 'sometimes the most difficult questions have the simpl@@ est solutions .', 'sometimes the har@@ dest questions have the easi@@ est solutions .', 'sometimes the har@@ dest questions have the simpl@@ est solutions .', 'sometimes the most challen@@ ging questions have the easi@@ est solutions .']
# sentence = sentences[0]
# sentence = "He might go"
# sentence_2 = "He could go"
# sentence = byte_pair_encoding(sentence=sentence.lower(),code_path="wmt14.en-de.fconv-py/bpecodes")
# sentence_2 = byte_pair_encoding(sentence=sentence_2.lower(),code_path="wmt14.en-de.fconv-py/bpecodes")
# sentence = byte_pair_encoding(sentence=byte_pair_decoding(sentences[0].lower()),code_path="wmt14.en-de.fconv-py/bpecodes")
# print("sentence:",sentence)
print("sentences:",sentences)


explicit_distractor_target_source = Constant(support=sentences,unfolded_model=unfolded_source_target_model,change_direction=False)


class Maximize_Entropy_Likelihood_Model:

	def __init__(self,underlying_model):

		# print("underlying_model",underlying_model)

		try: a = underlying_model.support
		except: raise Exception("underlying model must be a model with explicit distractors")

		self.underlying_model=underlying_model

	def likelihood(self,sequence,source_sentence, target):

		probs,support = self.underlying_model.forward(source_sentence=source_sentence,sequence=sequence)

		uniform = np.ones(len(self.underlying_model.support))*(1/len(self.underlying_model.support))

		kl_div = kl(uniform,np.exp(probs))

		print("KL",kl_div)
		print("uniform",uniform)
		print("np.exp probs",np.exp(probs))
		return -kl_div

maximize_entropy_likelihood_model = Maximize_Entropy_Likelihood_Model(explicit_distractor_target_source)

# print("RESULT",maximize_entropy_likelihood_model.likelihood(source_sentence=sentences[0],sequence=[],target="f"))
# raise Exception

# probs,support = explicit_distractor_target_source.forward(sequence=[],source_sentence="der .",debug=True)
# print(display(probs=probs,support=support))

# probs,support = explicit_distractor_target_source.likelihood(sequence=[],source_sentence="der .",target="das")
# print(display(probs=probs,support=support))

# probs,support = explicit_distractor_target_source.likelihood(sequence=[],source_sentence="der .",target="das")
# print(display(probs=probs,support=support))

# raise Exception


rightward_pragmatic = Pragmatic(
	rightward_model=source_target_translation_model,
	leftward_model=None, 
	unfolded_leftward_model=maximize_entropy_likelihood_model,
	width=10,rat=100.0,
	EXTEND=True)
# leftward_pragmatic = Pragmatic(de_en_translation_model,en_de_translation_model,width=2,rat=1.0)
# double_pragmatic = Pragmatic(rightward_pragmatic,leftward_pragmatic,width=2,rat=1.0)
# probs,support = speaker.forward(sequence=[],source_sentence=sentence,width=2)
# print(display(probs=probs,support=support))
# probs,support = double_pragmatic.forward(sequence=[],source_sentence=sentence,debug=True)
# print(display(probs=new_probs,support=support))
results_dict = {}
for sentence in sentences[:1]:

	unrolled_Pragmatic = Anamorphism(underlying_model=rightward_pragmatic)
	probs,support = unrolled_Pragmatic.forward(sequence=[],source_sentence=sentence,debug=True,beam_width=1)
	print(display(probs=probs,support=support))
	results_dict[sentence]=display(probs=probs,support=support)

print(results_dict)


