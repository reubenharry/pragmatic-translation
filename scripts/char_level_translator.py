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
from translators.translators import Coalgebra, Anamorphism, Factor_To_Character
from bayesian_agents.bayesian_pragmatics import Pragmatic


sentence = "he might go . "
sentence = byte_pair_encoding(sentence=sentence.lower(),code_path="wmt14.en-de.fconv-py/bpecodes")
print("sentence:",sentence)

en_de_translation_model = Coalgebra(path='wmt14.en-de.fconv-py',source='en',target='de')
de_en_translation_model = Coalgebra(path='iwslt14.tokenized.de-en',source='de',target='en')

unfolded_en_de_model = Anamorphism(underlying_model=en_de_translation_model)


char_level_translator_en_de = Factor_To_Character(en_de_translation_model)
char_level_translator_de_en = Factor_To_Character(de_en_translation_model)

# probs,support = char_level_translator_en_de.forward(sequence=" ".join(['Er', 'könnte', 'gehen', '.'])+" ",source_sentence=sentence)
# print(display(probs=probs,support=support))
# raise Exception


# probs,support = char_level_translator.forward(sequence=['e', 'r', ' ', 'k'],source_sentence=sentence)
# print(display(probs=probs,support=support))
# raise Exception
# sop

# unfolded_char_level_translator = Anamorphism(underlying_model=char_level_translator_en_de)
# probs,support = unfolded_char_level_translator.forward(sequence=[],source_sentence=sentence)
# print(display(probs=probs,support=support))
# raise Exception





# class Pragmatic:

# 	def __init__(self,rightward_model,leftward_model,width=2,rat=2.0):
# 		assert (rightward_model.seg_type==leftward_model.seg_type)
# 		self.seg_type = rightward_model.seg_type
# 		self.rightward_model = rightward_model
# 		self.leftward_model = leftward_model
# 		self.unfolded_leftward_model = Anamorphism(underlying_model=self.leftward_model)
# 		self.unfolded_rightward_model = Anamorphism(underlying_model=self.rightward_model)
# 		self.width=width
# 		self.rat=rat

# 	def forward(self,sequence,source_sentence):

# 		s0_probs, s0_support = self.rightward_model.forward(sequence=sequence,source_sentence=source_sentence)

# 		s0_probs, s0_support = np.asarray(s0_probs[:self.width]),s0_support[:self.width]

# 		print("s0",display(probs=s0_probs,support=s0_support))
# 		unrolled_sents = []
# 		l0_scores = []
# 		for i,word in enumerate(s0_support):
# 			if word!=stop_token["word"]:
# 				_, unrolled_sent = self.unfolded_rightward_model.forward(sequence=sequence+[word],source_sentence=source_sentence,beam_width=1)
# 				unrolled_sent = unrolled_sent[0]
# 				# print("unrolled sent",unrolled_sent)
# 				unrolled_sents.append(unrolled_sent)
# 				score = self.unfolded_leftward_model.likelihood(sequence=[],source_sentence=unrolled_sent.lower(),target_sentence=source_sentence)
# 				# score = self.unfolded_leftward_model.likelihood(sequence=source_sentence[:len(sequence)],source_sentence=s,target_sentence=source_sentence[len(sequence):])
# 				print(s0_support[i])
# 				print("source",unrolled_sent)
# 				print("target",source_sentence)
# 				print("score",score)

# 			else: 
# 				score = self.unfolded_leftward_model.likelihood(sequence=[],source_sentence=" ".join(sequence+[word]).lower(),target_sentence=source_sentence)
# 				# unrolled_sents.append(" ".join(sequence+[word]))

# 			l0_scores.append(score)


# 		print(unrolled_sents, "unrolled_sents")

# 		# print("seq",source_sentence.split()[:len(sequence)])
# 		# print("targ",source_sentence.split()[len(sequence):])
# 		# for i,s in enumerate(unrolled_sents):
# 		# 	l0_scores.append(score)

# 		print("rat",self.rat)
# 		l0_scores = self.rat*np.asarray(l0_scores)

# 		print(type(s0_probs),type(l0_scores))

# 		print("l0 scores",np.exp(l0_scores))


# 		# l0_scores = [(unfolded_de_en_model.likelihood(sequence=[],source_sentence=s,target_sentence=sentence.split())) for s in unrolled_sents]
# 		# l0_scores = np.asarray(l0_scores)
# 		# print(np.exp(l0_scores))

# 		unnormed_probs = s0_probs + l0_scores
# 		normed_probs = unnormed_probs - scipy.special.logsumexp(unnormed_probs)

# 		return normed_probs,s0_support

# 	def likelihood(self,sequence,source_sentence,target_word):
# 		probs,support = self.forward(sequence,source_sentence)

# 		try: out = probs[support.index(target_word)]
# 		except: 
# 			print("target word failed:",target_word)
# 			raise Exception
# 		return out

# rightward_pragmatic = Pragmatic(en_de_translation_model,de_en_translation_model,width=2,rat=1.0)
# leftward_pragmatic = Pragmatic(de_en_translation_model,en_de_translation_model,width=2,rat=1.0)
# double_pragmatic = Pragmatic(rightward_pragmatic,leftward_pragmatic,width=2,rat=1.0)
# unrolled_Pragmatic = Anamorphism(underlying_model=double_pragmatic)
# probs,support = unrolled_Pragmatic.forward(sequence=[],source_sentence=sentence,beam_width=1)
# print(display(probs=probs,support=support))

speaker = Pragmatic(char_level_translator_en_de,char_level_translator_de_en,width=20,rat=1000.0,EXTEND=True)
unrolled_Pragmatic = Anamorphism(underlying_model=speaker)
probs,support = unrolled_Pragmatic.forward(sequence=[],source_sentence=sentence,beam_width=1,debug=True)
print(display(probs=probs,support=support))
