import matplotlib
matplotlib.use('Agg')
import re
import requests
import time
import pickle
#import tensorflow as tf
import numpy as np
#from keras.preprocessing import image
from collections import defaultdict
from utils.config import *
from utils.helper_functions import uniform_vector, make_initial_prior, display,byte_pair_encoding
from recursion_schemes.recursion_schemes import ana_greedy,ana_beam,ana_monad,cata
# from bayesian_agents.rsa_for_presup import RSA
from bayesian_agents.rsa import RSA
from train.Model import Neural_Model, Unrolled_Model, Language_Model, Uniform_Model,Translation_Model,Compose_Model
from bayesian_agents.rsaState import RSA_State
from bayesian_agents.rsaWorld import RSA_World


# [('Certains de ses présidents sont de couleur bleue .', 0.09550389766674683), 
# ('Certains des ch@@ aises sont de couleur bleue .', 0.07888596456083233), 
# ('Parmi les ch@@ aises certaines sont de couleur bleue .', 0.05472958515892882), ('une partie des ch@@ aises sont de couleur bleue .', 0.049461813903291636), ('Parmi les ch@@ aises certaines sont en bleu .', 0.030983092702221517), ('une partie des ch@@ aises sont en bleu .', 0.02069884323163199), ('Parmi les ch@@ aises certains sont en bleu .', 0.01987945692138585), ('une partie des ch@@ aises sont de couleur bleue , par exemple .', 7.924890539797931e-05), ('une partie des ch@@ aises sont de couleur bleue , par exemple :', 6.495228938847462e-05), ('une partie des ch@@ aises sont de couleur bleue , etc. ) .', 3.6342650764707474e-05)]

sentences = [

	
	
	# "the animals run fast .",
	# "animals run fast .",

	# "einige der stühle sind blau",

	# byte_pair_encoding(sentence="einige der stühle sind blau .",code_path="iwslt14.tokenized.de-en/code"),
	# byte_pair_encoding(sentence="mein name ist john .",code_path="iwslt14.tokenized.de-en/code"),
	
	# "einige der stühle sind blau .",
	# "einige das stühle sind blau .",
	# "mein name ist john . ",
	"some of the chairs are blue . ",
	# "some of the chairs are not blue",
	"some chairs are blue . ",

	# "mein name ist john . ",
	# "ich habe ein buch . ",

	# "He can go . ",
	# "He is able to go . ",

	# "He needs to go . ",
	# "He must go . ",

	# "some of the chairs are blue . ",
	# "all of the chairs are blue . ",

	# "The boy saw the girl . ",
	# "The girl was seen by the boy . ",

	# "He should be in the office .",
	# "He ought to be in the office .",


	# "My name is John . ",
	# "I call myself John . "

	]

rats = [10.0]
number_of_sentences = len(sentences)
quds = [lambda x : [x]]
hyperpriors = [np.log(np.array([0.5,0.5]))]




# de_en_translation_model = Translation_Model(path='iwslt14.tokenized.de-en',source='de',target='en',sentences=sentences,rationalities=rats,hyperpriors=hyperpriors,code_path="iwslt14.tokenized.de-en/code")
en_de_translation_model = Translation_Model(path='wmt14.en-de.fconv-py',source='en',target='de',sentences=sentences,rationalities=rats,hyperpriors=hyperpriors,code_path='wmt14.en-de.fconv-py/bpecodes')

en_fr_translation_model = Translation_Model(path='wmt14.en-fr.fconv-py',source='en',target='fr',sentences=sentences,rationalities=rats,hyperpriors=hyperpriors,code_path='wmt14.en-fr.fconv-py/bpecodes')

# Compose_Model(model1=de_en_translation_model,model2=en_de_translation_model,composition_beam_width=1)
# lang_mods = [de_en_translation_model]
# models = [de_en_translation_model]
models = [en_fr_translation_model]
lang_mods = [en_fr_translation_model]

initial_image_prior = np.asarray([0.5,0.5])
initial_rationality_prior=uniform_vector(len(rats))
initial_speaker_prior=uniform_vector(len(models))
initial_hyperprior_prior = uniform_vector(len(hyperpriors))
initial_langmod_prior = uniform_vector(len(lang_mods))
initial_qud_prior = uniform_vector(len(quds))

initial_world_prior = make_initial_prior(initial_image_prior,initial_rationality_prior,initial_speaker_prior,initial_hyperprior_prior,initial_langmod_prior,initial_qud_prior)


# unrolling_of_rsa_d0_de_en = Unrolled_Model(images=sentences,rationalities=rats,hyperpriors=hyperpriors, underlying_model=base_rsa_d0_de_en,max_sentence_length=max_sentence_length)




# print("next/n/n/n/n")
base_rsa_d0_en_de = RSA(depth=0,models=[en_de_translation_model],urls=sentences,lang_mods=[en_de_translation_model],quds=quds)
world = RSA_World(target=0,speaker=0,rationality=0,hyperprior=0,language_model=0,qud=0)
state = RSA_State(initial_world_prior,seg_type="word")
# state.context_sentence = ["some"]
# de_en_translation_model.sentences = [byte_pair_encoding(sentence="Der Junge traf das Mädchen.".lower(),code_path="iwslt14.tokenized.de-en/code"),
									 # byte_pair_encoding(sentence="mein name ist john .",code_path="iwslt14.tokenized.de-en/code"),
										# ]
# unrolled = Unrolled_Model(images=sentences,rationalities=rats,hyperpriors=hyperpriors, underlying_model=[base_rsa_d0_en_de],max_sentence_length=max_sentence_length)
# probs_d00,support_d00 = de_en_translation_model.world_to_utterance(state=state,world=world)
probs_d00,support_d00 = base_rsa_d0_en_de.world_to_utterance(state=state,world=world,beam_width=1)
print(display(probs=probs_d00,support=support_d00))
world = RSA_World(target=0,speaker=0,rationality=0,hyperprior=0,language_model=0,qud=0)
state = RSA_State(initial_world_prior,seg_type="word")
score = base_rsa_d0_en_de.speaker_likelihood(state=state,world=world,utterance="Einige")
print(score,np.exp(score))
# print(unrolling_of_rsa_d0_de_en.speaker_likelihood(state,world,'the'))
raise Exception




# print("next/n/n/n/n")
world = RSA_World(target=0,speaker=0,rationality=0,hyperprior=0,language_model=0,qud=0)
state = RSA_State(initial_world_prior,seg_type="word")
# unrolled = Unrolled_Model(images=sentences,rationalities=rats,hyperpriors=hyperpriors, underlying_model=en_fr_translation_model,max_sentence_length=max_sentence_length,beam_width=2)
# state.context_sentence = ["some"]
# r = RSA(depth=1,models=models,urls=sentences,lang_mods=lang_mods,quds=quds)
# en_de_translation_model.sentences = [byte_pair_encoding(sentence="some of .".lower(),code_path="iwslt14.tokenized.de-en/code"),
# 									 byte_pair_encoding(sentence="mein name ist john .",code_path="iwslt14.tokenized.de-en/code"),
										# ]
# print(r.world_to_utterance(state=state,world=world))
# raise Exception
# unrolled = Unrolled_Model(images=sentences,rationalities=rats,hyperpriors=hyperpriors, underlying_model=en_fr_translation_model,max_sentence_length=max_sentence_length,beam_width=10)
# pragmatic_unrolled = RSA(depth=1,models=[unrolled],urls=sentences,lang_mods=[unrolled],quds=quds)

pragmatic = RSA(depth=1,models=[en_fr_translation_model],urls=sentences,lang_mods=[en_fr_translation_model],quds=quds)
unrolled_pragmatic = Unrolled_Model(images=sentences,rationalities=rats,hyperpriors=hyperpriors, underlying_model=pragmatic,max_sentence_length=max_sentence_length,beam_width=10)

probs_d00,support_d00 = unrolled_pragmatic.world_to_utterance(state=state,world=world)

# probs_d00,support_d00 = pragmatic_unrolled.world_to_utterance(state=state,world=world)
print(probs_d00,support_d00)
print(display(probs=probs_d00,support=support_d00))
# print(unrolling_of_rsa_d0_de_en.speaker_likelihood(state,world,'the'))
# probs_d00,support_d00 = de_en_translation_model.world_to_utterance(state=state,world=world)
raise Exception


composed_model = Compose_Model(model1=en_de_translation_model,model2=de_en_translation_model,composition_beam_width=1)
composed_model_rsa = RSA(depth=1,models=[composed_model],urls=sentences,lang_mods=[composed_model],quds=quds)
unrolled_composed = Unrolled_Model(images=sentences,rationalities=rats,hyperpriors=hyperpriors, underlying_model=composed_model_rsa,max_sentence_length=max_sentence_length,beam_width=1)
world = RSA_World(target=0,speaker=0,rationality=0,hyperprior=0,language_model=0,qud=0)
state = RSA_State(initial_world_prior,seg_type="word")
# out = unrolled_composed.speaker_likelihood(state,world,'some of the cha@@ irs are blue . </s>'.split())
# print(out)
# print(np.exp(out))
# probs_d00,support_d00 = composed_model_rsa.world_to_utterance(state=state,world=world)
probs_d00,support_d00 = unrolled_composed.world_to_utterance(pass_prior=True,cut_rate=1,decay_rate=0,state=state,world=world)
print(display(probs=probs_d00,support=support_d00))






# print(unrolled_composed.speaker_likelihood(state,world,'the cha@@ irs are blue .') )

# world = RSA_World(target=1,speaker=0,rationality=0,hyperprior=0,language_model=0,qud=0)
# state = RSA_State(initial_world_prior,seg_type=base_rsa_d0.seg_type)
# probs_d01,support_d01 = unrolling_of_rsa_d0.beam_search(beam_width=1,pass_prior=True,cut_rate=1,decay_rate=0,state=state,world=world)

# state = RSA_State(initial_world_prior,seg_type=base_rsa_d0.seg_type)
# world = RSA_World(target=0,rationality=0,speaker=0,hyperprior=0,language_model=0,qud=0)
# state.context_sentence = ["Il","est"]
# probs, support = base_rsa_d0.world_to_utterance(state=state,world=world)
# print(sorted(list(zip(probs,support)),key=lambda x :x[0],reverse=True)[:10])
# print(unrolling_of_rsa_d0.speaker_likelihood(state,world,"Il est autorisé à aller".split(" ")+["</s>"]) )
# probs_d0,support_d0 = unrolling_of_rsa_d0.beam_search(beam_width=1,pass_prior=True,cut_rate=1,decay_rate=0,state=state,world=RSA_World(target=0,rationality=0,speaker=0,hyperprior=0,language_model=0,qud=0))
# print("UNROLLING OF RSA d0",display(probs=probs_d0,support=support_d0))

# base_rsa_d1 = RSA(depth=1,models=models,urls=sentences,lang_mods=lang_mods,quds=quds)
# world = RSA_World(target=0,speaker=0,rationality=0,hyperprior=0,language_model=0,qud=0)
# state = RSA_State(initial_world_prior,seg_type=base_rsa_d1.seg_type)
# unrolling_of_rsa_d1 = Unrolled_Model(images=sentences,rationalities=rats,hyperpriors=hyperpriors, underlying_model=base_rsa_d1,max_sentence_length=max_sentence_length)
# probs_d10,support_d10 = unrolling_of_rsa_d1.beam_search(beam_width=1,pass_prior=True,cut_rate=1,decay_rate=0,state=state,world=world)

# base_rsa_d1 = RSA(depth=1,models=models,urls=sentences,lang_mods=lang_mods,quds=quds)
# world = RSA_World(target=1,speaker=0,rationality=0,hyperprior=0,language_model=0,qud=0)
# state = RSA_State(initial_world_prior,seg_type=base_rsa_d1.seg_type)
# unrolling_of_rsa_d1 = Unrolled_Model(images=sentences,rationalities=rats,hyperpriors=hyperpriors, underlying_model=base_rsa_d1,max_sentence_length=max_sentence_length)
# probs_d11,support_d11 = unrolling_of_rsa_d1.beam_search(beam_width=1,pass_prior=True,cut_rate=1,decay_rate=0,state=state,world=world)

# print("UNROLLING OF RSA d0",display(probs=probs_d00,support=support_d00))
# print("UNROLLING OF RSA d0",display(probs=probs_d01,support=support_d01))
# print("UNROLLING OF RSA d1",display(probs=probs_d10,support=support_d10))
# print("UNROLLING OF RSA d1",display(probs=probs_d11,support=support_d11))

