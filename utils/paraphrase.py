import matplotlib
matplotlib.use('Agg')
import re
from py_translator import Translator
import requests
import time
import pickle
import numpy as np
import scipy
import scipy.special
import pickle
import nltk
from collections import defaultdict
from utils.config import *
from utils.helper_functions import uniform_vector, make_initial_prior, display,byte_pair_encoding,byte_pair_decoding,unique_list
from translators.translators import Coalgebra, Anamorphism
from experiment.exp1_data import single_sentence_generator
def get_paraphrases(
	sentence,
	rightward_model,
	leftward_model,
	):

	unfolded_rightward_model = Anamorphism(underlying_model=rightward_model,beam_width=1)
	unfolded_leftward_model = Anamorphism(underlying_model=leftward_model,beam_width=2)

	# sentence = byte_pair_encoding(sentence=sentence.lower(),code_path=unfolded_rightward_model.bpe_code_path)

	# _ , target_sents = unfolded_rightward_model.forward(sequence=[],source_sentence=sentence)
	# target_sent = target_sents[0].lower()

	target_sent = Translator().translate(text=sentence,src='en', dest='fr').text

	# target_sent = "Der mann lacht."

	# input_target_sent = byte_pair_encoding(sentence=target_sent,code_path=leftward_code_path)
	print("target sentence:",target_sent)
	# print("INPUT TARGET SENTENCE",input_target_sent)
	probs , paraphrases = unfolded_leftward_model.forward(sequence=[],source_sentence=target_sent)
	unfolded_leftward_model.empty_cache()
	print("ALL PARAPHRASES",display(probs=probs,support=paraphrases))
	# print("paraphrases",paraphrases)
	# raise Exception

	paraphrase1=paraphrases[0]
	paraphrase2=paraphrases[1]

	_ , backtrans1=unfolded_rightward_model.forward(sequence=[],source_sentence=paraphrase1)
	_ , backtrans2=unfolded_rightward_model.forward(sequence=[],source_sentence=paraphrase2)
	backtrans1=backtrans1[0]
	backtrans2=backtrans2[0]

	print("tok",nltk.word_tokenize(byte_pair_decoding(backtrans1)),nltk.word_tokenize(byte_pair_decoding(backtrans2)))
	condition1 = nltk.word_tokenize(byte_pair_decoding(backtrans1))==nltk.word_tokenize(byte_pair_decoding(backtrans2))
	if condition1:

		return [paraphrase1,paraphrase2] 

	# counter = 0
	# # many_to_ones = []
	# for paraphrase in paraphrases:

	# 	print("paraphrase",byte_pair_decoding(paraphrase))

	# 	# print(nltk.word_tokenize(back),nltk.word_tokenize(sent))
	# 	condition1=nltk.word_tokenize(sentence)!=nltk.word_tokenize(byte_pair_decoding(paraphrase))
	# 	print("condition1",condition1)

	# 	if condition1:
	# 		_ , back_trans = unfolded_rightward_model.forward(sequence=[],source_sentence=paraphrase)
	# 		back_trans = back_trans[0].lower()
	# 		print("back trans",back_trans)
	# 		# target_sent=target_sent.lower()
	# 		condition2=nltk.word_tokenize(back_trans)==nltk.word_tokenize(target_sent)
	# 		print("condition2",condition2)

	# 		if condition2:
	# 			counter+=1
	# 			return [sentence,paraphrase]




		# input_paraphrase = re.sub("@@ ","",paraphrase)
		# input_paraphrase = re.sub("\.","",input_paraphrase)

		# input_paraphrase = input_target_sent = byte_pair_encoding(sentence=input_paraphrase.lower(),code_path=rightward_code_path)
		


		# print("back_trans","|"+back_trans+"|")
		# print("target_sentence:","|"+target_sent+"|")

		# print(target_sent==back_trans)

		# if target_sent==back_trans:

		# 	counter+=1
		# 	many_to_ones.append(paraphrase)

	# print("COUNTER:",counter)
	# return many_to_ones

if __name__=="__main__":

	# from scripts.base_case import source_target_model,target_source_model,unfolded_source_target

	path = "_french"

	source_target_model = Coalgebra(path='wmt14.en-fr.joined-dict.transformer',source='en',target='fr',bpe_code_path='/bpecodes')
	target_source_model = Coalgebra(path='fconv_wmt_fr_en',source='fr',target='en',bpe_code_path='/code')


	results = []
	for sent in single_sentence_generator():
		print("sent",sent)
		out = get_paraphrases(rightward_model=source_target_model,leftward_model=target_source_model,sentence=sent)
		results.append(out)

		print(results)
		pickle.dump(results,open('experiment/paraphrase_pairs'+path+'.pkl','wb'))

	# "There is a lovely time in the evening . "
	# print(out)
	# raise Exception

	# if out is not None:

	# 	from translators.translators import Coalgebra, Anamorphism, Factor_To_Character, Constant, Compose
	# 	from bayesian_agents.bayesian_pragmatics import Pragmatic


	# 	explicit_distractor_target_source = Constant(support=out,unfolded_model=unfolded_source_target,change_direction=True)

	# 	rightward_pragmatic_explicit = Pragmatic(
	# 		rightward_model=source_target_model,
	# 		leftward_model=None, 
	# 		unfolded_leftward_model=explicit_distractor_target_source,
	# 		width=100,rat=10.0,
	# 		EXTEND=False)

	# 	unfolded_rightward_pragmatic_explicit = Anamorphism(underlying_model=rightward_pragmatic_explicit,beam_width=2)


	# 	results_dict = {"literal":{},"pragmatic":{}}
		
	# 	for sentence in out:

	# 		# unfolded_source_target.cache={}
	# 		# unfolded_rightward_pragmatic_explicit.cache


	# 		probs,support = unfolded_source_target.forward(sequence=[],source_sentence=sentence,debug=False)
	# 		print(display(probs=probs,support=support))
	# 		results_dict["literal"][sentence]=display(probs=probs,support=support)[0]

	# 		probs,support = unfolded_rightward_pragmatic_explicit.forward(sequence=[],source_sentence=sentence,debug=False)
	# 		print(display(probs=probs,support=support))
	# 		results_dict["pragmatic"][sentence]=display(probs=probs,support=support)[0]

	# 	print(results_dict)
	# google_paraphrases = pickle.load(open("paraphrase_data.pkl",'rb'))

	# google_paraphrases = [x for x in google_paraphrases if x!=None]

	# new_paraphrases=[]
	# for pair in google_paraphrases:
	# 	new_pair = []
	# 	for s in pair:

	# 		# print(s)
	# 		if s[-1]=="\n":
	# 			s=s[:-1]
	# 		if s[-1] not in ["!","?","."]:
	# 			s+="."

	# 		new_pair.append(s)
	# 	new_paraphrases.append(new_pair)
	# google_paraphrases=new_paraphrases
	# print(google_paraphrases)


	# # raise Exception
	# # data = {sentence:translation}

	# # condition: first and second equal under translation s0

	# # filter by condition:
	# # 	return short list

	# # beam search 2

	# # pickle results as "s0collapses"
	# # return results



	# s0_dict = {}

	# LOAD = True
	# if not LOAD:

	# 	from scripts.base_case import unfolded_source_target
	# 	for pair in google_paraphrases:
	# 		for s in pair:

	# 			# print(s)
	# 			# if s[-1]=="\n":
	# 			# 	s=s[:-1]
	# 			# if s[-1] not in ["!","?","."]:
	# 			# 	s+="."
	# 			# print(s)
	# 			translation = unfolded_source_target.forward(sequence=[],source_sentence=s,debug=False)[1][0]
	# 			print(translation)
	# 			s0_dict[s]=translation

	# 	pickle.dump(s0_dict,open("s0_dict.pkl",'wb'))

	# elif LOAD:

	# 	s0_dict = pickle.load(open("s0_dict.pkl",'rb'))

	# 	for pair in google_paraphrases:
	# 		print(s0_dict[pair[0]])
	# 		print(s0_dict[pair[1]])
	# 		print(s0_dict[pair[0]]==s0_dict[pair[1]])

		# for x in s0_dict:
		# 	print(x)
		# 	print(s0_dict[x])
			# s0_dict[x] = s0_dict[x][1][0]
			# pickle.dump(s0_dict,open("s0_dict.pkl",'wb'))

			


	# from utils.collect_sentences import sentences as dev_sentences

# sentences = ["he is gone .".lower(), "he has gone .".lower(),"he must have gone .", "he could have gone .", "he might have gone .", "he might go .", "he could go .", "he must go .", "he may go ."]
# "This is the seventh incident in the past few days . "
	
	
	# viable = []
	# print("DEV SENTS 1",dev_sentences[:1])
	# for dev_sentence in dev_sentences[:1]:

	# 	print("SENTENCE")

	# 	paraphrases = get_paraphrases(
	# 		sentence=dev_sentence,
	# 		rightward_model=source_target_model,
	# 		leftward_model=target_source_model)

	# 	paraphrases = unique_list([dev_sentence]+paraphrases)
	# 	paraphrases = [p.lower() for p in paraphrases]

	# 	if len(paraphrases)>1:
	# 		viable.append(paraphrases)
	# 		print("\n\n\nSUCCESS\n\n\n",paraphrases)

	# print(viable)
	
