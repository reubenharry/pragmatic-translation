import matplotlib
matplotlib.use('Agg')
import re
import requests
from py_translator import Translator
import time
import pickle
import numpy as np
import scipy
import scipy.special
from collections import defaultdict
from utils.config import *
from utils.helper_functions import display,byte_pair_encoding,byte_pair_decoding
from translators.translators import Coalgebra, Anamorphism, Factor_To_Character, Constant, Compose
from utils.paraphrase import get_paraphrases
from bayesian_agents.bayesian_pragmatics import Pragmatic
from experiment.exp1_data import french_sentence_pairs,german_sentence_pairs
from nltk.translate.bleu_score import sentence_bleu
import nltk


LOAD=True

if LOAD:

	results_dict = pickle.load(open("experiment/results_dict.pkl",'rb'))

elif not LOAD:
	source_lang = "en"
	target_lang = "de"

	source_target_word = Coalgebra(path='wmt16.en-de.joined-dict.transformer',source='en',target='de',bpe_code_path='/bpecodes')
	target_source_word = Coalgebra(path='fconv_wmt_de_en',source='de',target='en',bpe_code_path='/code')



	source_target = source_target_word
	target_source = target_source_word

	unfolded_source_target = Anamorphism(underlying_model=source_target,beam_width=2)


	rightward_pragmatic = Pragmatic(
		rightward_model=source_target,
		leftward_model=target_source, 
		unfolded_leftward_model=None,
		EXTEND=True,
		width=2,rat=5.0)

	explicit_distractor_target_source = Constant(support=["sentence"],unfolded_model=unfolded_source_target,change_direction=True)

	rightward_pragmatic_explicit = Pragmatic(
		rightward_model=source_target,
		leftward_model=None, 
		unfolded_leftward_model=explicit_distractor_target_source,
		width=100,rat=5.0,
		EXTEND=False)

	global_unfolded_rightward_pragmatic_explicit = Pragmatic(
		rightward_model=unfolded_source_target,
		leftward_model=None, 
		unfolded_leftward_model=explicit_distractor_target_source,
		width=100,rat=5.0,
		EXTEND=False)

	unfolded_rightward_pragmatic = Anamorphism(underlying_model=rightward_pragmatic,beam_width=1)
	unfolded_rightward_pragmatic_explicit = Anamorphism(underlying_model=rightward_pragmatic_explicit,beam_width=1,diverse=False)


	from experiment.exp1_data import sentence_generator
	sentences = sentence_generator()
	# sentences = german_sentence_pairs

	results_dict = {
		"target":{"literal":{},"incremental pragmatic":{},"global pragmatic":{},"nonparametric incremental":{}},
		"source":{"literal":{},"incremental pragmatic":{},"global pragmatic":{},"nonparametric incremental":{}}
		}

	for pair in sentences:
		
		pair = [byte_pair_encoding(sentence=p,code_path=source_target.bpe_code_path) for p in pair]
		explicit_distractor_target_source.support=pair
		print("PAIR",pair)

		# sentence = pair[1]
		# if True:
		
		for sentence in pair:

		# unfolded_source_target.cache={}
		# unfolded_rightward_pragmatic_explicit.cache
	# byte_pair_encoding(sentence="My cousin",code_path=rightward_pragmatic_explicit.bpe_code_path).split()
			

			probs,support = global_unfolded_rightward_pragmatic_explicit.forward(sequence=[],source_sentence=sentence,debug=False)
			translated_sentence = byte_pair_decoding(display(probs=probs,support=support)[0][0])
			results_dict["target"]["global pragmatic"][sentence]=translated_sentence
			results_dict["source"]["global pragmatic"][sentence]=Translator().translate(text=translated_sentence,src=target_lang, dest=source_lang).text

			unfolded_source_target.empty_cache()
			global_unfolded_rightward_pragmatic_explicit.empty_cache()
			unfolded_rightward_pragmatic_explicit.empty_cache()

			probs,support = unfolded_source_target.forward(sequence=[],source_sentence=sentence,debug=False)
			translated_sentence = byte_pair_decoding(display(probs=probs,support=support)[0][0])
			results_dict["target"]["literal"][sentence]=translated_sentence
			results_dict["source"]["literal"][sentence]=Translator().translate(text=translated_sentence,src=target_lang, dest=source_lang).text

			unfolded_source_target.empty_cache()
			global_unfolded_rightward_pragmatic_explicit.empty_cache()
			unfolded_rightward_pragmatic_explicit.empty_cache()


			probs,support = unfolded_rightward_pragmatic_explicit.forward(sequence=[],source_sentence=sentence,debug=False)
			translated_sentence = byte_pair_decoding(display(probs=probs,support=support)[0][0])
			results_dict["target"]["incremental pragmatic"][sentence]=translated_sentence
			results_dict["source"]["incremental pragmatic"][sentence]=Translator().translate(text=translated_sentence,src=target_lang, dest=source_lang).text


			unfolded_source_target.empty_cache()
			global_unfolded_rightward_pragmatic_explicit.empty_cache()
			unfolded_rightward_pragmatic_explicit.empty_cache()

			probs,support = unfolded_rightward_pragmatic.forward(sequence=[],source_sentence=sentence,debug=False)
			translated_sentence = byte_pair_decoding(display(probs=probs,support=support)[0][0])
			results_dict["target"]["nonparametric incremental"][sentence]=translated_sentence
			results_dict["source"]["nonparametric incremental"][sentence]=Translator().translate(text=translated_sentence,src=target_lang, dest=source_lang).text


			unfolded_source_target.empty_cache()
			global_unfolded_rightward_pragmatic_explicit.empty_cache()
			unfolded_rightward_pragmatic_explicit.empty_cache()


			print(results_dict)
			with open("experiment/result_dict.txt",'w') as fw:
				fw.write(str(results_dict))
			pickle.dump(results_dict,open("experiment/results_dict.pkl",'wb'))

		# break

print(results_dict)
for orig in results_dict["source"]['literal']:

	# lit = results_dict["source"]["literal"]
	# prag = results_dict['source']['incremental pragmatic']

	target = nltk.word_tokenize(byte_pair_decoding(orig))
	lit_translation = nltk.word_tokenize(byte_pair_decoding(results_dict["source"]["literal"][orig]))
	inc_prag_translation = nltk.word_tokenize(byte_pair_decoding(results_dict["source"]["incremental pragmatic"][orig]))
	non_param_inc_prag_translation = nltk.word_tokenize(byte_pair_decoding(results_dict["source"]["nonparametric incremental"][orig]))
	glob_prag_translation = nltk.word_tokenize(byte_pair_decoding(results_dict["source"]["global pragmatic"][orig]))


	print("target",target)
	print("lit translation",lit_translation)
	print("prag translation",inc_prag_translation)
	print("non_param_inc_prag_translation",non_param_inc_prag_translation)
	print("glob_prag_translation",glob_prag_translation)


	lit_score = sentence_bleu([target],lit_translation, weights=(1, 0, 0, 0))
	prag_score = sentence_bleu([target],inc_prag_translation, weights=(1, 0, 0, 0))

	print("lit score",lit_score)
	print("prag score",prag_score)



