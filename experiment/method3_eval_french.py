from utils.collect_sentences import dev_sentences,test_sentences
print(len(test_sentences))
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
from utils.get_sents_from_wmt_test_set import english_test_sentences



RAT = 2.0
LOAD=False
path = "_2018_wmt_test_french"
source_lang='en'
target_lang='fr'


if LOAD:

	results_dict = pickle.load(open("experiment/results/rat"+str(RAT)+"_bleu_test_sentences"+path+".pkl",'rb'))

elif not LOAD:

	# from scripts.base_case import source_target_model,target_source_model,unfolded_source_target,unfolded_target_source
	
	source_target_word_model = Coalgebra(path='wmt14.en-fr.joined-dict.transformer',source='en',target='fr',bpe_code_path='/bpecodes')
	target_source_word_model = Coalgebra(path='fconv_wmt_fr_en',source='fr',target='en',bpe_code_path='/code')

	source_target = source_target_word_model
	target_source = target_source_word_model

	unfolded_source_target = Anamorphism(underlying_model=source_target,beam_width=2)
	unfolded_target_source = Anamorphism(underlying_model=target_source,beam_width=1)


	rightward_pragmatic = Pragmatic(
		rightward_model=source_target,
		leftward_model=target_source, 
		unfolded_leftward_model=None,
		EXTEND=True,
		width=2,rat=RAT)

#	rightward_pragmatic_

	rightward_pragmatic_global = Pragmatic(
		rightward_model=unfolded_source_target,
		leftward_model=unfolded_target_source, 
		unfolded_leftward_model=None,
		EXTEND=False,
		width=2,rat=RAT)


	unfolded_rightward_pragmatic = Anamorphism(underlying_model=rightward_pragmatic,beam_width=1)

	results_dict = {
		"target":{"literal":{},"incremental pragmatic":{},"global pragmatic":{},"nonparametric incremental":{}},
		"source":{"literal":{},"incremental pragmatic":{},"global pragmatic":{},"nonparametric incremental":{}}
		}
	for sentence in english_test_sentences[:]:

		# sentence = sent_dict["en"]
		try:
			print("SENTENCE",sentence)
			
			probs,support = unfolded_rightward_pragmatic.forward(sequence=[],source_sentence=sentence,debug=False)
			translated_sentence = byte_pair_decoding(display(probs=probs,support=support)[0][0])
			results_dict["target"]["nonparametric incremental"][sentence]=translated_sentence
			results_dict["source"]["nonparametric incremental"][sentence]=Translator().translate(text=translated_sentence,src=target_lang, dest=source_lang).text

			unfolded_source_target.empty_cache()
			rightward_pragmatic_global.empty_cache()
			unfolded_rightward_pragmatic.empty_cache()

			probs,support = unfolded_source_target.forward(sequence=[],source_sentence=sentence,debug=False)
			translated_sentence = byte_pair_decoding(display(probs=probs,support=support)[0][0])
			results_dict["target"]["literal"][sentence]=translated_sentence
			results_dict["source"]["literal"][sentence]=Translator().translate(text=translated_sentence,src=target_lang, dest=source_lang).text

			unfolded_source_target.empty_cache()
			rightward_pragmatic_global.empty_cache()
			unfolded_rightward_pragmatic.empty_cache()

			probs,support = rightward_pragmatic_global.forward(sequence=[],source_sentence=sentence,debug=False)
			translated_sentence = byte_pair_decoding(display(probs=probs,support=support)[0][0])
			results_dict["target"]["global pragmatic"][sentence]=translated_sentence
			results_dict["source"]["global pragmatic"][sentence]=Translator().translate(text=translated_sentence,src=target_lang, dest=source_lang).text

			unfolded_source_target.empty_cache()
			rightward_pragmatic_global.empty_cache()
			unfolded_rightward_pragmatic.empty_cache()


			print(results_dict)
			with open("experiment/results/rat"+str(RAT)+"_bleu_test_sentences"+path+".txt",'w') as fw:
				fw.write(str(results_dict))
			pickle.dump(results_dict,open("experiment/results/rat"+str(RAT)+"_bleu_test_sentences"+path+".pkl",'wb'))

		except: continue
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



