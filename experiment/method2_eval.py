from py_translator import Translator
import pickle
import nltk
from utils.config import *
from utils.helper_functions import uniform_vector, make_initial_prior, display,byte_pair_encoding,byte_pair_decoding,unique_list
from experiment.exp1_data import single_sentence_generator
# from scripts.base_case import source_target,target_source,unfolded_source_target
from utils.paraphrase import get_paraphrases
import re
import requests
import time
from collections import defaultdict
from utils.config import *
from translators.translators import Coalgebra, Anamorphism, Factor_To_Character, Constant, Compose
from utils.paraphrase import get_paraphrases
from bayesian_agents.bayesian_pragmatics import Pragmatic
from experiment.exp1_data import french_sentence_pairs,german_sentence_pairs
from nltk.translate.bleu_score import sentence_bleu
import nltk

collect_pairs=False

if collect_pairs:

	pairs = []
	for sent in single_sentence_generator():
		print("sent",sent)
		out = get_paraphrases(rightward_model=source_target,leftward_model=target_source,sentence=sent)
		pairs.append(out)

		pickle.dump(pairs,open('experiment/results/paraphrase_pairs.pkl','wb'))
		with open("experiment/results/paraphrase_pairs.txt",'w') as f:
			f.write(str(pairs))
		print(pairs)

else: pairs = pickle.load(open("experiment/results/paraphrase_pairs.pkl","rb"))

pairs = [pair for pair in pairs if pair is not None]




RAT = 5.0
LOAD=False

source_lang='en'
target_lang='de'

path = "beam"

if LOAD:

	results_dict = pickle.load(open("experiment/results/rat"+str(RAT)+"_method2_sentences_"+path+".pkl",'rb'))

elif not LOAD:

	# from scripts.base_case import source_target_model,target_source_model,unfolded_source_target,unfolded_target_source
	
	source_target_word_model = Coalgebra(path='wmt16.en-de.joined-dict.transformer',source='en',target='de',bpe_code_path='/bpecodes')
	target_source_word_model = Coalgebra(path='fconv_wmt_de_en',source='de',target='en',bpe_code_path='/code')

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


	unfolded_rightward_pragmatic = Anamorphism(underlying_model=rightward_pragmatic,beam_width=2)

	results_dict = {
		"target":{"literal":{},"incremental pragmatic":{},"global pragmatic":{},"nonparametric incremental":{}},
		"source":{"literal":{},"incremental pragmatic":{},"global pragmatic":{},"nonparametric incremental":{}}
		}
	for pair in pairs:



		for sentence in pair:
	
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
				with open("experiment/results/rat"+str(RAT)+"_method2_sentences_"+path+".txt",'w') as fw:
					fw.write(str(results_dict))
				pickle.dump(results_dict,open("experiment/results/rat"+str(RAT)+"_method2_sentences_"+path+".pkl",'wb'))

			except: continue

# "There is a lovely time in the evening . "
