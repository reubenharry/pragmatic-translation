
import pickle
import nltk
from nltk.translate.bleu_score import sentence_bleu
from utils.collect_sentences import dev_sentences
from utils.helper_functions import display,byte_pair_encoding,byte_pair_decoding



# reference = [['this', 'is', 'small', 'test']]
# candidate = ['this', 'is', 'a', 'test']
# score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
# print(score)



LOAD=True

if LOAD:

	results_dict = pickle.load(open("bleu_test_sentences.pkl",'rb'))

elif not LOAD:
	from scripts.base_case import source_target_model,target_source_model,unfolded_source_target,unfolded_target_source
	from bayesian_agents.bayesian_pragmatics import Pragmatic
	from translators.translators import Anamorphism
	
	rightward_pragmatic = Pragmatic(
		rightward_model=source_target_model,
		leftward_model=target_source_model, 
		unfolded_leftward_model=None,
		EXTEND=True,
		width=2,rat=2.0)

#	rightward_pragmatic_

	rightward_pragmatic_global = Pragmatic(
		rightward_model=unfolded_source_target,
		leftward_model=unfolded_target_source, 
		unfolded_leftward_model=None,
		EXTEND=False,
		width=2,rat=2.0)

	unfolded_rightward_pragmatic = Anamorphism(underlying_model=rightward_pragmatic,beam_width=2)

	results_dict = {"literal":{},"global pragmatic":{},"nonparametric incremental":{},'nonparametric incremental global':{}}
	for sent_dict in dev_sentences[:]:

		try:
			sentence = sent_dict["en"]

			# unfolded_source_target.cache={}
			# unfolded_rightward_pragmatic_explicit.cache


			probs,support = unfolded_source_target.forward(sequence=[],source_sentence=sentence,debug=False)
			print(display(probs=probs,support=support))
			results_dict["literal"][sentence]=display(probs=probs,support=support)[0]

			probs,support = unfolded_rightward_pragmatic.forward(sequence=[],source_sentence=sentence,debug=False)
			print(display(probs=probs,support=support))
			results_dict["nonparametric incremental"][sentence]=display(probs=probs,support=support)[0]

			probs,support = rightward_pragmatic_global.forward(sequence=[],source_sentence=sentence,debug=False)
			print(display(probs=probs,support=support))
			results_dict["nonparametric incremental global"][sentence]=display(probs=probs,support=support)[0]


			unfolded_source_target.empty_cache()
			unfolded_rightward_pragmatic.empty_cache()

		except: continue

	pickle.dump(results_dict,open("bleu_test_sentences.pkl",'wb'))

scores = {"lit":0,"prag":0,"global prag":0}
for sent_dict in dev_sentences[:]:

	try:
		lit_translation = nltk.word_tokenize(byte_pair_decoding(results_dict["literal"][sent_dict["en"]][0]))
		prag_translation = nltk.word_tokenize(byte_pair_decoding(results_dict["nonparametric incremental"][sent_dict["en"]][0]))
		global_prag_translation = nltk.word_tokenize(byte_pair_decoding(results_dict["nonparametric incremental global"][sent_dict["en"]][0]))
		target = nltk.word_tokenize(sent_dict["de"])
		print("sentence",sent_dict["en"])
		print("target",target)
		print("lit candidate",lit_translation)
		print("prag candidate",prag_translation)
		print("global prag candidate",global_prag_translation)

		lit_score = sentence_bleu([target],lit_translation, weights=(1, 0, 0, 0))
		prag_score = sentence_bleu([target],prag_translation, weights=(1, 0, 0, 0))
		global_prag_score = sentence_bleu([target],global_prag_translation, weights=(1, 0, 0, 0))
		scores["lit"]+=lit_score
		scores["prag"]+=prag_score
		scores["global prag"]+=global_prag_score

		print("lit score",lit_score)
		print("prag_score",prag_score)
	except:continue
print("results",scores)
