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
from utils.helper_functions import display,byte_pair_encoding,byte_pair_decoding
from translators.translators import Coalgebra, Anamorphism, Factor_To_Character, Constant, Compose
from utils.paraphrase import get_paraphrases
from bayesian_agents.bayesian_pragmatics import Pragmatic
from experiment.exp1_data import french_sentence_pairs,german_sentence_pairs
# from utils.collect_sentences import sentences as dev_sentences

# sentences = pickle.load(open("paraphrase_data.pkl",'rb'))

# "Am I meant to understand?"
# sentences = ["Once again, the car broke . ",
# "The car broke again . ",
# "Yet again, the car broke . "]

# sentences = french_sentence_pairs[-2]
# sentences = ["I am not happy about what we have on the ground and in the air in Afghanistan.","I am not content about what we have on the ground and in the air in Afghanistan."]

# sentences = ["The man cried tears.","The man wept tears."]
# sentences = ["I made a cake.","I baked a cake."]
# sentences = ["Tu manges.","Vous mangez."]
# sentences = ["I eat no fish.","I do not eat fish."]
# sentences = ["Mon cousin est gros.","Ma cousine est grosse."]
# sentences = ["Don&apos;t go . ","Do not go . "]
# sentences = ["Where is your friend ? ", "Where is your friend at ? "]
# sentences = ["I am about to go . ", "I am going to go . ","I will go . "]
# sentences = ["What's your issue ? ","What's your problem ? "]
# sentences = ["I laughed and cried also . ", "I laughed and cried too . "]
# sentences = ["I attempted to go . ", "I tried to go . "]

	# I have yet to go, I have to go
	# I must be going, I must go
	# I will go, I'm going to go 
# sentences = ["I love you . ","I like you . "]
# sentences = ["I don't like you . ", "I do not like you . "]
# sentences = ["je m&apos appelle content . "]
# sentences = ["The book is on the table . ","The book is on top of the table . "]
# ,"J'ai vu ma cousine . ", "J'ai vu mon cousin . "]
# print(sentences)





# raise Exception
# sentences = sentences[-3]
# print("SENTENCES",sentences)
# source_target_word = Coalgebra(path='wmt14.en-fr.fconv-py',source='en',target='fr',bpe_code_path='/bpecodes')

# source_target_word = Coalgebra(path='fconv_wmt_fr_en',source='fr',target='en',bpe_code_path='/code')

# source_target_word = Coalgebra(path='wmt14.en-fr.joined-dict.transformer',source='en',target='fr',bpe_code_path='/bpecodes')
# source_target_word = Coalgebra(path='wmt16.en-de.joined-dict.transformer',source='en',target='de',bpe_code_path='/bpecodes')
# target_source_word = Coalgebra(path='fconv_wmt_fr_en',source='fr',target='en',bpe_code_path='/code')
# target_source_word = Coalgebra(path='fconv_wmt_de_en',source='de',target='en',bpe_code_path='/code')

# source_target_word = Coalgebra(path='fconv_wmt_fr_en',source='fr',target='en',bpe_code_path='/code')


# source_target_word = Coalgebra(path='fconv_wmt_fr_en',source='fr',target='en',bpe_code_path='/code')


source_target_word = Coalgebra(path='fconv_wmt_en_es',source='en',target='es',bpe_code_path='/code')
# target_source_word = None

# source_target_word = Coalgebra(path='fconv_wmt_de_en',source='de',target='en',bpe_code_path='/code')

# target_source_word = Coalgebra(path='iwslt14.tokenized.de-en',source='de',target='en',bpe_code_path='/code')
# target_source_word = Coalgebra(path='wmt14.de-en.fconv-py',source='de',target='en',bpe_code_path='/code')


# source_target_char = Factor_To_Character(source_target_word)
# target_source_char = Factor_To_Character(target_source_word)


source_target = source_target_word
# target_source = target_source_word
# source_target = source_target_char
# target_source = target_source_char

unfolded_source_target = Anamorphism(underlying_model=source_target,beam_width=2)
# unfolded_target_source = Anamorphism(underlying_model=target_source,beam_width=1)

# sentences = ["He squirmed. "]
# sentences = ["lebens@@ mittel : wo die europä@@ ische inf@@ lation ab@@ geru@@ t@@ scht ist <@@ /@@ s@@ >".lower()]
# sentences = dev_sentences

# probs,support = unfolded_source_target.forward(source_sentence=sentence,sequence=[])
# print(display(probs=probs,support=support))
# raise Exception


# sentences = ["he has gone .".lower(),"he is gone .".lower(),"he went . ",
# "Die Studenten der Tatra-Gemeinde demonstrieren ihre Fantasie, indem sie die fiktive Festung \"die Stadt\" schufen",
# "Students of the Dattatreya city Municipal corporation secondary school demonstrated their imagination power by creating the fictitious fort 'Duttgarh'.".lower(),
# "This is the seventh incident in the past few days . ".lower(),]

# sentences = ["Er hat gegangen .".lower(),]

# paraphrases = get_paraphrases(
# 	sentence=sentences[-1],
# 	rightward=source_target,
# 	leftward=target_source)
# print(paraphrases)
# raise Exception

# sentence = sentences[0]

# rightward_pragmatic = Pragmatic(
# 	rightward_model=source_target,
# 	leftward_model=target_source, 
# 	unfolded_leftward_model=None,
# 	EXTEND=True,
# 	width=2,rat=5.0)

# leftward_pragmatic = Pragmatic(
# 	rightward=target_source, 
# 	leftward=source_target,
# 	unfolded_leftward=None,
# 	EXTEND=True,
# 	width=2,rat=10.0)

# sentences = ["large dogs run . ","big dogs run . "]

# explicit_distractor_target_source = Constant(support=["sentence"],unfolded_model=unfolded_target_source,change_direction=False)

explicit_distractor_target_source = Constant(support=["sentence"],unfolded_model=unfolded_source_target,change_direction=True)

rightward_pragmatic_explicit = Pragmatic(
	rightward_model=source_target,
	leftward_model=None, 
	unfolded_leftward_model=explicit_distractor_target_source,
	width=2,rat=10.0,
	EXTEND=False)

# global_unfolded_rightward_pragmatic_explicit = Pragmatic(
# 	rightward_model=unfolded_source_target,
# 	leftward_model=None, 
# 	unfolded_leftward_model=explicit_distractor_target_source,
# 	width=100,rat=5.0,
# 	EXTEND=False)



# unfolded_rightward_pragmatic = Anamorphism(underlying_model=rightward_pragmatic,beam_width=1)
unfolded_rightward_pragmatic_explicit = Anamorphism(underlying_model=rightward_pragmatic_explicit,beam_width=2)

# unfolded_leftward = Anamorphism(underlying_model=leftward_pragmatic)



# source_source = Compose(rightward_model=source_target_model,leftward_model=target_source_model, unfolded_rightward_model=None)

# unfolded_source_source = Anamorphism(underlying_model=source_source,beam_width=1)

# source_source_pragmatic = Compose(rightward_model=rightward_pragmatic,leftward_model=target_source_model, unfolded_rightward_model=None)
# unfolded_source_source_pragmatic = Anamorphism(underlying_model=source_source_pragmatic,beam_width=1)

# sentences = french_sentence_pairs

# sentences = [["Some of the chairs are blue.","Some chairs are blue."]]
# sentences = [["She has arrived . ", "She arrived . "]]
# sentences = [["I saw this movie.","I saw that movie."],["What is this?","What is that?"]]
# sentences = [["I chopped up the tree.","I chopped down the tree."]]
sentences = [["The animals run quickly.","Animals run quickly."]]
# sentences = [["She went away.","She is gone."],["She chopped up the tree.","She chopped down the tree."]]
# sentences = [["some of the chairs are blue . ","some chairs are blue . "]]
# from experiment.exp1_data import sentence_generator
# sentences = sentence_generator()
# print(sentences)
# sentences = [["The teams don't play this week.","The teams don't play that week."]]

# sentences = [[byte_pair_decoding("Je m'appelle Claude.")]]

results_dict = {"literal":{},"incremental pragmatic":{},"global pragmatic":{},"nonparametric incremental":{}}
for pair in sentences:
	
	pair = [byte_pair_encoding(sentence=p,code_path=source_target.bpe_code_path) for p in pair]
	explicit_distractor_target_source.support=pair
	print("PAIR",pair)
	
	for sentence in pair:

		print("sents",explicit_distractor_target_source.support)

	# unfolded_source_target.cache={}
	# unfolded_rightward_pragmatic_explicit.cache
# byte_pair_encoding(sentence="My cousin",code_path=rightward_pragmatic_explicit.bpe_code_path).split()
		# probs,support = global_unfolded_rightward_pragmatic_explicit.forward(sequence=[],source_sentence=sentence,debug=False)
		# print(display(probs=probs,support=support))
		# results_dict["global pragmatic"][sentence]=display(probs=probs,support=support)[0]


		probs,support = unfolded_rightward_pragmatic_explicit.forward(sequence=[],source_sentence=sentence,debug=False)
		print(display(probs=probs,support=support))
		results_dict["incremental pragmatic"][sentence]=display(probs=probs,support=support)


		unfolded_source_target.empty_cache()
		# global_unfolded_rightward_pragmatic_explicit.empty_cache()
		unfolded_rightward_pragmatic_explicit.empty_cache()
		# probs,support = unfolded_rightward_pragmatic.forward(sequence=[],source_sentence=sentence,debug=False)
		# print(display(probs=probs,support=support))
		# results_dict["nonparametric incremental"][sentence]=display(probs=probs,support=support)[0]

		probs,support = unfolded_source_target.forward(sequence=[],source_sentence=sentence,debug=False)
		print(display(probs=probs,support=support))
		results_dict["literal"][sentence]=display(probs=probs,support=support)


		unfolded_source_target.empty_cache()
		# global_unfolded_rightward_pragmatic_explicit.empty_cache()
		unfolded_rightward_pragmatic_explicit.empty_cache()


		print(results_dict)
		# with open("experiment/result_dict.txt",'w') as fw:
		# 	fw.write(str(results_dict))
		# pickle.dump(results_dict,open("experiment/results_dict.pkl",'wb'))


	# break
	# raise Exception
print(results_dict)

# {'literal': {'She is gone . ': ('Elle est partie .', 1.0), 'She has gone . ': ('Elle est partie .', 1.0), 'She ch@@ op@@ ped up the tree . ': ('Elle a coup@@ é l&apos; arbre .', 1.0), 'She ch@@ op@@ ped down the tree . ': ('Elle a coup@@ é l&apos; arbre .', 1.0)}, 'incremental pragmatic': {'She is gone . ': ('Elle n&apos; existe plus .', 1.0), 'She has gone . ': ('Elle s&apos; en est all@@ ée .', 1.0), 'She ch@@ op@@ ped up the tree . ': ('Elle coup@@ a l&apos; arbre .', 1.0), 'She ch@@ op@@ ped down the tree . ': ('Elle a abatt@@ u l&apos; arbre .', 1.0)}, 'global pragmatic': {}, 'nonparametric incremental': {}}
# {'literal': {'I saw this movie . ': ('Ich habe diesen Film gesehen .', 1.0), 'I saw that movie . ': ('Ich habe diesen Film gesehen .', 1.0), 'What is this ? ': ('Was ist das ?', 1.0), 'What is that ? ': ('Was ist das ?', 1.0)}, 'incremental pragmatic': {'I saw this movie . ': ('Ich habe diesen Film gesehen .', 1.0), 'I saw that movie . ': ('Den Film habe ich gesehen .', 1.0), 'What is this ? ': ('Was sind das ?', 1.0), 'What is that ? ': ('Was ist das ?', 1.0)}, 'global pragmatic': {}, 'nonparametric incremental': {}}
