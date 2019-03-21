import matplotlib
matplotlib.use('Agg')
import re
import json
import os
import requests
import copy
import random
import time
import math
import pickle
import numpy as np
import scipy
import scipy.special
from collections import defaultdict
from utils.config import *
from utils.helper_functions import display,byte_pair_encoding,byte_pair_decoding
from translators.translators import Coalgebra,Coalgebra_Image, Anamorphism, Factor_To_Character, Constant, Compose
from utils.paraphrase import get_paraphrases
from bayesian_agents.bayesian_pragmatics import Pragmatic
from experiment.exp1_data import french_sentence_pairs,german_sentence_pairs

random.seed(9001)

# from utils.collect_sentences import sentences as dev_sentences
captioning_chars = '&^$ abcdefghijklmnopqrstuvwxyz'
iso = list_to_index_iso(list(captioning_chars))


captioner = Coalgebra_Image(path="coco",dictionaries=(iso["rightward"],iso["leftward"]))

# unfolded_captioner = Anamorphism(underlying_model=captioner,beam_width=1)

# probs,support = unfolded_captioner.forward(
# 	source_sentence="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Arriva_T6_nearside.JPG/1200px-Arriva_T6_nearside.JPG",
# 	sequence=list("^the most "))


# probs = unfolded_captioner.likelihood(sequence=[],source_sentence="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Arriva_T6_nearside.JPG/1200px-Arriva_T6_nearside.JPG",target=list("^a red bus"))


source_target = captioner
# target_source = target_source_word
# source_target = source_target_char
# target_source = target_source_char

unfolded_source_target = Anamorphism(underlying_model=source_target,beam_width=1)
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
# annotations = open("/Users/reuben/Downloads/annotations/captions_val2014.json",'r').read()
# annotations = json.loads(annotations)

id_to_cap = pickle.load(open("/Users/reuben/Recursive-RSA/data/id_to_cap.pkl",'rb'))

def strip_letters(num):
	return num[15:-4]

def remove_initial_zeros(num):
	# print("CALLING")
	new_num = num[1:]
	init = num[0]
	# print(new_num,"new_num")
	# print(init,"init")
	if init=='0': return remove_initial_zeros(new_num)
	else: return num

# explicit_distractor_target_source = Constant(support=["sentence"],unfolded_model=unfolded_target_source,change_direction=False)

# img1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/First_Student_IC_school_bus_202076.jpg/220px-First_Student_IC_school_bus_202076.jpg"
# img2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Arriva_T6_nearside.JPG/1200px-Arriva_T6_nearside.JPG"
# img3 = "https://www.gannett-cdn.com/presto/2018/09/24/USAT/de21227f-309f-4f54-bb86-86524420beb9-BUSAP_BUS_DRIVERS_NEEDED.JPG?width=534&height=401&fit=bounds&auto=webp"

explicit_distractor_target_source = Constant(support=[],unfolded_model=unfolded_source_target,change_direction=True)

rightward_pragmatic_explicit = Pragmatic(
	rightward_model=source_target,
	leftward_model=None, 
	unfolded_leftward_model=explicit_distractor_target_source,
	width=len(sym_set),rat=100.0,
	EXTEND=False)

# global_unfolded_rightward_pragmatic_explicit = Pragmatic(
# 	rightward_model=unfolded_source_target,
# 	leftward_model=None, 
# 	unfolded_leftward_model=explicit_distractor_target_source,
# 	width=100,rat=5.0,
# 	EXTEND=False)


unfolded_rightward_pragmatic_explicit = Anamorphism(underlying_model=rightward_pragmatic_explicit,beam_width=2)


# probs,support = unfolded_source_target.forward(
# 	source_sentence=img1,
# 	sequence=list("^"))
# print(display(probs=probs,support=support))
# raise Exception


# probs,support = explicit_distractor_target_source.forward(
# 	source_sentence=list("^yellow"),
# 	sequence=list(""),
# 	debug=False)
# print(display(probs=probs,support=support))
# print("BLAH")



# print(unfolded_rightward_pragmatic_explicit.likelihood(sequence=[],source_sentence=img1,target="^yellow bus"))


# explicit_distractor_target_source.support = [img1,img3]
# unfolded_rightward_pragmatic_explicit.empty_cache()
# print(unfolded_rightward_pragmatic_explicit.likelihood(sequence=[],source_sentence=img1,target="^yellow bus"))


# print(unfolded_source_target.likelihood(sequence=[],source_sentence=img2,target="^yellow"))

path = "/Users/reuben/Documents/captionproject/project/external/coco/images/train2014/"

distractors_to_scores_list = []

target_ids = [file_name for file_name in os.listdir(path)]
# random.shuffle(target_ids)
target_ids = target_ids[:1]

# distractor_ids = [file_name for file_name in os.listdir(path)]
# random.shuffle(distractor_ids)
# distractor_ids = distractor_ids[:10]



for target_id in target_ids:


	target = path+target_id
	# utt = list(annotations[img1])

# distractors = [img2,img3]
	distractors = [target]+[path+file_name for file_name in os.listdir(path)]
	# random.shuffle(distractors)
	distractors = distractors[:50]
	distractors_to_scores = {}

	# utt = list("a white vase")

	for LONG in [True]:
		
		utt = id_to_cap[int(remove_initial_zeros(strip_letters(target_id)))]
		utt = ''.join([x for x in utt.lower() if x in captioning_chars])
		if not LONG: utt=utt[:10]
		print(utt,"CAPTION",target_id)
		
		for distractor in distractors:
			# print("caption",utt)

			# try:

			# sentences = ["The animals run quickly.",distractor]
			# pair = [byte_pair_encoding(sentence=p,code_path=source_target.bpe_code_path) for p in sentences]
			explicit_distractor_target_source.support=[target,distractor]
			# print(pair)

			# sentence = pair[0]
			# probs,support = rightward_pragmatic_explicit.forward(sequence=[],source_sentence=sentence,debug=False)
		# "^white vase with different colored flowers sitting inside of it"
			logprobs = unfolded_rightward_pragmatic_explicit.likelihood(sequence=["^"],source_sentence=target,target=utt,debug=True)
			# print(list(zip([math.exp(p) for p in logprobs],utt)),"prob",distractor)
			# raise Exception
			unfolded_rightward_pragmatic_explicit.empty_cache()

			probs,support = unfolded_source_target.forward(sequence=["^"],source_sentence=distractor)
			uDist = display(probs=probs,support=support)
			probs,support = explicit_distractor_target_source.forward(sequence=["^"],source_sentence=utt)
			imgDist = display(probs=probs,support=support)

			probs,support = unfolded_rightward_pragmatic_explicit.forward(sequence=["^"],source_sentence=target)
			s1Dist = display(probs=probs,support=support)

			# ,target=target)
			# logprobs = unfolded_source_target.likelihood(sequence=[],source_sentence=target,target="^vase with flowers")
			# print(np.exp(logprobs),"logprobs")
			# raise Exception
			distractors_to_scores[distractor]=logprobs,uDist,imgDist,s1Dist
			unfolded_rightward_pragmatic_explicit.empty_cache()

			# except: continue
			# pickle.dump((distractors_to_scores,utt,target), open("data/pickles/distractors.pkl",'wb'))

			unfolded_rightward_pragmatic_explicit.empty_cache()
			explicit_distractor_target_source.empty_cache()
			rightward_pragmatic_explicit.empty_cache()
			unfolded_source_target.empty_cache()
			source_target.empty_cache()
			# except:
			# 	continue
		# print(distractors_to_scores)
		# best_distractor = max(distractors_to_scores.keys(), key=(lambda key: distractors_to_scores[key][0]))
		# print("best_distractor",best_distractor)
		distractors_to_scores_list.append((copy.deepcopy(distractors_to_scores),utt,target))

pickle.dump((distractors_to_scores_list), open("data/pickles/distractors.pkl",'wb'))

# out = open("distractors.txt",'w')
# out.write(json.dumps(distractors_to_scores))
# out.close()
