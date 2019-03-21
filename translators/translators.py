# an api giving functions of type ([word] -> [word] -> dist word) and ([word] -> [word] -> dist [word])

import re
import time
import copy
import itertools
import math
import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.autograd import Variable
# from torchvision import transforms
from utils.build_vocab import Vocabulary
# from train.image_captioning.char_model import EncoderCNN, DecoderRNN
# from PIL import Image
from utils.config import *
from utils.helper_functions import softmax,byte_pair_encoding,byte_pair_decoding
from tqdm import tqdm
# from interactive import world_to_utterance as translate_world_to_utterance
# import torch
# import os
# import argparse
# import numpy as np

# from train.helpers import *
# from train.char_model import *
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from utils.build_vocab import Vocabulary
# from train.image_captioning.char_model import EncoderCNN, DecoderRNN
from PIL import Image
import torch
from utils.config import *
from utils.numpy_functions import softmax
from utils.sample import to_var,load_image,load_image_from_path



class Coalgebra:

	def __init__(self,path,source,target,bpe_code_path=None):
		
		# from interactive import Args
		from collections import namedtuple
		import numpy as np
		import sys
		import scipy.misc

		import torch
		from torch.autograd import Variable

		from fairseq import data, options, tasks, tokenizer, utils
		# from fairseq.sequence_generator import SequenceGenerator
		from utils.seqgen import SequenceGenerator

		Batch = namedtuple('Batch', 'srcs tokens lengths')
		Translation = namedtuple('Translation', 'src_str hypos alignments')

		self.bpe_code_path=path+bpe_code_path
		# print("bpe code 1",self.bpe_code_path)

		class Args:
			def __init__(self):
				pass

		args = Args()
		args.sampling_topk=-1
		args.cpu=False
		args.gen_subset='test'
		args.log_format=None
		args.log_interval=1000
		args.max_len_a=0
		args.max_len_b=200
		args.max_source_positions=1024
		args.max_target_positions=1024
		args.min_len = 1
		args.model_overrides={}
		args.no_progress_bar=False
		args.num_shards=1
		args.prefix_size=0
		args.quiet=False
		args.raw_text=False
		args.replace_unk=None
		args.sampling_temperature=1
		args.score_reference=False
		args.seed=1
		args.shard_id=0
		args.skip_invalid_size_inputs_valid_test=False
		args.lenpen=1
		args.unkpen=0
		args.unnormalized=False
		args.no_early_stop=False
		args.fp16=False
		args.no_beamable_mm=False
		args.data=path
		args.path=path+'/model.pt'
		args.target_lang=target
		args.source_lang=source

		args.beam=1
		args.buffer_size=0
		args.max_tokens=None
		args.max_sentences=None
		args.nbest=1
		args.sampling=False
		args.remove_bpe=True
		args.task='translation'
		args.left_pad_source=True
		args.left_pad_target=False

		self.args=args

		# self.utterance_to_world=None

		self.cache = {}

		if args.buffer_size < 1:
			args.buffer_size = 1
		if args.max_tokens is None and args.max_sentences is None:
			args.max_sentences = 1

		assert not args.sampling or args.nbest == args.beam, \
			'--sampling requires --nbest to be equal to --beam'
		assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
			'--max-sentences/--batch-size cannot be larger than --buffer-size'

		# print(args)

		use_cuda = torch.cuda.is_available() and not args.cpu

		# Setup task, e.g., translation
		task = tasks.setup_task(args)

		# Load ensemble
		# print('| loading model(s) from {}'.format(args.path))
		model_paths = args.path.split(':')
		self.models, self.model_args = utils.load_ensemble_for_inference(model_paths, task)

		# Set dictionaries
		self.src_dict = task.source_dictionary
		self.tgt_dict = task.target_dictionary

		# print(tgt_dict)
		u_support = [self.tgt_dict[x] for x in range(len(self.tgt_dict))]
		# list(self.tgt_dict.values())
		mappings = list_to_index_iso(u_support)
		self.idx2seg=mappings["leftward"]
		self.seg2idx=mappings["rightward"]

		self.path=path
		self.seg_type='word'
		self.u_support=u_support

		# Optimize ensemble for generation
		for model in self.models:
			model.make_generation_fast_(beamable_mm_beam_size=None if args.no_beamable_mm else args.beam)
			if args.fp16:
				model.half()

		# Initialize generator
		self.translator = SequenceGenerator(
			self.models, self.tgt_dict, beam_size=args.beam, stop_early=(not args.no_early_stop),
			normalize_scores=(not args.unnormalized), len_penalty=args.lenpen,
			unk_penalty=args.unkpen, sampling=args.sampling, sampling_topk=args.sampling_topk,
			minlen=args.min_len,
		)

		if use_cuda:
			translator.cuda()

		# Load alignment dictionary for unknown word replacement
		# (None if no unknown word replacement, empty if no path to align dictionary)
		self.align_dict = utils.load_align_dict(args.replace_unk)

	def empty_cache(self):
		self.cache={}

	def memoize_forward(f):
		def helper(self,sequence,source_sentence,debug=False):
	

			# print("source sent",source_sentence)
			# world_prior_list = np.ndarray.tolist(np.ndarray.flatten(state.world_priors))
			hashable_args = (tuple(sequence),tuple(source_sentence))
			if hashable_args not in self.cache:
				# print("MEM\n\n")
				self.cache[hashable_args] = f(self,sequence=sequence,source_sentence=source_sentence,debug=debug)
			# else: 
			# 	print("LOOKING UP")
			# 	print("cache keys",list(self.cache))
			return self.cache[hashable_args]
		return helper

	@memoize_forward
	def forward(self,sequence,source_sentence,debug=False):

		# print("CALLING COALGEBRA",source_sentence)
		# source_sentence = re.sub("@@ ","",source_sentence).lower()
		# source_sentence = re.sub("@@@","",source_sentence).lower()



		# source_sentence = byte_pair_decoding(source_sentence)
		# source_sentence = byte_pair_encoding(sentence=source_sentence,code_path=self.bpe_code_path)




		# print("DECODED SENTENCE IN COALGEBRA",source_sentence)
		# print("bpe path",self.bpe_code_path)

		# print("ENCODED SENTENCE IN COALGEBRA",source_sentence)


		from interactive import make_batches, buffered_read

		def make_result(src_str, hypos):
			result = Translation(
				src_str='O\t{}'.format(src_str),
				hypos=[],
				alignments=[],
			)

			# Process top predictions
			for hypo in hypos[:min(len(hypos), self.args.nbest)]:
				hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
					hypo_tokens=hypo['tokens'].int().cpu(),
					src_str=src_str,
					alignment=hypo['alignment'].int().cpu(),
					align_dict=selfalign_dict,
					tgt_dict=self.tgt_dict,
					remove_bpe=self.args.remove_bpe,
				)
				result.hypos.append('H\t{}\t{}'.format(hypo['score'], hypo_str))
				result.alignments.append('A\t{}'.format(' '.join(map(lambda x: str(utils.item(x)), alignment))))
			return result

		def process_batch(batch):
			tokens = batch.tokens
			lengths = batch.lengths

			if use_cuda:
				tokens = tokens.cuda()
				lengths = lengths.cuda()

			translations = self.translator.generate(
				Variable(tokens),
				Variable(lengths),
				maxlen=int(self.args.max_len_a * tokens.size(1) + self.args.max_len_b),
			)

			# print("translations",translations)
			# print(batch.srcs[0])

			return [make_result(batch.srcs[i], t) for i, t in enumerate(translations)]

		# print("SENTENCES",self.sentences)
		batch, batch_indices = next(make_batches([source_sentence], self.args, self.src_dict, self.models[0].max_positions()))
		# print("batch shape",len(batch),len(batch[0]))
		translations = self.translator.generate(
				Variable(batch.tokens),
				Variable(batch.lengths),
				maxlen=int(self.args.max_len_a * batch.tokens.size(1) + self.args.max_len_b),
				prefix_tokens=sequence,
			)
		# print("translations",translations)
		# print(translations[:5],"translations")
		probs, support = translations, [self.tgt_dict[x] for x in range(translations.shape[0])]
		
		# probs = list(probs)
		# support = list(support)
		# for seg in source_sentence.split():
		# 	if seg in support:
		# 		ind = support.index(seg)
		# 		del support[ind]
		# 		del probs[ind]
		probs, support = list(zip(*sorted(list(zip(probs,support)),key=lambda x : x[0],reverse=True)))

		return probs, support

	def likelihood(self,sequence,source_sentence,target,debug=False):

		probs,support = self.forward(sequence=sequence,source_sentence=source_sentence)
		probs = np.asarray(probs)
		# print(target,"target word")
		try: out = probs[support.index(target)]
		except: 
			support = np.asarray(support)
			print("target word failed:",target)
			print("sequence:",sequence)
			print("source sentence:",source_sentence)
			# print(support[np.argsort(-probs)][:5])
			print(sorted(support)[5000:5500])
			# raise Exception
			out = np.log(1.0)
		return out

class Factor_To_Character:

	def __init__(self,word_model):
		self.word_model = word_model
		self.seg_type = "char"
		self.cache = {}
		self.bpe_code_path=word_model.bpe_code_path
	
	def empty_cache(self):
		self.cache={}
		self.word_model.empty_cache()

	def memoize_forward(f):
		def helper(self,sequence,source_sentence,debug=False):
	
			# world_prior_list = np.ndarray.tolist(np.ndarray.flatten(state.world_priors))
			hashable_args = (tuple(sequence),tuple(source_sentence))
			if hashable_args not in self.cache:
				# print("MEM\n\n")
				self.cache[hashable_args] = f(self,sequence=sequence,source_sentence=source_sentence)
			return self.cache[hashable_args]
		return helper

	@memoize_forward
	def forward(self,sequence,source_sentence,debug=False):
		if sequence!=[]:
			ends_with_space = (sequence[-1]==' ')
		else: ends_with_space = False
		sequence = "".join(sequence).split()
		word_sequence = sequence[:-1]
		if ends_with_space: word_sequence = sequence
		
		char_remainder = sequence[-1:]
		if char_remainder != []: char_remainder = char_remainder[0]
		else: char_remainder = ""
		# if ends_with_space: char_remainder = ' '

		# print(sequence,source_sentence)
		# raise Exception
		word_probs,word_support = self.word_model.forward(sequence=word_sequence,source_sentence=source_sentence)
		if ends_with_space: char_remainder = ""
		# print("char remainder",char_remainder)
		# print(word_sequence,"word_sequence")
		# print(word_support[np.asarray(np.argmax(word_probs))])
		# print("ends_with_space",ends_with_space,"ends_with_space")
			# ,len(char_remainder),["abc"][:len(char_remainder)],["abc"][:len(char_remainder)]==char_remainder)
		
		# print(word_probs)
		# print(word_sequence)

		# pairs = []
		# for x,y in zip(word_support,word_probs):
		# 	cond = (x[:len(char_remainder)]==char_remainder)
		# 	print("x:",x[:len(char_remainder)])
		# 	print("char rem:",char_remainder)
		# 	print(cond)
		# 	if cond:
		# 		pairs.append((x,y))

		# word_dict = dict(pairs)
		toc = time.time()
		word_dict = dict([(x,y) for (x,y) in zip(word_support,word_probs) if (x[:len(char_remainder)]==char_remainder)])
		# print(word_dict)
		# raise Exception
		# print(word_dict)

		char_dist = np.zeros(len(sym_set))

		# how far along in the word to look
		index = len(char_remainder)


		for i in range(len(sym_set)):
			for word in word_dict:
				# print("word",word,word[0],sym_set[i])
				# if word==stop_token["word"]:

					# print("BLAH BLAH")

				if word==stop_token["word"] or sym_set[i]==stop_token:

					if sym_set[i]==stop_token["char"]:
						# print("TRUEEUEUE\n\n\n")
					# print("stop token is word")
						char_dist[i]+=np.exp(word_dict[word])

				elif len(word)>len(char_remainder):

					if sym_set[i]==word[index]:
						char_dist[i]+=np.exp(word_dict[word])
					

					elif (word[index] not in set(sym_set)) and sym_set[i]=='&':
						char_dist[i]+=np.exp(word_dict[word])

				else:
					if sym_set[i]==" ":
						char_dist[i]+=np.exp(word_dict[word])

					else: continue



		# print(char_dist[i])
		# smooth
		# print("CHAR DIST UNNORMED UNSMOOTHED",char_dist[sym_set.index('$')])
		tic = time.time()
		# print("TIME (in factor to char):",tic-toc)
		char_dist = char_dist+1e-10
		char_probs = np.log(char_dist) - scipy.special.logsumexp(np.log(char_dist))
		char_probs, support = list(zip(*sorted(list(zip(char_probs,sym_set)),key=lambda x : x[0],reverse=True)))
		# print("NORMED AND SMOOTHED",char_probs[sym_set.index('$')])

		return char_probs, support

	def likelihood(self,sequence,source_sentence,target,debug=False):

		probs,support = self.forward(sequence=sequence,source_sentence=source_sentence)
		# print(target,"target word")
		try: out = probs[support.index(target)]
		except: 
			print("target word failed:",target)
			raise Exception
		return out


# useful for retrieving standard RSA by have an L0 defined as: const blah 
class Constant:

	# change_direction specifies whether the unfolded_model goes in the same direction as the output model or the other
	def __init__(self,support,unfolded_model,prior_ps=None,change_direction=True):
		
		self.seg_type="sentence"
		# self.support = [byte_pair_encoding(sentence=s,code_path=unfolded_model.bpe_code_path) for s in support]

		self.support = support

		# support
		# if not prior_ps is None:
		self.prior_ps=prior_ps
		self.unfolded_model = unfolded_model
		self.change_direction=change_direction
		# self.max_sentence_length = max_sentence_length[self.underlying_model.seg_type]
		self.cache = {}
		self.bpe_code_path=unfolded_model.bpe_code_path

	def empty_cache(self):
		self.cache={}
		self.unfolded_model.empty_cache()

	def memoize_forward(f):
		def helper(self,sequence,source_sentence,debug=False):
	
			# world_prior_list = np.ndarray.tolist(np.ndarray.flatten(state.world_priors))
			hashable_args = (tuple(sequence),tuple(source_sentence))
			if hashable_args not in self.cache:
				# print("MEM\n\n")
				self.cache[hashable_args] = f(self,sequence=sequence,source_sentence=source_sentence,debug=debug)
			return self.cache[hashable_args]
		return helper
		
	@memoize_forward
	def forward(self,source_sentence,sequence,debug=False):

		support_size = len(self.support)

		if self.prior_ps is None:
			self.prior_ps = np.log(np.zeros(len(self.support))+(1/support_size))

		scores = np.zeros(support_size)

		# print("STARTING")
		for i,sent in enumerate(self.support):
			if debug:
				pass
				# print("CONSTANT")
			if not self.change_direction:
				if debug:
					pass
					# print("CONSTANT")
					# print("source sentence",source_sentence)
					# print("sequence",sequence)
					# print("target",sent)
				# print("source_sentence",source_sentence)
				score = self.unfolded_model.likelihood(source_sentence=source_sentence,sequence=sequence,target=sent)
			else: 
				if debug:
					pass
					print("CONSTANT")
					print("source sentence",sent)
					print("sequence",sequence)
					print("target",source_sentence)
				# print("THING",source_sentence)
				score = self.unfolded_model.likelihood(source_sentence=sent,sequence=[],target=source_sentence)
				if debug:print("score",score)
				# print("score",source_sentence)
			# print("score",score,sent,"source sent:",source_sentence)
			scores[i]=score
		# print("STOPPING")

		# print(np.exp(scores)/np.sum(np.exp(scores)))
		unnormed_posterior_ps = scores+self.prior_ps
		# print("POST",unnormed_posterior_ps, source_sentence,sequence)
		norm = scipy.special.logsumexp(unnormed_posterior_ps)
		normed_posterior_ps = unnormed_posterior_ps - norm
		# print("finished")
		# print("results\n\n",normed_posterior_ps)
		return normed_posterior_ps,self.support

	def likelihood(self,sequence,source_sentence,target,debug=False):


		probs,support = self.forward(sequence=sequence,source_sentence=source_sentence,debug=debug)
		try: out = probs[support.index(target)]
		except: 
			print("target word failed:",target)
			print(support[:10])
			raise Exception
			# out = np.log(1.0)
		return out

		# list(zip(normed_posterior_ps,self.support))

class Compose:

	def __init__(self,rightward_model,leftward_model,unfolded_rightward_model=None):

		self.rightward_model = rightward_model
		if unfolded_rightward_model is None:
			unfolded_rightward_model = Anamorphism(underlying_model=self.rightward_model)
		self.unfolded_rightward_model=unfolded_rightward_model
		self.leftward_model = leftward_model
		self.seg_type = leftward_model.seg_type
		self.cache={}
		try:self.bpe_code_path=rightward_model.bpe_code_path
		except: self.bpe_code_path=unfolded_rightward_model.underlying_model.bpe_code_path
	
	def memoize_forward(f):
		def helper(self,sequence,source_sentence,debug=False,):
	
			hashable_args = (tuple(sequence),tuple(source_sentence),debug)
			if hashable_args not in self.cache:
				# print("MEM\n\n")
				self.cache[hashable_args] = f(self,sequence=sequence,source_sentence=source_sentence,debug=debug,)
			return self.cache[hashable_args]
		return helper
		
	@memoize_forward
	def forward(self,source_sentence,sequence,debug=False,):


		inter_probs,inter_support = self.unfolded_rightward_model.forward(source_sentence=source_sentence,sequence=[],debug=debug)

		# print("PROBS SUPP",list(zip(inter_probs,inter_support)))

		target_lang_input = list(zip(*(sorted(list(zip(inter_support,inter_probs)),key = lambda x : x[0],reverse=True))))[0][0]
		target_lang_input = re.sub("@@ ","",target_lang_input)
		target_lang_input+="."
		# target_sent = re.sub("\.","",target_sent)

		# target_lang_input = byte_pair_encoding(sentence=target_lang_input,code_path=self.leftward_model.bpe_code_path)


		# print("target_lang_input",target_lang_input)

		final_probs,final_support = self.leftward_model.forward(source_sentence=target_lang_input,sequence=sequence,debug=debug)
		
		if debug:
			pass
			# print("target support",final_support)

		# if many_to_one:

		# 	print("support", final_support)

		# 	new_final_support = []
		# 	new_final_probs = []

		# 	back_translations = [self.unfolded_rightward_model.forward(source_sentence=s,sequence=[]) for s in final_support]

		# 	for i, s in enumerate(final_support):

		# 		p,bt = self.unfolded_rightward_model.forward(source_sentence=s,sequence=[]) for s in final_support

		# 		if bt==target_lang_input:

		# 			new_final_support.append(final_support[i])
		# 			new_final_probs.append(final_probs[i])

		# 	return new_final_probs, new_final_support

		# print("final_support",final_support[:2])
		return final_probs,final_support

	def likelihood(self,sequence,source_sentence,target,debug=False):


		probs,support = self.forward(sequence=sequence,source_sentence=source_sentence)
		try: out = probs[support.index(target)]
		except: 
			print("target word failed:",target)
			raise Exception
		return out

class Anamorphism:

	def __init__(self,underlying_model,beam_width=1,diverse=False,stop_on=None):
		
		self.seg_type="sentence"
		self.underlying_model = underlying_model
		self.max_sentence_length = max_sentence_length[self.underlying_model.seg_type]
		self.cache = {}
		self.beam_width=beam_width
		self.diverse = diverse
		self.bpe_code_path=underlying_model.bpe_code_path
		self.stop_on = stop_on

	def empty_cache(self):
		self.cache={}
		self.underlying_model.empty_cache()

	def memoize_forward(f):
		def helper(self,sequence,source_sentence,debug=False):
	
			hashable_args = (tuple(sequence),tuple(source_sentence),debug)
			if hashable_args not in self.cache:
				# print("MEM\n\n")
				# print(self.cache)
				self.cache[hashable_args] = f(self,sequence=sequence,source_sentence=source_sentence,debug=debug)
			return self.cache[hashable_args]
		return helper
		
	@memoize_forward
	def forward(self, source_sentence, sequence,debug=False):
		# print("DEBUG?",debug,"STOP ON",stop_on)

		# log = open('log','w')
		# source_words = byte_pair_encoding(sentence=byte_pair_decoding(source_sentence),code_path=self.bpe_code_path).split()
		# print("source words",source_words)

		decay_rate=0
		cut_rate=1

		beam_width=self.beam_width
		# print("BEAM SEARCH",beam_width)

		greedy = beam_width==1
		if greedy and debug:
			print("GREEDY SEARCH")

		sent_prob = [(sequence,0.0)]


		final_sentences=[]
		for step in tqdm(range(1,self.max_sentence_length)):

			
			new_sent_prob = []
			for sent,old_prob in sent_prob:

				sequence = sent
				# print(sequence,"SEQ")
				dist,support = self.underlying_model.forward(sequence=sequence,source_sentence=source_sentence,debug=debug)
				dist,support = dist[:],support[:]
				# print(dist,support)
				# raise Exception
				# print(dist,support)
				# raise Exception
				for seg,prob in enumerate(np.squeeze(dist)):

					# cond1 = support[seg] not in source_words
					# cond2 = support[seg][0]!=support[seg][0].lower()
					# cond3 = step>1
					# print(sent)
					# cond4 = len(sent)>0  and sent[-1][-1]=='@'
					if True:
					# if (cond2 and cond3) or (cond4 or cond1):
					# if (support[seg] not in source_words) or (support[seg][0]!=support[seg][0].lower() and step>1) :
						new_sentence = copy.deepcopy(sent)
						new_sentence += [support[seg]]
						new_prob = (prob*(1/math.pow(step,decay_rate)))+old_prob
						new_sent_prob.append((new_sentence,new_prob))
						# if seg=='The':
							# print(seg,source_words,seg in source_words)
							# raise Exception
						# print("tick")

					if False:
						prior = 1
						self.underlying_model.unfolded_leftward_model.support=prior

			sent_prob = sorted(new_sent_prob,key=lambda x:x[-1],reverse=True)

			if (step-1)%cut_rate == 0:
				# cut down to size
				# if False:
				if self.diverse and step>2:

					index = -2
					# if step==1: index = -1
					# elif step>1: index = -2

					# print("sent_prob", sent_prob[-10:],len(sent_prob))
					new_sent_prob = []
					penultimate_sequence = []
					for sent,prob in sent_prob:

						if sent[:index+1] not in penultimate_sequence:
							new_sent_prob.append((sent,prob))

							penultimate_sequence.append(sent[:index+1])

					# print("penultimate_sequence",penultimate_sequence[:10])
					print(list(set([sent[0] for (sent,prob) in sent_prob])))
					# raise Exception
					sent_prob = new_sent_prob
					# print("new sent_prob",sent_prob[:10])

					#identify last word variants: remove all but max

				sent_prob = sent_prob[:beam_width]
				new_sent_prob = []
				for sent,prob in sent_prob:
					# print(prob)
					st = stop_token[self.underlying_model.seg_type]
					# sent[-1] in ['.','?','!'] or
					if sent[-1]==st or (len(sent)>1 and sent[-2] in ['.','?','!','<eos>']) or sent[-1]=="<Lua heritage>" or sent[-1]==self.stop_on or sent[-1]=="&":
						# if debug: print("\n\nENDING TOKEN\n\n",sent[-1])
						final_sentence = copy.deepcopy(sent)
						final_sentences.append((final_sentence,prob))
					else: 
						new_tuple = copy.deepcopy((sent,prob))
						new_sent_prob.append(new_tuple)


				sent_prob = new_sent_prob

				if len(final_sentences)==beam_width:
					# print("LEN FINAL SENTENCES = beam width")
					break

			# if debug and sent_prob!=[]: print(sent_prob,"SELECTION")


			# try:
			# 	for x in sent_prob:
			# 		print("sentence")
			# 		if self.underlying_model.seg_type=='word':
			# 			print(" ".join(x[0]),x[1][step],x[2])
			# 		else:print("".join(x[0])[1:])
			# except: pass

		probs=[]
		support=[]
		

		if final_sentences == []: final_sentences = sent_prob
		for sent,prob in final_sentences:
			probs.append(prob)
			if self.underlying_model.seg_type=='char':
				support.append("".join(sent[:-1]))
				# if support[-1][-1] not in 
			else: support.append(" ".join(sent[:-1]))


		# probs = np.array(probs)
		# return probs,support
		# normed_probs = probs - scipy.misc.logsumexp(probs)

		# raise Exception
		probs,support = list(zip(*sorted(zip(probs,support),key=lambda x : x[0],reverse=True)))
		probs,support = probs[:beam_width],support[:beam_width]
		probs = np.array(list(probs))
		support = list(support)
		# assert (probs.shape[0] == beam_width)
		# print("CHECKS")
		# print("len probs",probs)
		# print("len support",support)
		# return probs,support
		return (probs-scipy.misc.logsumexp(probs)),support

	def likelihood(self,sequence,source_sentence,target,debug=False):
		if self.underlying_model.seg_type=="word":
			new_target = target.split()
		else: new_target = target
		likelihood = {}
		new_sequence = sequence[:]
		for i in range(0,len(new_target)):
			seg = new_target[i]
			likelihood[i] = self.underlying_model.likelihood(sequence=new_sequence, source_sentence=source_sentence,target=seg,debug=debug)
			new_sequence+=[new_target[i]]

		
		return np.sum([p for (l,p) in likelihood.items()])


