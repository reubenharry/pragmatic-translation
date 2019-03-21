import re
import numpy as np
import scipy
import scipy.special
from utils.config import *
from translators.translators import Unfold
from utils.helper_functions import display,byte_pair_encoding,byte_pair_decoding



class Pragmatic:

	def __init__(self,rightward_model,leftward_model,unfolded_leftward_model=None, width=2,rat=2.0,EXTEND=True,):
		# assert (rightward_model.seg_type==leftward_model.seg_type)

		self.seg_type = rightward_model.seg_type
		# it's leftward because we want the bpe encoding that the leftward model would yield, since we're looking up from leftward model outputs
		
		try:self.bpe_code_path=leftward_model.bpe_code_path
		except: self.bpe_code_path=self.bpe_code_path=unfolded_leftward_model.bpe_code_path
		if rightward_model.seg_type=="sentence":
			self.underlying_model=rightward_model.underlying_model
		# except: 
		# 	print("RIGHTWARD MODEL HAS NO SEG TYPE")
		# 	self.seg_type=leftward_model.seg.type
		# 	self.bpe_code_path=leftward_model.bpe_code_path

		self.rightward_model = rightward_model
		self.leftward_model = leftward_model

		if unfolded_leftward_model is None:
			unfolded_leftward_model = Unfold(underlying_model=self.leftward_model)
		# else: self.bpe_code_path = unfolded_leftward_model.underlying_model.bpe_code_path
		self.unfolded_leftward_model=unfolded_leftward_model
		self.unfolded_rightward_model = Unfold(underlying_model=self.rightward_model,stop_on=None)
		self.width=width
		self.rat=rat
		self.EXTEND=EXTEND
		self.cache = {}

	def empty_cache(self):
		self.cache={}
		self.unfolded_leftward_model.empty_cache()
		self.unfolded_rightward_model.empty_cache()

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
	def forward(self,sequence,source_sentence,debug=False):

		# print("Starting pragmatics")


		# print("ENCODED SOurCE SENTENCE",source_sentence,self.bpe_code_path)
		# left_coded_source_sentence = byte_pair_decoding(source_sentence)
		# left_coded_source_sentence = byte_pair_encoding(sentence=left_coded_source_sentence,code_path=self.bpe_code_path)
		left_coded_source_sentence = source_sentence
		EXTEND = self.EXTEND

		s0_probs, s0_support = self.rightward_model.forward(sequence=sequence,source_sentence=source_sentence)

		s0_probs, s0_support = np.asarray(s0_probs[:self.width]),s0_support[:self.width]

		# print("S0 complete")

		# if debug:
		# 	print("s0",display(probs=s0_probs,support=s0_support))
			# print("probs and support",sym_set[np.argmax(s0_probs)])
			# print("EXTEND?",EXTEND)
		# unrolled_sents = []
		l0_scores = []
		for i,seg in enumerate(s0_support):

			# if debug: print("SEG",seg)

			if seg!=stop_token[self.seg_type]:
				# _, unrolled_sent = self.unfolded_rightward_model.forward(sequence=sequence+[seg],source_sentence=source_sentence,beam_width=1)
				# unrolled_sent = unrolled_sent[0]
				# if debug:
				# 	print("\nunrolled sent\n",unrolled_sent)
				# # unrolled_sents.append(unrolled_sent)
				if debug:
					print("SEG",seg)
				# 	print("stuff",unrolled_sent.lower(),source_sentence.split())
				# 	p,s = self.leftward_model.forward(sequence=[],source_sentence=unrolled_sent.lower())
				# 	print(s[:200])
				
				if EXTEND:

					if debug:print("STARTING EXTENSION")
					_, unrolled_sent = self.unfolded_rightward_model.forward(sequence=sequence+[seg],source_sentence=source_sentence,debug=debug)
					unrolled_sent = unrolled_sent[0]

					# if True:
						# print("EXTENSION COMPLETE")
						# print("\nunrolled sent\n",unrolled_sent)
					# unrolled_sents.append(unrolled_sent)
						# print("stuff",unrolled_sent.lower(),source_sentence.split())
						# p,s = self.leftward_model.forward(sequence=[],source_sentence=unrolled_sent.lower())
						# print(s[:200])
					score = self.unfolded_leftward_model.likelihood(sequence=[],source_sentence=unrolled_sent,target=left_coded_source_sentence)
					# if debug:
					# 	print("score",score)

				else: 
					# print("SEQUENCE AND WORD",sequence+[seg])
					if self.rightward_model.seg_type=='char': new_seq = "".join(sequence+[seg])
					elif self.rightward_model.seg_type in ['word','sentence']: 
						new_seq = " ".join(sequence+[seg])
						if debug:
							print("BP")
							print("source_sentence",new_seq)
							print("target",source_sentence)
					score = self.unfolded_leftward_model.likelihood(sequence=[],source_sentence=new_seq,target=left_coded_source_sentence,debug=debug)

				# score = self.unfolded_leftward_model.likelihood(sequence=[],source_sentence=unrolled_sent.lower(),target=source_sentence.split())
				# score = self.unfolded_leftward_model.likelihood(sequence=source_sentence.split()[:len(sequence)],source_sentence=s,target=source_sentence.split()[len(sequence):])
				if debug:
				# print(s0_support[i])
				# print("source"," ".join(sequence+[seg]).lower())
				# print("target",source_sentence.split())
					print("score",score)

			else: 
				score = self.unfolded_leftward_model.likelihood(sequence=[],source_sentence=" ".join(sequence+[seg]),target=left_coded_source_sentence)
				# unrolled_sents.append(" ".join(sequence+[seg]))

			if debug:
				print("SCORE, SEG",score,seg)
			l0_scores.append(score)


		# print(unrolled_sents, "unrolled_sents")

		# print("seq",source_sentence.split()[:len(sequence)])
		# print("targ",source_sentence.split()[len(sequence):])
		# for i,s in enumerate(unrolled_sents):
		# 	l0_scores.append(score)

		# print("rat",self.rat)
		l0_scores = self.rat*np.asarray(l0_scores)

		# print(type(s0_probs),type(l0_scores))

		# if debug:print("l0 scores",sym_set[np.argmax(l0_scores)])


		# l0_scores = [(unfolded_de_en_model.likelihood(sequence=[],source_sentence=s,target=sentence.split())) for s in unrolled_sents]
		# l0_scores = np.asarray(l0_scores)
		# print(np.exp(l0_scores))

		unnormed_probs = s0_probs + l0_scores
		normed_probs = unnormed_probs - scipy.special.logsumexp(unnormed_probs)

		# print("MAX",sym_set[np.argmax(normed_probs)])

		return normed_probs,s0_support

	def likelihood(self,sequence,source_sentence,target,debug=False):
		probs,support = self.forward(sequence,source_sentence)

		try: out = probs[support.index(target)]
		except: 
			print("target failed:",target)
			# out = np.log(1.0)
			raise Exception
		return out