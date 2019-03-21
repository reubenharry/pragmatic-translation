print("PRESUP CURRENTLY DISABLED")
import time
import itertools
import scipy
import scipy.stats
import numpy as np
import math
import copy
from PIL import Image as PIL_Image
from keras.preprocessing import image
from keras.models import load_model
from utils.config import *
from bayesian_agents.rsaWorld import RSA_World
from utils.helper_functions import softmax, kl, uniform_vector

class RSA:

	def __init__(
		self,
		models,
		urls,
		depth,
		lang_mods,
		quds,
		tf=False,
		speaker_prior=None,
		):

			self.tf=tf
			self.depth=depth



			try: self.iso=list_to_index_iso(models[0].u_support)
			except: pass
			#caches for memoization
			self._speaker_cache = {}
			self._listener_cache = {}
			self._speaker_prior_cache = {}

			self.seg_type = models[0].seg_type


			# mappings = list_to_index_iso(u_support)
			# self.idx2seg=mappings["leftward"]
			# self.seg2idx=mappings["rightward"]

			# self.initialize_speakers(models)
			self.initial_speakers=models
			self.lang_mods=lang_mods
			self.quds=quds
			if speaker_prior: self.speaker_prior=speaker_prior
			else: self.speaker_prior = None
			# [self.initial_speakers[i].set_features(images=urls,tf=False,rationaliti for i in range(len(models))]




	
	# input the base conditional probability distribution p(u|w)
		# in actual fact, I have uncertainty over this distribution
		# there's also a separate language model "speaker_prior"
		# so you need to supply a list of at least one speakers, and one speaker prior
		# i don't actually use the speaker prior presently
	# def initialize_speakers(self,models):

		# if hand_coded:
		# self.initial_speakers=models
			# self.speaker_prior = Hand_Coded_Model()
			# return None
		# self.initial_speakers = [Neural_Model(path=path,
		# 	dictionaries=(self.seg2idx,self.idx2seg)) for path in paths] 
		# self.speaker_prior = Neural_Model(path="lang_mod",
		# 	dictionaries=(self.seg2idx,self.idx2seg))
		# self.initial_speaker.set_features()

		# self.speaker_prior
		
		# self.images=images





		

	def flush_cache(self):

		self._speaker_cache = {}
		self._listener_cache = {}
		self._speaker_prior_cache = {}	

	# memoization is crucial for speed of the RSA, which is recursive: memoization via decorators for speaker and listener
	
	# def memoize_speaker_prior(f):
	# 	def helper(self,state,world):
	
	# 		# world_prior_list = np.ndarray.tolist(np.ndarray.flatten(state.world_priors))
	# 		hashable_args = state,world

	# 		if hashable_args not in self._speaker_cache:
	# 			self._speaker_prior_cache[hashable_args] = f(self,state,world)
	# 		return self._speaker_prior_cache[hashable_args]
	# 	return helper

	def memoize_speaker(f):
		def helper(self,state,world,depth):
	
			# world_prior_list = np.ndarray.tolist(np.ndarray.flatten(state.world_priors))
			hashable_args = state,world,depth

			if hashable_args not in self._speaker_cache:
				self._speaker_cache[hashable_args] = f(self,state,world,depth)
			return self._speaker_cache[hashable_args]
		return helper

	def memoize_listener(f):
		def helper(self,state,utterance,depth):
	
			# world_prior_list = np.ndarray.tolist(np.ndarray.flatten(state.world_priors))
			hashable_args = state,str(utterance),depth

			if hashable_args not in self._listener_cache:
				self._listener_cache[hashable_args] = f(self,state,utterance,depth)

			return self._listener_cache[hashable_args]
		return helper




	# @memoize_speaker_prior
	# def speaker_prior(self,state,world):

	# 	pass


	@memoize_speaker
	def speaker(self,state,world,depth):
		
		# print(depth, "world",world)
		if depth==0:

			
			probs,support = self.initial_speakers[world.speaker].world_to_utterance(state=state,world=world)
			print(probs,"PROBS at 0")
			return probs,support
			
		else: 


			prior,support = self.lang_mods[world.language_model].world_to_utterance(state=state,world=world)

		elif depth==1:

			scores = np.zeros((prior.shape[0]))
			for k in range(prior.shape[0]):
				score = self.listener(state=state,utterance=k,depth=depth-1)
				# score = out[world.target,0,world.speaker,0,0,world.qud]
				scores[k]=score

			scores = np.asarray(scores)
			scores = scores*(self.initial_speakers[world.speaker].rationality_support[world.rationality])
			posterior = (scores + prior) - scipy.misc.logsumexp(scores + prior)
			return posterior,support

		else:
			raise Exception("depth too high")


	@memoize_listener
	def listener(self,state,utterance,depth):

		if depth==0:
			score = self.initial_speakers[world.speaker].speaker_likelihood(state=state,world=world,utterance=utterance)

		else:
			raise Exception("depth too high")

		# world_prior = state.world_priors[state.timestep-1]

		# scores = np.zeros((world_prior.shape))



		# for n_tuple in itertools.product(*[list(range(x)) for x in world_prior.shape]):

		# 	world = RSA_World( 	target=n_tuple[state.dim["image"]],
		# 						rationality=n_tuple[state.dim["rationality"]],
		# 						speaker=n_tuple[state.dim["speaker"]],
		# 						hyperprior=n_tuple[state.dim["hyperprior"]],
		# 						language_model=n_tuple[state.dim["language_model"]],
		# 						qud=n_tuple[state.dim["qud"]])

		# 	probs,support = self.speaker(state=state,world=world,depth=depth)
		# 	# print("TESTS")
		# 	# print(probs)
		# 	# print(utterance)
		# 	scores[n_tuple]=probs[utterance]

		# scores = scores*state.listener_rationality

		# world_posterior = (scores + world_prior) - scipy.misc.logsumexp(scores + world_prior)
		# return world_posterior



	def world_to_utterance(self,state,world,beam_width=None):
		return self.speaker(depth=self.depth,state=state,world=world)

	# THIS ONLY CALCULATES THE HEARD UTTERANCE AT SPEAKER LEVEL NOT ALL, so is infinitely more efficient
	def utterance_to_world(self,state,utterance):
		return self.listener(depth=self.depth,state=state,utterance=self.iso['rightward'][utterance])

	def speaker_likelihood(self,state,world,utterance):

		probs,support = self.world_to_utterance(state=state,world=world)
		return probs[self.iso['rightward'][utterance]]


	# doesn't require s0 to calculate every possible utterance: for RSA of unrolled model, to calculate l_eval
	def lazy_l0(self,state,utterance):
		world_prior = state.world_priors[state.timestep-1]
		scores = np.zeros((world_prior.shape))
		for n_tuple in itertools.product(*[list(range(x)) for x in world_prior.shape]):

			world = RSA_World(target=n_tuple[state.dim["image"]],rationality=n_tuple[state.dim["rationality"]],speaker=n_tuple[state.dim["speaker"]],hyperprior=n_tuple[state.dim["hyperprior"]])

			likelihood = self.initial_speakers[world.speaker].speaker_likelihood(state=state,world=world,utterance=utterance)
			scores[n_tuple] = likelihood
			
		# scores = scores*state.listener_rationality
		world_posterior = (scores + world_prior) - scipy.misc.logsumexp(scores + world_prior)
		return world_posterior






