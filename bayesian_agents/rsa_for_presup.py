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

			total=[]
			# print("call")
			# print(world.target,world.qud)
			for w in self.quds[world.qud](world.target):
				# print(w,"w")
				new_world = copy.deepcopy(world)
				new_world.target=w
				# print(self.initial_speakers)
				# print(world.speaker)
				probs,support = self.initial_speakers[world.speaker].world_to_utterance(state=state,world=new_world)
				total.append(probs)

			probs = scipy.misc.logsumexp(np.array(total),axis=0)
			probs -= scipy.misc.logsumexp(probs,axis=0)
			# print("s0",probs)
			return probs,support
			
		else: 

			# self.initial_speakers[world.speaker].world_to_utterance(state=state,world=world)
			# if self.speaker_prior: 
			prior,support = self.lang_mods[world.language_model].world_to_utterance(state=state,world=world)
			# print("result",prior)
				# prior,support = uniform_vector(len(sym_set)),sym_set
			# else: 
			# 	prior,support = self.speaker(state,world,depth=0)

		if depth==1:


			state.hyperprior = self.initial_speakers[world.speaker].hyperprior_support[world.hyperprior]
			# print("l0 prior", state.hyperprior, world.hyperprior)

			scores = np.zeros((prior.shape[0]))
			for k in range(prior.shape[0]):
				out = self.listener(state=state,utterance=k,depth=depth-1)

				if state.style_transfer and self.depth==1:
					
					scores[k]=(-kl(np.exp(state.world_posteriors[state.timestep]),np.exp(out)))
				
				else: 
					# print(self.quds)
					# print(world.qud)
					# print(self.quds[world.qud])
					# print(out.shape)
					score = out[world.target,0,world.speaker,0,0,world.qud]
					# score =  scipy.misc.logsumexp([ out[w,0,world.speaker,0,0,0] for w in self.quds[world.qud](world.target)])
					# print(score,state.hyperprior,"score")
					scores[k]=score

			# print(scores)
			scores = np.asarray(scores)

			scores = scores*(self.initial_speakers[world.speaker].rationality_support[world.rationality])
			# scores = np.where(np.isinf(scores),np.log(np.zeros_like(scores)+1e-20),scores)
			# update prior to posterior
			# print(scores+prior,prior,scipy.misc.logsumexp(scores + prior))
			posterior = (scores + prior) - scipy.misc.logsumexp(scores + prior)
			# print("post",posterior)

			return posterior,support

		elif depth>1:

			# prior,support = self.speaker(state,world,depth=depth-1)

			scores = np.zeros((prior.shape[0]))
			# print("shape 1",scores.shape)
			for k in range(prior.shape[0]):

				out = self.listener(state=state,utterance=k,depth=depth-1)
				if state.style_transfer and self.depth==2: 

					scores[k]=(-kl(np.exp(state.world_posteriors[state.timestep]),np.exp(out)))
				else: 
					score = out[world.target,world.rationality,world.speaker,world.hyperprior,world.language_model,world.qud]
					# print(score,"score")
					scores[k]=score
				# scores.append(-kl([0.499,0.499,0.002],np.exp(out[:,world.rationality,world.speaker])))
			scores = np.asarray(scores)*state.s2_rat

			# print("SHAPES",prior.shape)
			# print("SCORES",np.array(scores).shape)
			# print("prior",prior.shape)
			# scores*=10
			# rationality not present at s2
			# update prior to posterior
			post = scores + prior
			normed_post = post - scipy.misc.logsumexp(post)

			# posterior_probs = (scores + prior) - scipy.misc.logsumexp(scores + prior)
			# print("PSOT SHAEP",posterior_probs.shape)
			return normed_post,support

	@memoize_listener
	def listener(self,state,utterance,depth):

		if depth==0:

			# print("state hyperprior",state.hyperprior.shape)


			# base case listener is inferred by bayes rule from neural s0, given the state's current prior on images
			# world_prior = state.hyperprior
			marginal_speaker_qud_prior = state.world_priors[state.timestep]
			marginal_speaker_qud_prior = scipy.misc.logsumexp(marginal_speaker_qud_prior,axis=0)
			marginal_speaker_qud_prior = scipy.misc.logsumexp(marginal_speaker_qud_prior,axis=0)
			marginal_speaker_qud_prior = scipy.misc.logsumexp(marginal_speaker_qud_prior,axis=1)
			marginal_speaker_qud_prior = scipy.misc.logsumexp(marginal_speaker_qud_prior,axis=1)
			# marginal_speaker_qud_prior = scipy.misc.logsumexp(marginal_speaker_qud_prior,axis=-1)

			# print("shape 1",marginal_speaker_qud_prior.shape)
			world_prior = np.log(np.multiply.outer(np.exp(state.world_priors[state.timestep-1]), np.exp(marginal_speaker_qud_prior)))
			# world_prior = np.log(np.multiply.outer(np.exp(state.hyperprior), np.exp(marginal_speaker_qud_prior)))
			# print(world_prior.shape,"shape")
			# print(world_prior,"world_prior")
			# world_prior = np.expand_dims(world_prior,-1)

			scores = np.zeros((world_prior.shape))
			# print(scores,scores.shape,"Scores")
			# raise Exception

			# print("scores and prior shape",scores.shape,world_prior.shape)
			for i,j,k in itertools.product(range(world_prior.shape[0]), range(world_prior.shape[1]),range(world_prior.shape[2]) ):

				n_tuple = [i,j,0,0,0,k]
				world = RSA_World( 	target=n_tuple[state.dim["image"]],
									rationality=n_tuple[state.dim["rationality"]],
									speaker=n_tuple[state.dim["speaker"]],
									hyperprior=n_tuple[state.dim["hyperprior"]],
									language_model=n_tuple[state.dim["language_model"]],
									qud=n_tuple[state.dim["qud"]])

				# print("created world",world)
				probs,support = self.speaker(state=state,world=world,depth=depth)
				# print(probs)

				# print("score",world,state.hyperprior,probs[utterance])
				# print(probs[utterance],world,"score")
				scores[i,j,k]=probs[utterance]

			# print("shapes",scores.shape,world_prior.shape)
			scores = scores*state.listener_rationality
			# print(scores,"scores")
			# print(scores+world_prior,"scores + prior")
			# print(scipy.misc.logsumexp(scores + world_prior),"norm")
			world_posterior = (scores + world_prior) - scipy.misc.logsumexp(scores + world_prior)
			world_posterior = np.expand_dims(world_posterior,1)
			world_posterior = np.expand_dims(world_posterior,-2)
			world_posterior = np.expand_dims(world_posterior,-2)
			# print(np.exp(world_posterior),"normed post")
			# if utterance==0:
				# print("stuff")
				# print(world_posterior.shape)
				# print(world_posterior[0,0,0,0,0],world_prior[0,0,0,0,0],state.hyperprior)
				# print(world_posterior[1,0,0,0,0],world_prior[1,0,0,0,0],state.hyperprior)
				# print(world_posterior[2,0,0,0,0],world_prior[2,0,0,0,0],state.hyperprior)
			# print(world_posterior[1,0,0,0,0])
			return world_posterior

		if depth>0:
			# base case listener is inferred by bayes rule from neural s0, given the state's current prior on images
			world_prior = state.world_priors[state.timestep-1]
			scores = np.zeros((world_prior.shape))
			# print(len(list(itertools.product(*[list(range(x)) for x in world_prior.shape]))))
			for n_tuple in itertools.product(*[list(range(x)) for x in world_prior.shape]):
				# print("ntuple",n_tuple)

				world = RSA_World( 	target=n_tuple[state.dim["image"]],
									rationality=n_tuple[state.dim["rationality"]],
									speaker=n_tuple[state.dim["speaker"]],
									hyperprior=n_tuple[state.dim["hyperprior"]],
									language_model=n_tuple[state.dim["language_model"]],
									qud=n_tuple[state.dim["qud"]])
				probs,support = self.speaker(state=state,world=world,depth=depth)

				# print("score",world,probs[utterance])

				scores[n_tuple]=probs[utterance]


			scores = scores*state.listener_rationality
			world_posterior = (scores + world_prior) - scipy.misc.logsumexp(scores + world_prior)
			return world_posterior

	def world_to_utterance(self,state,world):
		return self.speaker(depth=self.depth,state=state,world=world)

	# THIS ONLY CALCULATES THE HEARD UTTERANCE AT SPEAKER LEVEL NOT ALL, so is infinitely more efficient
	def utterance_to_world(self,state,utterance):

		return self.listener(depth=self.depth,state=state,utterance=self.iso['rightward'][utterance])

	def speaker_likelihood(self,state,world,utterance):

		# print(self.iso['rightward'][utterance])
		# print(self.world_to_utterance(state=state,world=world))
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






