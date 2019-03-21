from tqdm import tqdm
import numpy as np
import time
import math
import scipy
import scipy.stats
import copy
from utils.config import *
from bayesian_agents.rsaState import RSA_State
from bayesian_agents.rsaWorld import RSA_World
from utils.image_and_text_utils import max_sentence_length,devectorize_caption,\
	sentence_likelihood,largest_indices,vectorize_caption

###
"""
Recursion schemes for the RSA
"""
###


# stepwise L0
def cata(initial_world_prior,rsa,depth=0,listener_rationality=1.0,start_from=""):

	state = RSA_State(initial_world_prior, listener_rationality=listener_rationality)
	start_from = ['^'] + start_from
	

	
	for timestep in range(1,len(start_from)):
		state.timestep=timestep
		seg = start_from[state.timestep]
		state.context_sentence=start_from[:state.timestep]
		l = rsa.listener(state=state,utterance=rsa.seg2idx[seg],depth=depth)
		state.world_priors[state.timestep]=l

	# marginal_on_images = np.sum(np.sum(np.exp(state.world_priors),axis=state.dim["speaker"]+1),axis=state.dim["rationality"]+1)

	return np.exp(state.world_priors[state.timestep])
	# return marginal_on_images[state.timestep]

###KEY: check that l_eval score of sentence is same thing that greedy or beam gives for generating that sentence
# def l_eval(self,depth=0,which_image=0,speaker_rationality=1.0,speaker=0,listener_rationality=1.0,img_prior=np.log(np.asarray([0.5,0.5])),start_from=""):

# 	self.speaker_rationality=speaker_rationality
# 	self.listener_rationality=listener_rationality
# 	self.image_priors=np.log(np.ones((max_sentence_length+1,self.images.shape[0]))*(1/self.images.shape[0]))
# 	self.image_priors[0]=img_prior
# 	sentence = np.expand_dims(np.expand_dims(vectorize_caption(start_from)[0],0),-1)
# 	self.context_sentence = copy.deepcopy(sentence)

# 	likelihood = {}

# 	for i in range(1,max_sentence_length):
# 		self.i = i
# 		char = np.squeeze(sentence)[i]
# 		s = np.squeeze(self.speaker(img_idx=which_image,depth=depth))
# 		likelihood[i] = s[char]
# 		if index_to_char[char] == stop_token:
# 			break

# 	print(start_from,np.prod([np.exp(p) for (l,p) in likelihood.items()]))


def ana_greedy(rsa,initial_world_prior,speaker_rationality,speaker, target, pass_prior=True,listener_rationality=1.0,depth=0,start_from=[],max_sentence_length=max_sentence_length):


	"""
	speaker_rationality,listener_rationality: 
		
		see speaker and listener code for what they do: control strength of conditioning
	
	depth: 
		
		the number of levels of RSA: depth 0 uses listeral speaker to unroll, depth n uses speaker n to unroll, and listener n to update at each step
	
	start_from:

		a partial caption you start the unrolling from

	img_prior:

		a prior on the world to start with 
	"""


	state = RSA_State(initial_world_prior, listener_rationality=listener_rationality)
	# this RSA passes along a state: see rsa_state

	# for x in range(15):
	# 	if x % 2 == 0:
	# state.world_priors[10:]=np.log(np.array([0.00001,0.9999])).reshape((2,1,1))
	# print(state.world_priors[:4])
	# print("WORLD PRIORS",np.sum(np.sum(np.exp(state.world_priors[:5]),axis=state.dim["speaker"]+1),axis=state.dim["rationality"]+1) )
	# state.image_priors[:]=img_prior

	context_sentence = ['^']+start_from
	state.context_sentence=context_sentence


	world=RSA_World(target=target,rationality=speaker_rationality,speaker=speaker)

	# print()

	probs=[]
	for timestep in tqdm(range(len(start_from)+1,max_sentence_length)):
		state.timestep=timestep
		dist,support = rsa.speaker(state=state,world=world,depth=depth)

		# print("S:",s)	
		# print(s)
		segment = np.argmax(dist)
		# print("s",rsa.idx2seg[segment])
		prob = np.max(dist)
		probs.append(prob)

		if depth>0 and pass_prior:


			l = rsa.listener(state=state,utterance=segment,depth=depth)
			# print("listener complex",np.exp(l),timestep)
			state.world_priors[state.timestep]=l

		state.context_sentence += [rsa.idx2seg[segment]]
		if (rsa.idx2seg[segment] == stop_token[rsa.initial_speakers[0].seg_type]):
			break

	summed_probs = np.sum(np.asarray(probs))

	world_posterior = state.world_priors[:state.timestep+1][:5]

	# print("JOINT",np.squeeze(np.exp(world_posterior)))
	print("MARGINAL IMAGE",np.sum(np.sum(np.exp(world_posterior),axis=state.dim["speaker"]+1),axis=state.dim["rationality"]+1) )
	print("MARGINAL RATIONALITY",np.sum(np.sum(np.exp(world_posterior),axis=state.dim["speaker"]+1),axis=state.dim["image"]+1) )
	print("MARGINAL SPEAKER",np.sum(np.sum(np.exp(world_posterior),axis=state.dim["rationality"]+1),axis=state.dim["image"]+1) )





	return [("".join(state.context_sentence),summed_probs)]

#beam search anamorphism
#But within the n-th order ethno-metapragmatic perspective, this creative indexical effect is the motivated realization, or performable execution, of an already constituted framework of semiotic value.
# def ana_beam(rsa, pass_prior=True, speaker_rationality=1.0, listener_rationality=1.0, beam_width=len(sym_set),cut_rate=1,decay_rate=0.0,beam_decay=0,depth=0,start_from=[],which_image=0,,img_prior=np.log(np.asarray([0.5,0.5]))):
def ana_beam(rsa,initial_world_prior,speaker_rationality,speaker, target, pass_prior=True,listener_rationality=1.0,depth=0,start_from=[],beam_width=len(sym_set),cut_rate=1,decay_rate=0.0,beam_decay=0,max_sentence_length=max_sentence_length):
	"""
	speaker_rationality,listener_rationality: 
		
		see speaker and listener code for what they do: control strength of conditioning
	
	depth: 
		
		the number of levels of RSA: depth 0 uses listeral speaker to unroll, depth n uses speaker n to unroll, and listener n to update at each step
	
	start_from:
		a partial caption you start the unrolling from
	img_prior:
		a prior on the world to start with 
	which_image: 
		which of the images in the prior should be targeted?
	beam width: width beam is cut down to every cut_rate iterations of the unrolling
	cut_rate: how often beam is cut down to beam_width
	beam_decay: amount by which beam_width is lessened after each iteration 
	decay_rate: a multiplier that makes later decisions in the unrolling matter less: 0.0 does no decay. negative decay makes start matter more
	"""

	state = RSA_State(initial_world_prior, listener_rationality=listener_rationality)
	# state.image_priors[:]=img_prior

	context_sentence = ['^']+start_from
	state.context_sentence=context_sentence


	world=RSA_World(target=target,rationality=speaker_rationality,speaker=speaker)
	# state.world_priors[10:]=np.log(np.array([0.9999,0.00001,])).reshape((2,1,1))



	context_sentence = start_from
	state.context_sentence=context_sentence


	sent_worldprior_prob = [(state.context_sentence,state.world_priors,0.0)]

	final_sentences=[]

	toc = time.time()
	for timestep in tqdm(range(len(start_from)+1,max_sentence_length)):
		
		
		state.timestep=timestep

		new_sent_worldprior_prob = []
		for sent,worldpriors,old_prob in sent_worldprior_prob:

			state.world_priors=worldpriors

			if state.timestep > 1:

				state.context_sentence = sent[:-1]
				seg = sent[-1]

				if depth>0 and pass_prior:

					l=rsa.listener(state=state,utterance=rsa.seg2idx[seg],depth=depth)
					state.world_priors[state.timestep-1]=copy.deepcopy(l)		

			state.context_sentence = sent

			# out = rsa.speaker(state=state,img_idx=which_image,depth=depth)
			dist,support = rsa.speaker(state=state,world=world,depth=depth)	
			
			for seg,prob in enumerate(np.squeeze(dist)):

				new_sentence = copy.deepcopy(sent)

				# conditional to deal with captions longer than max sentence length
				# if state.timestep<max_sentence_length+1:
				new_sentence += [rsa.idx2seg[seg]]
				# else: new_sentence = np.expand_dims(np.expand_dims(np.concat([np.squeeze(new_sentence)[:-1],[seg]],axis=0),0),-1)
				
				state.context_sentence = new_sentence

				new_prob = (prob*(1/math.pow(state.timestep,decay_rate)))+old_prob


				# print("beam listener",rsa.word2ord[seg], l)

				new_sent_worldprior_prob.append((new_sentence,worldpriors,new_prob))

		rsa.flush_cache()
		sent_worldprior_prob = sorted(new_sent_worldprior_prob,key=lambda x:x[-1],reverse=True)

		if state.timestep%cut_rate == 0:
			# cut down to size
			sent_worldprior_prob = sent_worldprior_prob[:beam_width]
			new_sent_worldprior_prob = []
			
			for sent,worldprior,prob in sent_worldprior_prob:
				# print("".join(sent),np.exp(prob))
				# print(state.timestep)
				if sent[-1] == stop_token[rsa.initial_speakers[0].seg_type]:
					final_sentence = copy.deepcopy(sent)
					final_sentences.append((final_sentence,prob))
					# print("REMOVED SENTENCE")
				else: 
					new_triple = copy.deepcopy((sent,worldprior,prob))
					new_sent_worldprior_prob.append(new_triple)

			sent_worldprior_prob = new_sent_worldprior_prob



			if len(final_sentences)>=50:
			# 	# print("beam unroll time",tic-toc)
			# 	# print(state.image_priors[:])
				sentences = sorted(final_sentences,key=lambda x : x[-1],reverse=True)
				output = []
				for i,(sent,prob) in enumerate(sentences):

					output.append(("".join(sent),prob))

				return output
			# 		# print(sentences)
			# 		for i,(sent,prob) in enumerate(sentences):

			# 			output.append(("".join([rsa.idx2word[idx] for idx in np.squeeze(sent)]),prob))

			# 		return output
			# 	return "COMPLETE"
			# 	return "".join([rsa.idx2word[idx] for idx in np.squeeze(final_sentences[0])])

		if beam_decay < beam_width:
			beam_width-=beam_decay
		# print("decayed beam width by "+str(beam_decay)+"; beam_width now: "+str(beam_width))

	sentences = sorted(final_sentences,key=lambda x : x[-1],reverse=True)

	output = []
	# print(sentences)
	for i,(sent,prob) in enumerate(sentences):

		output.append(("".join(sent),prob))

	print(state.world_priors)
	return output

# write it to take a stateful model and unroll: keeping type as a speaker
# you should do this for all your recursion schemes actually
def ana_monad(kleisli_coalgebra,initial_world_prior, pass_prior=True,listener_rationality=1.0,start_from=[],max_sentence_length=max_sentence_length):
	pass

	# out = foo(speaker_rationality,speaker,target):


	# 	state = RSA_State(initial_world_prior, listener_rationality=listener_rationality)
	# 	context_sentence = ['^']+start_from
	# 	state.context_sentence=context_sentence
		

	# 	world=RSA_World(target=target,rationality=speaker_rationality,speaker=speaker)

	# 	sent_worldprior_prob = [(state.context_sentence,state.world_priors,0.0)]

	# 	for timestep in tqdm(range(len(start_from)+1,max_sentence_length)):
			
			
	# 		state.timestep=timestep

	# 		new_sent_worldprior_prob = []
	# 		for sent,worldpriors,old_prob in sent_worldprior_prob:

	# 			state.world_priors=worldpriors

	# 			# if state.timestep > 1:

	# 				# state.context_sentence = sent[:-1]
	# 				# seg = sent[-1]

	# 				# if depth>0 and pass_prior:

	# 				# 	l=rsa.listener(state=state,utterance=rsa.seg2idx[seg],depth=depth)
	# 				# 	state.world_priors[state.timestep-1]=copy.deepcopy(l)		

	# 			state.context_sentence = sent

	# 			s = kleisli_coalgebra(state=state,world=world)	
				
	# 			for seg,prob in enumerate(np.squeeze(s)):

	# 				new_sentence = copy.deepcopy(sent)
	# 				new_sentence += [rsa.idx2seg[seg]]
	# 				state.context_sentence = new_sentence
	# 				new_prob = prob+old_prob

	# 				new_sent_worldprior_prob.append((new_sentence,worldpriors,new_prob))

	# 		rsa.flush_cache()
	# 		sent_worldprior_prob = sorted(new_sent_worldprior_prob,key=lambda x:x[-1],reverse=True)

	# 	final_sentences=[]
	# 	for sent,worldprior,prob in sent_worldprior_prob:
	# 		final_sentence = copy.deepcopy(sent)
	# 		# print("sentence",final_sentence)
	# 		final_sentences.append((final_sentence,prob))


	# 	sentences = sorted(final_sentences,key=lambda x : x[-1],reverse=True)

	# 	output = []
	# 	for i,(sent,prob) in enumerate(sentences):

	# 		output.append(("".join(sent),prob))

	# 	return output

