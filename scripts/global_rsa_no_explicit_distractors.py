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
from utils.helper_functions import uniform_vector, make_initial_prior, display,byte_pair_encoding,byte_pair_decoding
from translators.translators import Coalgebra, Anamorphism, Factor_To_Character, Constant, Compose
from bayesian_agents.bayesian_pragmatics import Pragmatic
from utils.paraphrase import get_paraphrases
from base_case import source_target_model, target_source_model, target_source_word_model

unfolded_source_target_model = Anamorphism(underlying_model=source_target_model,beam_width=20)
unfolded_target_source_model = Anamorphism(underlying_model=target_source_model)

rightward_pragmatic = Pragmatic(
	rightward_model=unfolded_source_target_model,
	leftward_model=target_source_model, 
	unfolded_leftward_model=unfolded_target_source_model,
	width=100,rat=1.0,
	EXTEND=False)

leftward_pragmatic = Pragmatic(
	rightward_model=unfolded_target_source_model,
	leftward_model=source_target_model, 
	unfolded_leftward_model=unfolded_source_target_model,
	width=100,rat=1.0,
	EXTEND=False)


# source_to_source = Compose(rightward_model=source_target_model,leftward_model=unfolded_target_source_model, unfolded_rightward_model=unfolded_source_target_model)
unfolded_source_to_source_pragmatic = Compose(rightward_model=None,leftward_model=unfolded_target_source_model, unfolded_rightward_model=rightward_pragmatic)
# unfolded_source_to_source_pragmatic = Anamorphism(underlying_model=source_to_source_pragmatic)


