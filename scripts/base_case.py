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
from utils.helper_functions import display,byte_pair_encoding
from translators.translators import Coalgebra, Anamorphism, Factor_To_Character, Constant, Compose
from bayesian_agents.bayesian_pragmatics import Pragmatic

# source_target_word_model = Coalgebra(path='wmt14.en-de.fconv-py',source='en',target='de',bpe_code_path='/bpecodes')
# target_source_word_model = Coalgebra(path='iwslt14.tokenized.de-en',source='de',target='en',bpe_code_path='/code')

# source_target_word_model = Coalgebra(path='wmt14.en-fr.joined-dict.transformer',source='en',target='fr',bpe_code_path='/bpecodes')
# target_source_word_model = Coalgebra(path='fconv_wmt_fr_en',source='fr',target='en',bpe_code_path='/code')

source_target_word_model = Coalgebra(path='wmt16.en-de.joined-dict.transformer',source='en',target='de',bpe_code_path='/bpecodes')
target_source_word_model = Coalgebra(path='wmt14.de-en.fconv-py',source='de',target='en',bpe_code_path='/code')



source_target = source_target_word_model
target_source = target_source_word_model

# source_target_char_model = Factor_To_Character(source_target_word_model)
# target_source_char_model = Factor_To_Character(target_source_word_model)
# source_target_model = source_target_char_model
# target_source_model = target_source_char_model

unfolded_source_target = Anamorphism(underlying_model=source_target,beam_width=2)
unfolded_target_source = Anamorphism(underlying_model=target_source,beam_width=1)


# source_to_source = Compose(rightward_model=source_target_model,leftward_model=target_source_model, unfolded_rightward_model=None)
# unfolded_source_to_source = Anamorphism(underlying_model=source_to_source,beam_width=2)


