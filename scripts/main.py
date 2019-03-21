from utils.config import *
from utils.helper_functions import display,byte_pair_encoding,byte_pair_decoding
from translators.translators import Coalgebra, Anamorphism, Factor_To_Character, Constant, Compose
from bayesian_agents.bayesian_pragmatics import Pragmatic


sentences = ["He is wearing glasses.","He wears glasses."]


s0_word = Coalgebra(path='wmt16.en-de.joined-dict.transformer',source='en',target='de',bpe_code_path='/bpecodes')

sentences = [byte_pair_encoding(sentence=s,code_path=s0_word.bpe_code_path) for s in sentences]

s0_sent = Anamorphism(underlying_model=s0_word,beam_width=2)

l1_sent = Constant(support=sentences,unfolded_model=s0_sent,change_direction=True)

s1_sent_gp = Pragmatic(
	rightward_model=s0_sent,
	leftward_model=None, 
	unfolded_leftward_model=l1_sent,
	width=2,rat=5.0,
	EXTEND=False)


l0_word = Coalgebra(path='fconv_wmt_de_en',source='de',target='en',bpe_code_path='/code')

s1_word = Pragmatic(
	rightward_model=s0_word,
	leftward_model=None, 
	unfolded_leftward_model=l1_sent,
	width=1000,rat=5.0,
	EXTEND=False)

s1_sent_ip = Anamorphism(underlying_model=s1_word,beam_width=2)



s1_sent_c_gp = Pragmatic(
	rightward_model=s0_sent,
	leftward_model=None, 
	unfolded_leftward_model=l1_sent,
	width=2,rat=5.0,
	EXTEND=True)

s1_word_c = Pragmatic(
	rightward_model=s0_word,
	leftward_model=l0_word, 
	unfolded_leftward_model=None,
	width=2,rat=5.0,
	EXTEND=True)

s1_sent_c_ip = Anamorphism(underlying_model=s1_word_c,beam_width=2)


models = {"s0_word":s0_word,"s0_sent":s0_sent,"s1_word":s1_word,"s1_sent_gp":s1_sent_gp,"s1_sent_ip":s1_sent_ip,"s1_sent_c_gp":s1_sent_c_gp,"s1_sent_c_ip":s1_sent_c_ip}

for model_type in models:

	print("model: "+model_type)
	model = models[model_type]

	probs, support = model.forward(sequence=[],source_sentence=sentences[0],debug=False)
	print(display(probs=probs,support=support))

	model.empty_cache()


