from utils.config import *
from utils.helper_functions import display,byte_pair_encoding,byte_pair_decoding
from translators.translators import BaseCase, Unfold, Factor_To_Character, Constant, Compose
from bayesian_agents.bayesian_pragmatics import Pragmatic


sentences = ["He is wearing glasses.","He wears glasses."]

# alpha, in the paper
rationality = 10.0
# a pretrained neural translator. Here using a fairseq transformer, but cnn, lstm, rnn etc would work too.
s0_word = BaseCase(path='models/wmt16.en-de.joined-dict.transformer',source='en',target='de',bpe_code_path='/bpecodes')

sentences = [byte_pair_encoding(sentence=s,code_path=s0_word.bpe_code_path) for s in sentences]

# produces a sentence level non-pragmatic speaker from s0_word
s0_sent = Unfold(underlying_model=s0_word,beam_width=2)

# returns distribution over the (here, two) source sentences, given target sentence. Just Bayes rule on s0_sent
l1_sent = Constant(support=sentences,unfolded_model=s0_sent,change_direction=True)

# Prefers translations which communicate the source sentence over any distractors. E.g. source_sent=sentences[0] and distractors = [sentences[1]]
# Because I restrict s0_sent to produce a distribution over only two output sentences (using beam_width=2), 
# s1_sent_gp is tractable here.
s1_sent_gp = Pragmatic(
	rightward_model=s0_sent,
	leftward_model=None, 
	unfolded_leftward_model=l1_sent,
	width=2,rat=rationality,
	EXTEND=False)

# a pretrained neural translator. Fairseq again.
l0_word = BaseCase(path='models/fconv_wmt_de_en',source='de',target='en',bpe_code_path='/code')

# prefers choosing next word of translation which communicates source sentence over any distractors.
s1_word = Pragmatic(
	rightward_model=s0_word,
	leftward_model=None, 
	unfolded_leftward_model=l1_sent,
	width=1000,rat=rationality,
	EXTEND=False)

# produces a sentence level pragmatic speaker from s1_word
s1_sent_ip = Unfold(underlying_model=s1_word,beam_width=2)

# no explicit distractors needed (implicitly, all possible lists of English words are distractors)
# again, only tractable when target sentence space is restricted (here to two sentences)
s1_sent_c_gp = Pragmatic(
	rightward_model=s0_sent,
	leftward_model=None, 
	unfolded_leftward_model=l1_sent,
	width=2,rat=rationality,
	EXTEND=True)

# avoids restriction of s1_sent_c_gp by generating a single word at a time. See paper for details.
s1_word_c = Pragmatic(
	rightward_model=s0_word,
	leftward_model=l0_word, 
	unfolded_leftward_model=None,
	width=2,rat=5.0,
	EXTEND=True)

# produces a sentence level pragmatic speaker that doesn't need explicit distractors from s1_word _c
s1_sent_c_ip = Unfold(underlying_model=s1_word_c,beam_width=2)


models = {"s0_word":s0_word,"s0_sent":s0_sent,"s1_word":s1_word,"s1_sent_gp":s1_sent_gp,"s1_sent_ip":s1_sent_ip,"s1_sent_c_gp":s1_sent_c_gp,"s1_sent_c_ip":s1_sent_c_ip}

for model_type in models:

	print("model: "+model_type)
	model = models[model_type]

	probs, support = model.forward(sequence=[],source_sentence=sentences[0],debug=False)
	print(display(probs=probs,support=support))

	model.empty_cache()


