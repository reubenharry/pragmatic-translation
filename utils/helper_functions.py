import numpy as np
import nltk
import re
from subword_nmt.apply_bpe import BPE
import codecs

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def uniform_vector(length):
	return np.ones((length))/length

def make_initial_prior(initial_image_prior,initial_rationality_prior,initial_speaker_prior,initial_hyperprior_prior,initial_langmod_prior, initial_qud_prior):

    langmod_qud = np.multiply.outer(initial_langmod_prior,initial_qud_prior)
    hyperprior_langmod_qud = np.multiply.outer(initial_hyperprior_prior,langmod_qud)
    speaker_hyperprior_langmod_qud = np.multiply.outer(initial_speaker_prior,hyperprior_langmod_qud)
    rat_speaker_hyperprior_langmod_qud = np.multiply.outer(initial_rationality_prior,speaker_hyperprior_langmod_qud)
    image_rat_speaker_hyperprior_langmod_qud = np.multiply.outer(initial_image_prior,rat_speaker_hyperprior_langmod_qud)
    return np.log(image_rat_speaker_hyperprior_langmod_qud)

def display(probs,support):
    # print("STUFFF")
    # print(probs)
    # print(support)
    # print(list(zip(list(support),list(np.exp(probs)))))
    # print(sorted(list(zip(list(support),list(np.exp(probs)))),key=lambda x : x[1],reverse=True))
    return sorted(list(zip(list(support),list(np.exp(probs)))),key=lambda x : x[1],reverse=True)[:10]

def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0.0, p * np.log(p / q), 0.0))

def marginalize_out(dimension1,dimension2):
    return np.sum(np.sum(np.exp(world_posterior),axis=state.dim[dimension1]+1),axis=state.dim[dimension2]+1)

# subword level encoding used by fairseq
def byte_pair_encoding(sentence,code_path):
    sentence = re.sub("'","&apos;",sentence)
    if sentence[-1] in list(".!?"): 
        sentence=sentence[:-1]+" "+sentence[-1]+" "
        # sentence+=" "

    code = codecs.open(code_path, encoding='utf-8')
    out = BPE(code).process_line(sentence)
    # print("ENCODED SENTENCE:",out)
    # print("CODE PATH",code_path)
    return out

def byte_pair_decoding(sentence):
    # sentence = " ".join(nltk.word_tokenize(sentence))+" . "
    # return re.sub("(\.$)|(@@ )","",sentence)
    # sentence = re.sub("\.","",sentence)
    sentence = re.sub("\n","",sentence)
    sentence = re.sub("&apos;","'",sentence)
    sentence = re.sub("@@ ","",sentence)
    sentence = re.sub(" @-@ ","",sentence)
    sentence = re.sub("@@@","",sentence)
    sentence = re.sub("\&quot","\"",sentence)
    sentence = re.sub("&amp","&",sentence)
    return sentence

def unique_list(l):

    new_l = []
    for item in l:
        if item not in new_l:
            new_l.append(item)

    return new_l

