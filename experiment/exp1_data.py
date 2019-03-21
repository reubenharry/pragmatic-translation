
import pickle
import re
import nltk
from nltk.corpus import brown
import spacy
nlp = spacy.load('en')


# brown_words = list(brown.words())
# pickle.dump(brown_words,open("experiment/brown_words.pkl",'wb'))
freqs = pickle.load(open("experiment/freqs",'rb'))

top_20000 = set(sorted(list(freqs),key = lambda x : freqs[x],reverse=True))

brown_words = pickle.load(open("experiment/brown_words.pkl",'rb'))

def all_words_in_brown(sent):
	return sum(0 if w in brown_words else 1 for w in sent) == 0

def all_words_are_frequent(sent):
	# print("checking",sent)

	return sum(0 if w.lower() in top_20000 else 1 for w in sent[:-1]) == 0

# feature = (" some "," some of the ")
feature = (" this "," that ")
# " is \w*ing "

def single_sentence_generator():

	counter = 0 
	for sent in brown.sents():
		# if counter>5: break

		cond1 = len(sent)<10
		cond2 = sent[-1]=='.'
		cond5 = all_words_are_frequent(sent)
		if cond1 and cond2 and cond5:
			counter+=1
			sent = " ".join(sent)
			yield sent


def pair_sentence_generator():

	# english_sentences = open('../Downloads/dev/newstest2013.en','r')
	counter = 0 
	# for sent in english_sentences:
	# 	sent = nltk.word_tokenize(sent)
		# print(sent)
	for sent in brown.sents():
		if counter>5: break

		cond1 = len(sent)<10
		cond2 = sent[-1]=='.'
		# cond3 = all_words_in_brown(sent)
		cond4 = search_for_feature(sent=sent,feature=feature)
		# print("cond5")
		cond5 = all_words_are_frequent(sent)
		# print(cond2,cond4)

		if cond1 and cond2 and cond4 and cond5:
			counter+=1
			sent = " ".join(sent)
			# new_sent = sent[:]

			new_sent = re.sub(feature[0],feature[1],sent)


			# new_sent[sent.index(feature[0])]=feature[1]
			# new_sent[sent.index(feature[0])]=feature[1]
			# sent,new_sent = " ".join(sent)," ".join(new_sent)

			yield [sent,new_sent]


# filtered_sents = (x for x in brown.sents() if len(x)<10 and x[-1]=='.' and all_words_in_brown(x))


def search_for_feature(sent,feature):

	# if feature=='this':
	# 	return 'this' in sent

	# if feature=='might':
	return bool(re.search(feature[0]," ".join(sent)))



german_sentence_pairs = [
	["This is a dog . ","That is a dog . "],
	["People go to the market . ", "People are going to the market . "],
	["I went to the store . ","I have gone to the store . "],
	["She might buy food . ", "She could buy food . "],
	["The weather is lovely . ","The weather is nice . "],
	["Where shall I go ? ","Where will I go ? "]
	]
	# is going vs goes?
# in a clever way vs cleverly


french_sentence_pairs = [
	# ["She likes hiking . ","She likes to hike ."],
	["The animals run fast . ","Animals run fast . "],
	["Some of the chairs are blue . ", "Some chairs are blue . "],
	["She has arrived . ", "She arrived . "],
	["I cut my finger . ","I cut my finger off . "],
	# ["I read a book . ", "I read a tome . "],
	# ["I am happy with your performance . ","I am content with your performance . ",],
	# ["He kept on falling . ", "He kept falling . "],
	["She chopped up the tree . ", "She chopped down the tree . "],
	]

if __name__=="__main__":


	counter = 0
	a = sentence_generator()
	for x in a:
		counter+=1
		print(x)

	print(counter)

# french_demonstration_pairs = [
# ]


# more on modals: has to vs must