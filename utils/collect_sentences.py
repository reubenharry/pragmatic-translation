# from py_translator import Translator
# import nltk
# import pickle

english_sentences = list(open('../Downloads/dev/news-test2013.en','r'))
german_sentences = list(open('../Downloads/dev/news-test2013.de','r'))

pairs = []
for i,line in enumerate(english_sentences):
	# if len(line.split())<10:
	if True:
		pairs.append({"en":line,"de":german_sentences[i]})

dev_sentences = pairs[:50]
test_sentences = pairs[50:]

# print(dev_sentences)

# def create_paraphrase(sent,source,target):

# 	there = Translator().translate(text=sent,src=source, dest=target).text
# 	# print("intermediate",there)
# 	back = Translator().translate(text=there,src=target, dest=source).text

# 	there_again = Translator().translate(text=back,src=source, dest=target).text

# 	condition_1 = nltk.word_tokenize(back)!=nltk.word_tokenize(sent)
# 	# print(nltk.word_tokenize(back),nltk.word_tokenize(sent))
# 	condition_2 = there==there_again

# 	if condition_1 and condition_2: return [sent,back]
# 	else: 
# 		print(nltk.word_tokenize(back),nltk.word_tokenize(sent))
# 		print("sent",sent,"there",there,"back",back,"there_again",there_again)
# 		return None

# results = []
# for i in range(20):
# 	inp = dev_sentences[i]
# 	try:
# 		out = create_paraphrase(inp,source="en",target="de")
# 	except: continue
# 	print(out)
# 	results.append(out)


# print(results)

# pickle.dump(results,open("paraphrase_data.pkl",'wb'))


