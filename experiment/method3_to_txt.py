import pickle
import nltk
import re
from nltk.translate.bleu_score import sentence_bleu
from utils.get_sents_from_wmt_test_set import english_test_sentences,german_test_sentences
from experiment.exp1_data import all_words_are_frequent

RAT = 0.1

# path = "_2018_wmt_test"
# results = pickle.load(open("experiment/results/rat"+str(RAT)+"_bleu_test_sentences"+path+".pkl",'rb'))
# results = pickle.load(open("experiment/results/rat"+str(2.0)+"_bleu_test_sentences"+path+"_lit.pkl",'rb'))
# english_sentences = english_test_sentences
# german_sentences = german_test_sentences
path = "_2018_wmt_0"
# path = "_2018_wmt_test_greedy_third_batch"
results = pickle.load(open("experiment/results/rat"+str(RAT)+"_bleu_test_sentences"+path+".pkl",'rb'))

# second_batch_results = pickle.load(open("experiment/results/rat0.1_bleu_test_sentences_2018_wmt_test_greedy_second_batch.pkl",'rb')) 

# third_batch_results = pickle.load(open("experiment/results/rat0.1_bleu_test_sentences_2018_wmt_test_greedy_third_batch.pkl",'rb')) 


# print(len(second_batch_results["target"]["literal"]))
# print(len(results["target"]["literal"]))

# results = 
# for s in second_batch_results["target"]["literal"]:

# 	results["target"]["literal"][s]=second_batch_results["target"]["literal"][s]

# for s in second_batch_results["target"]["nonparametric incremental"]:
# 	if (s in results["target"]["nonparametric incremental"]):
# 		print(s)

# 	results["target"]["nonparametric incremental"][s]=second_batch_results["target"]["nonparametric incremental"][s]


# for s in third_batch_results["target"]["literal"]:

# 	results["target"]["literal"][s]=third_batch_results["target"]["literal"][s]

# for s in third_batch_results["target"]["nonparametric incremental"]:
# 	if (s in results["target"]["nonparametric incremental"]):
# 		print(s)

# 	results["target"]["nonparametric incremental"][s]=third_batch_results["target"]["nonparametric incremental"][s]


# print(len(results["target"]["literal"]))
# raise Exception

english_sentences = english_test_sentences
german_sentences = german_test_sentences
# english_sentences = list(open('../Downloads/dev/news-test2013.en','r'))
# german_sentences = list(open('../Downloads/dev/news-test2013.de','r'))


# lit_scores = 0
# prag_scores = 0
# glob_prag_scores = 0
# print(results)
# print(list(results["source"]["global pragmatic"]))

write_path="new"

gold_standard_source = open("experiment/results/method3/gold_standard_source"+write_path,'w')
lit_results_source = open("experiment/results/method3/lit_results_source"+write_path,'w')
inc_prag_results_source = open("experiment/results/method3/inc_prag_results_source"+write_path,'w')
glob_prag_results_source = open("experiment/results/method3/glob_prag_results_source"+write_path,'w')

gold_standard_target = open("experiment/results/method3/gold_standard_target"+write_path,'w')
lit_results_target = open("experiment/results/method3/lit_results_target"+write_path,'w')
inc_prag_results_target = open("experiment/results/method3/inc_prag_results_target"+write_path,'w')
glob_prag_results_target = open("experiment/results/method3/glob_prag_results_target"+write_path,'w')
counter = 0

for sent in results["target"]["literal"]:

	if sent in results["target"]["nonparametric incremental"]:
	# and all_words_are_frequent(sent.split()):
	# and sent in results["source"]["global pragmatic"]:
		try:
			counter+=1
			gold_standard_target.write(german_sentences[english_sentences.index(sent)])
			gold_standard_target.write("\n")
			lit_results_target.write(re.sub(" \.",".",results["target"]["literal"][sent]))
			gold_standard_source.write(sent)
			gold_standard_source.write("\n")
			# lit_results_source.write(results["source"]["literal"][sent])
			# lit_results_source.write("\n")
			# inc_prag_results_source.write(results["source"]["nonparametric incremental"][sent])
			# inc_prag_results_source.write("\n")

			# glob_prag_results_source.write(results["source"]["global pragmatic"][sent])
			# glob_prag_results_source.write("\n")
			lit_results_target.write("\n")
			inc_prag_results_target.write(re.sub(" \.",".",results["target"]["nonparametric incremental"][sent]))
			# if counter==751:break
			inc_prag_results_target.write("\n")

			# glob_prag_results_target.write(results["target"]["global pragmatic"][sent])
			# glob_prag_results_target.write("\n")
		except: 
			print("CONTINUE")
			continue


print(counter)

lit_results_source.close()
inc_prag_results_source.close()
gold_standard_source.close()
lit_results_target.close()
inc_prag_results_target.close()
gold_standard_target.close()

# print(len(results["target"]["nonparametric incremental"]))
# print(len(results["target"]["global pragmatic"]))

	# 	try:
	# 		print(sent)
	# 		lit = results["source"]["literal"][sent]
	# 		inc_prag = results["source"]["nonparametric incremental"][sent]
	# 		glob_prag = results["source"]["global pragmatic"][sent]

	# 		lit_score = sentence_bleu([sent.split()],lit.split(), weights=(1, 0, 0, 0))
	# 		prag_score = sentence_bleu([sent.split()],inc_prag.split(), weights=(1, 0, 0, 0))
	# 		glob_prag_score = sentence_bleu([sent.split()],glob_prag.split(), weights=(1, 0, 0, 0))

	# 		lit_scores += lit_score
	# 		prag_scores += prag_score
	# 		glob_prag_scores += glob_prag_score
	# 	except:continue

	# print("lit score",lit_scores)
	# print("prag score",prag_scores)
	# print("glob prag score",glob_prag_scores)



