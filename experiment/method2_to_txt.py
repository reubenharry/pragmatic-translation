import pickle
import nltk
from nltk.translate.bleu_score import sentence_bleu
from utils.helper_functions import byte_pair_decoding


RAT = 5.0
path = ""

results = pickle.load(open("experiment/results/rat"+str(RAT)+"_method2_sentences"+path+".pkl",'rb'))

# lit_scores = 0
# prag_scores = 0
# glob_prag_scores = 0
print(results)
# print(list(results["source"]["global pragmatic"]))

gold_standard = open("experiment/results/method2/gold_standard",'w')
# lit_results = open("experiment/results/method2/lit_results",'w')
# inc_prag_results = open("experiment/results/method2/inc_prag_results",'w')
# glob_prag_results = open("experiment/results/method2/glob_prag_results",'w')

lit_results_target = open("experiment/results/method2/lit_results_target",'w')
inc_prag_results_target = open("experiment/results/method2/inc_prag_results_target",'w')
glob_prag_results_target = open("experiment/results/method2/glob_prag_results_target",'w')
counter=0

for i,sent in enumerate(results["target"]["literal"]):

	if i>50 and i<=1000: 

		if sent in results["target"]["nonparametric incremental"]:

			gold_standard.write(byte_pair_decoding(sent))
			gold_standard.write("\n")
			# lit_results.write(results["source"]["literal"][sent])
			# lit_results.write("\n")
			# inc_prag_results.write(results["source"]["nonparametric incremental"][sent])
			# inc_prag_results.write("\n")

			# glob_prag_results.write(results["source"]["global pragmatic"][sent])
			# glob_prag_results.write("\n")

			# gold_standard_target.write(german_sentences[english_sentences.index(sent)])
			# gold_standard_target.write("\n")
			lit_results_target.write(results["target"]["literal"][sent])
			lit_results_target.write("\n")
			inc_prag_results_target.write(results["target"]["nonparametric incremental"][sent])
			inc_prag_results_target.write("\n")

			counter+=1
print("counter",counter)
# lit_results.close()
# inc_prag_results.close()
# gold_standard.close()

# print(len(results["source"]["nonparametric incremental"]))
# print(len(results["source"]["global pragmatic"]))

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



