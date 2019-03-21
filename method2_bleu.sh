#cat experiment/results/method3/inc_prag_results_sourcelong_sents | sacrebleu experiment/results/method3/gold_standard_sourcelong_sents

cat experiment/results/method2/lit_results | sacrebleu experiment/results/method2/gold_standard
cat experiment/results/method2/inc_prag_results | sacrebleu experiment/results/method2/gold_standard
cat experiment/results/method2/glob_prag_results | sacrebleu experiment/results/method2/gold_standard
