cat experiment/results/method3/lit_results_source_manual | sacrebleu experiment/results/method3/gold_standard_source
cat experiment/results/method3/inc_prag_results_source_manual | sacrebleu experiment/results/method3/gold_standard_source
# cat experiment/results/method3/glob_prag_results_source | sacrebleu experiment/results/method3/gold_standard_source



cat experiment/results/method3/lit_results_target | sacrebleu experiment/results/method3/gold_standard_target
cat experiment/results/method3/inc_prag_results_target | sacrebleu experiment/results/method3/gold_standard_target
# cat experiment/results/method3/glob_prag_results_target | sacrebleu experiment/results/method3/gold_standard_target


