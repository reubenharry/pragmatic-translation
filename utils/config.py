from collections import defaultdict
IMG_DATA_PATH="charpragcap/resources/visual_genome_data/"
REP_DATA_PATH="charpragcap/resources/resnet_reps/"
TRAINED_MODEL_PATH="data/models/"
WEIGHTS_PATH="charpragcap/resources/weights/"
S0_WEIGHTS_PATH="s0_weights"
S0_PRIME_WEIGHTS_PATH="s0_prime_weights"
caption,region = 0,1
start_token = {"word":"</s>","char":'^',"sentence":"<start>"}
stop_token = {"word":"</s>","char":'$',"sentence":"<stop>"}
pad_token = '&'
# sym_set = list('&^$ abcdefghijklmnopqrstuvwxyz')
sym_set = list('&$ abcdefghijklmnopqrstuvwxyzÄäöÖÜüABCDEFGHIJKLMNOPQRSTUVWXYZ.,éÇâêîôûàèùëï@ß-')

stride_length = 10
start_index = 1
stop_index = 2
pad_index = 0
batch_size = 50
max_sentence_length = {"word":45, "char": 60,"sentence":1}

train_size,val_size,test_size = 0.98,0.01,0.01
rep_size = 2048
img_rep_layer = 'hiddenrep'

# e.g. char_to_index,index_to_char=list_to_index_iso(sym_set)["rightward"],list_to_index_iso(sym_set)["leftward"]
def list_to_index_iso(l):
	rightward = defaultdict(int)
	for i,x in enumerate(l):
		rightward[x] = i
	leftward = defaultdict(int)
	for i,x in enumerate(sym_set):
		leftward[i] = x

	out = {}
	out["rightward"]=rightward
	out["leftward"]=leftward
	return out



