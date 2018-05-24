'''Train Bidirectional LSTM network for relation extraction task.
'''
from __future__ import print_function


#import tensorflow as tf
#tf.python.control_flow_ops = tf
# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation,  Input, merge
from keras.layers import Embedding
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from keras.models import load_model

#arg parsing
import argparse
import preprocessingscript
import read_CRFFeaturesFile_i2b22010_rel_onemodel

def build_model(args,X_train, Y_train, X_sentence):
	# def build_model(args,X_train, Y_train):
	print('Build model...')
	if not args.embedding:
		sequence = Input(shape=(args.maxlen,))
		embedded = Embedding(args.max_features, 128, input_length=args.maxlen)(sequence)
		forwards = LSTM(64)(embedded)
		backwards = LSTM(64, go_backwards=True)(embedded)
		merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
		after_dp = Dropout(0.5)(merged)
		output = Dense(len(args.tagset), activation='softmax')(after_dp)
		args.model = Model(input=sequence, output=output)
	else:
		
		#single network
		# sequence = Input(shape=(args.maxlen, args.embedding_size))
		# forwards = LSTM(64)(sequence)
		# backwards = LSTM(64, go_backwards=True)(sequence)
		# merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
		# after_dp = Dropout(0.5)(merged)
		# output = Dense(len(args.tagset), activation='softmax')(after_dp)

		#siamese network
		input_name = Input(shape=(args.maxlen, args.embedding_size))
		#first layer first birectional
		forwards = LSTM(64)(input_name)
		backwards = LSTM(64, go_backwards=True)(input_name)
		merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
		after_dp = Dropout(0.5)(merged)
		# output = Dense(len(args.tagset), activation='softmax')(after_dp)
		# args.model = Model(input=input_name, output=output)

		#second layer sentence level input
		sentence_level_input = Input(shape=(args.maxlen,))
		#forwards_sent = LSTM(64)(sentence_level_input)
		#backwards_sent = LSTM(64, go_backwards=True)(sentence_level_input)
		#merged_sent = merge([forwards_sent, backwards_sent], mode='concat', concat_axis=-1)
		#after_dp_sent = Dropout(0.5)(merged_sent)

		merged_networks = merge([after_dp, sentence_level_input], mode='concat')
		merged_networks_dense = Dense(64, activation='relu')(merged_networks)
		output = Dense(len(args.tagset), activation='softmax')(merged_networks_dense)
		args.model = Model(input=[input_name, sentence_level_input], output=output)



	if args.numberofclasses == 2:
		args.model.compile(loss='binary_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])
	else:
		args.model.compile(loss='categorical_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])


	print('Train...')
	#'TrWP', 'TeCP', 'TrCP', '0', 'TrNAP', 'TrAP', 'PIP', 'TrIP', 'TeRP']), 9
	args.model.fit([X_train, X_sentence], Y_train, batch_size=args.batch_size, nb_epoch=args.nb_epoch, \
	# args.model.fit(X_train, Y_train, batch_size=args.batch_size, nb_epoch=args.nb_epoch, \
	class_weight={0:10, 1:1, 2:4, 3:1, 4:10,  5:1,6:1, 7:5, 8:1})#,
	          #validation_data=(X_test, Y_test))
	#args.model.save(args.modelfile)


def test_model(args):
	if not args.model:
		args.model = load_model(args.modelfile);
	print('model loaded')
	# if not args.model:
	# 	args.model = load_model(args.modelfile);
	#test file
	#read_CRFFeaturesFile_i2b22010.testTensorsForRelationExtraction(args,args.test_crfFeaturesFile, args.output_crfFeaturesFile)

	#test dir
	# args.test_crfFeatures_dir = "../finaldata/i2b22010Dataset/test/features"
	# args.output_crfFeatures_dir = "../finaldata/i2b22010Dataset/test/imdb_bidirectional_lstm_hitachi_ner_cat_B-prob.model.h5_features_sep24_outputdir"
	read_CRFFeaturesFile_i2b22010_rel_onemodel.postProcessing_createTestCRFFeaturesFile_dir(args)

def main():
	parser = argparse.ArgumentParser(description='This is a demo script by nixCraft.')

	# Embedding
	parser.add_argument('--max_features', type=int, default=160380,
	                        help='Max number of word index for embeddings')
	parser.add_argument('--vocab_size', type=int, default=160380,
	                        help='vocabulary size')
	parser.add_argument('--maxlen', type=int, default=11,
	                        help='length of the sequence')
	parser.add_argument('--embedding_size', type=int, default=128,
	                        help='Size of the embeddings. pre-trained:40, embeddings=128')
	parser.add_argument('--pretrained_embedding_size', type=int, default=40,
	                        help='Size of the embeddings. pre-trained:40')
	parser.add_argument('--embedding', action='store_true', help='Use pre-trained embeddings')
	#args.max_features = 160380 #vocabulary size
	# args.vocab_size = 160380
	# args.maxlen = 11 #length of the sequence
	# args.embedding_size = 40 # word embedding size
	#args.vector_dim = 40;
	# args.embedding=True

	# Convolution
	parser.add_argument('--filter_length', type=int, default=3,
	                        help='length of each Filter in Convolution')
	parser.add_argument('--nb_filter', type=int, default=10,
	                        help='number of filters for Convolution')
	parser.add_argument('--pool_length', type=int, default=4,
	                        help='Max pooling length in Convolution1D')
	# args.filter_length = 3
	# args.nb_filter = 10
	# args.pool_length = 4

	# LSTM
	parser.add_argument('--lstm_output_size', type=int, default=64,
	                        help='lstm output size')
	# args.lstm_output_size = 64

	#network
	parser.add_argument('--num_hidden_layers', type=int, default=1,
	                        help='number of hidden layers')
	# args.num_hidden_layers = 1

	# Training
	parser.add_argument('--batch_size', type=int, default=30,
	                        help='batch_size')
	parser.add_argument('--nb_epoch', type=int, default=2,
	                        help='number of epochs')
	# args.batch_size = 30
	# args.nb_epoch = 2

	#char initializations
	parser.add_argument('--max_charlen', type=int, default=20,
	                        help='max character length of a word.')
	parser.add_argument('--char_vocab_size', type=int, default=100,
	                        help='number of characters vocabulary')
	parser.add_argument('--char_embedding_size', type=int, default=50,
	                        help='embedding_size for each character')
	parser.add_argument('--include_chars', type=bool, default=False,
	                        help='whether to include chars in building model.')
	# args.max_charlen=20
	# args.char_vocab_size = 100
	# args.char_embedding_size = 50
	# args.include_chars = False

	#char ngrams
	parser.add_argument('--num_of_ngram', type=int, default=3,
	                        help='whether to include character ngrams.')
	# args.num_of_ngram=3
	parser.add_argument('--include_char_ngram_hash', type=bool, default=False,
	                        help='whether to include char ngram in building model.')
	# args.include_char_ngram_hash =False
	parser.add_argument('--ngram_vocab_file', type=str, default='../../finaldata/thyme_ngram_vocab.txt',
	                        help='ngram vocabulary file.')
	# args.ngram_vocab_file="../finaldata/thyme_ngram_vocab.txt"
	#preprocessingscript.read_ngram_vocab_file(args)

	parser.add_argument('--conceptfield', type=int, default=11,
	                        help='concept (ner) column number in crf feature File.')
	parser.add_argument('--tagfield', type=int, default=11,
	                        help='tag column number in crf feature File.')
	parser.add_argument('--testconceptfield', type=int, default=11,
	                        help='concept (ner) column number in crf feature File.')
	parser.add_argument('--testtagfield', type=int, default=11,
	                        help='tag column number in crf feature File.')
	# args.conceptfield=11
	# args.tagfield=11

	#preprocess
	parser.add_argument('--word2vecFile', type=str, 
		default='../../finaldata/i2b22010_clef2013_clef2014_thyme_MIMICIII.txt_cleaned_glove.op_40.txt',
	                        help='ngram vocabulary file.')
	parser.add_argument('--train_crfFeaturesFile', type=str, 
		default='/home/raghavendra/BackUP/MyWorks/workspace/TemporalMining/THYMEWorkingDir/semeval2016/featureFiles/deepnlTrain_DevFeatureFiles_all_es_props',
	                        help='ngram vocabulary file.')
	parser.add_argument('--test_crfFeaturesFile', type=str, 
		default='/home/raghavendra/BackUP/MyWorks/workspace/TemporalMining/THYMEWorkingDir/semeval2016/featureFiles/deepnltestphase2FeatureFiles_es_props/ID006_clinic_016',
	                        help='ngram vocabulary file.')
	parser.add_argument('--output_crfFeaturesFile', type=str, 
		default='/home/raghavendra/BackUP/MyWorks/workspace/TemporalMining/THYMEWorkingDir/semeval2016/featureFiles_OP/oct14_ID006_clinic_016_op',
	                        help='ngram vocabulary file.')
	parser.add_argument('--test_crfFeatures_dir', type=str, 
		default='/home/raghavendra/BackUP/MyWorks/workspace/TemporalMining/THYMEWorkingDir/semeval2016/featureFiles/deepnltestphase2FeatureFiles_es_props/',
	                        help='ngram vocabulary file.')
	parser.add_argument('--output_crfFeatures_dir', type=str, 
		default='/home/raghavendra/BackUP/MyWorks/workspace/TemporalMining/THYMEWorkingDir/semeval2016/featureFiles_OP/oct14_deepnltestphase2FeatureFiles_dnnOP_semeval2016.es',
	                        help='ngram vocabulary file.')
	parser.add_argument('--modelfile', type=str, 
		default='models/bidirectional_lstm_ner_categorical.embedded.model.h5_oct14', help='save the model in the file')

	parser.add_argument('--posfile', type=str, default='../../finaldata/word2VecFiles/posTags.txt', help='pos tags')
	parser.add_argument('--chunkfile', type=str, default='../../finaldata/word2VecFiles/chunkTags.txt', help='chunk tags')
	# parser.add_argument('--capfil2', type=str, default='False', help='capitalized tags')
	# parser.add_argument('--gazetterfile', type=str, default='False', help='is in gazetter')
	parser.add_argument('--suffile', type=str, default='../../finaldata/word2VecFiles/suffixes.txt', help='suffix')
	parser.add_argument('--prefile', type=str, default='../../finaldata/word2VecFiles/prefixes.txt', help='prefix')
	parser.add_argument('--gazetterfile', type=str, default='False', help='is in gazetter')

	#features
	# p.add_argument('-f', '--foo', action='store_true')
	parser.add_argument('--pos', action='store_true', help='pos tags')
	parser.add_argument('--chunk',action='store_true', help='chunk tags')
	parser.add_argument('--cap', action='store_true', help='capitalized tags')
	parser.add_argument('--gazetter', action='store_true', help='is in gazetter')
	parser.add_argument('--suf', action='store_true', help='suffix')
	parser.add_argument('--pre', action='store_true', help='prefix')
	parser.add_argument('--others', action='store_true', help='other features')

	parser.add_argument('--sentencelevelfeatures', action='store_true', help='sentence level features')

	parser.add_argument('--train', action='store_true')
	parser.add_argument('--test', action='store_true')

	parser.add_argument('--withgroundtags', action='store_true')
	parser.add_argument('--model', type=str, default='', help='model')

	#parse arguments
	args = parser.parse_args()
	#initializations
	args.ngram2index={}
	args.vocab={} #-- word frequency map
	args.index2word={}
	args.word2index={}
	args.word2vec={}
	args.features2Tokens={}#{pos:[], chunk:[], suffix={}, prefix={}}
	args.tagset=[]
	args.conceptset=[]
	args.all_chars=[chr(k) for k in xrange(32, 127)] #list
	args.char2index=dict((k, v) for v,k in enumerate(args.all_chars)) #dict

	#preprocessing
	preprocessingscript.readWord2Vec(args)
	preprocessingscript.loadFeatures2Tokens(args)
	#preprocessingscript.readTags(args)
	
	print ("args.embedding_size: ", args.embedding_size)
	print('Loading data...')
	#preprocessingscript.loadTrainAndTestFromCRFFeatures(args)
	args.tagset = ['TrWP', 'TeCP', 'TrCP', '0', 'TrNAP', 'TrAP', 'PIP', 'TrIP', 'TeRP']
	args.numberofclasses = len(args.tagset)
	print ("numberofclasses: " ,args.numberofclasses)
	print ("classes: ", args.tagset)
	#args.embedding_size = 90
	#for word features this is already added in preprocessing
	#args.embedding_size+=4;
	#for relation phrase
	# args.embedding_size+=40;
	#for pos

	# positions
	args.embedding_size+=2
	# if args.conceptindex:
	args.embedding_size +=4
	if args.others:
		# args.embedding_size += 221;
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/relation_words/relation_verbs_words.txt', 'r') as f: args.relation_verbs_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/relation_words/worsen_problems.txt', 'r') as f: args.worsen_problems =  f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/relation_words/causes_problems.txt', 'r') as f: args.causes_problems = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/relation_words/all_i2b22010_words_lower.txt', 'r') as f: args.all_i2b22010_words_lower = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/relation_words/conjunction_words.txt', 'r') as f: args.conjunction_words = f.read().splitlines();
		#pos sequences
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/relation_words/TrAP_posseq.txt', 'r') as f: args.TrAP_posseq_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/relation_words/TrCP_posseq.txt', 'r') as f: args.TrCP_posseq_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/relation_words/TrIP_posseq.txt', 'r') as f: args.TrIP_posseq_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/relation_words/TrNAP_posseq.txt', 'r') as f: args.TrNAP_posseq_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/relation_words/TrWP_posseq.txt', 'r') as f: args.TrWP_posseq_words = f.read().splitlines();

		#21 assertion semantic classes
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/assertion_semantic_classes/allergy.txt', 'r') as f: args.allergy_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/assertion_semantic_classes/cause.txt', 'r') as f: args.cause_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/assertion_semantic_classes/continue.txt', 'r') as f: args.continue_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/assertion_semantic_classes/def_determiner.txt', 'r') as f: args.def_determiner_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/assertion_semantic_classes/evidence.txt', 'r') as f: args.evidence_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/assertion_semantic_classes/fail.txt', 'r') as f: args.fail_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/assertion_semantic_classes/instance.txt', 'r') as f: args.instance_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/assertion_semantic_classes/association.txt', 'r') as f: args.association_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/assertion_semantic_classes/certainty.txt', 'r') as f: args.certainty_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/assertion_semantic_classes/copular.txt', 'r') as f: args.copular_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/assertion_semantic_classes/deny.txt', 'r') as f: args.deny_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/assertion_semantic_classes/exception.txt', 'r') as f: args.exception_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/assertion_semantic_classes/history.txt', 'r') as f: args.history_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/assertion_semantic_classes/stop.txt', 'r') as f: args.stop_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/assertion_semantic_classes/avoid.txt', 'r') as f: args.avoid_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/assertion_semantic_classes/change_state.txt', 'r') as f: args.change_state_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/assertion_semantic_classes/decline.txt', 'r') as f: args.decline_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/assertion_semantic_classes/disappear.txt', 'r') as f: args.disappear_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/assertion_semantic_classes/experience.txt', 'r') as f: args.experience_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/assertion_semantic_classes/hypothetical_cue_list.txt', 'r') as f: args.hypothetical_cue_list_words = f.read().splitlines();
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/assertion_semantic_classes/uncertainty.txt', 'r') as f: args.uncertainty_words = f.read().splitlines();
		#load pmi scores
		args.pmiscores = {}
		with open('/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/pmi_i2b22010_scores.txt', 'r') as f: 
			for line in f:
				split = line.strip().split('&123#')
				# print ('split:',(split[0], split[1]),'-',split[2])
				args.pmiscores[(split[0], split[1])]=float(split[2])

		

	print ("args.embedding_size: ", args.embedding_size)
	if args.train:
		X_train, Y_train, X_sentence = read_CRFFeaturesFile_i2b22010_rel_onemodel.readTensorsForEachRelationFromCRFFeaturesFile(args, args.train_crfFeaturesFile)
		# X_train, Y_train = read_CRFFeaturesFile_i2b22010_rel_onemodel.readTensorsForEachRelationFromCRFFeaturesFile(args, args.train_crfFeaturesFile)
		build_model(args,X_train, Y_train, X_sentence)
	test_model(args)

if __name__ == '__main__':
	main()
