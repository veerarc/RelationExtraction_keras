'''Train Bidirectional LSTM network for Medical term (Concept) identification task.
'''

from __future__ import print_function


# import tensorflow as tf
# tf.python.control_flow_ops = tf
# # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


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

# import viterbi

#arg parsing
import argparse
import preprocessingscript as preprocessingscript
import lstm_model_creator

def build_model(args,X_train, Y_train):
	print('Build model...')
	if not args.embedding:
		sequence = Input(shape=(args.maxlen,))
		dropout_0 = Embedding(args.max_features, 128, input_length=args.maxlen)(sequence)
		for i in xrange(args.num_hidden_layers):
			last_dropout_name = "dropout_%s" % i
			forward_name, backward_name, merged_name, dropout_name = ["%s_%s" % (k, i + 1) for k in ["forward", "backward","merged", "dropout"]]
			forward_name = LSTM(64)(last_dropout_name)
			backward_name = LSTM(64, go_backwards=True)(last_dropout_name)
			merged_name = merge([forward_name, backward_name], mode='concat', concat_axis=-1)
			dropout_name = Dropout(0.5)(merged_name)
		output = Dense(len(args.tagset), activation='softmax')(dropout_name)
		args.model = Model(input=sequence, output=output)
		if args.numberofclasses == 2:
			args.model.compile(loss='binary_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])
		else:
			args.model.compile(loss='categorical_crossentropy',
				  optimizer='rmsprop',
				  metrics=['accuracy'])
	else:
		# input_name = Input(shape=(args.maxlen, args.embedding_size))
		# forward_name = LSTM(128)(input_name)
		# backward_name = LSTM(128, go_backwards=True)(input_name)
		# merged_name = merge([forward_name, backward_name], mode='concat', concat_axis=-1)
		# dropout_name = Dropout(0.5)(merged_name)
		# output = Dense(len(args.tagset), activation='softmax')(dropout_name)
		# args.model = Model(input=input_name, output=output)
		args.model = lstm_model_creator.get_bilstm_model(
			maxlen=args.maxlen, 
			embedding_size=args.embedding_size, 
			lstm_output_size=args.lstm_output_size, 
			number_of_lstm_layers=args.number_of_lstm_layers, 
			numberofclasses=args.numberofclasses,
			dropout=args.dropout
			)

	print('Train...')
	args.model.fit(X_train, Y_train, batch_size=args.batch_size, nb_epoch=args.nb_epoch)#,
			  #validation_data=(X_test, Y_test))
	args.model.save(args.modelfile) 
	#score, acc = args.model.evaluate(X_test, Y_test, batch_size=args.batch_size)
	#print('Test score:', score)
	#print('Test accuracy:', acc)

def test_model(args):
	
	if not args.model:
		args.model = load_model(args.modelfile);
	# if args.numberofclasses == 2:
	# 	predict_test = args.model.predict(X_test, batch_size=args.batch_size, verbose=0)
	# 	#round the probabilities to 0 or 1
	# 	predict_test = np.round(predict_test)
	# else:
	# 	#predict_classes directly gives the index of maximum probability. predict() probabilities of each class in an array.
	# 	predict_test = args.model.predict(X_test, batch_size=args.batch_size, verbose=0)
	# 	predict_test = np.argmax(predict_test, axis=1)

	#preprocessingscript.postProcessingCreateTestCRFFeaturesFile(args, args.test_crfFeaturesFile, predict_test, args.output_crfFeaturesFile, args.numberofclasses)
	#directory
	preprocessingscript.postProcessing_triword_adding_pretrainedemb_and_features_dir(args, args.model)

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
	# args.include_char_ngram_hash =False
	parser.add_argument('--modeltype', type=str, default='bilstm',
							help='Type of model lstm, gru, bilstm, bigru.')
	parser.add_argument('--number_of_lstm_layers', type=int, default=1,
							help='lstm output size')
	parser.add_argument('--lstm_output_size', type=int, default=64,
							help='lstm output size')
	parser.add_argument('--dropout', type=float, default=0.01,
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

	parser.add_argument('--testconceptfield', type=int, default=28,
							help='concept (ner) column number in crf feature File.')
	parser.add_argument('--testtagfield', type=int, default=28,
							help='tag column number in crf feature File.')
	# args.conceptfield=11
	# args.tagfield=11

	#preprocess
	parser.add_argument('--word2vecFile', type=str, 
		default='../../finaldata/i2b22010_clef2013_clef2014_thyme_MIMICIII.txt_cleaned_glove.op_40.txt',
							help='ngram vocabulary file.')
	parser.add_argument('--train_crfFeaturesFile', type=str, 
		default='',
							help='ngram vocabulary file.')
	parser.add_argument('--test_crfFeaturesFile', type=str, 
		default='',
							help='ngram vocabulary file.')
	parser.add_argument('--output_crfFeaturesFile', type=str, 
		default='',
							help='ngram vocabulary file.')
	parser.add_argument('--test_crfFeatures_dir', type=str, 
		default='',
							help='ngram vocabulary file.')
	parser.add_argument('--output_crfFeatures_dir', type=str, 
		default='',
							help='ngram vocabulary file.')
	parser.add_argument('--modelfile', type=str, 
		default='models/bidirectional_lstm_ner_categorical.embedded.model.h5_oct14', help='save the model in the file')

	parser.add_argument('--posfile', type=str, default='../../finaldata/word2VecFiles/posTags.txt', help='pos tags')
	parser.add_argument('--chunkfile', type=str, default='../../finaldata/word2VecFiles/chunkTags.txt', help='chunk tags')
	# parser.add_argument('--capfil2', type=str, default='False', help='capitalized tags')
	parser.add_argument('--gazetterfile', type=str, default='False', help='is in gazetter')
	parser.add_argument('--suffile', type=str, default='../../finaldata/word2VecFiles/suffixes.txt', help='suffix')
	parser.add_argument('--prefile', type=str, default='../../finaldata/word2VecFiles/prefixes.txt', help='prefix')
	parser.add_argument('--ngramhashfile', type=str, default='False', help='is in ngramhashfile')
	#features
	# p.add_argument('-f', '--foo', action='store_true')
	parser.add_argument('--pos', action='store_true', help='pos tags')
	parser.add_argument('--chunk',action='store_true', help='chunk tags')
	parser.add_argument('--cap', action='store_true', help='capitalized tags')
	parser.add_argument('--gazetter', action='store_true', help='is in gazetter')
	parser.add_argument('--suf', action='store_true', help='suffix')
	parser.add_argument('--pre', action='store_true', help='prefix')

	parser.add_argument('--train', action='store_true')
	parser.add_argument('--test', action='store_true')
	parser.add_argument('--model', type=str, default='', help='model')

	parser.add_argument('--loadXYfrompickle', action='store_true')
	parser.add_argument('--XYpicklefile', type=str, default='models/thyme_colon_train_XY.pickle', help='is in ngramhashfile')
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
	preprocessingscript.readTags(args)
	args.numberofclasses=len(args.tagset)

	print ("numberofclasses: " ,args.numberofclasses)
	print ("args.embedding_size: ", args.embedding_size)
	print('Loading data...')
	#preprocessingscript.loadTrainAndTestFromCRFFeatures(args)
	if args.train:
		import pickle
		PIK = args.XYpicklefile
		if args.loadXYfrompickle:
			print('Loading from Pickle file!')
			with open(PIK, "rb") as f:
				[X_train, Y_train] = pickle.load(f)
		else:
			print('Creating XY from crf features files!')
			args.test = False
			X_train, Y_train = preprocessingscript.read_featuresfile_triword_adding_pretrainedemb_and_features(args, args.train_crfFeaturesFile)
			#save X_train and Y_train
			print('Dumping the Pickle file!')
			# data = [X_train, Y_train]
			# with open(PIK, "wb") as f:
			# 	pickle.dump(data, f)

		build_model(args, X_train, Y_train)
	test_model(args)

if __name__ == '__main__':
	main()
