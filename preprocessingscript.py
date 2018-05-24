import numpy as np
import re
import os
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

import itertools
import pos_features

global args

'''Read glove word2vector outputfile and store in the args.word2vec dict.
Each line is of the form: <word> <40 dim vector-space delimited>
'''
def readWord2Vec(args1):
	print 'Reading word2vecFile...'
	global args
	args = args1
	with open(args.word2vecFile) as f:
		lines = f.readlines();

	for line in lines:
		split = line.strip().split();
		# if(len(split)==args.pretrained_embedding_size+1):
		# 	#edit for np
		# 	args.word2vec[split[0]] = np.asarray(split[1:], dtype='float32');
		# 	array = []
		# 	array.append([int(x) for x in split[1:]])
		# 	args.word2vec[split[0]] = array
		args.word2vec[split[0]] = map(float,split[1:])
		args.word2index[split[0]] = len(args.word2index)+1;
		args.index2word[len(args.word2index)] = split[0];

	#vocab['<UK>']=1
	#edit for np
	#args.word2vec['<UK>'] = np.zeros(args.pretrained_embedding_size);
	args.word2vec['<UK>'] = [0] * args.pretrained_embedding_size;
	args.word2index['<UK>']=len(args.word2index)+1;
	args.index2word[len(args.word2index)]='<UK>'
	
	#vocab_size= #config.index2word

	print 'word2vec length: ',len(args.word2vec)
	print 'word2index length: ', len(args.word2index)
	print 'index2word length: ',len(args.index2word)

	# args.embedding_matrix = np.zeros((len(args.word2index) + 1, args.pretrained_embedding_size))
	# for word, i in args.word2index.items():
	# 	embedding_vector = args.word2vec.get(word)
	# 	if embedding_vector is not None:
	# 		# words not found in embedding index will be all-zeros.
	# 		args.embedding_matrix[i] = embedding_vector

	return;

'''Read train CRF features files and read the tags of conceptfield and store in args.tagset
'''
def readTags(args):
	print "Reading training features file for tags..."
	with open(args.train_crfFeaturesFile) as f:
		lines = f.readlines();

	args.alltokens=set()
	tags = set()
	for i in range(len(lines)):
		#print(i, lines[i]);
		line = lines[i].strip('\n');
		split = line.split('\t');
		#print "length", len(split)
		if (len(split)>1) and len(split)>args.tagfield:
			tags.add(split[args.tagfield]);

	args.tagset = list(tags)
	#args.tagset.append(args.tagset[0])
	#args.tagset[0]='<UK>'
	print 'WARNING: no args.tagset: <UK>'
	print ("number of tags: " , tags, len(tags))


def loadFeatures2Tokens(args):
	print 'Loading features to tokens...'
	args.embedding_size=0;
	if args.embedding:
		args.embedding_size = args.pretrained_embedding_size;
	if args.pos:
		with open(args.posfile) as f:
			lines = f.readlines();
			args.features2Tokens['pos'] = map(lambda s: s.strip(), lines)
			args.embedding_size+=len(lines)
			print('loading posfile:',len(args.features2Tokens['pos']))
	if args.chunk:
		with open(args.chunkfile) as f:
			lines = f.readlines();
			args.features2Tokens['chunk'] = map(lambda s: s.strip(), lines)
			args.embedding_size+=len(lines)
			print('loading chunkfile:',len(args.features2Tokens['chunk']))
	if args.pre:
		with open(args.prefile) as f:
			lines = f.readlines();
			args.features2Tokens['pre'] = map(lambda s: s.strip(), lines)
			args.embedding_size+=len(lines)
			print('loading prefile:',len(args.features2Tokens['pre']))
	if args.suf:
		with open(args.suffile) as f:
			lines = f.readlines();
			args.features2Tokens['suf'] = map(lambda s: s.strip(), lines)
			args.embedding_size+=len(lines)
			print('loading suffile:',len(args.features2Tokens['suf']))
	if args.gazetter:
		with open(args.gazetterfile) as f:
			lines = f.readlines();
			args.features2Tokens['gazetter'] = map(lambda s: s.strip(), lines)
			args.embedding_size+=1
			print('loading gazetterfile:',len(args.features2Tokens['gazetter']))
	#ngram does not go for embedding size. this is separate from that input
	#Here we only create ngram2index
	if args.include_char_ngram_hash:
		with open(args.ngramhashfile) as f:
			lines = f.readlines();
		all_ngrams = set()
		for line in lines:
			all_ngrams.add(line)	
		args.ngram2index=dict((k, v) for v,k in enumerate(all_ngrams)) 
		print('loading ngramfile:',len(args.ngram2index))
		all_ngrams.clear()
	# only for thyme
#	if args.tagfield$ != args.conceptfield:
#		args.embedding_size+=1;
#		print('WARNING: args.tagfield != args.conceptfield. This should be only for THYME corpus.')
	# print('INFO: args.tagfield != args.conceptfield. NOT THYME.')
	#word_features caps
	args.embedding_size+=4;

#Lipitor NNP	 B-NP	START-CAP	   X	   or	  X	   isInTrainDict   B-treatment
def readInputTensors (args, crffeaturesFile):
	with open(crffeaturesFile) as f:
		lines = f.readlines();

	print 'word2vec length: ',len(args.word2vec)
	
	empty_tensor = np.zeros(args.embedding_size);
	#print empty_tensor; 
	#x = np.ndarray(1,40)
	#mask = np.ones(len(x), dtype=bool);
	#mask[0:]=False
	#declare_tensor = x[mask]
	input_tensor = np.zeros(40);
	input_tensors= [];
	target_tensor = np.zeros(40)#array([1.0,0.]);
	target_tensors=[];

	i=0;
	for line in lines:
		split = line.split('\t');		
		#print "length", len(split)
		if (len(split)>1):
			word = split[0].lower();
			#print word
			if(word in args.word2vec):
				#print "Has Key", word
				input_tensor = np.vstack((input_tensor,args.word2vec[word]));
				print 'input_tensor shape: ',input_tensor.shape
			else:
				input_tensor = np.vstack((input_tensor,empty_tensor));
			i+=1;
			if (split[len(split)-1].find("-M")!= -1):
				target_tensor = np.append(target_tensor, 1);
				#target_tensor = np.vstack((target_tensor, 1));
				#print 'target',target_tensor.shape
			else:
				target_tensor = np.append(target_tensor, 0);
				#target_tensor = np.vstack((target_tensor, 0));
		else:
			if(len(input_tensor)>1):
				input_tensors.append(input_tensor);
				target_tensors.append(target_tensor);
				input_tensor = np.zeros(40);
				target_tensor = np.zeros(1);
			i=0;
	#print input_tensors
	return input_tensors, target_tensors

def loadTrainAndTestFromCRFFeaturesSeqToSeq(args):
	args.train_input_tensors={}
	args.train_target_tensors={}
	args.test_input_tensors={}
	args.test_target_tensors={}

	args.train_input_tensors, args.train_target_tensors = readInputTensors(args, args.train_crfFeaturesFile)
	args.test_input_tensors, args.test_target_tensors = readInputTensors(args, args.test_crfFeaturesFile)

	print 'train_input_tensors length: ',args.train_input_tensors.shape
	print 'train_target_tensors length: ',args.train_target_tensors.shape
	print 'test_input_tensors length: ',args.test_input_tensors.shape
	print 'test_target_tensors length: ',args.test_target_tensors.shape


def readTensorsWithIndexesSeqToSeq (args, crffeaturesFile):
	with open(crffeaturesFile) as f:
		lines = f.readlines();

	print len(args.word2vec)
	
	empty_tensor = np.zeros(args.embedding_size);
	#print empty_tensor; 
	#x = np.ndarray(1,40)
	#mask = np.ones(len(x), dtype=bool);
	#mask[0:]=False
	#declare_tensor = x[mask]
	input_tensor = np.zeros(40);
	input_tensors= [];
	target_tensor = np.asarray(40);
	target_tensors=[];

	i=0;
	for line in lines:
		split = line.split('\t');		
		#print "length", len(split)
		if (len(split)>1):
			word = split[0].lower();
			#print word
			if(len(input_tensor)==39):
				input_tensors.append(input_tensor);
				target_tensors.append(target_tensor);
				input_tensor = np.zeros(40);
				target_tensor = np.asarray([1,0])
				i=0;

			if(word in args.word2vec):
				#print "Has Key", word
				input_tensor = np.append(input_tensor,args.word2index[word]);
				#print 'input_tensor',input_tensor.shape
			else:
				input_tensor = np.append(input_tensor, args.word2index['<UK>']);
			i+=1;
			if (split[len(split)-1] == 'O'):
				target_tensor = np.append(target_tensor, [1,0]);
				#target_tensor = np.vstack((target_tensor, 1));
				#print 'target',target_tensor.shape
			else:
				target_tensor = np.append(target_tensor, [0,1]);
				#target_tensor = np.vstack((target_tensor, 0));
		else:
			if(len(input_tensor)>1):
				input_tensors.append(input_tensor);
				target_tensors.append(target_tensor);
				input_tensor = np.zeros(40);
				target_tensor = np.asarray([0,1])
				i=0;
	#print input_tensors
	return input_tensors, target_tensors

def readTensorsWithIndexesSeqToSeqSimplified (args, crffeaturesFile):
	#raw = open('wikigold.conll.txt', 'r').readlines()
	raw = open(crffeaturesFile, 'r').readlines()
	all_x = []
	point = []
	for line in raw:
		stripped_line = line.strip().split('\t')
		point.append(stripped_line)
		if line == '\n':
			all_x.append(point[:-1])
			point = []
	# all_x = all_x[:-1] #[[['Admission', 'Admission', 'NN', 'B-NP', 'X', 'X', 'X', 'ion', 'X', 'X', 'X', 'X', 'X', 'O'],
 #  ['Date', 'Date', 'NNP', 'I-NP', 'X', 'START-CAP', 'X', 'ate', 'X', 'X', 'ALPHANUMERIC', 'X', 'X', 'O'],
 #  [':', ':', ':', 'O', 'X', 'X', 'X', 'X', 'X', 'NUMERIC', 'ALPHANUMERIC', 'X', 'X', 'O']],
 # [['2012-10-31', '2012-10-31', 'CD', 'O', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'O']],
 # [['Date', 'Date', 'NNP', 'B-NP', 'X', 'START-CAP', 'X', 'ate', 'X', 'X', 'ALPHANUMERIC', 'X', 'X', 'O'],
 #  ['of', 'of', 'IN', 'B-PP', 'STOPWORD', 'X', 'X', 'X', 'of', 'X', 'ALPHANUMERIC', 'X', 'X', 'O'],
 #  ['Birth', 'Birth', 'NNP', 'B-NP', 'X', 'START-CAP', 'X', 'X', 'bi', 'X', 'ALPHANUMERIC', 'X', 'X', 'O'],
 #  [':', ':', ':', 'O', 'X', 'X', 'X', 'X', 'X', 'NUMERIC', 'ALPHANUMERIC', 'X', 'X', 'O']],
 # [['1941-03-23', '1941-03-23', 'CD', 'O', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'O']]]

 # [allSentences, [sentence, [line]]
	lengths = [len(x) for x in all_x] # [3, 1, 3, 1]
	print "all lengths:", lengths
	#short_x = [x for x in all_x if len(x) < 64] # only lengths which are < 64

	X = [[c[0] for c in x] for x in all_x] #words sequences
	Y = [[c[13] for c in x] for x in all_x] #tags sequeces  [['B-treatment', 'I-treatment', 'O', 'B-treatment']]

	#all_text = [c for x in X for c in x] # all words in a single list
	#words = list(set(all_text)) # removing duplicates
	#word2ind = {word: index for index, word in enumerate(words)} #word:index dict
	#ind2word = {index: word for index, word in enumerate(words)} #index:word dict
	#labels = list(set([c for x in y for c in x])) #unique labels
	#label2ind = {label: (index + 1) for index, label in enumerate(labels)} #label:labelindex
	#ind2label = {(index + 1): label for index, label in enumerate(labels)} #labelindex:label
	#print 'Input sequence length range: ', max(lengths), min(lengths)
	#maxlen = max([len(x) for x in X])
	#print 'Maximum sequence length:', maxlen
	# def encode(x, n):
	#	 result = np.zeros(n)
	#	 result[x] = 1
	#	 return result

	# print [[args.word2index[word] for word in x if word in args.word2index ] for x in X]
	#max_label = len(args.tagset) #number of labels + 1 
	#Y_enc = [[0] * (args.maxlen - len(y)) + [args.tagset.index(tag) for tag in y] for y in Y] #padd each sentence with max len
	#Y_enc = [[encode(c, max_label) for c in ey] for ey in Y_enc] # to categorical

	X_enc = [[args.word2index[word] if word in args.word2index else args.word2index['<UK>'] for word in x] for x in X] #word indexes
	Y_enc = [[args.tagset.index(tag) for tag in y] for y in Y]
	Y_enc = [to_categorical(np.asarray(y), len(args.tagset)) for y in Y_enc]
	X = pad_sequences(X_enc, maxlen=args.maxlen)
	Y = pad_sequences(Y_enc, maxlen=args.maxlen)

	return np.asarray(X).astype(dtype='int32'), np.asarray(Y).astype(dtype='int32')


def readTrainAndTestTensorsWithIndexesSeqToSeq(args):
	args.train_input_tensors={}
	args.train_target_tensors={}
	args.test_input_tensors={}
	args.test_target_tensors={}

	args.train_input_tensors, args.train_target_tensors = readTensorsWithIndexesSeqToSeqSimplified(args, args.train_crfFeaturesFile)
	args.test_input_tensors, args.test_target_tensors = readTensorsWithIndexesSeqToSeqSimplified(args, args.test_crfFeaturesFile)

	print 'train_input_tensors length: ',len(args.train_input_tensors), args.train_input_tensors.shape
	print 'train_target_tensors length: ',len(args.train_target_tensors), args.train_target_tensors.shape
	print 'test_input_tensors length: ',len(args.test_input_tensors), args.test_input_tensors.shape
	print 'test_target_tensors length: ',len(args.test_target_tensors), args.test_target_tensors.shape
	return args.train_input_tensors, args.train_target_tensors, args.test_input_tensors, args.test_target_tensors

def getNeighbourWords(lines, index, number):
	list = []
	split = lines[index].strip('\n').split('\t');		
	#print "length", len(split)
	if (len(split)<=1):
		for j in range(number+number+1):
			list.append('<UK>')
		return list

	flag_pre = False
	for j in reversed(range(number)):
		#print index-number+j
		if(index-number+j >= 0 and flag_pre == False) :
			split = lines[index-number+j].strip('\n').split('\t');		
			if (len(split)>1):
				word = split[0];
				list.append(word)
			else:
				list.append('<UK>')
				flag_pre = True
		else:
			flag_pre = True
			list.append('<UK>')

	list.reverse()

	flag_post = False
	for j in range(number+1):
		#print index+j
		if(index+j < len(lines) and flag_post == False):
			split = lines[index+j].strip('\n').split('\t');
			if (len(split)>1):
				word = split[0];
				list.append(word)
			else:
				flag_post = True;
				list.append('<UK>')
		else:
			flag_post = True;
			list.append('<UK>')

	#print list
	return list

#get neighbour lines in the crffeaturelines.
def getNeighbourLinesInFeatureFile(lines, index, number):
	list = []
	split = lines[index].strip('\n').split('\t');		
	# print "length", len(split)
	split_length = len(split)
	if (len(split)<=1):
		return list

	flag_pre = False
	for j in reversed(range(number)):
		#print index-number+j
		if(index-number+j >= 0 and flag_pre == False) :
			split = lines[index-number+j].strip('\n').split('\t');		
			if (len(split)>1):
				#word = split[0];
				list.append(lines[index-number+j])
			else:
				list.append(('<UK>\t'*split_length).strip('\t'))
				flag_pre = True
		else:
			flag_pre = True
			list.append(('<UK>\t'*split_length).strip('\t'))

	list.reverse()

	flag_post = False
	for j in range(number+1):
		#print index+j
		if(index+j < len(lines) and flag_post == False):
			split = lines[index+j].strip('\n').split('\t');
			if (len(split)>1):
				# word = split[0];
				list.append(lines[index+j])
			else:
				flag_post = True;
				list.append(('<UK>\t'*split_length).strip('\t'))
		else:
			flag_post = True;
			list.append(('<UK>\t'*split_length).strip('\t'))

	#print list
	return list


def get_char_ngrams(word,n):
	word = '#'+word+'#';
	return [word[i:i+n] for i in range(len(word)-n+1)]

#this only works for embeddings. Not pretrained
def readInputTensorsForEmbeddingTriWordToLabel (args, crffeaturesFile):
	with open(crffeaturesFile) as f:
		lines = f.readlines();

	print 'word2vec length: ',len(args.word2vec)
	
	empty_tensor = np.zeros(args.embedding_size);
	#print empty_tensor;

	
	X=[]# np.zeros(11);#np.empty(shape=(0,0));] for whole dataset
	Y=[]#np.zeros(1)
	X_char = []
	X_char_ngram = []

	#i=0;
	for i in range(len(lines)):
		window_words = getNeighbourWords(lines, i, args.maxlen/2);

		#for each sequence extract indexes
		x=[]
		y=[]
		x_char=[]
		x_char_ngram=[]

		#print(i, lines[i]);
		line = lines[i].strip('\n');
		split = line.split('\t');
		#print "length", len(split)
		if (len(split)>1):
			for word in window_words:
				word_lower = word.lower()
				
				if(word_lower in args.word2vec):
					#print "Has Key", word
					x.append(args.word2index[word_lower])#x = np.append(x,args.word2index[word]);
					#print 'x',x.shape
				else:
					x.append(args.word2index['<UK>'])#x = np.append(x, args.word2index['<UK>']);
				#for char embeddings of each word
				if args.include_chars:
					char_seq = [1 + args.char2index.get(k, -1) for k in word] # 0 for OOV char
					x_char.append((1 + np.asarray(char_seq)).tolist())
					if len(char_seq) == 0:
						x_char.append(1)
				if args.include_char_ngram_hash:
					char_ngrams = get_char_ngrams(word_lower, args.num_of_ngram)
					char_ngram_indexes = [1 + args.ngram2index.get(k, -1) for k in char_ngrams]
					x_char_ngram.append((1 + np.asarray(char_ngram_indexes)).tolist())

			y.append(args.tagset.index(split[args.tagfield]))
			
			#append the sequence to matrix
			X.append(x)#input_tensors = np.vstack((input_tensors, x))
			if args.include_chars:
				padded_sequence = sequence.pad_sequences([[] for k in xrange(args.maxlen - len(x_char))], maxlen=args.max_charlen).tolist() +\
						sequence.pad_sequences(x_char[:args.maxlen], maxlen=args.max_charlen).tolist()
				X_char.append(padded_sequence)
			if args.include_char_ngram_hash:
				padded_sequence = sequence.pad_sequences([[] for k in xrange(args.maxlen - len(x_char))], maxlen=args.max_charlen).tolist() +\
						sequence.pad_sequences(x_char_ngram[:args.maxlen], maxlen=args.max_charlen).tolist()
				X_char_ngram.append(padded_sequence)
			Y.append(y)#target_tensors = np.append(target_tensors, target_tensor)
	
	#print 'input_tensors: ',X.shape
	#print 'target_tensors: ',Y.shape

	if args.include_chars:
		return np.asarray(X).astype(dtype='int32'), np.asarray(Y).astype(dtype='int32'),\
		np.asarray(X_char).astype(dtype='int32')
	if args.include_chars and args.include_char_ngram_hash:
		return np.asarray(X).astype(dtype='int32'), np.asarray(Y).astype(dtype='int32'),\
		np.asarray(X_char).astype(dtype='int32'), np.asarray(X_char_ngram).astype(dtype='int32')
	return np.asarray(X).astype(dtype='int32'), np.asarray(Y).astype(dtype='int32')


def preScanHitachiTriWordToLabel(args):
	args.X_train=[]
	args.Y_train=[]
	args.X_test=[]
	args.Y_test=[]

	args.X_train, args.Y_train = readInputTensorsForEmbeddingTriWordToLabel(args, args.train_crfFeaturesFile)
	args.X_test, args.Y_test = readInputTensorsForEmbeddingTriWordToLabel(args, args.test_crfFeaturesFile)

	print 'X_train length: ',args.X_train.shape
	print 'Y_train length: ',args.Y_train.shape
	print 'X_test length: ',args.X_test.shape
	print 'Y_test length: ',args.Y_test.shape
	return args.X_train, args.Y_train, args.X_test, args.Y_test

def preScanHitachiTriWordToLabel_includechar(args):
	args.X_train=[]
	args.Y_train=[]
	args.X_test=[]
	args.Y_test=[]

	args.X_train, args.Y_train, args.X_char_train = readInputTensorsForEmbeddingTriWordToLabel(args, args.train_crfFeaturesFile)
	args.X_test, args.Y_test, args.X_char_test = readInputTensorsForEmbeddingTriWordToLabel(args, args.test_crfFeaturesFile)

	print 'X_train length: ',args.X_train.shape
	print 'Y_train length: ',args.Y_train.shape
	print 'X_test length: ',args.X_test.shape
	print 'Y_test length: ',args.Y_test.shape
	return args.X_train, args.Y_train, args.X_test, args.Y_test, args.X_char_train, args.X_char_test


#this only works for embeddings.
def read_featuresfile_triword_adding_pretrainedemb_and_features (args, crffeaturesFile):
	with open(crffeaturesFile) as f:
		lines = f.readlines();

	print 'Reading features file: word2vec length: ',len(args.word2vec)

	X=[]# np.zeros(11);#np.empty(shape=(0,0));] for whole dataset
	Y=[]#np.zeros(1)
	X_char = []
	X_char_ngram = []

	#i=0;
	for i in range(len(lines)):
		window_lines = getNeighbourLinesInFeatureFile(lines, i, args.maxlen/2);

		if not window_lines:
			continue;
		#for each sequence extract indexes
		x=[]
		y=[]
		x_char=[]
		x_char_ngram=[]


		#print(i, lines[i]);
		line = lines[i].strip('\n');
		split = line.split('\t');
		#print "length", len(split)
		if (len(split)>2):
			for window_line in window_lines:
				line_list = window_line.split('\t');
				#print "length line_list ", len(line_list)
				if(line_list[0].lower() in args.word2vec):
					#print "Has Key", word
					# x.append(args.word2index[line_list[0].lower()])#x = np.append(x,args.word2index[word]);
					#print 'x',x.shape
					
					x_line=[];
					# print('b4 appending: ', len(x_line))
					#print('In get_relntype_vectortensor: search in word2vec: word:',line_list[0])
					if args.embedding:
						if line_list[0].lower() in args.word2vec:
							#edit for np
							#x_line.append(args.word2vec[line_list[0].lower()].tolist())
							x_line.append(args.word2vec[line_list[0].lower()])
						else:
							#edit for np
							#x_line.append(args.word2vec['<UK>'].tolist())
							x_line.append(args.word2vec['<UK>'])
					x_line.append(pos_features.word_features(line_list[0]))
					if args.pos:
						# print 'pos: ',args.pos,  args.features2Tokens['pos'].index(line_list[2])
						x_line.append(pos_features.get_pos_features(line_list[2],args.features2Tokens['pos']))
					if args.chunk:
						x_line.append(pos_features.get_chunk_features(line_list[3],args.features2Tokens['chunk']))
					if args.pre:
						x_line.append(pos_features.get_pre_features(line_list[0],args.features2Tokens['pre']))
					if args.suf:
						x_line.append(pos_features.get_suf_features(line_list[0],args.features2Tokens['suf']))
					if args.gazetter:
						x_line.append(pos_features.get_gazetter_features(line_list[0],args.features2Tokens['gazetter']))
					if args.tagfield != args.conceptfield:
						if args.train:
							if split[args.conceptfield] != 'O':
								x_line.append([1])
							else:
								x_line.append([0])
						else: #test
							#print 'split: ', split
							if split[args.testconceptfield] != 'O':
								x_line.append([1])
							else:
								x_line.append([0])
					# print('Each line x_line: ', len([val for sublist in x_line for val in sublist]))
					x.append([val for sublist in x_line for val in sublist])
					# print 'inside x: ',len([val for sublist in x_line for val in sublist])
					# print [val for sublist in x_line for val in sublist]
				else:
					x.append([0]*args.embedding_size)#x = np.append(x, args.word2index['<UK>']);
				
			#append the sequence to matrix
			# print 'x:',len(x)
			X.append(x)#input_tensors = np.vstack((input_tensors, x))

			# print 'split length: ', len(split), split
			# print 'split[args.tagfield]: ' , split[args.tagfield]
			y.append(args.tagset.index(split[args.tagfield]))
			Y.append(y)#target_tensors = np.append(target_tensors, target_tensor)
	print 'X',len(X)
	# print 'input_tensors: ',X.shape
	# print 'target_tensors: ',Y.shape
	print 'include chars: ',args.include_chars, args.include_char_ngram_hash 
	print 'Iterating train features file completed!'
	#X = sequence.pad_sequences(np.asarray(X))#, maxlen=args.maxlen)
	#X = np.asarray(X)
	print 'X padded sequence'
	if not args.test:
		Y = to_categorical(np.asarray(Y), nb_classes=args.numberofclasses)
	#Y = sequence.pad_sequences(Y, maxlen=args.numberofclasses)
	print 'Y padded sequence'
	
	if args.include_chars and args.include_char_ngram_hash:
		#X_char = np.asarray(X_char).astype(dtype='int32')
		#X_char_ngram = np.asarray(X_char_ngram).astype(dtype='int32')
		return X, Y, X_char, X_char_ngram 
	if args.include_chars:
		#X_char = np.asarray(X_char).astype(dtype='int32')
		return X, Y, X_char
	else:
		return X , Y




def read_trainandtest_triword_adding_pretrainedemb_and_features(args):
	args.X_train=[]
	args.Y_train=[]
	args.X_test=[]
	args.Y_test=[]

	args.X_train, args.Y_train = read_featuresfile_triword_adding_pretrainedemb_and_features(args, args.train_crfFeaturesFile)
	args.X_test, args.Y_test = read_featuresfile_triword_adding_pretrainedemb_and_features(args, args.test_crfFeaturesFile)
	
	# print('X_train shape:', args.X_train.shape, args.Y_train.shape)
	# print('X_test shape:', args.X_test.shape, args.Y_test.shape)

	# print('Pad sequences (samples x time)')
	# args.X_train = sequence.pad_sequences(args.X_train, maxlen=args.maxlen)
	# args.X_test = sequence.pad_sequences(args.X_test, maxlen=args.maxlen)

	# args.Y_train = to_categorical(np.asarray(args.Y_train))
	# args.Y_train = sequence.pad_sequences(args.Y_train, maxlen=args.numberofclasses)
	# args.Y_test = to_categorical(np.asarray(args.Y_test))
	# args.Y_test = sequence.pad_sequences(args.Y_test, maxlen=args.numberofclasses)

	# print(X_train)
	print('X_train shape:', args.X_train.shape, args.Y_train.shape)
	print('X_test shape:', args.X_test.shape, args.Y_test.shape)

	return args.X_train, args.Y_train, args.X_test, args.Y_test


def postProcessingCreateTestCRFFeaturesFile(args,testCRFFeaturesFile, predicted_classes,output_crfFeaturesFile, numberofclasses):
	with open(testCRFFeaturesFile) as f:
		lines = f.readlines();

	print('test lines:', testCRFFeaturesFile, len(lines))
	print('predicted_classes: ' ,predicted_classes.shape)
	print('outputfile: ', output_crfFeaturesFile)
	outputfile = open(output_crfFeaturesFile, 'w')
	i=0
	for line in lines:
		split = line.split('\t');
		# print "length", line , len(split), 
		# print "predicted_classes[i]:", predicted_classes[i]
		if (len(split)>1):
			tag = args.tagset[predicted_classes[i]]
			outputfile.write("{}\t{}\n".format(line.strip('\n'),tag))
			i+=1;
		else:
			outputfile.write("\n")

def postProcessing_createTestCRFFeaturesFile_dir(args, model):
	listFiles = os.listdir(args.test_crfFeatures_dir)
	if not os.path.exists(args.output_crfFeatures_dir):
		os.makedirs(args.output_crfFeatures_dir)

	for file in listFiles:
		X_test, y_test = readInputTensorsForEmbeddingTriWordToLabel(args,os.path.join(args.test_crfFeatures_dir, file))
		if args.embedding:
			X_test = sequence.pad_sequences(X_test, maxlen=args.maxlen)
		#print 'testing file:', file, ', test shape:',X_test.shape,y_test.shape
		X_test = sequence.pad_sequences(X_test, maxlen=args.maxlen)
		#print 'After padding testing file:', file, ', test shape:',X_test.shape,y_test.shape

		if args.numberofclasses == 2:
			predict_test = model.predict(X_test, batch_size=32, verbose=0)
			#round the probabilities to 0 or 1
			predict_test = np.round(predict_test)
		else:
			predict_test = model.predict(X_test, batch_size=32, verbose=0)
			predict_test = np.argmax(predict_test, axis=1)

		postProcessingCreateTestCRFFeaturesFile(args,os.path.join(args.test_crfFeatures_dir, file), predict_test, os.path.join(args.output_crfFeatures_dir, file), args.numberofclasses)

def postProcessing_triword_adding_pretrainedemb_and_features_dir(args, model):
	listFiles = os.listdir(args.test_crfFeatures_dir)
	if not os.path.exists(args.output_crfFeatures_dir):
		os.makedirs(args.output_crfFeatures_dir)
	args.train = False
	args.test = True
	for file in listFiles:
		X_test, y_test = read_featuresfile_triword_adding_pretrainedemb_and_features(args,os.path.join(args.test_crfFeatures_dir, file))
		
		if args.numberofclasses == 2:
			predict_test = model.predict(X_test, batch_size=32, verbose=0)
			#round the probabilities to 0 or 1
			predict_test = np.round(predict_test)
		else:
			predict_test = model.predict(X_test, batch_size=32, verbose=0)
			predict_test = np.argmax(predict_test, axis=1)

		postProcessingCreateTestCRFFeaturesFile(args,os.path.join(args.test_crfFeatures_dir, file), predict_test, os.path.join(args.output_crfFeatures_dir, file), args.numberofclasses)

def postProcessing_triword_adding_pretrainedemb_and_features_dir_SeqToSeq(args, model):
	listFiles = os.listdir(args.test_crfFeatures_dir)
	if not os.path.exists(args.output_crfFeatures_dir):
		os.makedirs(args.output_crfFeatures_dir)
	args.train = False
	args.test = True
	for file in listFiles:
		X_test, y_test = readTensorsWithIndexesSeqToSeqSimplified(args,os.path.join(args.test_crfFeatures_dir, file))
		
		if args.numberofclasses == 2:
			predict_test = model.predict(X_test, batch_size=32, verbose=0)
			#round the probabilities to 0 or 1
			predict_test = np.round(predict_test)
		else:
			predict_test = model.predict(X_test, batch_size=32, verbose=0)
			# print 'predict_test1: predict:',predict_test, predict_test.shape
			predict_test = predict_test.argmax(2)

		print 'X_test, Y_test, predict_test2 after argmax:', X_test.shape, y_test.shape, predict_test.shape
		# predict_test = np.argmax(predict_test, axis=1)
		print 'predict_test3:', predict_test
		postProcessingCreateTestCRFFeaturesFile_SeqToSeq(args,os.path.join(args.test_crfFeatures_dir, file), predict_test, os.path.join(args.output_crfFeatures_dir, file), args.numberofclasses)

def postProcessingCreateTestCRFFeaturesFile_SeqToSeq(args,testCRFFeaturesFile, predicted_classes,output_crfFeaturesFile, numberofclasses):
	with open(testCRFFeaturesFile) as f:
		lines = f.readlines();

	print('test lines:', testCRFFeaturesFile, len(lines))
	print('predicted_classes: ' ,predicted_classes.shape)
	print('outputfile: ', output_crfFeaturesFile)
	outputfile = open(output_crfFeaturesFile, 'w')
	i=0;
	j=0
	for line in lines:
		split = line.split('\t');
		# print "length", line.rstrip() , len(split), 
		# print 'i,j:', i,j
		# print "predicted_classes[i]:", predicted_classes[i][j]
		if (len(split)>1) and j<args.maxlen:
			tag = args.tagset[predicted_classes[i][j]]
			outputfile.write("{}\t{}\n".format(line.strip('\n'),tag))
			j+=1
		elif (len(split)>1):
			outputfile.write("{}\t{}\n".format(line.strip('\n'),'O'))
		else:
			i+=1;
			j=0
			outputfile.write("\n")

if __name__ == '__main__':
	readWord2Vec("sampleword2vec.txt");
	readInputTensors("sampleCrfFeatures1.txt");
