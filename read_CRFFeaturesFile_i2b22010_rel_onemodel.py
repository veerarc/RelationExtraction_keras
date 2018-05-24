import numpy as np
import re
import os
from keras.models import load_model
import itertools
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical

import pos_features
# Mesenteric	Mesenter	NNP	B-NP	X	START-CAP	X	ic	X	X	ALPHANUMERIC	X	X	B-test	TeCP	0
# angiograpm	angiograpm	IN	B-PP	X	X	X	X	ang	X	ALPHANUMERIC	X	X	O	0	0
# w/	w/	DT	B-NP	X	X	X	X	X	X	X	X	X	O	0	0
# coil	coil	NN	I-NP	X	X	X	X	co	X	ALPHANUMERIC	X	X	B-treatment	0	0
# embolization	embol	NN	I-NP	X	X	X	ion	em	X	ALPHANUMERIC	X	X	O	0	0
# of	of	IN	B-PP	STOPWORD	X	X	X	of	X	ALPHANUMERIC	X	X	O	0	0
# bleeding	bleed	VBG	B-NP	X	X	X	ing	X	X	ALPHANUMERIC	X	X	B-problem	TeCP	0
# vessel	vessel	NN	I-NP	X	X	X	X	X	X	ALPHANUMERIC	X	X	O	0	0
# .	.	.	O	X	X	X	X	X	NUMERIC	ALPHANUMERIC	X	X	O	0	0

def readTensorsForEachRelationFromCRFFeaturesFile (args, crffeaturesFile):
	
	with open(crffeaturesFile) as f:
		lines = f.readlines();

	X =[]
	X_char =[]
	X_char_ngram =[]
	Y =[]
	X_sentence =[]

	all_pairs_size = 0;
	gold_pairs_size = 0;
	#go through all lines of crf features file
	i=0;
	while i < len(lines):
		#print lines[i]
		#sentence_tensor contains the lines of single sentence.
		sentence_tensor = []
		#search for the sentence end.
		while i < len(lines):
			#break when newline. That is at sentence end.
			if not lines[i].strip():
				break;
			line = lines[i].rstrip();
			split = line.split('\t');
			sentence_tensor.append(split);
			i+=1

		#if sentence tensor is empty. Continue for next sentence. This happens when when two consecutive newlines occur.
		if not sentence_tensor:
			i+=1;
			continue;

		#total treatment pairs
		all_pairs = get_all_index_pairs(np.asarray(sentence_tensor)[:,args.conceptfield]);
		all_pairs_size += len(all_pairs)
		gold_relation_pairs = []
		#Range from 14th column to last column of the sentence crf feature format.
		for j in range(args.tagfield, len(sentence_tensor[0])):
			#print('sentence_tensor length: ', len(sentence_tensor))
			
			sentence_tensor_nparray = np.asarray(sentence_tensor)
			# rel_type, x = get_relntype_vectortensor(args, myarray, j)

			rel_type, index1, index2 = get_indexes_of_relns(sentence_tensor_nparray[:,args.conceptfield],sentence_tensor_nparray[:,j]);

			#all relationships of treatment Tr
			if rel_type.startswith('T') or rel_type.startswith('P'): 
				gold_relation_pairs.append([index1, index2]);
				gold_pairs_size += 1
				#get features with indexes included
				if args.include_chars and args.include_char_ngram_hash:
					x, x_char, x_char_ngram = get_test_features_for_sentencetensor(args,sentence_tensor, index1, index2)
					#print 'rel_type:',rel_type, ', x:', x
					if x:
						if rel_type in args.tagset:
							X.append(x)
							X_char.append(sequence.pad_sequences([[] for k in xrange(args.maxlen - len(x_char))], maxlen=args.max_charlen ).tolist() +\
										sequence.pad_sequences(x_char[:args.maxlen], maxlen=args.max_charlen ).tolist())
							X_char_ngram.append(sequence.pad_sequences([[] for k in xrange(args.maxlen - len(x_char))], maxlen=args.max_charlen ).tolist() +\
										sequence.pad_sequences(x_char_ngram[:args.maxlen], maxlen=args.max_charlen ).tolist())
							Y.append(args.tagset.index(rel_type))
						else:
							print ('tagset doesnot contain: ', rel_type)
				elif args.include_chars:
					x, x_char = get_test_features_for_sentencetensor(args,sentence_tensor, index1, index2)
					#print 'rel_type:',rel_type, ', x:', x
					if x:
						if rel_type in args.tagset:
							X.append(x)
							X_char.append(sequence.pad_sequences([[] for k in xrange(args.maxlen - len(x_char))], maxlen=args.max_charlen ).tolist() +\
										sequence.pad_sequences(x_char[:args.maxlen], maxlen=args.max_charlen ).tolist())
							#print x_char
							Y.append(args.tagset.index(rel_type))
						else:
							print ('tagset doesnot contain: ', rel_type)
				else:
					x = get_test_features_for_sentencetensor(args,sentence_tensor, index1, index2)
					#print 'rel_type:',rel_type, ', x:', x
					if x:
						if rel_type in args.tagset:
							X.append(x)
							Y.append(args.tagset.index(rel_type))
							#double up the samples other than TrAP
							# if rel_type.startswith('Tr') and rel_type != 'TrAP':
							# 	X.append(x)
							# 	Y.append(args.tagset.index(rel_type))
							#append sentence level features
							if args.sentencelevelfeatures:
								X_sentence.append(get_sentencelevel_features_for_sentencetensor(args,sentence_tensor, index1, index2))
						else:
							print ('tagset doesnot contain: ', rel_type)



		#null pairs of treatment remaining for balancing the classes
		for index1, index2 in all_pairs:
			#if the pair is not having treatment relation
			if [index1, index2] not in gold_relation_pairs:
				x = get_test_features_for_sentencetensor(args,sentence_tensor, index1, index2)
					#print 'rel_type:',rel_type, ', x:', x
				if x:
					if len(X) < 15000:
						X.append(x)
						Y.append(args.tagset.index('0'))
						if args.sentencelevelfeatures:
								X_sentence.append(get_sentencelevel_features_for_sentencetensor(args,sentence_tensor, index1, index2))
					
	print len(X),np.array(X).shape
	
	X = sequence.pad_sequences(np.array(X), maxlen=args.maxlen)
	Y = to_categorical(np.asarray(Y), len(args.tagset))
	Y = sequence.pad_sequences(Y, maxlen=args.numberofclasses )

	if args.include_chars and args.include_char_ngram_hash:
		X_char = np.array(X_char)
		X_char_ngram = np.array(X_char_ngram)
		return X, Y, X_char, X_char_ngram 
	if args.include_chars:
		X_char = np.array(X_char)
		print 'X, Y, X_char shape: ', X.shape, Y.shape, X_char.shape
		return X, Y, X_char

	if args.sentencelevelfeatures:
		X_sentence = sequence.pad_sequences(np.array(X_sentence), maxlen=args.maxlen )
		print 'X, Y, X_char shape: ', X.shape, Y.shape, X_sentence
		return X, Y, X_sentence		
	print 'X, Y 	shape: ', X.shape, Y.shape
	print "tr pairs size: ", all_pairs_size, gold_pairs_size
	return X , Y

		
#get relationship_type and feature vector of the sentence.
# def get_relntype_vectortensor(args,sentence_tensor, j):
# 	#feature vector contains sequence with features of each word.
# 	x_sentence = []
# 	print('column index of relation in the line: ',j)
# 	relns_column = sentence_tensor[:,j];
# 	#TrAP 3 6
# 	rel_type, index1, index2 = get_indexes_of_relns(sentence_tensor[:,14], relns_column);
# 	print 'Return of get_indexes_of_relns: ', rel_type, index1, index2
# 	# Mesenteric	Mesenter	NNP	B-NP	X	START-CAP	X	ic	X	X	ALPHANUMERIC	X	X	B-test	TeCP	0
# 	get_test_features_for_sentencetensor(args,sentence_tensor, i, j)

# 	for i in range(len(sentence_tensor)):
# 		line_list = sentence_tensor[i]; 
# 		x_line=[];
# 		#print('b4 appending: ', len(x_line))
# 		#print('In get_relntype_vectortensor: search in word2vec: word:',line_list[0])
# 		if line_list[0].lower() in args.word2vec:
# 			x_line.append(args.word2vec[line_list[0].lower()].tolist())
# 		else:
# 			x_line.append(args.word2vec['<UK>'].tolist())
# 		#print('Each line x_line: ', len(x_line))
# 		x_line.append([i-index1])
# 		x_line.append([i-index2])
# 		#line_tensor.append(args.pos2vec(line_tensor[2]))
# 		#line_tensor.append(args.chunk2vec(line_tensor[3]))
# 		#line_tensor.append(args.nertag2vec(line_tensor[13]))
# 		x_sentence.append([val for sublist in x_line for val in sublist])

# 	return rel_type, x_sentence

#get relationship type and indexes of the relation entities for that relation column of the crf sentence
def get_indexes_of_relns(con_column, relns_column):
	# print 'get_indexes_of_relns: relations column: ', relns_column
	indexes = []
	rel_type=''
	index1=0
	index2=0
	# 0 0 0 TrAP 0 0 TrAP 0 
	for i,val in enumerate(relns_column):
		if val != '0' and val != 0 and val != '0\n':
			# print "value of relns_column: ",val
			rel_type = val;
			indexes.append(i);
			# if not rel_type.startswith('Tr'):
			# 	rel_type = '0'

	if len(indexes) > 1:
		if con_column[indexes[0]] == 'B-problem':
			index1 = indexes[0]
			index2 = indexes[1]
		else:
			index1 = indexes[1]
			index2 = indexes[0]

	return rel_type, index1, index2


def getTrainFromRelnDictionary(reln_features_dictionary,rel_type):
	X_train = reln_features_dictionary.get(rel_type);
	Y_train = []
	Y_train.append([1] * len(X_train))
	print ('len of rel_type samples: ' , len(X_train))
	for k,v in reln_features_dictionary.items():
		if rel_type != k:
			print ('len of other reln_types: ' , len(v))
			#append each sample of the reln_types
			for samples in v:
				X_train.append(samples)
			Y_train.append([0]*len(v)) 
	
	return X_train, [val for sublist in Y_train for val in sublist]

#test tensors for each sentence and put the output tags in the file
def testTensorsForRelationExtraction_onemodel(args, crffeaturesFile, output_crfFeaturesFile):
	print ("testing", crffeaturesFile, output_crfFeaturesFile)
	with open(crffeaturesFile) as f:
		lines = f.readlines();

	outputfile = open(output_crfFeaturesFile, 'w')

	#go through all lines of crf features file
	i=0;
	while i < len(lines):
		#print lines[i]
		#sentence_tensor contains the lines of single sentence.
		sentence_tensor = []
		#search for the sentence end.
		while i < len(lines):
			#break when newline. That is at sentence end.
			if not lines[i].strip():
				break;
			line = lines[i].rstrip();
			split = line.split('\t');
			sentence_tensor.append(split);
			i+=1

		# print 'sentence_tensor length:', len(sentence_tensor)
		#if sentence tensor is empty. Continue for next sentence. This happens when when two consecutive newlines occur.
		if not sentence_tensor:
			i+=1;
			continue;

		#pass event column and timex column
		if args.withgroundtags:
			all_pairs = get_all_index_pairs(np.asarray(sentence_tensor)[:,args.conceptfield]);
		else:
			all_pairs = get_all_index_pairs(np.asarray(sentence_tensor)[:,args.testconceptfield]);

		if all_pairs:
			#sentence_tensors = []
			for index1, index2 in all_pairs:
				if index1 != index2:
					# sentence_tensors.append(get_test_features_for_sentencetensor(args,sentence_tensor, i, j))
					#x_sentence = get_test_features_for_sentencetensor(args,sentence_tensor, index1, index2);
					X_test =[]
					X_char_test =[]
					X_char_ngram_test =[]
					X_sentence = []
					if args.include_chars and args.include_char_ngram_hash:
						#lists
						x, x_char, x_char_ngram = get_test_features_for_sentencetensor(args,sentence_tensor, index1, index2)
						#(sentence_length, emb)
						X_test.append(x)
						#(max_char_len, char_emb)
						X_char_test.append(sequence.pad_sequences([[] for k in xrange(args.maxlen - len(x_char))], maxlen=args.max_charlen ).tolist() +\
									sequence.pad_sequences(x_char[:args.maxlen], maxlen=args.max_charlen ).tolist())
						X_char_ngram_test.append(sequence.pad_sequences([[] for k in xrange(args.maxlen - len(x_char))], maxlen=args.max_charlen ).tolist() +\
									sequence.pad_sequences(x_char_ngram[:args.maxlen], maxlen=args.max_charlen ).tolist())

						#(None, sentence_max_len, emb)
						X_test = sequence.pad_sequences(np.array(X_test), maxlen=args.maxlen )
						X_char_test = np.asarray(X_char_test)
						X_char_ngram_test = np.asarray(X_char_ngram_test)
						predict_test = args.model.predict({'input1': X_test, "input2": X_char_test}, batch_size=32, verbose=0)
				
					elif args.include_chars:
						x, x_char = get_test_features_for_sentencetensor(args,sentence_tensor, index1, index2)
						
						X_test.append(x)
						X_char_test.append(sequence.pad_sequences([[] for k in xrange(args.maxlen - len(x_char))], maxlen=args.max_charlen ).tolist() +\
									sequence.pad_sequences(x_char[:args.maxlen], maxlen=args.max_charlen ).tolist())
						#print type(X_test), type(X_char_test)
						X_test = sequence.pad_sequences(np.array(X_test), maxlen=args.maxlen )
						X_char_test = np.asarray(X_char_test)
						#print type(X_test), type(X_char_test)
						predict_test = args.model.predict({'input1': X_test, "input2": X_char_test}, batch_size=32, verbose=0)
						print 'predicting'
					else:
						X_test.append(get_test_features_for_sentencetensor(args,sentence_tensor, index1, index2))
						X_test = sequence.pad_sequences(np.array(X_test), maxlen=args.maxlen )
						if args.sentencelevelfeatures:
							X_sentence.append(get_sentencelevel_features_for_sentencetensor(args,sentence_tensor, index1, index2))
							X_sentence = sequence.pad_sequences(np.array(X_sentence), maxlen=args.maxlen )
							predict_test = args.model.predict([X_test, X_sentence])
						else:
							predict_test = args.model.predict(X_test)

					predict_test = np.argmax(predict_test, axis=1)
					#print('output', predict_test[0])
					if '0' not in args.tagset[predict_test[0]] :
						append_relation_to_sentence_tensor(sentence_tensor, args.tagset[predict_test] , index1, index2)

		outputfile.write("\n".join("\t".join(map(str, x)) for x in sentence_tensor))
		outputfile.write("\n\n")


#get test features for the sentencetensor with positions of the pair of concepts
def get_test_features_for_sentencetensor(args, sentence_tensor, index1, index2):
	# print ('get test features file for:', sentence_tensor, index1, index2)
	#print ('get test features file for:', sentence_tensor, index1, index2)
	x=[]# np.zeros(11);#np.empty(shape=(0,0));] for whole dataset
	x_char = []
	x_char_ngram = []
	end_index1 = get_end_index_for_entity (args,sentence_tensor, index1)
	end_index2 = get_end_index_for_entity (args,sentence_tensor, index2)
	# Mesenteric	Mesenter	NNP	B-NP	X	START-CAP	X	ic	X	X	ALPHANUMERIC	X	X	B-test	TeCP	0
	for i in range(len(sentence_tensor)):
		line_list = sentence_tensor[i]; 
		x_1=[]
		#print('b4 appending: ', len(x))
		#print('In get_relntype_vectortensor: search in word2vec: word:',line_list[0])
		if args.embedding:

			x_1 = pos_features.get_wordlevel_features(args, x_1, line_list)
			# x_1 = pos_features.get_other_relation_features(args, x_1, line_list)
		
			if args.include_chars:
				word = line_list[0]
				#get indexes of each char of the word
				char_seq = [1 + args.char2index.get(k, -1) for k in word] # 0 for OOV char
				x_char.append((1 + np.array(char_seq)).tolist())
				if len(char_seq) == 0:
					x_char.append([1])
			if args.include_char_ngram_hash:
				char_ngrams = get_char_ngrams(line_list[0].lower(), args.num_of_ngram)
				char_ngram_indexes = [1 + args.ngram2index.get(k, -1) for k in char_ngrams]
				x_char_ngram.append((1 + np.array(char_ngram_indexes)).tolist())
		#print('Each line x: ', len(x))
		if i < index1:
			x_1.append([i - index1])
		elif end_index1 < i:
			x_1.append([i - end_index1])
		else:
			x_1.append([0])

		if i < index2: 
			x_1.append([i - index2])
		elif end_index2 < i:
			x_1.append([end_index2 - i])
		else:	
			x_1.append([0])
		x_1.append(getConceptIndexForTag(line_list[args.conceptfield]))
		#line_tensor.append(args.pos2vec(line_tensor[2]))
		#line_tensor.append(args.chunk2vec(line_tensor[3]))
		#line_tensor.append(args.nertag2vec(line_tensor[13]))
		#print ('embedding size of each word: ', len([val for sublist in x for val in sublist]))
		#print('sublist: ', len([val for sublist in x_1 for val in sublist]))
		x.append([val for sublist in x_1 for val in sublist])


	#let this be list. Convert as np arrays in the next step.
	#this only constitute a sample taking whole sentence (window) into consideration
	if args.include_chars:
		# print ('X_char length: ', len(x_char))
		return x, x_char
	if args.include_chars and args.include_char_ngram_hash:
		return x, x_char, x_char_ngram 
	#print('x length total: ', len(x),x)
	return x

#get sentence level features for the sentencetensor with positions of the pair of concepts
def get_sentencelevel_features_for_sentencetensor(args, sentence_tensor, index1, index2):
	x_1 = []
	#sorting the indexes
	index1, index2 = sorted([index1, index2])
	#get end indexes of index1 and index2
	end_index1, index1_entity = get_end_index_for_entity (args,sentence_tensor, index1)
	end_index2, index2_entity = get_end_index_for_entity (args,sentence_tensor, index2)
	#append sum of all word vectors which are in-berween index1 and index2
	#x_1.append(getrelationphrase_vector(args,sentence_tensor, end_index1, index1, index2))
	#distance between the pair
	x_1.append([index2-end_index1])
	x_1 = pos_features.get_other_relationphrase_features(args, x_1, \
		getrelationphrase_complete(sentence_tensor, index1, end_index2), \
		getrelationphrase_middle(sentence_tensor, end_index1, index2), \
		index1_entity, index2_entity)
	# x_1 = pos_features.get_posseq_relationphrase_features(args, x_1, \
	# 	getrelationphrase_complete(sentence_tensor, index1, end_index2), \
	# 	getrelationphrase_middle(sentence_tensor, end_index1, index2), \
	# 	index1_entity, index2_entity)
	# x_1 = pos_features.get_pmi_relationphrase_features(args, x_1, \
	# 	getrelationphrase_complete(sentence_tensor, index1, end_index2), \
	# 	getrelationphrase_middle(sentence_tensor, end_index1, index2), \
	# 	index1_entity, index2_entity)
	x_1 = pos_features.get_assertion_semantic_classes_relationphrase_features(args, x_1, \
		getrelationphrase_complete(sentence_tensor, index1, end_index2))
	# Mesenteric	Mesenter	NNP	B-NP	X	START-CAP	X	ic	X	X	ALPHANUMERIC	X	X	B-test	TeCP	0
	# for i in range(len(sentence_tensor)):
	# 	line_list = sentence_tensor[i]; 

	# 	#print('b4 appending: ', len(x))
	# 	#print('In get_relntype_vectortensor: search in word2vec: word:',line_list[0])
	# 	if args.embedding:
	# 		if line_list[0].lower() in args.word2vec:
	# 			x_1.append(args.word2vec[line_list[0].lower()].tolist())
	# 		else:
	# 			x_1.append(args.word2vec['<UK>'].tolist())

	# 	x_1.append(getConceptIndexForTag(line_list[args.conceptfield]))
	return [val for sublist in x_1 for val in sublist]

#get all index pairs of the concepts for identifying relations
def get_all_index_pairs(con_column):
	problem_indexes=[]
	other_indexes = []
	# treatment_indexes = []
	for i,con in enumerate(con_column):
		if 'B-problem' in con:
			problem_indexes.append(i);
		elif 'B-treatment' in con:
			other_indexes.append(i)
		elif 'B-test' in con :
			other_indexes.append(i);
	#only treatment pairs
	other_pairs = [x for x in itertools.product(problem_indexes,other_indexes)]	
	#return other_pairs
	problem_pairs = [x for x in itertools.product(problem_indexes,problem_indexes)]
	return problem_pairs + other_pairs
	# treatment_pairs = [x for x in itertools.product(problem_indexes,treatment_indexes)]
	# test_pairs = [x for x in itertools.product(problem_indexes,test_indexes)]
	# #print ('all pairs for: ', con_column, problem_pairs, test_pairs, treatment_pairs)
	# return problem_pairs, treatment_pairs, test_pairs

def append_relation_to_sentence_tensor(sentence_tensor, rel_type, index1, index2):
	for i in range(len(sentence_tensor)):
		if i == index1 or i == index2:
			sentence_tensor[i] = sentence_tensor[i]+[rel_type];
		else:
			sentence_tensor[i] = sentence_tensor[i]+[0];
	return sentence_tensor

def get_end_index_for_entity (args,sentence_tensor, index1):
	end_index1 = index1;
	entity = sentence_tensor[index1][0]
	for i in range(index1+1, len(sentence_tensor)):
		line_list = sentence_tensor[i]; 
		concepttag = line_list[args.conceptfield];
		if 'I-' in concepttag:
			entity = entity + ' ' + line_list[0]
			end_index1 = i;
		else:
			break;
	return end_index1, entity

def getConceptIndexForTag(concept):
	if 'problem' in concept:
		return [1,0,0,0]
	if 'test' in concept:
		return [0,1,0,0]
	if 'treatment' in concept:
		return [0,0,1,0]
	if 'medication' in concept:
		return [0,0,0,1]
	return [0,0,0,0]

def getrelationphrase_vector(args,sentence_tensor, current_position, index1, index2):
	min_index = min(index1, index2)
	max_index = max(index1, index2)
	x = np.zeros(40);
	if current_position < min_index:
		for i in range(current_position, min_index):
			line_list = sentence_tensor[i]; 
			if line_list[0].lower() in args.word2vec:# and len(line_list[0])>3:
				x += args.word2vec[line_list[0].lower()]
	elif min_index <= current_position and current_position < max_index:
		for i in range(current_position, max_index):
			line_list = sentence_tensor[i]; 
			if line_list[0].lower() in args.word2vec:# and len(line_list[0])>3:
				x += args.word2vec[line_list[0].lower()]
	else:
		for i in range(current_position, len(sentence_tensor)):
			line_list = sentence_tensor[i]; 
			if line_list[0].lower() in args.word2vec:# and len(line_list[0])>3:
				x += args.word2vec[line_list[0].lower()]
	return x
			
def getrelationphrase_complete(sentence_tensor, index1, end_index2):
	list1 = []
	for i in range(index1, end_index2):
		list1.append(sentence_tensor[i]);
	return ' '.join(str(e) for e in list1)

def getrelationphrase_middle(sentence_tensor, end_index1, index2):
	list1 = []
	for i in range(end_index1, index2):
		list1.append(sentence_tensor[i]);
	return ' '.join(str(e) for e in list1)


def postProcessing_createTestCRFFeaturesFile_dir(args):

	listFiles = os.listdir(args.test_crfFeatures_dir)
	if not os.path.exists(args.output_crfFeatures_dir):
		os.makedirs(args.output_crfFeatures_dir)
	for file in listFiles:
		testTensorsForRelationExtraction_onemodel(args,os.path.join(args.test_crfFeatures_dir, file),\
			os.path.join(args.output_crfFeatures_dir, file))


