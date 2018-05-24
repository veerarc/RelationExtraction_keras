import numpy as np
import nltk
from nltk.tokenize import SpaceTokenizer

from keras.utils.np_utils import to_categorical

def get_wordlevel_features(args, x_1, line_list):
	if line_list[0].lower() in args.word2vec:
		x_1.append(args.word2vec[line_list[0].lower()])
	else:
		x_1.append(args.word2vec['<UK>'])

	#add word features
	x_1.append(word_features(line_list[0]))

	#add relation phrase vector
	# x_1.append(getrelationphrase_vector(args,sentence_tensor, i, index1, index2))

	if args.pos:
		# print 'pos: ',args.pos,  args.features2Tokens['pos'].index(line_list[2])
		x_1.append(get_pos_features(line_list[2],args.features2Tokens['pos']))
	if args.chunk:
		x_1.append(get_chunk_features(line_list[3],args.features2Tokens['chunk']))
	if args.pre:
		x_1.append(get_pre_features(line_list[0],args.features2Tokens['pre']))
	if args.suf:
		x_1.append(get_suf_features(line_list[0],args.features2Tokens['suf']))
	if args.gazetter:
		x_1.append(get_gazetter_features(line_list[0],args.features2Tokens['gazetter']))

	return x_1;

def get_features_genericlist (word, tokensList):
	if word in tokensList:
		pos_index = tokensList.index(word)
		return np.lib.pad(to_categorical(np.asarray([pos_index]))[0], 
			(0,len(tokensList) - pos_index - 1), 
			'constant',constant_values=0)
	else:
		return np.asarray([0] * len(tokensList)).tolist()
	
def get_pos_features (word, tokensList):
	if word in tokensList:
		pos_index = tokensList.index(word)
		return np.lib.pad(to_categorical(np.asarray([pos_index]))[0], 
			(0,len(tokensList) - pos_index - 1), 
			'constant',constant_values=0)
	else:
		return np.asarray([0] * len(tokensList)).tolist()

def get_chunk_features (word, tokensList):
	if word in tokensList:
		chunk_index = tokensList.index(word)
		return np.lib.pad(to_categorical(np.asarray([chunk_index]))[0], 
			(0,len(tokensList) - chunk_index - 1), 
			'constant',constant_values=0)
	else:
		return np.asarray([0] * len(tokensList)).tolist()

def get_pre_features (word, tokensList):
	word =word.lower()
	#check for siz chars from start 6 to 0
	for i in reversed(range(7)):
		if word[:i] in tokensList:
			pos_index = tokensList.index(word[:i])
			return np.lib.pad(to_categorical(np.asarray([pos_index]))[0], 
			(0,len(tokensList) - pos_index - 1), 
			'constant',constant_values=0)
		else:
			return np.asarray([0] * len(tokensList)).tolist()

def get_suf_features (word,tokensList):
	word =word.lower()
	#check for siz chars from start 6 to 0
	for i in reversed(range(7)):
		if word[-i:] in tokensList:
			pos_index = tokensList.index(word[-i:])
			return np.lib.pad(to_categorical(np.asarray([pos_index]))[0], 
			(0,len(tokensList) - pos_index - 1), 
			'constant',constant_values=0)
		else:
			return np.asarray([0] * len(tokensList)).tolist()

def get_gazetter_features (word, tokensList):
	word =word.lower()
	if word in tokensList:
		return np.asarray([1]).tolist()
	else:
		for var in tokensList:
			if word in var and len(word) > 3:
				return np.asarray([1]).tolist()		
	return np.asarray([0]).tolist()

def word_features(word, config=None):

	"""Return array of surface form features for given word."""
	if not word:
		return [0,0,0,1]
	if word.isupper():
		return [1,0,0,0]    # all caps
	elif word[0].isupper():
		return [0,1,0,0]    # init cap
	elif any(c.isupper() for c in word):
		return [0,0,1,0]    # has cap
	else:
		return [0,0,0,1]    # no caps

def convertWordIndexesToEmbeddings(args, X_in):
	X_out = []
	for seq in X_in:
		X_out.append(getEmbeddingOfSequence(args, seq))
	return X_out

def getEmbeddingOfSequence(args, sequence):
	list = []
	for i in sequence:
		if i in args.index2word:
			list.append(args.word2vec[args.index2word[i]])
		else:
			list.append(args.word2vec['<UK>'])
	#print list
	return list

stringTokenizer = SpaceTokenizer()
'''Other features over relationphrase
'''
def get_other_relationphrase_features(args, x_1, relationphrase_complete, relationphrase_middle, entity_1, entity_2):
	
	if len(relationphrase_complete) < 4:
		return x_1

	#nltk.word_tokenize(sentence)
	relationphrase_complete_tokens = stringTokenizer.tokenize(relationphrase_complete)
	relationphrase_complete_postags = nltk.pos_tag(relationphrase_complete_tokens)

	#100 any word of relation phrase
	x_temp = []
	for word in relationphrase_complete_tokens:
		word = word.lower()
		if word in args.all_i2b22010_words_lower:
			x_temp.append(args.all_i2b22010_words_lower.index(word))
	x_1.append((x_temp + [0] * 100)[:100]) #multiples appends 100 zeros. return first 10 values

	#100 any pos tag of relation phrase
	x_temp = []
	for pos_tag in relationphrase_complete_postags:
		if pos_tag[1] in args.features2Tokens['pos']:
			x_temp.append(args.features2Tokens['pos'].index(pos_tag[1]))
	x_1.append((x_temp + [0] * 100)[:100]) #multiples appends 100 zeros. return first 10 values

	#100 any word of relation phrase
	# x_temp = []
	# for chunk_tag in relationphrase_complete_postags:
	# 	if pos_tag in args.features2Tokens['pos']:
	# 		x_temp.append(args.features2Tokens['pos'].index(word)
	# x_1.append(x_temp + [0] * 100)[:100]

	#10 conjunction words of relation phrase
	#10 verbs of relation phrase
	x_temp = []
	for i,conjunction_word in enumerate(args.conjunction_words):
		if ' '+conjunction_word+' ' in relationphrase_complete:
			x_temp.append(i);
	x_1.append((x_temp + [0] * 10)[:10])

	#10 verbs of relation phrase
	x_temp = []
	for i,verb in enumerate(args.relation_verbs_words):
		if ' '+verb+' ' in relationphrase_complete:
			x_temp.append(i);
	x_1.append((x_temp + [0] * 10)[:10])

	#1 worsen problems and causes problems
	if entity_1 in args.worsen_problems or entity_2  in args.worsen_problems or entity_1 in args.causes_problems or entity_2 in args.causes_problems:
		x_1.append([1]);
	else:
		x_1.append([0]);

	return x_1


'''Other features over relationphrase
'''
def get_posseq_relationphrase_features(args, x_1, relationphrase_complete, relationphrase_middle, entity_1, entity_2):
	
	if len(relationphrase_complete) < 4:
		return x_1

	#nltk.word_tokenize(sentence)
	relationphrase_complete_tokens = stringTokenizer.tokenize(relationphrase_complete)
	relationphrase_complete_postags = nltk.pos_tag(relationphrase_complete_tokens)
	pos_seq = posTagsToString (relationphrase_complete_postags)
	#1 is in TrAP posseq
	x_temp = []
	for i,pos_tag_local in enumerate(args.TrAP_posseq_words):
		if pos_seq in pos_tag_local:
			x_temp.append(i)
	x_1.append((x_temp + [0] * 1)[:1])

	#2 is in TrCP posseq
	x_temp = []
	for i,pos_tag_local in enumerate(args.TrCP_posseq_words):
		if pos_seq in pos_tag_local:
			x_temp.append(i)
	x_1.append((x_temp + [0] * 1)[:1])
	#3 is in TrIP posseq
	x_temp = []
	for i,pos_tag_local in enumerate(args.TrIP_posseq_words):
		if pos_seq in pos_tag_local:
			x_temp.append(i)
	x_1.append((x_temp + [0] * 1)[:1])
	#4 is in TrNAP posseq
	x_temp = []
	for i,pos_tag_local in enumerate(args.TrNAP_posseq_words):
		if pos_seq in pos_tag_local:
			x_temp.append(i)
	x_1.append((x_temp + [0] * 1)[:1])
	#5 is in TrWP posseq
	x_temp = []
	for i,pos_tag_local in enumerate(args.TrWP_posseq_words):
		if pos_seq in pos_tag_local:
			x_temp.append(i)
	x_1.append((x_temp + [0] * 1)[:1])

	return x_1

def posTagsToString(relationphrase_complete_postags):
	pos_seq='';
	for pos_tag in relationphrase_complete_postags:
		pos_seq = pos_seq+' '+pos_tag[1]
	return pos_seq.strip()

def get_pmi_relationphrase_features(args, x_1, relationphrase_complete, relationphrase_middle, entity_1, entity_2):
	x_temp = []
	index1, index2 = sorted([entity_1, entity_2])
	index1 = index1.lower()
	index2 = index2.lower()
	# print len(args.pmiscores), index1, index2
	if (index1,index2) in args.pmiscores:
		# print 'pmi score:',index1, index2,args.pmiscores[(index1,index2)]
		x_temp.append(args.pmiscores[(index1,index2)]);
	x_1.append((x_temp + [0.0] * 1)[:1])

	return x_1


'''Other features over relationphrase
'''
def get_assertion_semantic_classes_relationphrase_features(args, x_1, relationphrase_complete):
	
	# allergy.txt
	# cause.txt         
	# continue.txt
	# def_determiner.txt
	# evidence.txt
	# fail.txt
	# instance.txt
	# association.txt
	# certainty.txt
	# copular.txt   
	# deny.txt
	# exception.txt
	# history.txt
	# stop.txt
	# avoid.txt
	# change_state.txt
	# decline.txt
	# disappear.txt
	# experience.txt
	# hypothetical_cue_list.txt
	# uncertainty.txt

	#1 allergy of relation phrase
	x_temp = []
	for i,word_local in enumerate(args.allergy_words):
		if ' '+word_local+' ' in relationphrase_complete:
			x_temp.append(i);
			break
	x_1.append((x_temp + [0] * 1)[:1])

	#2 cause of relation phrase
	x_temp = []
	for i,word_local in enumerate(args.cause_words):
		if ' '+word_local+' ' in relationphrase_complete:
			x_temp.append(i);
			break
	x_1.append((x_temp + [0] * 1)[:1])

#3 continue of relation phrase
	x_temp = []
	for i,word_local in enumerate(args.continue_words):
		if ' '+word_local+' ' in relationphrase_complete:
			x_temp.append(i);
			break
	x_1.append((x_temp + [0] * 1)[:1])

#4 def_determiner of relation phrase
	x_temp = []
	for i,word_local in enumerate(args.def_determiner_words):
		if ' '+word_local+' ' in relationphrase_complete:
			x_temp.append(i);
			break
	x_1.append((x_temp + [0] * 1)[:1])

#5 evidence of relation phrase
	x_temp = []
	for i,word_local in enumerate(args.evidence_words):
		if ' '+word_local+' ' in relationphrase_complete:
			x_temp.append(i);
			break
	x_1.append((x_temp + [0] * 1)[:1])

#6 fail of relation phrase
	x_temp = []
	for i,word_local in enumerate(args.fail_words):
		if ' '+word_local+' ' in relationphrase_complete:
			x_temp.append(i);
			break
	x_1.append((x_temp + [0] * 1)[:1])

#7 instance of relation phrase
	x_temp = []
	for i,word_local in enumerate(args.instance_words):
		if ' '+word_local+' ' in relationphrase_complete:
			x_temp.append(i);
			break
	x_1.append((x_temp + [0] * 1)[:1])

#8 association of relation phrase
	x_temp = []
	for i,word_local in enumerate(args.association_words):
		if ' '+word_local+' ' in relationphrase_complete:
			x_temp.append(i);
			break
	x_1.append((x_temp + [0] * 1)[:1])

#9 certainty of relation phrase
	x_temp = []
	for i,word_local in enumerate(args.certainty_words):
		if ' '+word_local+' ' in relationphrase_complete:
			x_temp.append(i);
			break
	x_1.append((x_temp + [0] * 1)[:1])

#10 copular of relation phrase
	x_temp = []
	for i,word_local in enumerate(args.copular_words):
		if ' '+word_local+' ' in relationphrase_complete:
			x_temp.append(i);
			break
	x_1.append((x_temp + [0] * 1)[:1])

#11 deny of relation phrase
	x_temp = []
	for i,word_local in enumerate(args.deny_words):
		if ' '+word_local+' ' in relationphrase_complete:
			x_temp.append(i);
			break
	x_1.append((x_temp + [0] * 1)[:1])

#12 exception of relation phrase
	x_temp = []
	for i,word_local in enumerate(args.exception_words):
		if ' '+word_local+' ' in relationphrase_complete:
			x_temp.append(i);
			break
	x_1.append((x_temp + [0] * 1)[:1])

#13 history of relation phrase
	x_temp = []
	for i,word_local in enumerate(args.history_words):
		if ' '+word_local+' ' in relationphrase_complete:
			x_temp.append(i);
			break
	x_1.append((x_temp + [0] * 1)[:1])

#14 stop of relation phrase
	x_temp = []
	for i,word_local in enumerate(args.stop_words):
		if ' '+word_local+' ' in relationphrase_complete:
			x_temp.append(i);
			break
	x_1.append((x_temp + [0] * 1)[:1])

#15 avoid of relation phrase
	x_temp = []
	for i,word_local in enumerate(args.avoid_words):
		if ' '+word_local+' ' in relationphrase_complete:
			x_temp.append(i);
			break
	x_1.append((x_temp + [0] * 1)[:1])

#16 change_state of relation phrase
	x_temp = []
	for i,word_local in enumerate(args.change_state_words):
		if ' '+word_local+' ' in relationphrase_complete:
			x_temp.append(i);
			break
	x_1.append((x_temp + [0] * 1)[:1])

#17 decline of relation phrase
	x_temp = []
	for i,word_local in enumerate(args.decline_words):
		if ' '+word_local+' ' in relationphrase_complete:
			x_temp.append(i);
			break
	x_1.append((x_temp + [0] * 1)[:1])

	#18 disappear of relation phrase
	x_temp = []
	for i,word_local in enumerate(args.disappear_words):		
		if ' '+word_local+' ' in relationphrase_complete: 
			x_temp.append(i);
			break
	x_1.append((x_temp + [0] * 1)[:1])

	#19 experience of relation phrase
	x_temp = []
	for i,word_local in enumerate(args.experience_words):		
		if ' '+word_local+' ' in relationphrase_complete: 
			x_temp.append(i);
			break
	x_1.append((x_temp + [0] * 1)[:1])
	#20 hypothetical_cue_list of relation phrase
	x_temp = []
	for i,word_local in enumerate(args.hypothetical_cue_list_words):		
		if ' '+word_local+' ' in relationphrase_complete: 
			x_temp.append(i);
			break
	x_1.append((x_temp + [0] * 1)[:1])
	#21 hypothetical_cue_list of relation phrase
	x_temp = []
	for i,word_local in enumerate(args.hypothetical_cue_list_words):		
		if ' '+word_local+' ' in relationphrase_complete: 
			x_temp.append(i);
			break
	x_1.append((x_temp + [0] * 1)[:1])

	#1 uncertainty of relation phrase
	x_temp = []
	for i,word_local in enumerate(args.uncertainty_words):		
		if ' '+word_local+' ' in relationphrase_complete: 
			x_temp.append(i);
			break
	x_1.append((x_temp + [0] * 1)[:1])

	return x_1	
