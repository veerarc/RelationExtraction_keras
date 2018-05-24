'''Methods to create neural network model.
'''

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation,  Input, merge
from keras.layers import Embedding
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers import Convolution1D, MaxPooling1D

def get_birnn_model(maxlen, embedding_size, lstm_output_size, 
	number_of_lstm_layers, numberofclasses,
	dropout=0.01):
	if number_of_lstm_layers == 1:
		input_name = Input(shape=(maxlen, embedding_size))
		forward_name = SimpleRNN(lstm_output_size)(input_name)
		backward_name = SimpleRNN(lstm_output_size, go_backwards=True)(input_name)
		merged_name = merge([forward_name, backward_name], mode='concat', concat_axis=-1)
		dropout_name = Dropout(dropout)(merged_name)
		output = Dense(numberofclasses, activation='softmax')(dropout_name)
		model = Model(input=input_name, output=output)

	if numberofclasses == 2:
		model.compile(loss='binary_crossentropy',
			optimizer='adam',
			metrics=['accuracy'])
	else:
		model.compile(loss='categorical_crossentropy',
			optimizer='rmsprop',
			metrics=['accuracy'])
	return model

def get_bilstm_model(maxlen, embedding_size, lstm_output_size, 
	number_of_lstm_layers, numberofclasses,
	dropout=0.01):
	if number_of_lstm_layers == 1:
		input_name = Input(shape=(maxlen, embedding_size))
		forward_name = LSTM(lstm_output_size)(input_name)
		backward_name = LSTM(lstm_output_size, go_backwards=True)(input_name)
		merged_name = merge([forward_name, backward_name], mode='concat', concat_axis=-1)
		dropout_name = Dropout(dropout)(merged_name)
		output = Dense(numberofclasses, activation='softmax')(dropout_name)
		model = Model(input=input_name, output=output)

	if numberofclasses == 2:
		model.compile(loss='binary_crossentropy',
			optimizer='adam',
			metrics=['accuracy'])
	else:
		model.compile(loss='categorical_crossentropy',
			optimizer='rmsprop',
			metrics=['accuracy'])
	return model

