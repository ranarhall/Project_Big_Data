import numpy as np

np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
import copy

np.random.seed(1)

def preprocessing(input):
    newInput = []
    maxLen = max([len(l) for l in input])
    for sentence in input:
        currEmbedding = copy.deepcopy(sentence)
        curLen = len(sentence)
        if(curLen < maxLen) :
            newSentence = np.zeros(len(sentence[0]))
            for i in range(curLen,maxLen) :
                currEmbedding.append(newSentence)
        newInput.append(currEmbedding)
    return newInput

def LSTMModel(input):
    #embedding = Input()
    net_input = Input(shape=np.array(input).shape[1:])
    X = LSTM(128, return_sequences=True)(net_input)
    X = Dropout(0.5)(X)
    X = LSTM(128, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(1, activation=None)(X)
    X = Activation('sigmoid')(X)
    
    model = Model(inputs=net_input, outputs=X)

    return model;
            

