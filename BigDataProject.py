import Preprocessing as pp
import TakeInput as ti
import ElmoWordEmbedding as Elm
import GetWordsDictionary as gw
import LSTMBinaryClassifier as lsc
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform


partition = 2500
nrows = 3000
epoch = 40


df = ti.Input("train.csv",[1,2],nrows)

#print(df.values[1:][:,1])

sentences = pp.findmatches( df.values[1:][:,1])


Words = gw.GetWords(sentences)

AllEmbeddings = Elm.ElmoEmbedding(Words)

UnshapedInput = Elm.GetTokensEmbedding(Words,AllEmbeddings,sentences)

Labels = [int(a) for a in df.values[1:][:,0]]

Input = np.array(lsc.preprocessing(UnshapedInput))

model = lsc.LSTMModel(Input[0:partition])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(Input[0:partition], Labels[0:partition], epochs = epoch, batch_size = 1000, shuffle=True)

loss, acc = model.evaluate(Input[partition:],Labels[partition:])

print("Test accuracy = ", acc)
