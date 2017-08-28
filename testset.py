# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 00:50:40 2017

@author: Sean Chang
"""

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras import backend as K

margin = 0.6
theta = lambda t: (K.sign(t)+1.)/2.

max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 10

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train[0]))
'''
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

loss = lambda y_true, y_pred: -(1-theta(y_true-margin)*theta(y_pred-margin)-theta(1-margin-y_true)*theta(1-margin-y_pred))*(y_true*K.log(y_pred)+(1-y_true)*K.log(1-y_pred))

model.compile(loss=loss,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
'''