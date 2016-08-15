import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation

import numpy as np

model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

data = np.random.random((1000,784))
labels = np.random.randint(2,size=(1000,1))

model.fit(data,labels, nb_epoch=10, batch_size=32)

model.save("model_keras.h5")

####

model2 = load_model("model_keras.h5")
