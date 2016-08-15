import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import numpy

## to run and see which outputs go to stderr and which to stdout
## do
## python test1.py 2>err 1>out

seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt("data/pima-indians-diabetes.data.csv", delimiter=",")

X = dataset[:,0:8]
Y = dataset[:,8]

model = Sequential()

## First layer is fully connected (Dense), has 12 neurons, 8 inputs, the 
## weight initialization is uniform (0..0.5), and it uses the rectifier (relu)
## activation function (max(0,x))
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
## Secpmd  layer fully connected, 8 units)
model.add(Dense(8, init='uniform', activation='relu'))
## final layer fully connected, one unit, sigmoid activation function
model.add(Dense(1, init='uniform', activation='sigmoid'))

## no specify how to train: loss function and optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

## actually do the training
model.fit(X,Y,nb_epoch=150, batch_size=10)

## Evaluation on the trianing set
scores = model.evaluate(X,Y)
model.summary()
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

modelname="model_ex1_"+keras.backend._BACKEND+".h5"
print("Saving model: "+modelname)

model.save(modelname)

## delete the model
del model

## restore and evaluate again
print("APPLICATION: load model")
model2 = load_model(modelname)
print("APPLICATION: evaluate")
scores2 = model2.evaluate(X,Y)
print("%s: %.2f%%" % (model2.metrics_names[1], scores2[1]*100))
print("APPLICATION: model summary")
model2.summary()
print("APPLICATION: creating prediction")
preds = model.predict(X)
print("APPLICATION: done")
