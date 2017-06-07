from __future__ import print_function
import sys
import numpy as np
from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.optimizers import SGD

# Example lfkerastrain.py file: this should get placed into the LearningFramework's DataDirectory
# The script must implement the actual code for training a network on the data and
# it MUST assign the resulting model to the variable model!

# The script gets access to all the variables from the calling kerasTrain.py script:
# indeps: the independent variables / attributes, usually a 2-D array of the shape (nrInstances,nrAttributes)
global indeps
# deps: the dependent variable / targets, usually a single class number, target value. For cost sensitive
#   classification, could be an array of costs where the i-th entry is the cost for class number i
global deps
# shape: the shape of the input data (=indeps.shape)
global shape
# model: the variable which must get assigned the trained model
global model
# mode: the kind of learning task, "regr" = regression, "class" = classification
global mode
# nrCl: the number of classes
global nrCl


# The example code below uses the following variables to set up a very basic network and train it:
# Number of examples to train on per iteration
batchsize = int(shape[0]/10+1)
# maximum number of epochs to train (can be stopped earlier if no improvement on validation set)
epochs = 50
# portion of data to use for validation
validation = 0.1
# Number of units and activation function for the first hidden layer
units1 = 10
act1 = 'relu'
# Number of units and activation function for the second hidden layer
# No second hidden layer if units2 == 0
units2 = 0
act2 = 'relu'

# ========================================================================================

print("Running lfkerastrain, batchsize=",batchsize," maxepochs=",epochs, " validation=",validation, file=sys.stderr)


inputs = shape[1]
examples = shape[0]


model = Sequential()
# the first layer has as many inputs as our data vectors
# and uses, for now half the number of inputs + 1 units
print("Input layer with inputs: ", inputs, " outputs: ", units1, " activation=", act1, file=sys.stderr)
# model.add(Dense(units1, input_dim=inputs, kernel_initializer='random_uniform', activation='relu'))
model.add(Dense(units1, input_dim=inputs, use_bias=True, kernel_initializer='random_uniform', activation=act1))

# second layer
if units2 > 0:
    print("Second layer with outputs: ", units2, " activation=", act2, file=sys.stderr)
    model.add(Dense(units2, kernel_initializer='random_uniform', activation=act2))

# top layer depends on our learning situation: if we have regr, we use a single unit
# sigmoid layer, otherwise we use as many output units as we have classes
# Also, our loss and metrics will be different depending on if we have classification or regression
if mode == "regr":
    print("Output layer for regression: ", " activation=sigmoid", file=sys.stderr)
    model.add(Dense(1, use_bias=True, kernel_initializer='random_normal', activation='sigmoid'))
    # maybe use optimizer rmsprop?
    # maybe use activation rely or tanh?
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model.fit(indeps, deps, batch_size=batchsize)
else:
    # NOTE: this probably works only if the mode is classind, not for classcost so far!!
    # If we have classcost, we use the cost vector directly as "one hot", otherwise
    # we convert to one hot representation
    if mode == "classcost":
        onehot = indeps
    else:
        onehot = np.eye(nrCl)[deps.astype(int)]
        # NOTE: apparently there is a keras utils method which does the same, probably just
        # wrapper:
        # from keras.utils import np_utils
        # np_utils.to_categorical(deps.astype(int))
        # Another
 
    # NOTE: with keras.utils.np_utils.to_categorical(deps) we could also do it the other
    # way round and use the shape of the resulting matrix to figure out the required number of
    # output units, because the keras routine determines the number of different classes itself.

    print("Output layer for classification, outputs=", nrCl, file=sys.stderr)
    model.add(Dense(nrCl, activation='softmax'))
    # explicitly specify the optimizer: stochastic gradient descend
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    # Alternatives: RMSprop, Adagrad, Adadelta ... see https://keras.io/optimizers/

    # NOTE: for classification, average crossentropy error (ACE) is usually better then acc or mse:
    # - acc does not consider how close the one hot units really are to the target
    # - mse over-uses the distances of the incorrect outputs
    #
    # an alternate optimizer could be 'adadelta'
    # Metrics: list of metrics to evaluate for the training progress output
    # model.compile(loss='categorical_crossentropy',metrics=['categorical_crossentropy','accuracy'],optimizer=sgd)
    model.compile(loss='categorical_crossentropy', metrics=['categorical_crossentropy', 'accuracy'], optimizer=adadelta)
    # alternate loss: mse, mae, mape, msle, squared_hinge, hinge, kld, poisson, cosine_proximity
    # see https://github.com/fchollet/keras/blob/master/keras/objectives.py

    # TODO: we should maybe automatically include support for maximum epochs, automatical adjustment of
    # learning rate, early termination if we go below a certain loss threshold, use of validation set, and other
    # details (we use decay for SGD which decreases the learning rate!)
    # - validation_split=0.1
    # - epochs=N
    # - verbose=1
    # - show_accuracy=True
    # NOTE: show_accuracy=True is not supported any more, TODO: need to
    # implement a callback for this!
    print("shape of indeps=", indeps.shape)
    print("shape of outputs=", onehot.shape)
    cb1 = EarlyStopping(monitor='val_loss', min_delta=0.0000001, patience=2, verbose=1, mode='auto')
    model.fit(indeps, onehot, batch_size=batchsize, callbacks=[cb1], validation_split=validation, epochs=epochs, verbose=2)

print("Finishing lfkerastrain", file=sys.stderr)

