# from __future__ import print_function

#import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

print("Running lfkerastrain",file=sys.stderr)

inputs=shape[1]
examples=shape[0]

batchsize=examples/100+1

model=Sequential()
## the first layer has as many inputs as our data vectors
## and uses, for now half the number of inputs + 1 units
units1=inputs/2+1
model.add(Dense(output_dim=units1,input_dim=inputs,init='uniform',activation='relu'))

## second layer is another rely layer, same number of units
## model.add(Dense(inputs/2+1,input_dim=inputs,init='uniform',activation='relu'))

## top layer depends on our learning situation: if we have regr, we use a single unit
## sigmoid layer, otherwise we use as many output units as we have classes
## Also, our loss and metrics will be different depending on if we have classification or regression
if(mode == "regr"):
  model.add(Dense(output_dim=1,input_dim=units1,init='uniform',activation='sigmoid'))
  model.compile(loss='',optimizer='adam',metrics=['mse'])
  model.fit(indeps,deps,batch_size=batchsize)
else:
  ## NOTE: this probably works only if the mode is classind, not for classcost so far!!
  ## If we have classcost, we use the cost vector directly as "one hot", otherwise 
  ## we convert to one hot representation
  if(mode == "classcost"):
    onehot=indeps
  else:
    onehot=np.eye(nrCl)[deps.astype(int)]
    ## NOTE: apparently there is a keras utils method which does the same, probably just
    ## wrapper:
    # from keras.utils import np_utils
    # np_utils.to_categorical(deps.astype(int))
 
  ## NOTE: with keras.utils.np_utils.to_categorical(deps) we could also do it the other 
  ## way round and use the shape of the resulting matrix to figure out the required number of 
  ## otput units, because the keras routine determines the number of different classes itself.

  print("Layer with inputs: ",units1," outputs: ",nrCl," shape of deps: ",onehot.shape,file=sys.stderr)
  model.add(Dense(output_dim=nrCl,input_dim=units1,activation='softmax'))
  ## explicitly specify the optimizer
  sgd=SGD(lr=0.1,decay=1e-6,momentum=0.9,nesterov=True)
  # an alternate optimizer could be 'adadelta'
  model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=sgd)
  model.fit(indeps,onehot,batch_size=batchsize)

print("Finishing lfkerastrain",file=sys.stderr)

