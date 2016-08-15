from __future__ import print_function
import sys
import tensorflow as tf 
import importlib
import os
import inspect
import numpy as np

## IMPORTANT: we need to pass/get more information about the learning problem, most
## importantly the target type: for example if we have classification, we may want to use
## a softmax layer of nrclass nodes by default while for regression we may just use a single 
## output unit.
## Depending on how the output is coded, our predict and predict_proba methods need to 
## do different things as well.
## 
## As an alternative we could also use algorithms or building blocks from tf learn or from
## tf-slim or from tflearn
## https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/learn/python/learn
## https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
## https://github.com/tflearn/tflearn
##
## Possible alternative: different commands and different engine for wrapping tf learn 
## since that is much more similar to sklearn

## IMPORTANT: the option values need to be valid python expressions, and they
## will get evaluated! So in order to pass a string, enclose the value in quotes!


print("tensorflowTrain - got args: ", sys.argv, file=sys.stderr)
if len(sys.argv) < 4:
	sys.exit("ERROR: Not at least three arguments: [script], data base name, model base name, and options")

data=sys.argv[1]
if not data:
	sys.exit("ERROR: No data path")

modelpath=sys.argv[2]
if not modelpath:
	sys.exit("ERROR: No model path")

options=sys.argv[3:]

## The option should control what exactly we want to do here
## One possibility would be to have an option that takes the file "algorithm.py" from
## modelpath and executes functions from there to create the graph.
## If necessary, the functions could take arguments that depend on the dimensionality
## and/or type of the data (e.g. number of input nodes based on shape, number of hidden
## nodes based, by default, on shape etc.)
## As a fallback, if that option is not given, we could instead import our own file from
## the wrapper directory and just use some simple perceptron or other simple dense network.

## load the data: we expect two files in Matrix 
## The parameter is the prefix to which we add "dep.mtx" and "indep.mtx" to get the final names
depfile = data+"dep.csv"
indepfile=data+"indep.csv"
## TODO: at some point, support, weights, costs etc
## weightsfile=data+"instweights.csv"


deps = np.loadtxt(depfile)
indeps = np.loadtxt(indepfile)

shape = indeps.shape

print("DOING NOTHING YET, TENSORFLOW TRAINING NEEDS TO GET IMPLEMENTED",file=sys.stderr)
print("MODEL NEEDS TO GET TRAINED MANUALLY AND SHOULD BE IN: "+modelpath)
print("AND SHOULD HAVE NAME: tensorflow")

