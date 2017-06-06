from __future__ import print_function
import sys
import keras
import importlib
import os
import inspect
import numpy as np
# import h5py just to check if it is installed
import h5py

# This gets invoked by the EngineKerasWrapper which is just an instance of EnginePythonNetworksbase which
# passes the following parameters:
# - [rootdirectory]: this gets removed by the invoking bash script and not passed on to us
# - path to the model/data directory with the directory separator at the end
# - path prefix for the model file: the complete path plus the beginning of the file name(s) for
#   the model
# - mode: classind/classcosts/regr
# - nrClasses: 0 for regression or the number of classes otherwise
# - any algorithm parameters specified for the PR


# IMPORTANT: the option values need to be valid python expressions, and they
# will get evaluated! So in order to pass a string, enclose the value in quotes!


print("kerasTrain - got args: ", sys.argv, file=sys.stderr)
if len(sys.argv) < 5:
    sys.exit("ERROR: Not at least 5 arguments: [script], model/data directory name, model base name, mode, nrClasses, and 0 to n options")

data = sys.argv[1]
modelpath = sys.argv[2]
mode = sys.argv[3]
nrCl = int(sys.argv[4])

options = sys.argv[5:]


# load the data: we expect two files
depfile = data+"dep.csv"
indepfile=data+"indep.csv"
# TODO: at some point, also support getting instance weights
# weightsfile=data+"instweights.csv"

print("Loading labels: "+depfile, file=sys.stderr)
deps = np.loadtxt(depfile, delimiter=",")
print("Loading attributes: "+depfile, file=sys.stderr)
indeps = np.loadtxt(indepfile, delimiter=",")

shape = indeps.shape
print("Attributes have shape: ", shape, file=sys.stderr)

# Now to keep things simple we always try to load execute the file "lfkerastrain.py" within
# the data directory. However if this file does not exist, we try to fall back to use our own
# copy of that file
dofile=data+"lfkerastrain.py"
if not os.path.isfile(dofile):
  dofile=os.path.dirname(sys.argv[0])+os.path.sep+"lfkerastrain.py"

print("Running train file: ", dofile, file=sys.stderr)
# this file has one task and one task only: create a trained model and assign it to the variable model
# Python2 execfile was used here, but that does not work in python3
with open(dofile, 'rb') as pythonfile:
    exec(pythonfile.read())

# Lets check if we got a model


# save the model
modelfile=modelpath+".h5"
model.save(modelfile)
print("Model saved to ", modelfile, file=sys.stderr)


print("kerasTrain: finishing", file=sys.stderr)
