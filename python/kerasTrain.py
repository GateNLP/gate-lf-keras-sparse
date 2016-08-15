from __future__ import print_function
import sys
import tensorflow as tf 
import importlib
import os
import inspect
import numpy as np

## This gets invoked by the EngineKerasWrapper which is just an instance of EnginePythonNetworksbase which
## passes the following parameters:
## - [rootdirectory]: this gets removed by the invoking bash script and not passed on to us
## - path to the model/data directory with the directory separator at the end
## - path prefix for the model file: the complete path plus the beginning of the file name(s) for 
##   the model
## - mode: classind/classcosts/regr
## - nrClasses: 0 for regression or the number of classes otherwise
## - any algorithm parameters specified for the PR


## IMPORTANT: the option values need to be valid python expressions, and they
## will get evaluated! So in order to pass a string, enclose the value in quotes!


print("kerasTrain - got args: ", sys.argv, file=sys.stderr)
if len(sys.argv) < 6:
	sys.exit("ERROR: Not at least 5 arguments: [script], model/data directory name, model base name, mode, nrClasses, and 0 to n options")

data=sys.argv[1]
modelpath=sys.argv[2]
mode=sys.argv[3]
nrCl=sys.argv[4]

options=sys.argv[5:]


## load the data: we expect two files
depfile = data+"dep.csv"
indepfile=data+"indep.csv"
## TODO: at some point, also support getting instance weights
## weightsfile=data+"instweights.csv"

deps = np.loadtxt(depfile)
indeps = np.loadtxt(indepfile)

shape = indeps.shape

## Now the actual training is done based on code which is in a file for which we 
## got the file name as an option, or we fall back to one of our own simple default files.

## for argument parsing use: argparse: https://docs.python.org/2/library/argparse.html#module-argparse
## for invoking another python file use one of:
## === Solution 1:
## import sys
## import os
## sys.path.append(os.path.abspath("/the/directory")
## ## if this directory ontains a file module.py
## from module import *
## ## imports all function defined in that file
## === Solution 2:
## ## use importlib, but better implemented in >3.1?
## ##then importlib.import_module(name,package=None)
## ## SEE http://stackoverflow.com/a/67692/1382437
## === Solution 3:
## ## see https://docs.python.org/3/library/runpy.html#runpy.run_module



print("CAUTION: NOT WORKING PROPERLY YET!!")


