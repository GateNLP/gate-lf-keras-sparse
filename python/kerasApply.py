from __future__ import print_function
import sys
import keras 
import json
from keras.models import load_model


print ("kerasApply - got args: ", sys.argv, file=sys.stderr)
if len(sys.argv) != 2:
	sys.exit("ERROR: Not exactly two arguments: [script] and model path")

modelpath=sys.argv[1]

## load the keras model
model = load_model("keras_model.h5")
## TODO: check for errors

## TODO: figure out model capabilities and other things we need to know
## for the applicaiton step!

## Now iterate through reading json from standard input
## We expect a map which either contains data to find predictions for or
## just an indication to stop processing.
## For this we expect the same JSON format we also use for server intercation:
## A map with the following keys:
## - values: an array of arrays of values
## - indices: an array of array of indices, if this is missing, values are dense
## - n: the number of dimensions of values, needed if sparse and we want to
##   to convert to dense
## - cmd: with value "STOP" if present and has this value, stop processing.
## The response gets written to standard output as a line of json with the following format
## - status: "OK" or some error message
## - targets: array of prediction values (regression value or class index)
## - probas: array of arrays of per-class probabilities 

nlines=0
## NOTE: apparently there is a bug in python prior to 3.3 
## that forces the use of Ctrl-D twice to get EOF from the command line!!
##print("sklearnApply: before loop",file=sys.stderr)
while True:
	line = sys.stdin.readline()
	print("kerasApply - got json line",file=sys.stderr)
	if line == "" :
	  break
	nlines = nlines + 1
	map=json.loads(line)
	##print("JSON parsed: ",map,file=sys.stderr)
	if map['cmd'] == "STOP":
		break
	## create an array of dense value arrays from the json	
	X = csr_matrix((map['values'],(map['rowinds'],map['colinds'])),shape=(map['shaperows'],map['shapecols']))
	## print "Matrix is: ",X.toarray()
	ret = {}
	ret["status"] = "OK"
	if canProbs:
		probs = model.predict_proba(X)
		targets = np.argmax(probs,axis=1).astype("float64")
		#print "Got probs: ",probs
		#print "Got targets: ",targets
		ret["targets"] = targets.tolist()
		ret["probas"] = probs.tolist()
	else:
		targets = model.predict(X)
		#print "Got targets: ",targets
		ret["targets"] = targets.tolist()

	##print("sklearnApply: sending response",file=sys.stderr)
	print(json.dumps(ret))
	sys.stdout.flush()
	##print("sklearnApply: response sent",file=sys.stderr)

	

##print("Lines read: ", nlines,file=sys.stderr)
