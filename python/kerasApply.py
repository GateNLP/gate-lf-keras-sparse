from __future__ import print_function
import sys
import os 
import json
import numpy as np
import keras
from keras.models import load_model

olderr = sys.stderr
oldout = sys.stdout
f = open("/dev/null", 'w')
sys.stderr = f
sys.stdout = f

# re-enable stderr, but keep suppressing stdout just to be safe ...
sys.stderr=olderr


print ("kerasApply - got args: ", sys.argv, file=sys.stderr)
if len(sys.argv) != 4:
    sys.exit("ERROR: Not exactly two arguments: [script], model path prefix, mode, nrclasses")

modelpath = sys.argv[1]
mode = sys.argv[2]
nrCl = int(sys.argv[3])

## load the keras model
model = load_model(modelpath+".h5")


# Now iterate through reading json from standard input
# We expect a map which either contains data to find predictions for or
# just an indication to stop processing.
# For this we expect the same JSON format we also use for server intercation:
# A map with the following keys:
# - values: an array of arrays of values
# - indices: an array of array of indices, if this is missing, values are dense
# - n: the number of dimensions of values, needed if sparse and we want to
#   to convert to dense
# - cmd: with value "STOP" if present and has this value, stop processing.
# The response gets written to standard output as a line of json with the following format
# - status: "OK" or some error message
# - targets: array of prediction values (regression value or class index)
# - probas: array of arrays of per-class probabilities

nlines = 0
# NOTE: apparently there is a bug in python prior to 3.3
# that forces the use of Ctrl-D twice to get EOF from the command line!!
# print("sklearnApply: before loop",file=sys.stderr)

while True:
    line = sys.stdin.readline()
    #print("kerasApply - got json line",line,file=sys.stderr)
    if line == "" :
      break
    nlines = nlines + 1
    map = json.loads(line)
    #print("JSON parsed: ",map,file=sys.stderr)
    if map['cmd'] == "STOP":
        break
    if "indices" in map:
        sys.exit("Sparse vectors not yet supported!!")
    values = np.array(map['values'])
    probs = model.predict_proba(values, verbose=False)
    # probs = model.predict(values) 
    ret = {}
    ret["status"] = "OK"
    # NOTE: instead of argmax, we could also use class index
    # prediction using model.predict_classes(values,verbose=0)
    targets = np.argmax(probs, axis=1).astype("float64")
    # print("Got probs: ",probs,file=sys.stderr)
    # print("Got targets: ",targets,file=sys.stderr)
    ret["targets"] = targets.tolist()
    ret["probas"] = probs.tolist()
    #print("sklearnApply: sending response",json.dumps(ret),file=sys.stderr)
    print(json.dumps(ret),file=oldout)
    oldout.flush()
    #print("sklearnApply: response sent",file=sys.stderr)



#print("Lines read: ", nlines,file=sys.stderr)
