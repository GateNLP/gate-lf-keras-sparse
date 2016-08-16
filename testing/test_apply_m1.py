from __future__ import print_function
import numpy as np
import sys
# f = open('/dev/null','w')
#oout=sys.stdout
#oerr=sys.stderr
#sys.stdout=f
#sys.stderr=f
from keras.models import load_model
#sys.stderr=oerr

X=np.loadtxt("m1/indep.csv",delimiter=",")
y=np.loadtxt("m1/dep.csv",delimiter=",")
model=load_model("m1/kerasmodel.h5")

probs=model.predict_proba(X[0:10],verbose=False)
## without verbose=False we get unwanted output
## probs=model.predict_proba(X[0:10],verbose=True)

## print("true labels: ",y[0:10])
## print("predictions: ",probs[0:10])

print("finish!",file=sys.stderr)

