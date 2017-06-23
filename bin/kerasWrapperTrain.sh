#!/bin/bash

ROOTDIR="$KERAS_WRAPPER_HOME"

mypython=${KERAS_WRAPPER_PYTHON:-python}

pushd "$ROOTDIR" >/dev/null

echo 1>&2 ${mypython} "${ROOTDIR}/python/kerasTrain.py"  $@
${mypython} "${ROOTDIR}/python/kerasTrain.py"  $@
popd >/dev/null
