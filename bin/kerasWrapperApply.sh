#!/bin/bash

ROOTDIR="$1"
shift

mypython=${SKLEARN_WRAPPER_PYTHON:-python}

pushd "$ROOTDIR" >/dev/null

echo 1>&2 ${mypython} "${ROOTDIR}/python/kerasApply.py"  $@
${mypython} "${ROOTDIR}/python/kerasApply.py" $@
popd >/dev/null
