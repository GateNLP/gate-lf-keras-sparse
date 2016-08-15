#!/bin/bash

ROOTDIR="$1"
shift
modeldir="$1"
shift

pushd "$ROOTDIR" >/dev/null

echo 1>&2 python "${ROOTDIR}/python/kerasApply.py" "${modeldir}" $@
python "${ROOTDIR}/python/kerasApply.py" "${modeldir}" $@
popd >/dev/null
