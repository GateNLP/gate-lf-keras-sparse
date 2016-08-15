#!/bin/bash

ROOTDIR="$1"
shift
data="$1"
shift
modeldir="$1"
shift
algorithmfile="$1"
shift

pushd "$ROOTDIR" >/dev/null

echo 1>&2 python "${ROOTDIR}/python/kerasTrain.py" "${data}" "${modeldir}" "${algorithm}" $@
python "${ROOTDIR}/python/kerasTrain.py" "${data}" "${modeldir}" "${algorithmfile}" $@
popd >/dev/null
