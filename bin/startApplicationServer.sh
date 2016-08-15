#!/bin/bash

ROOTDIR="$1"
shift
modeldir="$1"
shift

pushd "$ROOTDIR" >/dev/null

echo 1>&2 python "${ROOTDIR}/python/kerasApplicationServer.py" "${modeldir}" $@
python "${ROOTDIR}/python/kerasApplicationServer.py" "${modeldir}" $@
popd >/dev/null
