#!/bin/bash

ROOTDIR="$1"
shift
model="$1"
shift

pushd "$ROOTDIR" >/dev/null

echo 1>&2 python "${ROOTDIR}/python/tensorflowApply.py" "${model}" $@
python "${ROOTDIR}/python/tensorflowApply.py" "${model}" $@
popd >/dev/null
