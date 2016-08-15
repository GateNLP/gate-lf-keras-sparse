#!/bin/bash

ROOTDIR="$1"
shift

pushd "$ROOTDIR" >/dev/null

echo 1>&2 python "${ROOTDIR}/python/kerasTrain.py"  $@
python "${ROOTDIR}/python/kerasTrain.py"  $@
popd >/dev/null
