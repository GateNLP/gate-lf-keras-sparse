#!/bin/bash

ROOTDIR="$1"
shift

pushd "$ROOTDIR" >/dev/null

echo 1>&2 python "${ROOTDIR}/python/kerasApply.py"  $@
python "${ROOTDIR}/python/kerasApply.py" $@
popd >/dev/null
