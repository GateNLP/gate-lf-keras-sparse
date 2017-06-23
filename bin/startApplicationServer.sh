#!/bin/bash

PRG="$0"
CURDIR="`pwd`"
# need this for relative symlinks
while [ -h "$PRG" ] ; do
  ls=`ls -ld "$PRG"`
  link=`expr "$ls" : '.*-> \(.*\)$'`
  if expr "$link" : '/.*' > /dev/null; then
    PRG="$link"
  else
    PRG=`dirname "$PRG"`"/$link"
  fi
done
SCRIPTDIR=`dirname "$PRG"`
SCRIPTDIR=`cd "$SCRIPTDIR"; pwd -P`
MYROOTDIR=`cd "$SCRIPTDIR/.."; pwd -P`


ROOTDIR="$KERAS_WRAPPER_HOME"
modeldir="$1"
shift

if [ "x$ROOTDIR" == "x" ]
then
  export ROOTDIR="$MYROOTDIR"
fi


pushd "$ROOTDIR" >/dev/null

echo 1>&2 python "${ROOTDIR}/python/kerasApplicationServer.py" "${modeldir}" $@
python "${ROOTDIR}/python/kerasApplicationServer.py" "${modeldir}" $@
popd >/dev/null
