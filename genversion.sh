#!/bin/env bash

BUILD_HASH=$(git show --pretty=format:%h -s HEAD 2>/dev/null)
BUILD_DATE=$(date "+%Y%m%d%H%M%S")


if [ "0" = "$?" ]; then
    echo "#define BUILD_HASH \"${BUILD_HASH}\"" > version_hash.h
else
    echo "#define BUILD_HASH \"undefined\"" > version_hash.h
fi

echo "#define BUILD_DATE \"${BUILD_DATE}\"" >> version_hash.h
