#!/bin/bash
. env.rc
echo "copy data from $ARIES_DATA_DIR/* to $ARIES_DATA_TARGET_DIR/$DB_NAME"
cp $ARIES_DATA_DIR/* $ARIES_DATA_TARGET_DIR/$DB_NAME
