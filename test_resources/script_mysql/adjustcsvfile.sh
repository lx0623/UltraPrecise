#!/bin/bash
. env.rc
for f in `ls $ORG_DATA_DIR/*.tbl`;
do
  echo "adjust csv file: $f"
  `sed -i 's/|$//g' $f`
done
