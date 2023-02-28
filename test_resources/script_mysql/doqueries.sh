#!/bin/bash
. env.rc
mkdir -p "$SQL_RESULT_DIR"
for i in {1..22};
do
  echo "do query: $i.sql"
  time ./query_1.sh $i > $SQL_RESULT_DIR/$i.result
done
