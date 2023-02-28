#!/bin/bash

DIR="/path/to/csv/of/tpch"
if [ $# -lt 2 ]; then
    echo "usage: ./loaddata.sh {dir} {db_name}"
    exit 1;
fi
DIR=$1
DB_NAME=$2

PASSWORD=123456
PORT=3306

DIR="${DIR%/}"
cd $DIR

tablename=""
fullpath=""
ext="tbl"
echo `date`
echo "--------load data from ${DIR} into database ${DB_NAME}"
# for f in `ls`; do
#     if [ -f $f ]; then
# 	  tablename="${f%.*}"
# 	  fullpath="$DIR/$f"
# 	  echo "Load $fullpath"
# 	  mysql -h:: ${DB_NAME} --execute="load data infile '$fullpath' into table $tablename fields terminated by '|';"
# 	  if [ $? != 0 ]; then
#           	echo "Load data error for : $fullpath"
# 	  fi
#     fi
# done
declare -a array
tables=("region" "nation" "part" "supplier" "partsupp" "customer" "orders" "lineitem")
for tablename in ${tables[@]}; do
    fullpath="${DIR}/${tablename}."${ext}
    echo "Load $fullpath"
    mysql -h:: -urateup -p${PASSWORD} -P ${PORT} ${DB_NAME} --execute="load data infile '$fullpath' into table $tablename fields terminated by '|';"
    if [ $? != 0 ]; then
          	echo "Load data error for : $fullpath"
    fi
done

echo `date`
