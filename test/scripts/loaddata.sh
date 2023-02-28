#!/bin/sh

DIR="/home/tengjp/git/data/scale_1"
if [ $# -gt 0 ]; then
    DIR=$1
fi
DIR="${DIR%/}"
cd $DIR
DB_NAME="${DIR##*/}"

tablename=""
fullpath=""
echo `date`
echo "--------load data from ${DIR} into database ${DB_NAME}"
for f in `ls`; do
    if [ -f $f ]; then
	  tablename="${f%.*}"
	  fullpath="$DIR/$f"
	  echo "Load $fullpath"
	  mysql -h:: ${DB_NAME} --execute="load data infile '$fullpath' into table $tablename fields terminated by '|';"
	  if [ $? != 0 ]; then
          	echo "Load data error for : $fullpath"
	  fi
    fi
done

echo `date`
