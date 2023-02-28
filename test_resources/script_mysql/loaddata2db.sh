#!/bin/bash
. env.rc
O_DATA_DIR=$(pwd)/$ORG_DATA_DIR
if [[ $DB_ENGINE = "mapd" ]]; then
  if [[ -n $LOADDATA_FILE ]]; then
    echo "do...."
    sed "s!{DATA_PATH}!$O_DATA_DIR!g" $LOADDATA_FILE | omnisql -u$USER -p$PASSWORD --db $DB_NAME
  fi
elif [[ $DB_ENGINE = "aries" ]]; then
  ./$LOADDATA_FILE
else
  if [[ -n $LOADDATA_FILE ]]; then
    sed "s!{DATA_PATH}!$O_DATA_DIR!g" $LOADDATA_FILE | mysql -h127.0.0.1 -u$USER -p$PASSWORD -D $DB_NAME
  fi
fi
./preloaddata.sh
