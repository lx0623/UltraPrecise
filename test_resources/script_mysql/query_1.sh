#!/bin/bash
. env.rc
if [[ $DB_ENGINE = "mapd" ]]; then
  cat $SQL_DIR/$1.sql | omnisql -u$USER -p$PASSWORD --db $DB_NAME
else
  cat $SQL_DIR/$1.sql | mysql -h127.0.0.1 -u$USER -p$PASSWORD -D $DB_NAME
fi
