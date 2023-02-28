#!/bin/bash
. env.rc
echo "create db: $DB_NAME"
if [[ $DB_ENGINE = "mapd" ]]; then
  sed "s!{DB_NAME}!$DB_NAME!g" $CREATE_TABLE_FILE | omnisql -uroot -pabc@123
else
  sed "s!{DB_NAME}!$DB_NAME!g" $CREATE_TABLE_FILE | mysql -h127.0.0.1 -uroot -pabc@123
fi
