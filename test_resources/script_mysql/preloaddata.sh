#!/bin/bash
. env.rc
echo "preload data ..."
if [[ $DB_ENGINE = "mapd" ]]; then
  cat select_all.sql | omnisql -u$USER -p$PASSWORD --db $DB_NAME > /dev/null
else
  cat select_all.sql | mysql -h127.0.0.1 -u$USER -p$PASSWORD -q -D $DB_NAME > /dev/null
fi
