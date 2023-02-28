#!/bin/bash

DIR="/path/to/csv/of/tpcds"
if [ $# -lt 2 ]; then
    echo "usage: ./load_tpcds_data.sh {dir} {db_name}"
    exit 1;
fi
DIR=$1
DB_NAME=$2

DIR="${DIR%/}"
cd $DIR

tablename=""
fullpath=""
ext="dat"
echo `date`
echo "--------load data from ${DIR} into database ${DB_NAME}"

declare -a array
tables=("call_center" "catalog_page" "catalog_returns" "catalog_sales" "customer" "customer_address" "customer_demographics" "date_dim" "dbgen_version" "household_demographics" "income_band" "inventory" "item" "promotion" "reason" "ship_mode" "store" "store_returns" "store_sales" "time_dim" "warehouse" "web_page" "web_returns" "web_sales" "web_site")

for tablename in ${tables[@]}; do
    fullpath="${DIR}/${tablename}."${ext}
    echo "Load $fullpath"
    # mysql -h:: ${DB_NAME} --local-infile=1 --execute="load data local infile '$fullpath' into table $tablename fields terminated by '|';"
    mysql -h:: ${DB_NAME} --execute="load data infile '$fullpath' into table $tablename fields terminated by '|';"
    if [ $? != 0 ]; then
          	echo "Load data error for : $fullpath"
    fi
done

echo `date`
