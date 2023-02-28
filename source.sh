#!/bin/bash

BUILD_DIR=debug
RATEUP_PORT=3333
DATA_DIR=$BUILD_DIR/data
PASSWORD=123456
ROOT_DIR=$PWD

alias serv-start="$BUILD_DIR/rateup --port=$RATEUP_PORT --datadir=$DATA_DIR"
alias cli-connect="mysql -h:: -u rateup --password=$PASSWORD -P $RATEUP_PORT"
# cli-exec-db scale1 < debug/test_resources/schema/tpch/create_table.sql 
alias cli-exec-db="mysql -h:: -u rateup --password=$PASSWORD -P $RATEUP_PORT -D"
# cli-exec "create database if not exists scale10"
alias cli-exec="mysql -h:: -u rateup --password=$PASSWORD -P $RATEUP_PORT -e "
alias log-error="tail -F $DATA_DIR/log/rateup.ERROR"
alias log-warning="tail -F $DATA_DIR/log/rateup.WARNING"
alias log-info="tail -F $DATA_DIR/log/rateup.INFO"
alias create-tags="ctags --langmap=C++:+.cu -e -R ."
alias root-dir="cd $ROOT_DIR"
