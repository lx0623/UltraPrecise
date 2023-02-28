#!/bin/bash
rm -rf output
mkdir output
./comp_err -F ./errmsg-utf8.txt -D ./output/
cp ./output/english/errmsg.sys ./english/
mv *.h ../include/
rm -rf output
