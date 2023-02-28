#!/bin/bash
declare -i num=0
for f in ../test_tpch_queries/*.sql; do
    echo "checking test case $f"
    basename=`basename ${f}`
    noext=${basename%.sql}
    cmd="valgrind --tool=memcheck --leak-check=yes ./aries --gtest_filter=query.q${noext}"
    echo "${cmd}"

    logName="valgrind-gtest-q${basename}.log"
    ${cmd} &> ${logName}
    # num=`expr $num + 1`
done

grep "definitely lost:" valgrind-gtest-q*.log | sort -t ':' -k 1 -r | awk '$4 > 0 {print}'
