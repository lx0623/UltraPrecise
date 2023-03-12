LEN_list=(one two thr)
for (( i = 0; i < ${#LEN_list[*]}; i++ )); do
    python3 ${LEN_list[$i]}/gen.py > ./${LEN_list[$i]}/data.tbl
done
for (( i = 0; i < ${#LEN_list[*]}; i++ )); do
    g++ -D${LEN_list[$i]} a.cpp -lmpfr -lgmp 
    ./a.out > ./${LEN_list[$i]}/ans.tbl
done
rm ./a.out