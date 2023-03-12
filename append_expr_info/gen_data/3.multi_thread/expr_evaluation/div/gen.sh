LEN_list=(LEN_4 LEN_8 LEN_16 LEN_32)
for (( i = 0; i < ${#LEN_list[*]}; i++ )); do
    mkdir ${LEN_list[$i]}
    g++ -D${LEN_list[$i]} a.cpp
    ./a.out
done
rm ./a.out