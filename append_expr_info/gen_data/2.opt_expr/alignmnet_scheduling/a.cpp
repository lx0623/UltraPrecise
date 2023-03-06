#include <iostream>
#include <cstdlib>
#include <time.h>
#include <string.h>
#include <fstream>
#include <cstring>
using namespace std;
#define random(x) rand()%(x)
string gen_data(int length){
    string data_str = "";
    int t = random(10);
    while( t == 0 ){
        t = random(10);
    }
    data_str += to_string(t);
    for(int i=1; i<length; i++){
        t = random(10);
        data_str += to_string(t);
    }
    return data_str;
}
int main()
{
    long long number = 10000000;
#ifdef LEN_2
    int prec[] = {2,17};
    int scale[] = {1,11};
    string file = "./LEN_2/data.tbl";
#elif LEN_4
    int prec[] = {20,18};
    int scale[] = {1,11};
    string file = "./LEN_4/data.tbl";
#elif LEN_8
    int prec[] = {56,18};
    int scale[] = {1,11};
    string file = "./LEN_8/data.tbl";
#elif LEN_16
    int prec[] = {128,18};
    int scale[] = {1,11};
    string file = "./LEN_16/data.tbl";
#elif LEN_32
    int prec[] = {272,18};
    int scale[] = {1,11};
    string file = "./LEN_32/data.tbl";
#endif
    int data_per_row = 2;
    srand((unsigned)time(NULL));
    string filename(file);
    fstream file_out;
    file_out.open(filename, ios_base::out);
    for (int index = 0; index < number; index++){
        string row_data = "";
        for(int row_index = 0; row_index < data_per_row; row_index++){
            int sign = random(2);
            if(sign == 1)
                row_data += "-"+ gen_data(prec[row_index] - scale[row_index]) + "." + gen_data(scale[row_index]);
            else
                row_data += gen_data(prec[row_index] - scale[row_index]) + "." + gen_data(scale[row_index]);
            if(row_index < data_per_row - 1)
                row_data += "|";
        }
        file_out<< row_data <<endl;
    }
    return 0; 
}