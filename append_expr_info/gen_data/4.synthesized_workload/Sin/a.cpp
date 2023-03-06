#include <stdio.h>
#include <gmp.h>
#include <stdlib.h>
#include <mpfr.h>
#include <string>
#include <iostream>
#include <fstream>
using namespace std;
int main (int argc, char *argv[])
{
    int percision = 2048;
    mp_prec_t p = percision;
    mpfr_t a;
    mpfr_inits2 (p, a, (mpfr_ptr) 0);

    // two or thr
    string inPath = "./one/";
    string filename( inPath+"data.tbl");
    fstream file_in;
    file_in.open(filename, ios_base::in);
    string str;
    while ( file_in>>str ){
        char data[11];
        if(str.length()!=10){
            cout<<"error data"<<endl;
            exit(0);
        }
        int i;
        for( i=0; i<str.length(); i++){
            data[i] = str[i];
        }
        data[i]='\0';
        mpfr_set_str (a, data, 10, GMP_RNDN);
        mpfr_sin (a, a, GMP_RNDN);
        cout<<str<<"|";
        mpfr_printf ("%.286Rf\n", a);
    }
    
    mpfr_clears (a, NULL);
    return 0;
}

// g++ a.cpp -lmpfr -lgmp