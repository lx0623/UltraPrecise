#pragma once
#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <malloc.h>
#include <iostream>
#include <fstream>
#include <sys/mman.h>
#include <gtest/gtest.h>
#include <CudaAcc/algorithm/kernel_join.hxx>
#include <CudaAcc/algorithm/kernel_merge.hxx>
#include <CudaAcc/algorithm/kernel_mergesort.hxx>
#include <CudaAcc/algorithm/kernel_radixsort.hxx>
#include <CudaAcc/algorithm/kernel_shuffle.hxx>
#include <CudaAcc/algorithm/kernel_sortedsearch.hxx>
#include <CudaAcc/algorithm/kernel_segsort.hxx>
#include <CudaAcc/algorithm/kernel_load_balance.hxx>
#include <CudaAcc/algorithm/kernel_segreduce.hxx>
#include <CudaAcc/algorithm/kernel_reduce.hxx>
#include <CudaAcc/algorithm/kernel_filter.hxx>
#include <CudaAcc/algorithm/kernel_util.hxx>
#include <CudaAcc/algorithm/kernel_intervalmove.hxx>
#include <CudaAcc/algorithm/kernel_adapter.hxx>
#include "AriesEngine/transaction/AriesInitialTable.h"
#include "CpuTimer.h"

using namespace aries_acc;
using namespace std;

struct ARIES_PACKED columnHeader
{
    int32_t version;
    uint64_t rows;
    int8_t containNull;
    int16_t itemLen;
    char padding[4081]; /* for futher use */
};

int loadColumn(const char *colFile, int& colSize, char **data);
int loadFloatCol( const char* file, float* &data );
int loadIntCol( const char* file, int* &data );
int loadNullIntCol( const char* file, nullable_type<int>* &data );
int loadCharCol(const char* file, char* &dates, int colSize);
int loadColumnToDevice(const char* file, int colSize, char* &data);

int loadLineItemShipdate(char* &dates);
int loadOrdersShipdate(char* &dates);
int loadPartType(char* &data);
int loadPartSize(char* &data);
int loadPsPartkey(int* &data);
int loadPsSuppkey(int* &data);
int loadReturnFlag(char* &data);
int loadLineStatus(char* &data);
int loadLineNumber(int* &nums);
int loadLDiscount(float* &data);
int loadOTotalprice(float* &data);
int loadOCustkey(int* &data);
int loadOOrderkey(int* &data);
int loadLOrderkey(int* &data);
int loadLPartkey(int* &data);

void  printShipDate(char* data, int index);
void printDates(const char* array, int count, const char* prefix=NULL);
void printArray(int* array, int count, const char* prefix=NULL);
void printArray(float* array, int count, const char* prefix=NULL);

void dumpIntArray(const char* filename, const int* array, int count);
int loadIntArray(const char* filename, int* &array);

void dumpFloatArray(const char* filename, const float* array, int count);
int loadFloatArray(const char* filename, float* &array);

void dumpDateArray(const char* filename, const char* array, int count);

int loadCharArray(const char* filename, char* &array, int colSize);
// namespace aries_acc {
// struct CPU_Timer{
//     long start;
//     long stop;
//     void begin()
//     {
//         struct timeval tv;
//         gettimeofday(&tv,NULL);
//         start = tv.tv_sec * 1000 + tv.tv_usec/1000;
//     }
//     long end()
//     {
//         struct timeval tv;
//         gettimeofday(&tv,NULL);
//         stop = tv.tv_sec * 1000 + tv.tv_usec/1000;
//         long elapsed = stop -start;
//         printf("cpu time: %ld\n", elapsed);
//         return elapsed;
//     }
// };
// }
__inline__ void printDates(const char* data, int offset, int count, const char* format)
{
    const int size = 10;
    char * buff = (char *) malloc(size + 1);
    buff[size] = 0;
    for(int i=offset; i<offset + count; i++)
    {
        memcpy(buff, data + i * size, size);
        printf("[%5d] ", i);
        printf(format, buff);
    }
    free(buff);
}

__inline__ void printString(const char* data, int size, int index, const char* format)
{
    char * buff = (char *) malloc(size + 1);
    buff[size] = 0;
    memcpy(buff, data + index * size, size);
    printf(format, buff);

    free(buff);
}

__inline__  void getDate(const char* dates, int index, char* d11)
{
    d11[10] = 0;
    memcpy(d11, dates+index*10, 10);
}

void warmup(char* data, int len);

int loadDecimalCol(const char* file, int precision, int scale, aries_acc::Decimal* &data);

AriesDataBufferSPtr ReadColumn( const string& dbName, const string& tableName, int columnId );

aries_engine::AriesTableBlockUPtr ReadTable( const string& dbName, const string& tableName, const vector< int >& columnIds );
