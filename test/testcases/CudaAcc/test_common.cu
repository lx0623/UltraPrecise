/*
 * test_common.cu
 *
 *  Created on: Jun 11, 2019
 *      Author: lichi
 */

#include "test_common.h"

int loadColumn( const char *colFile, int& colSize, char **data )
{
    //printf("%s\n", attributeFile);
    columnHeader header;
    long outSize;
    char *outTable;
    int offset = sizeof(struct columnHeader);
    int outFd = open( colFile, O_RDONLY );
    if( outFd == -1 )
    {
        printf( "[error] cannot open %s", colFile );
        exit( -1 );
    }
    //long pos = lseek( outFd, 0, SEEK_END ) - 4096;
    read( outFd, &header, sizeof(struct columnHeader) );
    colSize = header.itemLen + header.containNull;
    long tupleNum = header.rows;
    outSize = tupleNum * colSize;
    lseek( outFd, 0, SEEK_SET );
    outTable = ( char * )mmap( 0, outSize, PROT_READ, MAP_SHARED, outFd, offset );

    *data = ( char * )memalign( 256, outSize );

    memcpy( *data, outTable, outSize );
    munmap( outTable, outSize );
    close( outFd );
    return tupleNum;
}

int loadColumnToDevice(const char* file, int colSize, char* &data)
{
    char **h_flag=(char**)malloc(sizeof(char**));
    int len = loadColumn(file, colSize, h_flag);
    cudaMallocManaged(&data, len * colSize * sizeof(char));
    cudaMemcpy(data, *h_flag, len * colSize, cudaMemcpyHostToDevice);
    free(*h_flag);
    free(h_flag);
    return len;
}

void cleanStringTail(char* data, int colSize, int count)
{
    for(int i=0; i<count; i++)
    {
        bool tail = false;
        char* item = data+i*colSize;
        for(int p=0; p<colSize; p++)
        {
            if (tail) item[p]=0;
            else if (item[p]==0) tail = true;
        }
    }
}

int loadCharCol(const char* file, char* &dates, int colSize)
{

    char **h_data=(char**)malloc(sizeof(char**));
    int len = loadColumn(file, colSize, h_data);

    cudaMallocManaged(&dates, len*colSize*sizeof(char));
    cudaMemcpy(dates, *h_data, len*colSize, cudaMemcpyHostToDevice);
    free(*h_data);
    free(h_data);
    return len;
}

int loadIntCol( const char* file, int* &data )
{
    int colSize = sizeof(int);
    char **h_data = ( char** )malloc( sizeof(char**) );
    int len = loadColumn( file, colSize, h_data );

    cudaMallocManaged( &data, len * sizeof(int) );
    cudaMemcpy( data, *h_data, len * sizeof(int), cudaMemcpyHostToDevice );
    free( *h_data );
    free( h_data );
    return len;
}
int loadNullIntCol( const char* file, nullable_type<int>* &data )
{
    int colSize = sizeof(nullable_type<int>);
    char **h_data = ( char** )malloc( sizeof(char**) );
    int len = loadColumn( file, colSize, h_data );

    cudaMallocManaged( &data, len * sizeof(nullable_type<int>) );
    cudaMemcpy( data, *h_data, len * sizeof(nullable_type<int>), cudaMemcpyHostToDevice );
    free( *h_data );
    free( h_data );
    return len;
}
int loadFloatCol( const char* file, float* &data )
{
    int colSize = sizeof(float);
    char **h_data = ( char** )malloc( sizeof(char**) );
    int len = loadColumn( file, colSize, h_data );

    cudaMallocManaged( &data, len * sizeof(float) );
    cudaMemcpy( data, *h_data, len * sizeof(float), cudaMemcpyHostToDevice );
    free( *h_data );
    free( h_data );
    return len;
}

int loadDecimalCol( const char* file, int precision, int scale, aries_acc::Decimal* &data )
{
    int colSize = sizeof(aries_acc::Decimal);
    char **h_data = ( char** )malloc( sizeof(char**) );
    int len = loadColumn( file, colSize, h_data );

    cudaMallocManaged( &data, len * sizeof(aries_acc::Decimal) );
    cudaMemcpy( data, *h_data, len * sizeof(aries_acc::Decimal), cudaMemcpyHostToDevice );
    free( *h_data );
    free( h_data );
    return len;
    return len;
}

int loadLineItemShipdate(char* &dates)
{
    int colSize=10;
    return loadCharCol("/home/lichi/oldaries/src/utility/lineitem10", dates, colSize);
}

int loadOrdersShipdate(char* &dates)
{
    int colSize=10;
    return loadCharCol("/var/rateup/data/scale_1/orders4", dates, colSize);
}

int loadPartType(char* &data)
{//part4
    int colSize=25;
    const char* file = "/var/rateup/data/scale_1/part4";
    int count = loadCharCol(file, data, colSize);
    cleanStringTail(data, colSize, count);
    return count;
}
int loadPartSize(int* &data)
{//part5
    const char* file = "/var/rateup/data/scale_1/part5";
    return loadIntCol(file, data);
}


int loadPsPartkey(int* &data)
{
    const char* file = "/var/rateup/data/scale_1/partsupp0";
    return loadIntCol(file, data);
}

int loadPsSuppkey(int* &data)
{
    const char* file = "/var/rateup/data/scale_1/partsupp1";
    return loadIntCol(file, data);
}

int loadLineStatus(char* &data)
{
    return loadCharCol("/var/rateup/data/scale_1/lineitem8", data, 1);
}

int loadReturnFlag(char* &data)
{
    return loadCharCol("/var/rateup/data/scale_1/lineitem9", data, 1);
}

int loadLineNumber(int* &nums)
{
    return loadIntCol("/var/rateup/data/scale_1/lineitem3", nums);
}

int loadLDiscount(float* &data)
{
    return loadFloatCol("/var/rateup/data/scale_1/lineitem6", data);
}

int loadOTotalprice(float* &data)
{
    return loadFloatCol("/var/rateup/data/scale_1/orders3", data);
}
int loadOCustkey(int* &data)
{
    return loadIntCol("/var/rateup/data/scale_1/orders1", data);
}

int loadOOrderkey(int* &data)
{
    return loadIntCol("/var/rateup/data/scale_1/orders0", data);
}

int loadLOrderkey(int* &data)
{
    return loadIntCol("/var/rateup/data/scale_1/lineitem0", data);
}


int loadLPartkey(int* &data)
{
    return loadIntCol("/var/rateup/data/scale_1/lineitem1", data);
}

void printArray(int* array, int count, const char* prefix)
{
    int row_num = count / 10;
    for (int row = 0; row < row_num; row++)
    {
        if(prefix != NULL) printf("%s", prefix);

        printf("[%3d] ", row);
        int col_num = min(row * 10 + 10, count) - row * 10;
        for( int col = 0; col < col_num; col++)
        {
            printf("%10d", array[row * 10 + col]);
        }
        printf("\n");
    }
    if (count % 10)
    {
        if(prefix != NULL) printf("%s", prefix);
        printf("[%3d] ", row_num);
        for(int i=row_num*10; i<count; i++)
        {
            printf("%10d", array[i]);
        }
        printf("\n");
    }
}

void printArray(float* array, int count, const char* prefix)
{
    int row_num = count / 10;
    for (int row = 0; row < row_num; row++)
    {
        if(prefix != NULL) printf("%s", prefix);

        printf("[%3d] ", row);
        int col_num = min(row * 10 + 10, count) - row * 10;
        for( int col = 0; col < col_num; col++)
        {
            printf("%3.5f", array[row * 10 + col]);
        }
        printf("\n");
    }
    if (count % 10)
    {
        if(prefix != NULL) printf("%s", prefix);
        printf("[%3d] ", row_num);
        for(int i=row_num*10; i<count; i++)
        {
            printf("%3.5f", array[i]);
        }
        printf("\n");
    }
}

void printDates(const char* array, int count, const char* prefix)
{
    char d[11];
    int row_num = count / 10;
    for (int row = 0; row < row_num; row++)
    {
        if(prefix != NULL) printf("%s", prefix);

        printf("[%3d] ", row);
        int col_num = min(row * 10 + 10, count) - row * 10;

        for( int col = 0; col < col_num; col++)
        {
            getDate(array, row * 10 + col, d);
            printf("%s  ", d);
        }
        printf("\n");
    }
    if (count % 10)
    {
        if(prefix != NULL) printf("%s", prefix);
        printf("[%3d] ", row_num);
        for(int i=row_num*10; i<count; i++)
        {
            getDate(array, i, d);
            printf("%s  ", d);

        }
        printf("\n");
    }
}
void dumpDateArray(const char* filename, const char* array, int count)
{
    std::ofstream outfile(filename);
    outfile<<count<<std::endl;
    for(int i=0; i<count; i++)
    {
        for(int k=0; k<10; k++)
            outfile<<array[i*10+k];
        outfile<<std::endl;
    }
    outfile.close();
}
void dumpIntArray(const char* filename, const int* array, int count)
{
    std::ofstream outfile(filename);
    outfile<<count<<std::endl;
    for(int i=0; i<count; i++)
    {
        outfile<<array[i]<<std::endl;
    }
    outfile.close();
}

int loadIntArray(const char* filename, int* &array)
{
    std::ifstream f(filename);
    char buff[20];
    f.getline(buff, 20);
    int count=0;
    sscanf(buff, "%d", &count);
    cudaMallocManaged(&array, count*sizeof(int));
    for(int i=0; i<count; i++)
    {
        f.getline(buff, 20);
        sscanf(buff, "%d", &array[i]);
    }
    f.close();
    return count;
}

void dumpFloatArray(const char* filename, const float* array, int count)
{
    std::ofstream outfile(filename);
    outfile<<count<<std::endl;
    for(int i=0; i<count; i++)
    {
        outfile<<array[i]<<std::endl;
    }
    outfile.close();
}

int loadFloatArray(const char* filename, float* &array)
{
    std::ifstream f(filename);
    char buff[20];
    f.getline(buff, 20);
    int count=0;
    sscanf(buff, "%d", &count);
    cudaMallocManaged(&array, count*sizeof(int));
    for(int i=0; i<count; i++)
    {
        f.getline(buff, 20);
        sscanf(buff, "%f", &array[i]);
    }
    f.close();
    return count;
}

int loadCharArray(const char* filename, char* &array, int colSize)
{
    std::ifstream f(filename);
    char buff[120];
    f.getline(buff, 120);
    int count=0;
    sscanf(buff, "%d", &count);
    cudaMallocManaged(&array, count*colSize);
    for(int i=0; i<count; i++)
    {
        f.getline(buff, 120);
        sscanf(buff, "%s", &array[i*colSize]);
    }
    f.close();
    return count;
}

__global__ void gpuwarmup(char* data, int len)
{
    char i = data[blockIdx.x * blockDim.x + threadIdx.x];
    i=i+1;
}

void warmup(char* data, int len)
{
    gpuwarmup<<<(len + 256 - 1)/256, 256>>>(data, len);
}

void  printShipDate(char* data, int index)
{
    char buff[11];
    buff[10]=0;

    memcpy(buff,data+index*10*sizeof(char), 10);
    printf("[%d] %s\n", index, buff);
}

AriesDataBufferSPtr ReadColumn( const string& dbName, const string& tableName, int columnId )
{
    vector< int > columnIds;
    columnIds.push_back( columnId );
    aries_engine::AriesInitialTableSPtr initialTable = make_shared< aries_engine::AriesInitialTable >( dbName, tableName );
    auto table = initialTable->GetTable( columnIds );
    auto column = table->GetColumnBuffer( 1 );
    return column;
}

aries_engine::AriesTableBlockUPtr ReadTable( const string& dbName, const string& tableName, const vector< int >& columnIds )
{
    aries_engine::AriesInitialTableSPtr initialTable = make_shared< aries_engine::AriesInitialTable >( dbName, tableName );
    return initialTable->GetTable( columnIds );
}
