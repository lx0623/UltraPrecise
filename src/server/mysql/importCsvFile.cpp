#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <sys/mman.h>
#include <sys/statfs.h>
#include <sys/stat.h>
#include <future>
#include <mutex>

#include "AriesColumnType.h"
// #include "utils/cpu_timer.h"
#include "utils/string_util.h"
#include "server/mysql/include/sql_class.h"
#include "server/mysql/include/my_thread_local.h"
#include "server/mysql/include/my_sys.h"
#include "server/mysql/include/mysys_err.h"
#include "server/mysql/include/sys_vars.h"
#include "boost/filesystem/operations.hpp"

#include "../datatypes/AriesDatetimeTrans.h"
#include "../datatypes/decimal.hxx"
#include "schema/SchemaManager.h"
#include "Compression/dict/AriesDictManager.h"
#include "schema/DatabaseEntry.h"
#include "schema/TableEntry.h"
#include "frontend/SQLExecutor.h"
#include "utils/thread.h"
#include "utils/utils.h"
#include "AriesEngine/transaction/AriesMvccTableManager.h"
#include "AriesEngine/transaction/AriesInitialTableManager.h"
#include "AriesEngine/transaction/AriesTuple.h"
#include "AriesEngine/transaction/AriesXLogManager.h"
#include "AriesEngine/index/AriesIndex.h"
#include "AriesEngine/AriesUtil.h"
#include "Compression/dict/AriesDict.h"
#include "server/Configuration.h"
#include "CpuTimer.h"

#include "CudaAcc/AriesEngineDef.h"
using aries_acc::AriesDataBuffer;
using aries_acc::AriesDataBufferSPtr;

bool IsThdKilled( THD* thd );

using namespace aries;
using namespace aries::schema;
using aries_engine::AriesInitialTable;
using aries_engine::AriesMvccTable;
using aries_engine::AriesMvccTableManager;
using aries_engine::AriesInitialTableManager;
using aries_engine::AriesXLogManager;

// static char gDelimiter = ',';
extern bool STRICT_MODE;

enum class ImportMode : int8_t
{
    APPEND,
    REPLACE
};

ImportMode importMode = ImportMode::REPLACE;

int FormatNullError( const ColumnEntryPtr& colEntry, int64_t lineIdx, string& errorMsg )
{
    char errmsg[ERRMSGSIZE] = {0};
    int errorCode = ER_WARN_NULL_TO_NOTNULL;
    snprintf( errmsg, sizeof(errmsg),
              "NULL supplied to NOT NULL column '%s' at row %ld",
              colEntry->GetName().data(), lineIdx + 1 );
    errorMsg.assign( errmsg );
    return errorCode;
}

void throwContainNulError( const string& colName, uint64_t lineIdx )
{
    char errmsg[ERRMSGSIZE] = {0};
    snprintf( errmsg, sizeof(errmsg),
              "NULL supplied to NOT NULL column '%s' at row %ld",
              colName.data(), lineIdx + 1 );
    ARIES_EXCEPTION_SIMPLE( ER_WARN_NULL_TO_NOTNULL, errmsg );
}

void writeAriesColumnFile( std::string &filename,
                           uint64_t numValues,
                           std::shared_ptr< char > &buf,
                           uint64_t totalSize,
                           int16_t itemLen,
                           int8_t containNull )
{
    char header[ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE] = { 0 };
    BlockFileHeader info;
    info.rows = numValues;
    info.containNull = containNull;
    info.itemLen = itemLen;
    memcpy( header, &info, sizeof( BlockFileHeader ) );

    std::ofstream file( filename );
    if ( file.is_open() )
    {
        file.write( header, ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE );
        file.write( buf.get(), totalSize );
        file.flush();
        file.close();
    }
    else
    {
        throw std::runtime_error( "can't create file [" + filename + "]" );
    }
}

size_t getMaxItemLenOfChar( const ColumnEntryPtr& colEntry, const vector<vector< string >>& lines, uint64_t numValues, uint64_t colIndex )
{
    size_t maxLen = 0;
    for ( uint64_t i = 0; i < numValues; ++i )
    {
        std::string col = lines[i][colIndex];
        CheckCharLen( colEntry->GetName(), col.length() );
        if ( col.length() > maxLen )
        {
            maxLen = col.length();
        }
    }
    if ( maxLen == 0 )
    {
        maxLen = 1;
    }
    return maxLen;
}

void writeAriesColumnChar( const ColumnEntryPtr& colEntry,
                           const vector<vector< string >>& lines,
                           uint64_t numValues,
                           uint64_t colIndex,
                           size_t itemLen,
                           int containNull,
                           std::string defValue,
                           std::string &filename )
{
    CheckCharLen( colEntry->GetName(), itemLen );

    size_t maxItemLen = getMaxItemLenOfChar( colEntry, lines, numValues, colIndex );
    if ( !containNull && defValue.length() > maxItemLen )
    {
        maxItemLen = defValue.length();
    }
    if ( maxItemLen + containNull < itemLen )
    {
        LOG(INFO) << "Info: column [" << colIndex << "]: actual item size is less than schema defined: " << itemLen << ", it will be adjusted to "
                  << maxItemLen << " and save size: " << ( itemLen - maxItemLen - containNull ) * numValues << " bytes!";
    }
    // schema define length is char_length(), NOT length() , from mysql
    // so use the actual length (length()) instead of schema define length
    itemLen = maxItemLen + containNull;
    CheckCharLen( colEntry->GetName(), itemLen );

    uint64_t totalSize = numValues * itemLen;
    std::shared_ptr< char > buf( new char[totalSize] );
    int index = 0;
    char tmp[ARIES_MAX_CHAR_WIDTH + 1];
    uint64_t offset = 0;

    for ( uint64_t i = 0; i < numValues; ++i )
    {
        string col = lines[i][colIndex];
        if ( !containNull && col.empty() && !defValue.empty() )
        {
            col = defValue;
        }
        index = 0;
        memset( tmp, 0x00, sizeof( tmp ) );
        if ( col.empty() )
        {
            if ( !containNull )
            {
                throwContainNulError( colEntry->GetName(), i );
            }
            tmp[index] = 0;
        }
        else
        {
            if ( containNull )
            {
                tmp[index++] = 1;
            }
            assert( itemLen - index >= col.size() );
            memcpy( tmp + index, col.c_str(), col.size() );
        }
        memcpy( buf.get() + offset, tmp, itemLen );
        offset += itemLen;
    }
    // write data
    writeAriesColumnFile( filename, numValues, buf, totalSize, itemLen, containNull );
}

void writeAriesColumnInt64( const ColumnEntryPtr& colEntry,
                            const vector<vector< string >>& lines,
                            uint64_t numValues,
                            uint64_t colIndex,
                            size_t itemLen,
                            int containNull,
                            std::string defValue,
                            std::string &filename )
{
    uint64_t totalSize = numValues * itemLen;
    std::shared_ptr< char > buf( new char[totalSize] );
    int index = 0;
    char tmp[128];
    uint64_t offset = 0;
    //
    size_t size = itemLen - containNull;
    assert( size == sizeof( int64_t ) );
    int64_t value;

    for ( uint64_t i = 0; i < numValues; ++i )
    {
        string col = lines[i][colIndex];
        if ( !containNull && col.empty() && !defValue.empty() )
        {
            col = defValue;
        }
        index = 0;
        memset( tmp, 0x00, sizeof( tmp ) );
        if ( col.empty() )
        {
            if ( !containNull )
            {
                throwContainNulError( colEntry->GetName(), i );
            }
            tmp[index] = 0;
        }
        else
        {
            if ( containNull )
            {
                tmp[index++] = 1;
            }
            char *tail;
            value = std::strtoll( col.c_str(), &tail, 10 );
            if ( *col.c_str() == '\0' || *tail != '\0' )
            {
                ARIES_EXCEPTION( ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "bigint", col.data(), colEntry->GetName().data(), i + 1 );
            }
            memcpy( tmp + index, &value, size );
        }
        memcpy( buf.get() + offset, tmp, itemLen );
        offset += itemLen;
    }
    // write data
    writeAriesColumnFile( filename, numValues, buf, totalSize, itemLen, containNull );
}

void writeAriesColumnInt32( const ColumnEntryPtr& colEntry,
                            const vector<vector< string >>& lines,
                            uint64_t numValues,
                            uint64_t colIndex,
                            size_t itemLen,
                            int containNull,
                            std::string defValue,
                            std::string &filename )
{
    uint64_t totalSize = numValues * itemLen;
    std::shared_ptr< char > buf( new char[totalSize] );
    int index = 0;
    char tmp[128];
    uint64_t offset = 0;
    //
    size_t size = itemLen - containNull;
    assert( size == sizeof( int32_t ) );
    int32_t value;

    for ( uint64_t i = 0; i < numValues; ++i )
    {
        string col = lines[i][colIndex];
        if ( !containNull && col.empty() && !defValue.empty() )
        {
            col = defValue;
        }
        index = 0;
        memset( tmp, 0x00, sizeof( tmp ) );
        if ( col.empty() )
        {
            if ( !containNull )
            {
                throwContainNulError( colEntry->GetName(), i );
            }
            tmp[index] = 0;
        }
        else
        {
            if ( containNull )
            {
                tmp[index++] = 1;
            }
            char *tail;
            value = std::strtol( col.c_str(), &tail, 10 );
            if ( *tail != '\0' || *col.c_str() == '\0' )
            {
                ARIES_EXCEPTION( ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "integer", col.data(), colEntry->GetName().data(), i + 1 );
            }
            memcpy( tmp + index, &value, size );
        }
        memcpy( buf.get() + offset, tmp, itemLen );
        offset += itemLen;
    }
    // write data
    writeAriesColumnFile( filename, numValues, buf, totalSize, itemLen, containNull );
}

void writeAriesColumnInt16( const ColumnEntryPtr& colEntry,
                            const vector<vector< string >>& lines,
                            uint64_t numValues,
                            uint64_t colIndex,
                            size_t itemLen,
                            int containNull,
                            std::string defValue,
                            std::string &filename )
{
    uint64_t totalSize = numValues * itemLen;
    std::shared_ptr< char > buf( new char[totalSize] );
    int index = 0;
    char tmp[128];
    uint64_t offset = 0;
    //
    size_t size = itemLen - containNull;
    assert( size == sizeof( int16_t ) );
    int16_t value;

    for ( uint64_t i = 0; i < numValues; ++i )
    {
        string col = lines[i][colIndex];
        if ( !containNull && col.empty() && !defValue.empty() )
        {
            col = defValue;
        }
        index = 0;
        memset( tmp, 0x00, sizeof( tmp ) );
        if ( col.empty() )
        {
            if ( !containNull )
            {
                throwContainNulError( colEntry->GetName(), i );
            }
            tmp[index] = 0;
        }
        else
        {
            if ( containNull )
            {
                tmp[index++] = 1;
            }
            char *tail;
            value = std::strtol( col.c_str(), &tail, 10 );
            if ( *tail != '\0' || *col.c_str() == '\0' )
            {
                ARIES_EXCEPTION( ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "integer", col.data(), colEntry->GetName().data(), i + 1 );
            }
            memcpy( tmp + index, &value, size );
        }
        memcpy( buf.get() + offset, tmp, itemLen );
        offset += itemLen;
    }
    // write data
    writeAriesColumnFile( filename, numValues, buf, totalSize, itemLen, containNull );
}

void writeAriesColumnInt8( const ColumnEntryPtr& colEntry,
                           const vector<vector< string >>& lines,
                           uint64_t numValues,
                           uint64_t colIndex,
                           size_t itemLen,
                           int containNull,
                           std::string defValue,
                           std::string &filename )
{
    uint64_t totalSize = numValues * itemLen;
    std::shared_ptr< char > buf( new char[totalSize] );
    int index = 0;
    char tmp[128];
    uint64_t offset = 0;
    //
    size_t size = itemLen - containNull;
    assert( size == sizeof( int8_t ) );
    int8_t value;

    for ( uint64_t i = 0; i < numValues; ++i )
    {
        string col = lines[i][colIndex];
        if ( !containNull && col.empty() && !defValue.empty() )
        {
            col = defValue;
        }
        index = 0;
        memset( tmp, 0x00, sizeof( tmp ) );
        if ( col.empty() )
        {
            if ( !containNull )
            {
                throwContainNulError( colEntry->GetName(), i );
            }
            tmp[index] = 0;
        }
        else
        {
            if ( containNull )
            {
                tmp[index++] = 1;
            }
            char *tail;
            value = std::strtol( col.c_str(), &tail, 10 );
            if ( *tail != '\0' || *col.c_str() == '\0' )
            {
                ARIES_EXCEPTION( ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "integer", col.data(), colEntry->GetName().data(), i + 1 );
            }
            memcpy( tmp + index, &value, size );
        }
        memcpy( buf.get() + offset, tmp, itemLen );
        offset += itemLen;
    }
    // write data
    writeAriesColumnFile( filename, numValues, buf, totalSize, itemLen, containNull );
}

void writeAriesColumnFloat( const ColumnEntryPtr& colEntry,
                            const vector<vector< string >>& lines,
                            uint64_t numValues,
                            uint64_t colIndex,
                            size_t itemLen,
                            int containNull,
                            std::string defValue,
                            std::string &filename )
{
    uint64_t totalSize = numValues * itemLen;
    std::shared_ptr< char > buf( new char[totalSize] );
    int index = 0;
    char tmp[128];
    uint64_t offset = 0;
    //
    size_t size = itemLen - containNull;
    assert( size == sizeof( float ) );
    float value;

    for ( uint64_t i = 0; i < numValues; ++i )
    {
        string col = lines[i][colIndex];
        if ( !containNull && col.empty() && !defValue.empty() )
        {
            col = defValue;
        }
        index = 0;
        memset( tmp, 0x00, sizeof( tmp ) );
        if ( col.empty() )
        {
            if ( !containNull )
            {
                throwContainNulError( colEntry->GetName(), i );
            }
            tmp[index] = 0;
        }
        else
        {
            if ( containNull )
            {
                tmp[index++] = 1;
            }
            char *tail;
            value = std::strtof( col.c_str(), &tail );
            if ( *tail != '\0' || *col.c_str() == '\0' )
            {
                ARIES_EXCEPTION( ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "float", col.data(), colEntry->GetName().data(), i + 1 );
            }
            memcpy( tmp + index, &value, size );
        }
        memcpy( buf.get() + offset, tmp, itemLen );
        offset += itemLen;
    }
    // write data
    writeAriesColumnFile( filename, numValues, buf, totalSize, itemLen, containNull );
}

void writeAriesColumnDouble( const ColumnEntryPtr& colEntry,
                             const vector<vector< string >>& lines,
                             uint64_t numValues,
                             uint64_t colIndex,
                             size_t itemLen,
                             int containNull,
                             std::string defValue,
                             std::string &filename )
{
    uint64_t totalSize = numValues * itemLen;
    std::shared_ptr< char > buf( new char[totalSize] );
    int index = 0;
    char tmp[128];
    uint64_t offset = 0;
    //
    size_t size = itemLen - containNull;
    assert( size == sizeof( double ) );
    double value;

    for ( uint64_t i = 0; i < numValues; ++i )
    {
        string col = lines[i][colIndex];
        if ( !containNull && col.empty() && !defValue.empty() )
        {
            col = defValue;
        }
        index = 0;
        memset( tmp, 0x00, sizeof( tmp ) );
        if ( col.empty() )
        {
            if ( !containNull )
            {
                throwContainNulError( colEntry->GetName(), i );
            }
            tmp[index] = 0;
        }
        else
        {
            if ( containNull )
            {
                tmp[index++] = 1;
            }
            char *tail;
            value = std::strtod( col.c_str(), &tail );
            if ( *tail != '\0' || *col.c_str() == '\0' )
            {
                ARIES_EXCEPTION( ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "double", col.data(), colEntry->GetName().data(), i + 1 );
            }
            memcpy( tmp + index, &value, size );
        }
        memcpy( buf.get() + offset, tmp, itemLen );
        offset += itemLen;
    }
    // write data
    writeAriesColumnFile( filename, numValues, buf, totalSize, itemLen, containNull );
}

void writeAriesColumnDecimal( const ColumnEntryPtr& colEntry,
                              const vector<vector< string >>& lines,
                              uint64_t numValues,
                              uint64_t colIndex,
                              size_t precision,
                              size_t scale,
                              size_t itemLen,
                              int containNull,
                              std::string defValue,
                              std::string &filename )
{
    uint64_t totalSize = numValues * itemLen;
    std::shared_ptr< char > buf( new char[totalSize] );
    int index = 0;
    char tmp[128];
    uint64_t offset = 0;
    //
    size_t size = itemLen - containNull;
    assert( size == sizeof( aries_acc::Decimal ) );
    aries_acc::Decimal value;

    for ( uint64_t i = 0; i < numValues; ++i )
    {
        string col = lines[i][colIndex];
        col = aries_utils::trim( col );
        if ( !containNull && col.empty() && !defValue.empty() )
        {
            col = defValue;
        }
        index = 0;
        memset( tmp, 0x00, sizeof( tmp ) );
        if ( col.empty() )
        {
            if ( !containNull )
            {
                throwContainNulError( colEntry->GetName(), i );
            }
            tmp[index] = 0;
        }
        else
        {
            if ( containNull )
            {
                tmp[index++] = 1;
            }
            aries_acc::Decimal dec( precision, scale, col.c_str() );
            if ( dec.GetError() == ERR_STR_2_DEC )
            {
                ARIES_EXCEPTION( ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "decimal", col.data(), colEntry->GetName().data(), i + 1 );
            }
            value = dec;
            memcpy( tmp + index, &value, size );
        }
        memcpy( buf.get() + offset, tmp, itemLen );
        offset += itemLen;
    }
    // write data
    writeAriesColumnFile( filename, numValues, buf, totalSize, itemLen, containNull );
}

void writeAriesColumnCompactDecimal( const ColumnEntryPtr& colEntry,
                                     const vector<vector< string >>& lines,
                                     uint64_t numValues,
                                     uint64_t colIndex,
                                     size_t precision,
                                     size_t scale,
                                     size_t itemLen,
                                     int containNull,
                                     std::string defValue,
                                     std::string &filename )
{
    uint64_t totalSize = numValues * itemLen;
    std::shared_ptr< char > buf( new char[totalSize] );
    int index = 0;
    char tmp[128];
    uint64_t offset = 0;
    //
    size_t size = itemLen - containNull;
    assert( size == (size_t) aries_acc::GetDecimalRealBytes( precision, scale ) );
    aries_acc::Decimal value;

    for ( uint64_t i = 0; i < numValues; ++i )
    {
        string col = lines[i][colIndex];
        col = aries_utils::trim( col );
        if ( !containNull && col.empty() && !defValue.empty() )
        {
            col = defValue;
        }
        index = 0;
        memset( tmp, 0x00, sizeof( tmp ) );
        if ( col.empty() )
        {
            if ( !containNull )
            {
                throwContainNulError( colEntry->GetName(), i );
            }
            tmp[index] = 0;
        }
        else
        {
            if ( containNull )
            {
                tmp[index++] = 1;
            }
            aries_acc::Decimal dec( precision, scale, col.c_str() );
            if ( dec.GetError() == ERR_STR_2_DEC )
            {
                ARIES_EXCEPTION( ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "decimal", col.data(), colEntry->GetName().data(), i + 1 );
            }
            value = dec;
            if ( !value.ToCompactDecimal( tmp + index, size ) )
            {
                ARIES_EXCEPTION( ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "decimal", col.data(), colEntry->GetName().data(), i + 1 );
            }
        }
        memcpy( buf.get() + offset, tmp, itemLen );
        offset += itemLen;
    }
    // write data
    writeAriesColumnFile( filename, numValues, buf, totalSize, itemLen, containNull );
}

void writeAriesColumnDate( const ColumnEntryPtr& colEntry,
                           const vector<vector< string >>& lines,
                           uint64_t numValues,
                           uint64_t colIndex,
                           size_t itemLen,
                           int containNull,
                           std::string defValue,
                           std::string &filename )
{
    uint64_t totalSize = numValues * itemLen;
    std::shared_ptr< char > buf( new char[totalSize] );
    int index = 0;
    char tmp[128];
    uint64_t offset = 0;
    //
    size_t size = itemLen - containNull;
    assert( size == sizeof( aries_acc::AriesDate ) );

    for ( uint64_t i = 0; i < numValues; ++i )
    {
        string col = lines[i][colIndex];
        if ( !containNull && col.empty() && !defValue.empty() )
        {
            col = defValue;
        }
        index = 0;
        memset( tmp, 0x00, sizeof( tmp ) );
        if ( col.empty() )
        {
            if ( !containNull )
            {
                throwContainNulError( colEntry->GetName(), i );
            }
            tmp[index] = 0;
        }
        else
        {
            if ( containNull )
            {
                tmp[index++] = 1;
            }
            aries_acc::AriesDate date;
            try
            {
                date = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDate( col );
            }
            catch ( ... )
            {
                ARIES_EXCEPTION( ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "date", col.data(), colEntry->GetName().data(), i + 1 );
            }
            memcpy( tmp + index, &date, size );
        }
        memcpy( buf.get() + offset, tmp, itemLen );
        offset += itemLen;
    }
    // write data
    writeAriesColumnFile( filename, numValues, buf, totalSize, itemLen, containNull );
}

void writeAriesColumnDatetime( const ColumnEntryPtr& colEntry,
                               const vector<vector< string >>& lines,
                               uint64_t numValues,
                               uint64_t colIndex,
                               size_t itemLen,
                               int containNull,
                               std::string defValue,
                               std::string &filename )
{
    uint64_t totalSize = numValues * itemLen;
    std::shared_ptr< char > buf( new char[totalSize] );
    int index = 0;
    char tmp[128];
    uint64_t offset = 0;
    //
    size_t size = itemLen - containNull;
    assert( size == sizeof( aries_acc::AriesDatetime ) );

    for ( uint64_t i = 0; i < numValues; ++i )
    {
        string col = lines[i][colIndex];
        if ( !containNull && col.empty() && !defValue.empty() )
        {
            col = defValue;
        }
        index = 0;
        memset( tmp, 0x00, sizeof( tmp ) );
        if ( col.empty() )
        {
            if ( !containNull )
            {
                throwContainNulError( colEntry->GetName(), i );
            }
            tmp[index] = 0;
        }
        else
        {
            if ( containNull )
            {
                tmp[index++] = 1;
            }
            aries_acc::AriesDatetime date;
            try
            {
                date = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDatetime( col );
            }
            catch ( ... )
            {
                ARIES_EXCEPTION( ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "datetime", col.data(), colEntry->GetName().data(), i + 1 );
            }
            memcpy( tmp + index, &date, size );
        }
        memcpy( buf.get() + offset, tmp, itemLen );
        offset += itemLen;
    }
    // write data
    writeAriesColumnFile( filename, numValues, buf, totalSize, itemLen, containNull );
}

void writeAriesColumnTime( const ColumnEntryPtr& colEntry,
                               const vector<vector< string >>& lines,
                               uint64_t numValues,
                               uint64_t colIndex,
                               size_t itemLen,
                               int containNull,
                               std::string defValue,
                               std::string &filename )
{
    uint64_t totalSize = numValues * itemLen;
    std::shared_ptr< char > buf( new char[totalSize] );
    int index = 0;
    char tmp[128];
    uint64_t offset = 0;
    //
    size_t size = itemLen - containNull;
    assert( size == sizeof( aries_acc::AriesTime ) );

    for ( uint64_t i = 0; i < numValues; ++i )
    {
        string col = lines[i][colIndex];
        if ( !containNull && col.empty() && !defValue.empty() )
        {
            col = defValue;
        }
        index = 0;
        memset( tmp, 0x00, sizeof( tmp ) );
        if ( col.empty() )
        {
            if ( !containNull )
            {
                throwContainNulError( colEntry->GetName(), i );
            }
            tmp[index] = 0;
        }
        else
        {
            if ( containNull )
            {
                tmp[index++] = 1;
            }
            aries_acc::AriesTime time;
            try
            {
                time = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesTime( col );
            }
            catch ( ... )
            {
                ARIES_EXCEPTION( ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "time", col.data(), colEntry->GetName().data(), i + 1 );
            }
            memcpy( tmp + index, &time, size );
        }
        memcpy( buf.get() + offset, tmp, itemLen );
        offset += itemLen;
    }
    // write data
    writeAriesColumnFile( filename, numValues, buf, totalSize, itemLen, containNull );
}

void writeAriesColumnYear( const ColumnEntryPtr& colEntry,
                           const vector<vector< string >>& lines,
                           uint64_t numValues,
                           uint64_t colIndex,
                           size_t itemLen,
                           int containNull,
                           std::string defValue,
                           std::string &filename )
{
    uint64_t totalSize = numValues * itemLen;
    std::shared_ptr< char > buf( new char[totalSize] );
    int index = 0;
    char tmp[128];
    uint64_t offset = 0;
    //
    size_t size = itemLen - containNull;
    assert( size == sizeof( aries_acc::AriesYear ) );

    for ( uint64_t i = 0; i < numValues; ++i )
    {
        string col = lines[i][colIndex];
        if ( !containNull && col.empty() && !defValue.empty() )
        {
            col = defValue;
        }
        index = 0;
        memset( tmp, 0x00, sizeof( tmp ) );
        if ( col.empty() )
        {
            if ( !containNull )
            {
                throwContainNulError( colEntry->GetName(), i );
            }
            tmp[index] = 0;
        }
        else
        {
            if ( containNull )
            {
                tmp[index++] = 1;
            }
            aries_acc::AriesYear year(0);
            try
            {
                year = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesYear( col );
            }
            catch ( ... )
            {
                ARIES_EXCEPTION( ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "year", col.data(), colEntry->GetName().data(), i + 1 );
            }
            memcpy( tmp + index, &year, size );
        }
        memcpy( buf.get() + offset, tmp, itemLen );
        offset += itemLen;
    }
    // write data
    writeAriesColumnFile( filename, numValues, buf, totalSize, itemLen, containNull );
}

void writeAriesColumnTimestamp( const ColumnEntryPtr& colEntry,
                                const vector<vector< string >>& lines,
                                uint64_t numValues,
                                uint64_t colIndex,
                                size_t itemLen,
                                int containNull,
                                std::string defValue,
                                std::string &filename )
{
    uint64_t totalSize = numValues * itemLen;
    std::shared_ptr< char > buf( new char[totalSize] );
    int index = 0;
    char tmp[128];
    uint64_t offset = 0;
    //
    size_t size = itemLen - containNull;
    assert( size == sizeof( aries_acc::AriesTimestamp ) );
    aries_acc::AriesTimestamp timestamp;

    for ( uint64_t i = 0; i < numValues; ++i )
    {
        string col = lines[i][colIndex];
        if ( !containNull && col.empty() && !defValue.empty() )
        {
            col = defValue;
        }
        index = 0;
        memset( tmp, 0x00, sizeof( tmp ) );
        if ( col.empty() )
        {
            if ( !containNull )
            {
                throwContainNulError( colEntry->GetName(), i );
            }
            tmp[index] = 0;
        }
        else
        {
            if ( containNull )
            {
                tmp[index++] = 1;
            }

            try
            {
                timestamp = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesTimestamp( col );
            }
            catch ( ... )
            {
                ARIES_EXCEPTION( ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, "timestamp", col.data(), colEntry->GetName().data(), i + 1 );
            }
            memcpy( tmp + index, &timestamp, size );
        }
        memcpy( buf.get() + offset, tmp, itemLen );
        offset += itemLen;
    }
    // write data
    writeAriesColumnFile( filename, numValues, buf, totalSize, itemLen, containNull );
}

bool moveData2Target(std::string targetDir, std::string tempDir, std::string backupDir)
{
    // rename()
    return true;
}
class ifstream_helper
{
public:
    explicit ifstream_helper( const string& s ) : mStream( new ifstream(s) )
    {
    }
    ~ifstream_helper()
    {
        if ( mStream->is_open() )
            mStream->close();
        delete mStream;
    }
    bool is_open() const { return mStream->is_open(); }
    ifstream& get() { return *mStream; }

private:
    ifstream* mStream = nullptr;
};

static const string LOAD_DATA_TMP_DIR_FORMAT = "rateup_load_data-%%%%%%%%";
string MakeTmpDir()
{
    auto s = Configuartion::GetInstance().GetTmpDirectory();

    // boost::filesystem::path tmpDirPath;
    // try {
    //     tmpDirPath = boost::filesystem::temp_directory_path();
    // }
    // catch ( boost::filesystem::filesystem_error& e ) {
    //     LOG(WARNING) << "make tmp dir error: " << e.what();
    //     tmpDirPath = P_tmpdir;
    // }
    boost::filesystem::path uniquePath = boost::filesystem::unique_path( s.append("/").append( LOAD_DATA_TMP_DIR_FORMAT ) );
    while ( boost::filesystem::exists( uniquePath) )
    {
        LOG(WARNING) << "tmp path: " << uniquePath.string() << " already exists";
        uniquePath = boost::filesystem::unique_path( s.append("/").append( LOAD_DATA_TMP_DIR_FORMAT ) );
    }
    boost::filesystem::create_directories( uniquePath );
    s = uniquePath.string();
    if ( !boost::filesystem::is_directory( uniquePath ) )
    {
        set_my_errno(errno);
        char errbuf[MYSYS_STRERROR_SIZE];
        ARIES_EXCEPTION( EE_CANT_MKDIR, s.data(), my_errno(), strerror_r( my_errno(), errbuf, sizeof(errbuf) ) );

    }
    LOG(INFO) << "tmp dir: " << s;
    return s;
}

string GetTableBackupDir( const string& dbName, const string& tableName )
{
    string bkDir = Configuartion::GetInstance().GetDataBackupDirectory();
    bkDir.append("/").append( dbName ).append( "/" ).append( tableName );
    return bkDir;
}

string MakeTableBackupDir(const string& dbName, const string& tableName, string& suffix)
{
    string backupDirPrefix = GetTableBackupDir( dbName, tableName );

    // 20191219_000000
    char tsbuff[16] = {0};
    time_t currentTime = time(NULL);
    struct tm tm_tmp;
    localtime_r(&currentTime, &tm_tmp);
    strftime( tsbuff, sizeof(tsbuff), "%Y%m%d-%H%M%S", &tm_tmp );

    string backupDir = backupDirPrefix;
    backupDir.append( "/" ).append( tsbuff );
    while ( boost::filesystem::exists( backupDir ) )
    {
        sleep( 1 );
        currentTime = time(NULL);
        localtime_r(&currentTime, &tm_tmp);
        strftime( tsbuff, sizeof(tsbuff), "%Y%m%d-%H%M%S", &tm_tmp );
        backupDir = backupDirPrefix;
        backupDir.append("/").append(tsbuff);
    }
    suffix.assign( tsbuff );

    backupDir.append("/");
    boost::filesystem::create_directories( backupDir );
    return backupDir;
}

#define GET (stack_pos != stack ? *--stack_pos : my_b_get(&cache))
#define PUSH(A) *(stack_pos++)=(A)

class READ_INFO {
    File	file;
    uchar	*buffer,			/* Buffer for read text */
            *end_of_buff;			/* Data in bufferts ends here */
    uint	buff_length;			/* Length of buffer */
    const uchar *field_term_ptr, *line_term_ptr;
    const char *line_start_ptr, *line_start_end;
    size_t	field_term_length,line_term_length,enclosed_length;
    int	field_term_char,line_term_char,enclosed_char,escape_char;
    int	*stack,*stack_pos;
    bool	found_end_of_line,start_of_line,eof;
    bool  need_end_io_cache;
    IO_CACHE cache;
    int level; /* for load xml */

public:
    bool error,line_cuted,found_null,enclosed;
    uchar	*row_start,			/* Found row starts here */
            *row_end;			/* Found row ends here */
    const CHARSET_INFO *read_charset;

    READ_INFO(File file,uint tot_length,const CHARSET_INFO *cs,
              const string &field_term,
              const string &line_start,
              const string &line_term,
              const string &enclosed,
              int escape,bool get_it_from_net, bool is_fifo);
    ~READ_INFO();
    int read_field();
    // int read_fixed_length(void);
    int next_line(void);
    char unescape(char chr);
    int terminator(const uchar *ptr, size_t length);
    bool find_start_of_fields();
    // int read_value(int delim, String *val);

    /**
      skip all data till the eof.
    */
    void skip_data_till_eof()
    {
        while (GET != my_b_EOF)
            ;
    }
};

/* Unescape all escape characters, mark \N as null */

char
READ_INFO::unescape(char chr)
{
    /* keep this switch synchornous with the ESCAPE_CHARS macro */
    switch(chr) {
        case 'n': return '\n';
        case 't': return '\t';
        case 'r': return '\r';
        case 'b': return '\b';
        case '0': return 0;				// Ascii null
        case 'Z': return '\032';			// Win32 end of file
        case 'N': found_null=1;

            /* fall through */
        default:  return chr;
    }
}


/*
  Read a line using buffering
  If last line is empty (in line mode) then it isn't outputed
*/


READ_INFO::READ_INFO(File file_par, uint tot_length, const CHARSET_INFO *cs,
                     const string &field_term,
                     const string &line_start,
                     const string &line_term,
                     const string &enclosed_par,
                     int escape, bool get_it_from_net, bool is_fifo)
        :file(file_par), buff_length(tot_length), escape_char(escape),
         found_end_of_line(false), eof(false), need_end_io_cache(false),
         error(false), line_cuted(false), found_null(false), read_charset(cs)
{
    /*
      Field and line terminators must be interpreted as sequence of unsigned char.
      Otherwise, non-ascii terminators will be negative on some platforms,
      and positive on others (depending on the implementation of char).
    */
    field_term_ptr=
            static_cast<const uchar*>(static_cast<const void*>(field_term.data()));
    field_term_length= field_term.length();
    line_term_ptr=
            static_cast<const uchar*>(static_cast<const void*>(line_term.data()));
    line_term_length= line_term.length();

    level= 0; /* for load xml */
    if (line_start.length() == 0)
    {
        line_start_ptr=0;
        start_of_line= 0;
    }
    else
    {
        line_start_ptr= line_start.data();
        line_start_end=line_start_ptr+line_start.length();
        start_of_line= 1;
    }
    /* If field_terminator == line_terminator, don't use line_terminator */
    if (field_term_length == line_term_length &&
        !memcmp(field_term_ptr,line_term_ptr,field_term_length))
    {
        line_term_length=0;
        line_term_ptr= NULL;
    }
    enclosed_char= (enclosed_length=enclosed_par.length()) ?
                   (uchar) enclosed_par[0] : INT_MAX;
    field_term_char= field_term_length ? field_term_ptr[0] : INT_MAX;
    line_term_char= line_term_length ? line_term_ptr[0] : INT_MAX;


    /* Set of a stack for unget if long terminators */
    size_t length= max<size_t>(cs->mbmaxlen, max(field_term_length, line_term_length)) + 1;
    set_if_bigger(length,line_start.length());
    stack=stack_pos=(int*) malloc(sizeof(int)*length);

    if (!(buffer=(uchar*) malloc( buff_length+1)))
        error= true; /* purecov: inspected */
    else
    {
        end_of_buff=buffer+buff_length;
        if (init_io_cache(&cache,(get_it_from_net) ? -1 : file, 0,
                          (get_it_from_net) ? READ_NET :
                          (is_fifo ? READ_FIFO : READ_CACHE),0L,1,
                          MYF(MY_WME)))
        {
            free(buffer); /* purecov: inspected */
            buffer= NULL;
            error= true;
        }
        else
        {
            /*
          init_io_cache() will not initialize read_function member
          if the cache is READ_NET. So we work around the problem with a
          manual assignment
            */
            need_end_io_cache = 1;

            if (get_it_from_net)
                cache.read_function = _my_b_net_read;
        }
    }
}


READ_INFO::~READ_INFO()
{
    if (need_end_io_cache)
        ::end_io_cache(&cache);

    if (stack != NULL)
        free(stack);

    if (buffer != NULL)
        free(buffer);
}

/**
  The logic here is similar with my_mbcharlen, except for GET and PUSH

  @param[in]  cs  charset info
  @param[in]  chr the first char of sequence
  @param[out] len the length of multi-byte char
*/
#define GET_MBCHARLEN(cs, chr, len)                                           \
  do {                                                                        \
    len= my_mbcharlen((cs), (chr));                                           \
    if (len == 0 && my_mbmaxlenlen((cs)) == 2)                                \
    {                                                                         \
      int chr1= GET;                                                          \
      if (chr1 != my_b_EOF)                                                   \
      {                                                                       \
        len= my_mbcharlen_2((cs), (chr), chr1);                               \
        /* Character is gb18030 or invalid (len = 0) */                       \
        DBUG_ASSERT(len == 0 || len == 2 || len == 4);                        \
      }                                                                       \
      if (len != 0)                                                           \
        PUSH(chr1);                                                           \
    }                                                                         \
  } while (0)

inline int READ_INFO::terminator(const uchar *ptr, size_t length)
{
    int chr=0;					// Keep gcc happy
    size_t i;
    for (i=1 ; i < length ; i++)
    {
        chr= GET;
        if (chr != *++ptr)
        {
            break;
        }
    }
    if (i == length)
        return 1;
    PUSH(chr);
    while (i-- > 1)
        PUSH(*--ptr);
    return 0;
}

int READ_INFO::read_field()
{
    int chr,found_enclosed_char;
    uchar *to,*new_buffer;

    found_null=0;
    if (found_end_of_line)
        return 1;					// One have to call next_line

    /* Skip until we find 'line_start' */

    if (start_of_line)
    {						// Skip until line_start
        start_of_line=0;
        if (find_start_of_fields())
            return 1;
    }
    if ((chr=GET) == my_b_EOF)
    {
        found_end_of_line=eof=1;
        return 1;
    }
    to=buffer;
    if (chr == enclosed_char)
    {
        found_enclosed_char=enclosed_char;
        *to++=(uchar) chr;				// If error
    }
    else
    {
        found_enclosed_char= INT_MAX;
        PUSH(chr);
    }

    for (;;)
    {
        bool escaped_mb= false;
        while ( to < end_of_buff)
        {
            chr = GET;
            if (chr == my_b_EOF)
                goto found_eof;
            if (chr == escape_char)
            {
                if ((chr=GET) == my_b_EOF)
                {
                    *to++= (uchar) escape_char;
                    goto found_eof;
                }
                /*
                  When escape_char == enclosed_char, we treat it like we do for
                  handling quotes in SQL parsing -- you can double-up the
                  escape_char to include it literally, but it doesn't do escapes
                  like \n. This allows: LOAD DATA ... ENCLOSED BY '"' ESCAPED BY '"'
                  with data like: "fie""ld1", "field2"
                 */
                if (escape_char != enclosed_char || chr == escape_char)
                {
                    uint ml;
                    GET_MBCHARLEN(read_charset, chr, ml);
                    /*
                      For escaped multibyte character, push back the first byte,
                      and will handle it below.
                      Because multibyte character's second byte is possible to be
                      0x5C, per Query_result_export::send_data, both head byte and
                      tail byte are escaped for such characters. So mark it if the
                      head byte is escaped and will handle it below.
                    */
                    if (ml == 1)
                        *to++= (uchar) unescape((char) chr);
                    else
                    {
                        escaped_mb= true;
                        PUSH(chr);
                    }
                    continue;
                }
                PUSH(chr);
                chr= escape_char;
            }
            if (chr == line_term_char && found_enclosed_char == INT_MAX)
            {
                if (terminator(line_term_ptr,line_term_length))
                {					// Maybe unexpected linefeed
                    enclosed=0;
                    found_end_of_line=1;
                    row_start=buffer;
                    row_end=  to;
                    return 0;
                }
            }
            if (chr == found_enclosed_char)
            {
                if ((chr=GET) == found_enclosed_char)
                {					// Remove dupplicated
                    *to++ = (uchar) chr;
                    continue;
                }
                // End of enclosed field if followed by field_term or line_term
                if (chr == my_b_EOF ||
                    (chr == line_term_char && terminator(line_term_ptr,
                                                         line_term_length)))
                {					// Maybe unexpected linefeed
                    enclosed=1;
                    found_end_of_line=1;
                    row_start=buffer+1;
                    row_end=  to;
                    return 0;
                }
                if (chr == field_term_char &&
                    terminator(field_term_ptr,field_term_length))
                {
                    enclosed=1;
                    row_start=buffer+1;
                    row_end=  to;
                    return 0;
                }
                /*
                  The string didn't terminate yet.
                  Store back next character for the loop
                */
                PUSH(chr);
                /* copy the found term character to 'to' */
                chr= found_enclosed_char;
            }
            else if (chr == field_term_char && found_enclosed_char == INT_MAX)
            {
                if (terminator(field_term_ptr,field_term_length))
                {
                    enclosed=0;
                    row_start=buffer;
                    row_end=  to;
                    return 0;
                }
            }

            uint ml;
            GET_MBCHARLEN(read_charset, chr, ml);
            if (ml == 0)
            {
                *to= '\0';
                error= true;
                ARIES_EXCEPTION( ER_INVALID_CHARACTER_STRING, read_charset->csname, buffer );
            }


            if (ml > 1 &&
                to + ml <= end_of_buff)
            {
                uchar* p= to;
                *to++ = chr;

                for (uint i= 1; i < ml; i++)
                {
                    chr= GET;
                    if (chr == my_b_EOF)
                    {
                        /*
                         Need to back up the bytes already ready from illformed
                         multi-byte char
                        */
                        to-= i;
                        goto found_eof;
                    }
                    else if (chr == escape_char && escaped_mb)
                    {
                        // Unescape the second byte if it is escaped.
                        chr= GET;
                        chr= (uchar) unescape((char) chr);
                    }
                    *to++ = chr;
                }
                if (escaped_mb)
                    escaped_mb= false;
                if (my_ismbchar(read_charset,
                                (const char *)p,
                                (const char *)to))
                    continue;
                for (uint i= 0; i < ml; i++)
                    PUSH(*--to);
                chr= GET;
            }
            else if (ml > 1)
            {
                // Buffer is too small, exit while loop, and reallocate.
                PUSH(chr);
                break;
            }
            *to++ = (uchar) chr;
        }
        /*
        ** We come here if buffer is too small. Enlarge it and continue
        */
        if (!(new_buffer=(uchar*) realloc( (char*) buffer,buff_length+1+IO_SIZE)))
            ARIES_EXCEPTION_SIMPLE( EE_OUTOFMEMORY, "Out of memory" );
        to=new_buffer + (to-buffer);
        buffer=new_buffer;
        buff_length+=IO_SIZE;
        end_of_buff=buffer+buff_length;
    }

found_eof:
    enclosed=0;
    found_end_of_line=eof=1;
    row_start=buffer;
    row_end=to;
    return 0;
}

int READ_INFO::next_line()
{
    line_cuted=0;
    start_of_line= line_start_ptr != 0;
    if (found_end_of_line || eof)
    {
        found_end_of_line=0;
        return eof;
    }
    found_end_of_line=0;
    if (!line_term_length)
        return 0;					// No lines
    for (;;)
    {
        int chr = GET;
        uint ml;
        if (chr == my_b_EOF)
        {
            eof= 1;
            return 1;
        }
        GET_MBCHARLEN(read_charset, chr, ml);
        if (ml > 1)
        {
            for (uint i=1;
                 chr != my_b_EOF && i < ml;
                 i++)
                chr = GET;
            if (chr == escape_char)
                continue;
        }
        if (chr == my_b_EOF)
        {
            eof=1;
            return 1;
        }
        if (chr == escape_char)
        {
            line_cuted=1;
            if (GET == my_b_EOF)
                return 1;
            continue;
        }
        if (chr == line_term_char && terminator(line_term_ptr,line_term_length))
            return 0;
        line_cuted=1;
    }
}

bool READ_INFO::find_start_of_fields()
{
    int chr;
    try_again:
    do
    {
        if ((chr=GET) == my_b_EOF)
        {
            found_end_of_line=eof=1;
            return 1;
        }
    } while ((char) chr != line_start_ptr[0]);
    for (const char *ptr=line_start_ptr+1 ; ptr != line_start_end ; ptr++)
    {
        chr=GET;					// Eof will be checked later
        if ((char) chr != *ptr)
        {						// Can't be line_start
            PUSH(chr);
            while (--ptr != line_start_ptr)
            {						// Restart with next char
                PUSH( *ptr);
            }
            goto try_again;
        }
    }
    return 0;
}

/*
 Implicit defaults are defined as follows:
• For numeric types, the default is 0, with the exception that for integer or floating-point types declared
with the AUTO_INCREMENT attribute, the default is the next value in the sequence.
Data Type Storage Requirements
1621
• For date and time types other than TIMESTAMP, the default is the appropriate “zero” value
for the type. This is also true for TIMESTAMP if the explicit_defaults_for_timestamp
system variable is enabled (see Section 5.1.7, “Server System Variables”). Otherwise, for the first
TIMESTAMP column in a table, the default value is the current date and time. See Section 11.3, “Date
and Time Types”.
• For string types other than ENUM, the default value is the empty string. For ENUM, the default is the
first enumeration value.
 */
static void
GetDataTypeImplicitDefaultValue( ColumnType colType,
                                 const int32_t precision,
                                 const int32_t scale,
                                 uchar *valueBuff,
                                 size_t buffLen,
                                 int8_t containNull)
{
    static const int8_t tinyIntDefaultVal = 0;
    static const int16_t smallIntDefaultVal = 0;
    static const int32_t intDefaultVal = 0;
    static const int64_t longIntDefaultVal = 0;
    static const float floatDefaultVal = 0.0;
    static const double doubleDefaultVal = 0.0;
    static const aries_acc::Decimal decimalDefaultVal;
    static const aries_acc::AriesDate dateDefaultVal;
    static const aries_acc::AriesTime timeDefaultVal;
    static const aries_acc::AriesDatetime datetimeDefaultVal;
    static const aries_acc::AriesTimestamp timestampDefaultVal;
    static const aries_acc::AriesYear yearDefaultVal;

    static const string tinyIntDefaultValStr((const char *)&tinyIntDefaultVal, 1);
    static const string smallIntDefaultValStr((const char *)&smallIntDefaultVal, 2);
    static const string intDefaultValStr((const char *)&intDefaultVal, 4);
    static const string longIntDefaultValStr((const char *)&longIntDefaultVal, 8);
    static const string floatDefaultValStr((const char *)&floatDefaultVal, sizeof(float));
    static const string doubleDefaultValStr((const char *)&doubleDefaultVal, sizeof(double));
    static const string decimalDefaultValStr((const char *)&decimalDefaultVal, sizeof(aries_acc::Decimal));
    static const string strDefaultValStr("", 1); // empty string, size is 1
    static const string dateDefaultValStr((const char *)&dateDefaultVal, sizeof(aries_acc::AriesDate));
    static const string timeDefaultValStr((const char *)&timeDefaultVal, sizeof(aries_acc::AriesTime));
    static const string datetimeDefaultValStr((const char *)&datetimeDefaultVal, sizeof(aries_acc::AriesDatetime));
    static const string timestampDefaultValStr((const char *)&timestampDefaultVal, sizeof(aries_acc::AriesTimestamp));
    static const string yearDefaultValStr((const char *)&yearDefaultVal, sizeof(aries_acc::AriesYear));
    static map<ColumnType, std::pair<const char *, size_t>> dataTypeDefValues =
        {
            {ColumnType::BOOL, {tinyIntDefaultValStr.data(), 1}},
            {ColumnType::TINY_INT, {tinyIntDefaultValStr.data(), 1}},
            {ColumnType::SMALL_INT, {smallIntDefaultValStr.data(), 2}},
            {ColumnType::INT, {intDefaultValStr.data(), 4}},
            {ColumnType::LONG_INT, {longIntDefaultValStr.data(), 8}},
            {ColumnType::FLOAT, {floatDefaultValStr.data(), sizeof(float)}},
            {ColumnType::DOUBLE, {doubleDefaultValStr.data(), sizeof(double)}},
            {ColumnType::DECIMAL, {decimalDefaultValStr.data(), sizeof(aries_acc::Decimal)}},
            {ColumnType::TEXT, {strDefaultValStr.data(), strDefaultValStr.size()}},
            {ColumnType::BINARY, {strDefaultValStr.data(), strDefaultValStr.size()}},
            {ColumnType::VARBINARY, {strDefaultValStr.data(), strDefaultValStr.size()}},
            {ColumnType::DATE, {dateDefaultValStr.data(), sizeof(aries_acc::AriesDate)}},
            {ColumnType::TIME, {timeDefaultValStr.data(), sizeof(aries_acc::AriesTime)}},
            {ColumnType::DATE_TIME, {datetimeDefaultValStr.data(), sizeof(aries_acc::AriesDatetime)}},
            {ColumnType::TIMESTAMP, {timestampDefaultValStr.data(), sizeof(aries_acc::AriesTimestamp)}},
            {ColumnType::YEAR, {yearDefaultValStr.data(), sizeof(aries_acc::AriesYear)}},
        };
    auto defValPair = dataTypeDefValues.find( colType );
    int offset = 0;
    if ( containNull )
    {
        valueBuff[offset++] = 1;
    }
    if ( ColumnType::DECIMAL == colType )
    {
        aries_acc::Decimal* dec = ( aries_acc::Decimal* )( defValPair->second.first );
        dec->ToCompactDecimal( ( char * )( valueBuff + offset ),
                               GetDecimalRealBytes( precision, scale ) );
    }
    else
        memcpy(valueBuff + offset, defValPair->second.first, defValPair->second.second);
}

/*
mysql 5.7.26 reference mannuals

Treatment of empty or incorrect field values differs from that just described if the SQL mode is set to
a restrictive value. For example, if sql_mode is set to TRADITIONAL, conversion of an empty value
or a value such as 'x' for a numeric column results in an error, not conversion to 0. (With LOCAL
or IGNORE, warnings occur rather than errors, even with a restrictive sql_mode value, and the row
is inserted using the same closest-value behavior used for nonrestrictive SQL modes. This occurs
because the server has no way to stop transmission of the file in the middle of the operation.)

signed int range: -128, 127, -32768, 32767, -2147483648, 2147483647, -9223372036854775808, 9223372036854775807
unsigned int range: 255, 65535, 4294967295, 18446744073709551615
*/
static int ToColumnValue( const ColumnEntryPtr& colEntry,
                             int8_t containNull,
                             const string& defValue,
                             string& col,
                             int64_t lineIdx,
                             uchar* valueBuff,
                             size_t buffLen,
                             string& errorMsg )// OUT
{
    int errorCode = 0;
    memset( valueBuff, 0x00, buffLen );
    if ( schema::ColumnType::DECIMAL == colEntry->GetType() )
    {
        col = aries_utils::trim( col );
    }
    /*
     * An attempt to load NULL into a NOT NULL column causes assignment of the implicit default value
     * for the column's data type and a warning, or an error in strict SQL mode.
     */
    if ( ( !containNull && col.empty() ) ||      // null value for not null column
         ( 1 == col.size() && '\0' == col[0] ) ) // empty value
    {
        GetDataTypeImplicitDefaultValue( colEntry->GetType(),
                                         colEntry->numeric_precision,
                                         colEntry->numeric_scale,
                                         valueBuff,
                                         buffLen,
                                         containNull);
        return 0;
    }
    /*
    in non strict node, for empty fields:
    mysql> show warnings;
    +---------+------+------------------------------------------------------+
    | Level   | Code | Message                                              |
    +---------+------+------------------------------------------------------+
    | Warning | 1366 | Incorrect integer value: '' for column 'f1' at row 1 |
    +---------+------+------------------------------------------------------+
    1 row in set (0.00 sec)
    */
    int offset = 0;
    size_t size = 0;
    if ( col.empty() ) // NULL value
    {
        valueBuff[offset] = 0;
    }
    else
    {
        if ( containNull )
        {
            valueBuff[offset++] = 1;
        }
        char *tail;
        long longVal = 0;
        unsigned long longValUnsigned = 0;
        switch (colEntry->GetType())
        {
            case schema::ColumnType::BOOL:
            case schema::ColumnType::TINY_INT: // 1 byte
            {
                size = sizeof( int8_t );
                if ( colEntry->is_unsigned )
                {
                    uint8_t value = 0;
                    longValUnsigned = std::strtoul(col.c_str(), &tail, 10);
                    CheckTruncError(colEntry->GetName(), "integer", col, lineIdx, errorMsg);

                    if (longValUnsigned > UINT8_MAX)
                    {
                        errorCode = FormatOutOfRangeValueError(colEntry->GetName(), lineIdx, errorMsg);
                        if (STRICT_MODE)
                            return errorCode;
                        value = UINT8_MAX;
                        LOG(WARNING) << "Convert data warning: " << errorMsg;
                    }
                    else
                        value = longValUnsigned;
                    memcpy( valueBuff + offset, &value, size );
                }
                else
                {
                    int8_t value = 0;
                    longVal = std::strtol(col.c_str(), &tail, 10);
                    CheckTruncError(colEntry->GetName(), "integer", col, lineIdx, errorMsg);

                    if (longVal > INT8_MAX || longVal < INT8_MIN)
                    {
                        errorCode = FormatOutOfRangeValueError(colEntry->GetName(), lineIdx, errorMsg);
                        if (STRICT_MODE)
                            return errorCode;
                        if (longVal > INT8_MAX)
                            value = INT8_MAX;
                        else
                            value = INT8_MIN;
                        LOG(WARNING) << "Convert data warning: " << errorMsg;
                    }
                    else
                        value = longVal;
                    memcpy(valueBuff + offset, &value, size);
                }
                break;
            }
            case schema::ColumnType::SMALL_INT:  // 2 bytes
            {
                size = sizeof( int16_t );
                if ( colEntry->is_unsigned )
                {
                    uint16_t value = 0;
                    longValUnsigned = std::strtoul(col.c_str(), &tail, 10);
                    CheckTruncError(colEntry->GetName(), "integer", col, lineIdx, errorMsg);

                    if (longValUnsigned > UINT16_MAX)
                    {
                        errorCode = FormatOutOfRangeValueError(colEntry->GetName(), lineIdx, errorMsg);
                        if (STRICT_MODE)
                            return errorCode;
                        value = UINT16_MAX;
                        LOG(WARNING) << "Convert data warning: " << errorMsg;
                    }
                    else
                        value = longValUnsigned;
                    memcpy( valueBuff + offset, &value, size );
                }
                else
                {
                    int16_t value = 0;
                    longVal = std::strtol(col.c_str(), &tail, 10);
                    CheckTruncError(colEntry->GetName(), "integer", col, lineIdx, errorMsg);

                    if (longVal > INT16_MAX || longVal < INT16_MIN)
                    {
                        errorCode = FormatOutOfRangeValueError(colEntry->GetName(), lineIdx, errorMsg);
                        if (STRICT_MODE)
                            return errorCode;
                        if (longVal > INT16_MAX)
                            value = INT16_MAX;
                        else
                            value = INT16_MIN;
                        LOG(WARNING) << "Convert data warning: " << errorMsg;
                    }
                    else
                        value = longVal;
                    memcpy(valueBuff + offset, &value, size);
                }
                break;
            }
            case schema::ColumnType::INT: // 4 bytes
            {
                size = sizeof( int32_t );
                errno = 0;
                if ( colEntry->is_unsigned )
                {
                    uint32_t value = 0;
                    longValUnsigned = std::strtoul(col.c_str(), &tail, 10);
                    CheckTruncError( colEntry->GetName(), "integer", col, lineIdx, errorMsg );

                    if (longValUnsigned > UINT32_MAX || ERANGE == errno)
                    {
                        errorCode = FormatOutOfRangeValueError(colEntry->GetName(), lineIdx, errorMsg);
                        if (STRICT_MODE)
                            return errorCode;
                        value = UINT32_MAX;
                        LOG(WARNING) << "Convert data warning: " << errorMsg;
                    }
                    else
                        value = longValUnsigned;
                    memcpy( valueBuff + offset, &value, size );
                }
                else
                {
                    int32_t value = 0;
                    longVal = std::strtol(col.c_str(), &tail, 10);
                    CheckTruncError( colEntry->GetName(), "integer", col, lineIdx, errorMsg );

                    if ( longVal > INT32_MAX || longVal < INT32_MIN )
                    {
                        errorCode = FormatOutOfRangeValueError(colEntry->GetName(), lineIdx, errorMsg);
                        if (STRICT_MODE)
                            return errorCode;
                        if (longVal > INT32_MAX)
                            value = INT32_MAX;
                        else
                            value = INT32_MIN;
                        LOG(WARNING) << "Convert data warning: " << errorMsg;
                    }
                    else
                    {
                        if (ERANGE == errno)
                        {
                            errorCode = FormatOutOfRangeValueError(colEntry->GetName(), lineIdx, errorMsg);
                            if (STRICT_MODE)
                                return errorCode;
                            LOG(WARNING) << "Convert data warning: " << errorMsg;
                        }
                        value = longVal;
                    }
                    memcpy(valueBuff + offset, &value, size);
                }
                break;
            }
            case schema::ColumnType::LONG_INT: // 8 bytes
            {
                size = sizeof( int64_t );
                errno = 0;
                if ( colEntry->is_unsigned )
                {
                    uint64_t value = std::strtoull(col.c_str(), &tail, 10);
                    CheckTruncError( colEntry->GetName(), "integer", col, lineIdx, errorMsg );

                    if (errno == ERANGE /*&& llValUnsigned == ULLONG_MAX*/)
                    {
                        errorCode = FormatOutOfRangeValueError(colEntry->GetName(), lineIdx, errorMsg);
                        if (STRICT_MODE)
                            return errorCode;
                        LOG(WARNING) << "Convert data warning: " << errorMsg;
                    }
                    memcpy( valueBuff + offset, &value, size );
                }
                else
                {
                    int64_t value = std::strtoll(col.c_str(), &tail, 10);
                    CheckTruncError( colEntry->GetName(), "integer", col, lineIdx, errorMsg );

                    if (errno == ERANGE /*&& llValUnsigned == ULLONG_MAX*/)
                    {
                        errorCode = FormatOutOfRangeValueError(colEntry->GetName(), lineIdx, errorMsg);
                        if (STRICT_MODE)
                            return errorCode;
                        LOG(WARNING) << "Convert data warning: " << errorMsg;
                    }
                    memcpy(valueBuff + offset, &value, size);
                }
                break;
            }
            case schema::ColumnType::DECIMAL:
            {
                aries_acc::Decimal value(colEntry->numeric_precision,
                                         colEntry->numeric_scale,
                                         ARIES_MODE_STRICT_ALL_TABLES,
                                         col.c_str());
                if ( value.GetError() == ERR_STR_2_DEC )
                {
                    errorCode = FormatTruncWrongValueError( colEntry->GetName(), col, lineIdx, "decimal", errorMsg );
                    if ( STRICT_MODE  )
                        return errorCode;
                    LOG(WARNING) << "Convert data warning: " << errorMsg;
                }
                if ( value.GetError() == ERR_OVER_FLOW )
                {
                    errorCode = FormatOutOfRangeValueError( colEntry->GetName(), lineIdx, errorMsg );
                    if ( STRICT_MODE  )
                        return errorCode;
                    LOG(WARNING) << "Convert data warning: " << errorMsg;
                }

                if (!value.ToCompactDecimal( ( char * )( valueBuff + offset ),
                                             GetDecimalRealBytes( colEntry->numeric_precision, colEntry->numeric_scale ) ) )
                {
                    errorCode = FormatTruncWrongValueError( colEntry->GetName(), col, lineIdx, "decimal", errorMsg );
                    if ( STRICT_MODE  )
                        return errorCode;
                    LOG(WARNING) << "Convert data warning: " << errorMsg;
                }
                break;
            }
            case schema::ColumnType::FLOAT:
            {
                // float unsinged max: 3.40282e38, min: 0
                // float signed min: -3.40282e38, max: 3.40282e38
                size = sizeof( float );
                errno = 0;
                float value = std::strtof( col.c_str(), &tail );
                CheckTruncError(colEntry->GetName(), "float", col, lineIdx, errorMsg);

                if ( ERANGE == errno )
                {
                    errorCode = FormatOutOfRangeValueError(colEntry->GetName(), lineIdx, errorMsg);
                    if (STRICT_MODE)
                        return errorCode;
                    LOG(WARNING) << "Convert data warning: " << errorMsg;
                }
                if (colEntry->is_unsigned && value < 0)
                {
                    errorCode = FormatOutOfRangeValueError(colEntry->GetName(), lineIdx, errorMsg);
                    if (STRICT_MODE)
                        return errorCode;
                    LOG(WARNING) << "Convert data warning: " << errorMsg;
                    value = 0;
                }

                memcpy( valueBuff + offset, &value, size );
                break;
            }
            case schema::ColumnType::DOUBLE:
            {
                // double max: 1.7976931348623157e308
                size = sizeof( double );
                errno = 0;
                double value = std::strtod( col.c_str(), &tail );
                CheckTruncError(colEntry->GetName(), "double", col, lineIdx, errorMsg);

                if ( ERANGE == errno )
                {
                    errorCode = FormatOutOfRangeValueError(colEntry->GetName(), lineIdx, errorMsg);
                    if (STRICT_MODE)
                        return errorCode;
                    LOG(WARNING) << "Convert data warning: " << errorMsg;
                }
                if (colEntry->is_unsigned && value < 0)
                {
                    errorCode = FormatOutOfRangeValueError(colEntry->GetName(), lineIdx, errorMsg);
                    if (STRICT_MODE)
                        return errorCode;
                    LOG(WARNING) << "Convert data warning: " << errorMsg;
                    value = 0;
                }

                memcpy( valueBuff + offset, &value, size );
                break;
            }

            // Date values with two-digit years are ambiguous because the century is unknown.
            // For DATETIME, DATE, and TIMESTAMP types,
            // MySQL interprets dates specified with ambiguous year
            // values using these rules:
            // • Year values in the range 00-69 are converted to 2000-2069.
            // • Year values in the range 70-99 are converted to 1970-1999.
            //
            // For YEAR, the rules are the same, with this exception:
            // A numeric 00 inserted into YEAR(4) results in
            // 0000 rather than 2000.
            // To specify zero for YEAR(4) and have it be interpreted as 2000,
            // specify it as a string '0' or '00'.

            // By default, when MySQL encounters a value for a date or time type
            // that is out of range or otherwise invalid for the type,
            // it converts the value to the “zero” value for that type.
            // The exception is that out-of range TIME values are clipped
            // to the appropriate endpoint of the TIME range.

            // by enabling the ALLOW_INVALID_DATES SQL mode, MySQL verifies
            // only that the month is in the range from 1 to 12 and
            // that the day is in the range from 1 to 31.

            // MySQL permits you to store dates where the day or month and day
            // are zero in a DATE or DATETIME column.
            // To disallow zero month or day parts in
            // dates, enable the NO_ZERO_IN_DATE mode.

            // MySQL permits you to store a “zero” value of '0000-00-00' as a “dummy date.”
            // To disallow '0000-00-00', enable the NO_ZERO_DATE mode.

            /*
            The following table shows the format of the “zero” value for each type.
            You can also do this using the values '0' or 0, which are easier to write.

            For temporal types that include a date part (DATE, DATETIME, and TIMESTAMP),
            use of these values produces warnings if the NO_ZERO_DATE SQL
            mode is enabled.
            ====================================
            | Data Type | “Zero” Value         |
            |-----------|----------------------|
            | DATE      | '0000-00-00'         |
            | TIME      | '00:00:00'           |
            | DATETIME  | '0000-00-00 00:00:00'|
            | TIMESTAMP | '0000-00-00 00:00:00'|
            | YEAR      |  0000                |
            ================================== |
            */
            // The supported range is '1000-01-01' to '9999-12-31'.
            case schema::ColumnType::DATE:
            {
                int mode = STRICT_MODE ? ARIES_DATE_STRICT_MODE : ARIES_DATE_NOT_STRICT_MODE;
                aries_acc::AriesDate date;
                try
                {
                    date = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDate( col, mode );
                }
                catch ( ... )
                {
                    errorCode = FormatTruncWrongValueError( colEntry->GetName(), col, lineIdx, "date", errorMsg );
                    if ( STRICT_MODE )
                        return errorCode;
                    LOG(WARNING) << "Convert data warning: " << errorMsg;
                }
                size = sizeof( aries_acc::AriesDate);
                memcpy( valueBuff + offset, &date,  size );
                break;
            }
            // The supported range is '1000-01-01 00:00:00' to '9999-12-31 23:59:59'
            case schema::ColumnType::DATE_TIME:
            {
                int mode = STRICT_MODE ? ARIES_DATE_STRICT_MODE : ARIES_DATE_NOT_STRICT_MODE;
                aries_acc::AriesDatetime datetime;
                try
                {
                    datetime = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDatetime( col, mode );
                }
                catch ( ... )
                {
                    errorCode = FormatTruncWrongValueError(colEntry->GetName(), col, lineIdx, "datetime", errorMsg);
                    if (STRICT_MODE)
                        return errorCode;
                    LOG(WARNING) << "Convert data warning: " << errorMsg;
                }
                size = sizeof( aries_acc::AriesDatetime );
                memcpy( valueBuff + offset, &datetime,  size );
                break;
            }
            // TIMESTAMP has a range of '1970-01-01 00:00:01' UTC to '2038-01-19 03:14:07' UTC.
            // MySQL does not accept TIMESTAMP values that include a zero in the day or month column or values
            // that are not a valid date. The sole exception to this rule is the special “zero” value '0000-00-00
            // 00:00:00'.
            case schema::ColumnType::TIMESTAMP:
            {
                int mode = STRICT_MODE ? ARIES_DATE_STRICT_MODE : ARIES_DATE_NOT_STRICT_MODE;
                aries_acc::AriesTimestamp timestamp;
                try
                {
                    timestamp = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesTimestamp( col, mode );
                }
                catch ( ... )
                {
                    errorCode = FormatTruncWrongValueError(colEntry->GetName(), col, lineIdx, "timestamp", errorMsg);
                    if (STRICT_MODE)
                        return errorCode;
                    LOG(WARNING) << "Convert data warning: " << errorMsg;
                }
                size = sizeof( aries_acc::AriesTimestamp );
                memcpy( valueBuff + offset, &timestamp,  size );
                break;
            }
            case schema::ColumnType::TEXT:
            case schema::ColumnType::VARBINARY:
            case schema::ColumnType::BINARY:
            {
                // mysql non-strict mode:
                // Warning | 1265 | Data truncated for column 'f1' at row 1
                size = col.size();
                if ( size > ( size_t )colEntry->GetLength() )
                {
                    errorMsg.assign( "Column length too big for column '");
                    errorMsg.append( colEntry->GetName() ).append( "' (max = ").append( std::to_string( ARIES_MAX_CHAR_WIDTH ) ).append( ")" );
                    errorCode = ER_TOO_BIG_FIELDLENGTH;
                    if (STRICT_MODE)
                        return errorCode;
                    LOG(INFO) << "Char data length " << size << " exceed schema defined length " << colEntry->GetLength();
                }
                memcpy( valueBuff + offset, col.c_str(), size );
                break;
            }
            case schema::ColumnType::YEAR:
            {
                int mode = STRICT_MODE ? ARIES_DATE_STRICT_MODE : ARIES_DATE_NOT_STRICT_MODE;
                aries_acc::AriesYear year(0);
                try
                {
                    year = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesYear( col, mode );
                }
                catch ( ... )
                {
                    errorCode = FormatTruncWrongValueError(colEntry->GetName(), col, lineIdx, "year", errorMsg);
                    if (STRICT_MODE)
                        return errorCode;
                    LOG(WARNING) << "Convert data warning: " << errorMsg;
                }
                size = sizeof( aries_acc::AriesYear );
                memcpy( valueBuff + offset, &year, size );
                break;
            }
            case schema::ColumnType::TIME:
            {
                int mode = STRICT_MODE ? ARIES_DATE_STRICT_MODE : ARIES_DATE_NOT_STRICT_MODE;
                aries_acc::AriesTime time;
                try
                {
                    time = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesTime( col, mode );
                }
                catch ( ... )
                {
                    errorCode = FormatTruncWrongValueError(colEntry->GetName(), col, lineIdx, "time", errorMsg);
                    if (STRICT_MODE)
                        return errorCode;
                    LOG(WARNING) << "Convert data warning: " << errorMsg;
                }
                size = sizeof( aries_acc::AriesTime );
                memcpy( valueBuff + offset, &time, size );
                break;
            }
            case schema::ColumnType::LIST:
            case schema::ColumnType::UNKNOWN:
            {
                errorMsg.assign( "load data for column type " + std::to_string((int) colEntry->GetType()) );
                errorCode = ER_UNKNOWN_ERROR;
                return errorCode;
                break;
            }
        }
    }

    return 0;
}

class ColumnBlockFile
{
public:
    int m_fd;
    string m_outDir;
    AriesInitialTableSPtr m_initTable;
    string m_tableName;
    int m_colIdx;
    ColumnEntryPtr m_colEntry;
    bool m_append;
    bool m_isDictColumn;
    uint32 m_blockIdx = UINT32_MAX;
    string m_fileName;
    string m_filePath;
    size_t m_rowCount;

    ColumnBlockFile( const string& outDir,
                     AriesInitialTableSPtr& initTable,
                     const string& tableName,
                     const int colIdx,
                     ColumnEntryPtr &colEntry,
                     const bool isDictColumn,
                     uint32 blockIdx,
                     const bool append = false )
    : m_fd( -1 ),
      m_outDir( outDir ),
      m_initTable( initTable ),
      m_tableName( tableName ),
      m_colIdx( colIdx ),
      m_colEntry( colEntry ),
      m_append( append ),
      m_isDictColumn( isDictColumn ),
      m_blockIdx( blockIdx ),
      m_rowCount( 0 )
    {
        BuildBaseFileName();
        BuildBlockFilePath();
    }

    int Open()
    {
        LOG(INFO) << "Creating column file: " << m_filePath;
        int oflags = O_RDWR;
        mode_t mode = S_IRUSR | S_IWUSR;
        if ( m_append )
        {
            auto oldFilePath = m_initTable->GetBlockFilePath( m_colIdx, m_blockIdx );
            boost::filesystem::copy_file( oldFilePath, m_filePath );
            oflags |= O_APPEND;
        }
        else
        {
            oflags |= O_CREAT | O_EXCL;
        }
        m_fd = open( m_filePath.data(), oflags, mode );
        if ( !m_append )
        {
            AriesInitialTable::WriteColumnBlockFileHeader( m_colEntry,
                                                           m_fd,
                                                           m_filePath,
                                                           0,
                                                           false );
        }
        return m_fd;
    }
    void Close()
    {
        if ( m_fd > 0 )
        {
            close( m_fd );
            m_fd = -1;
        }
    }
    int Reset( uint32 blockIdx )
    {
        Close();

        m_rowCount = 0;

        m_blockIdx = blockIdx;
        m_append = false;

        BuildBlockFilePath();

        return Open();
    }
    size_t GetRowCount() const
    {
         return m_rowCount;
    }
    size_t IncRowCount()
    {
        return ++m_rowCount;
    }
    ~ColumnBlockFile()
    {
        Close();
    }

private:
    void BuildBaseFileName()
    {
        m_fileName = m_tableName + std::to_string( m_colIdx );
        if ( m_isDictColumn )
            m_fileName.append( "_" ).append( ARIES_DICT_FILE_NAME_SUFFIX ).append( "_" ).append( ARIES_DICT_INDEX_FILE_NAME_SUFFIX );
    }
    void BuildBlockFilePath()
    {
        m_filePath = m_outDir + "/" + m_fileName;
        m_filePath.append( "_" ).append( std::to_string( m_blockIdx ) );
    }
};

using ColumnBlockFileSPtr = std::shared_ptr<ColumnBlockFile>;

void restoreDataFiles(const map<string, string>& files)
{
    char errbuf[MYSYS_STRERROR_SIZE];
    for ( auto& it : files )
    {
        if ( 0 != rename( it.second.data(), it.first.data() ) )
        {
            int tmpErrno = errno;
            LOG(INFO) << "Failed to restore original data file: " << it.second << " ==> " << it.first;
            LOG(INFO) << "Error: " << std::to_string( tmpErrno ) << ", " << strerror_r(tmpErrno, errbuf, sizeof(errbuf));
        }
    }
}

class REMOVE_OLD_BACKUP_INFO
{
public:
    REMOVE_OLD_BACKUP_INFO(const string& arg_dbName, const string& arg_tableName, const string& arg_latestBkTs)
            : dbName( arg_dbName ),
              tableName( arg_tableName),
              latestBkTs( arg_latestBkTs )
    {}
    string dbName;
    string tableName;
    string latestBkTs; // timestamp of the latest backup
};

void* removeOlderBackups(void* arg)
{
    REMOVE_OLD_BACKUP_INFO bkInfo = *( REMOVE_OLD_BACKUP_INFO* ) arg;
    delete ( REMOVE_OLD_BACKUP_INFO* )arg;
    string tableBkDir = GetTableBackupDir( bkInfo.dbName, bkInfo.tableName );
    vector<string> bkDirs = listFiles( tableBkDir, false );
    for ( const auto& dir : bkDirs )
    {
        if ( dir < bkInfo.latestBkTs )
        {
            string fullDir = tableBkDir + "/" + dir;
            LOG(INFO) << "Delete backup dir: " << fullDir;
            boost::filesystem::remove_all( fullDir );
        }
    }
    return nullptr;
}

size_t GetColumnItemSize( const ColumnEntryPtr& colEntry )
{
    size_t itemSize = 0;
    switch ( colEntry->GetType() )
    {
        case schema::ColumnType::BOOL:
        case schema::ColumnType::TINY_INT: // 1 byte
            itemSize = sizeof(int8_t);
            break;
        case schema::ColumnType::SMALL_INT:  // 2 bytes
            itemSize = sizeof(int16_t);
            break;
        case schema::ColumnType::INT: // 4 bytes
            itemSize = sizeof(int32_t);
            break;
        case schema::ColumnType::LONG_INT: // 8 bytes
            itemSize = sizeof(int64_t);
            break;
        case schema::ColumnType::DECIMAL:
            itemSize = aries_acc::GetDecimalRealBytes( colEntry->numeric_precision, colEntry->numeric_scale );
            break;
        case schema::ColumnType::FLOAT:
            itemSize = sizeof(float);
            break;
        case schema::ColumnType::DOUBLE:
            itemSize = sizeof(double);
            break;
        case schema::ColumnType::DATE:
            itemSize = sizeof(aries_acc::AriesDate);
            break;
        case schema::ColumnType::DATE_TIME:
            itemSize = sizeof(aries_acc::AriesDatetime);
            break;
        case schema::ColumnType::TIMESTAMP:
            itemSize = sizeof(aries_acc::AriesTimestamp);
            break;
        case schema::ColumnType::TEXT:
        case schema::ColumnType::VARBINARY:
        case schema::ColumnType::BINARY:
            itemSize = colEntry->GetLength();
            break;
        case schema::ColumnType::YEAR:
            itemSize = sizeof(aries_acc::AriesYear);
            break;
        case schema::ColumnType::TIME:
            itemSize = sizeof(aries_acc::AriesTime);
            break;
        case schema::ColumnType::LIST:
        case schema::ColumnType::UNKNOWN:
        {
            string msg = "Get column size for type " + std::to_string((int) colEntry->GetType());
            ARIES_EXCEPTION( ER_UNKNOWN_ERROR,  msg.data() );
            break;
        }
    }
    return itemSize;
}

static int backupTable( const DatabaseEntrySPtr& dbEntry,
                         const TableEntrySPtr& tableEntry,
                         string& tableBackupDir, // OUT
                         string& bkDirSuffix, // OUT
                         std::map<string, string>& backedUpFiles ) // OUT
{
    backedUpFiles.clear(); 

    auto initTable = 
        AriesInitialTableManager::GetInstance().getTable( dbEntry->GetName(),
                                                          tableEntry->GetName() );
    auto blockIndex = initTable->GetBlockCount() - 1;
    auto blockRowCount = initTable->GetBlockRowCount( blockIndex );
    // last block is not full, we started with appending to the last block, backup the last block
    if ( blockRowCount < ARIES_BLOCK_FILE_ROW_COUNT )
    {
        auto tableBackupDir = MakeTableBackupDir( dbEntry->GetName(), tableEntry->GetName(), bkDirSuffix );
        for ( auto& colEntry : tableEntry->GetColumns() )
        {
            auto oldPath = initTable->GetBlockFilePath( colEntry->GetColumnIndex(), blockIndex );
            auto fileName = AriesInitialTable::GetBlockFileName( tableEntry->GetName(),
                                                                 colEntry->GetColumnIndex(),
                                                                 blockIndex,
                                                                 EncodeType::DICT == colEntry->encode_type );
            auto newPath = tableBackupDir + fileName;
            if ( 0 != rename( oldPath.data(), newPath.data() ) )
            {
                set_my_errno(errno);
                return -1;
            }
            backedUpFiles[ oldPath ] = newPath;
        }
    }

    LOG(INFO) << "Backup data files done";
    return 0;
}

class ColumnBuff
{
public:
    ColumnBuff( const ColumnEntryPtr& colEntry,
                const size_t capacity = ARIES_BLOCK_FILE_ROW_COUNT )
            : m_itemCnt( 0 ),
              m_capacity( capacity ),
              m_colEntry( colEntry )
    {
        m_containNull = m_colEntry->IsAllowNull() ? 1 : 0;
        auto defValuePtr = m_colEntry->GetDefault();
        m_defValue = defValuePtr ? *defValuePtr : "";
        m_itemStoreSize = m_colEntry->GetItemStoreSize();
    }

    const ColumnEntryPtr& getColumnEntry() const
    {
        return m_colEntry;
    }
    size_t getResultItemSize() const
    {
        return m_itemStoreSize;
    }
    /*
    void pushData( const string& s )
    {
        data.append( s );
        dataSize += s.size();
        ++itemCnt;
        index.push_back( dataSize );
    }
    */
    virtual void pushData( const uint64_t rowIdx, const char* s, size_t size ) = 0;
    virtual const char* getResultData() const = 0;
    virtual size_t getResultDataSize() const
    {
        return m_itemStoreSize * m_itemCnt;
    }
    size_t getItemCount() const
    {
        return m_itemCnt;
    }
    virtual string getOriginalItem( size_t idx ) const = 0;
    virtual char* getResultItem( size_t idx ) const = 0;
    size_t getCapacity() const
    {
        return m_capacity;
    }
    virtual void clear()
    {
        m_data.clear();
        m_itemCnt = 0;
    }
protected:
    size_t m_itemCnt;
    size_t m_capacity;
    string m_data;
    ColumnEntryPtr m_colEntry;
    int8_t m_containNull;
    string m_defValue;
    size_t m_itemStoreSize;
};
using ColumnBuffPtr = shared_ptr< ColumnBuff >;

class FixedSizeColumnBuff : public ColumnBuff
{
public:
    FixedSizeColumnBuff( const ColumnEntryPtr& colEntry )
            : ColumnBuff( colEntry ),
              m_dataSize( 0 )
    {
        m_index.reserve( m_capacity );
        m_index.push_back( 0 );

        
        auto valueType = colEntry->GetType();
        auto length = colEntry->GetLength();
        auto nullable = colEntry->IsAllowNull();
        AriesColumnType dataType =
                CovertToAriesColumnType( valueType,
                                         length,
                                         nullable,
                                         true,
                                         colEntry->numeric_precision,
                                         colEntry->numeric_scale );
        m_result = std::make_shared< AriesDataBuffer >( dataType );
        int8_t* buff = ( int8_t* ) malloc( m_capacity * m_itemStoreSize );
        m_result->AttachBuffer( buff, m_capacity );
    }
    aries_acc::AriesDataBufferSPtr getResultBuffer() const
    {
        return m_result;
    }
    void pushData( const uint64_t rowIdx, const char* s, size_t size )
    {
        m_data.append( s, size );
        m_dataSize += size;
        ++m_itemCnt;
        m_index.push_back( m_dataSize );
    }
    void pushResult( char* buff )
    {
        // m_result.append( buff, m_itemStoreSize );
    }
    char* getResultItem( size_t idx ) const
    {
        assert( idx < m_itemCnt  );
        return ( char* )m_result->GetData( idx );
    }
    string getOriginalItem( size_t idx ) const
    {
        assert( idx < m_itemCnt  );
        const char* itemData = m_data.data() + m_index[ idx ];
        size_t itemDataSize = m_index[ idx + 1 ] - m_index[ idx ];
        string s;
        return s.assign( itemData, itemDataSize );
    }
    const char* getResultData() const
    {
        return ( const char* )m_result->GetData();
    }
    bool isFull() const
    {
        return m_itemCnt == m_capacity;
    }
    void clear()
    {
        ColumnBuff::clear();
        m_dataSize = 0;
        // m_result.clear();
        m_index.clear();
        m_index.push_back( 0 );
    }

private:
    size_t m_dataSize;
    vector< int64_t > m_index;
    aries_acc::AriesDataBufferSPtr m_result;
};
using FixedSizeColumnBuffPtr = shared_ptr<FixedSizeColumnBuff>;

class CharColumnBuff : public ColumnBuff
{
public:
    CharColumnBuff( const ColumnEntryPtr& colEntry )
            : ColumnBuff( colEntry )
    {
        m_data.resize( m_capacity * m_itemStoreSize, 0 );
    }
    void pushData( const uint64_t rowIdx, const char* s, size_t size )
    {
        size_t buffPos = 0;
        char* buffPtr = ( char* )m_data.data() + m_itemCnt * m_itemStoreSize;
        if ( m_containNull )
        {
            if ( size > 0 )
            {
                *buffPtr = 1;
            }
            buffPos = 1;
        }
        if ( size > 0 )
        {
            memcpy( buffPtr + buffPos, s, size );
        }
        ++m_itemCnt;
    }
    const char* getResultData() const
    {
        return m_data.data();
    }
    char* getResultItem( size_t idx ) const
    {
        assert( idx < m_itemCnt  );
        return ( char* )( m_data.data() + idx * m_itemStoreSize );
    }
    virtual size_t getResultDataSize() const
    {
        return m_itemStoreSize * m_itemCnt;
    }

    string getOriginalItem( size_t idx ) const
    {
        if ( m_containNull )
        {
            std::string value;
            if( *( ( char* )m_data.data() + idx * m_itemStoreSize ) )
            {
                std::string tmp;
                tmp.assign( ( char* )m_data.data() + idx * m_itemStoreSize + 1, m_itemStoreSize - 1 );
                value = tmp.c_str();
            }
            else
                value = "NULL";
            return value;
        }
        else
            return string( getResultItem( idx ), m_itemStoreSize );
    }
    void clear()
    {
        memset( ( char* )m_data.data(), 0, m_data.size() );
        ColumnBuff::clear();
        m_data.resize( m_capacity * m_itemStoreSize, 0 );
    }
};
using CharColumnBuffPtr = shared_ptr< CharColumnBuff >;

class DictEncodedColumnBuff : public ColumnBuff
{
public:

    DictEncodedColumnBuff( const ColumnEntryPtr& colEntry )
            : ColumnBuff( colEntry )
    {
        m_dict = AriesDictManager::GetInstance().GetDict( colEntry->GetDictId() );
        AriesDictManager::GetInstance().ReadDictData( m_dict );

        m_indices = make_shared< aries_acc::AriesDataBuffer >( colEntry->GetDictIndexColumnType(), m_capacity );
        m_newDictIndices.reserve( m_capacity );
    }
    AriesDictSPtr getDict() const
    {
        return m_dict;
    }
    void pushData( const uint64_t rowIdx, const char* s, size_t size )
    {
        addDataRow( rowIdx, s, size, m_colEntry->GetName() );
    }
    const char* getResultData() const
    {
        return ( const char* )m_indices->GetData();
    }
    virtual size_t getResultDataSize() const
    {
        return m_indices->GetItemSizeInBytes() * m_itemCnt;
    }
    const char* getDictData()
    {
        return m_dict->getDictData();
    }
    size_t getDictDataSize()
    {
        return m_dict->getDictDataSize();
    }
    char* getResultItem( size_t idx ) const
    {
        assert( idx < m_itemCnt  );
        return ( char* )getResultData() + idx * m_dict->getDictIndexItemSize();
    }
    const size_t getDictItemCount()
    {
        return m_dict->getDictItemCount();
    }

    string getOriginalItem( size_t idx ) const
    {
        int8_t* itemData = m_indices->GetItemDataAt( idx );
        if ( m_indices->isNullableColumn() && 0 == *itemData ) // null value
        {
            return string( "", 0 );
        }
        else
        {
            if ( m_indices->isNullableColumn() )
                ++itemData;
            int32_t index = -1;
            switch ( m_dict->GetIndexDataType() )
            {
            case schema::ColumnType::TINY_INT:
                index = *( int8_t* )itemData;
                break;
            case schema::ColumnType::SMALL_INT:
                index = *( int16_t* )itemData;
                break;
            case schema::ColumnType::INT:
                index = *( int32_t* )itemData;
                break;
            default:
                aries::ThrowNotSupportedException("dict encoding type: " + get_name_of_value_type( m_dict->GetIndexDataType() ) );
                break;
            }
            return m_dict->getDictItem( index );
        }
    }
    void clear()
    {
        m_itemCnt = 0;
        m_newDictIndices.clear();
    }
    int32_t addDataRow( int tupleIdx, const char* item, size_t size,
                        const string& colName )
    {
        int32_t idx;
        int errorCode;
        string errorMsg;
        bool newDictItem = m_dict->addDict( item, size, tupleIdx, &idx, errorCode, errorMsg );
        if ( 0 != errorCode )
            ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg );

        ARIES_ASSERT( m_itemCnt < m_capacity, "data row count exceeded capacity" );
        int8_t* pIndex = m_indices->GetItemDataAt( m_itemCnt++ );
        if ( m_dict->IsNullable() )
        {
            if ( 0 == size ) // NULL value
            {
                *pIndex = 0;
            }
            else
            {
                *pIndex = 1;
            }
            memcpy( pIndex + 1, &idx, m_indices->GetItemSizeInBytes() - 1 );
        }
        else
        {
            memcpy( pIndex, &idx, m_indices->GetItemSizeInBytes() );
        }
        if ( newDictItem )
        {
            m_newDictIndices.push_back( idx );
        }
        
        return idx;
    }

    const vector< int32_t >& GetNewDictIndices() const
    {
        return m_newDictIndices;
    }
private:
    AriesDictSPtr m_dict;
    aries_acc::AriesDataBufferSPtr m_indices;
    vector< int32_t > m_newDictIndices;

};
using DictEncodedColumnBuffSPtr = std::shared_ptr< DictEncodedColumnBuff >; 

static size_t readLine2( READ_INFO &read_info,
                      uint64_t dataRowIdx,
                      const vector<ColumnEntryPtr>& cols,
                      const vector< int8_t >& strTypeFlags,
                      const vector< int8_t >& nullableFlags,
                      size_t enclosed_length,
                      vector< FixedSizeColumnBuffPtr >& fixedSizeColumnBuffs,
                      vector< CharColumnBuffPtr >& charColumnBuffs,
                      vector< DictEncodedColumnBuffSPtr >& dictColumnBuffs/*,
                      int64_t& readAndScanTime,
                      int64_t& scanVectorOpTIme,
                      int64_t& otherTime*/ )
{
    size_t colCnt = cols.size();
    size_t readColCnt = 0;
    int fixedSizeColIdx = 0, charColIdx = 0, dictColIdx = 0;
    const char* colData;
    size_t colDataSize;
    // aries::CPU_Timer t;

    // read a line
    for (; readColCnt < colCnt; ++readColCnt)
    {
        uint length;
        uchar *pos;
        auto& col = cols[ readColCnt ];

        // t.begin();
        if (read_info.read_field())
            break;
        // readAndScanTime += t.end();

        // t.begin();
        pos = read_info.row_start;
        length = (uint) (read_info.row_end - pos);

        // null
        /**
         If FIELDS ENCLOSED BY is not empty, a field containing the literal word NULL as its value is read as
         a NULL value. This differs from the word NULL enclosed within FIELDS ENCLOSED BY characters,
         which is read as the string 'NULL'.

         If an input line has too many fields, the extra fields are ignored and the number of warnings is
         incremented.
         If an input line has too few fields, the table columns for which input fields are missing are set to their
         default values. Default value assignment is described in Section 11.7, “Data Type Default Values”.
         An empty field value is interpreted different from a missing field:
         • For string types, the column is set to the empty string.
         • For numeric types, the column is set to 0.
         • For date and time types, the column is set to the appropriate “zero” value for the type. See
         Section 11.3, “Date and Time Types”.
         These are the same values that result if you assign an empty string explicitly to a string, numeric, or
         date or time type explicitly in an INSERT or UPDATE statement.
         *
         */
        if ((!read_info.enclosed &&
             (enclosed_length && length == 4 &&
              !memcmp(pos, STRING_WITH_LEN("NULL")))) ||
            (length == 1 && read_info.found_null))
        {
            colData = "";
            colDataSize = 0;
        }
        else
        {
            if ( 0 == length )
            {
                colData = "";
                colDataSize = 1; // empty value
            }
            else
            {
                colData = (const char *)pos;
                colDataSize = length;
            }
        }
        // otherTime += t.end();

        // t.begin();
        if ( strTypeFlags[ readColCnt ] )
        {
            CheckCharLen( col->GetName(), colDataSize );
            if ( colDataSize > ( size_t )col->GetLength() )
            {
                if ( STRICT_MODE )
                    ARIES_EXCEPTION( ER_DATA_TOO_LONG, col->GetName().data(), dataRowIdx + 1 );
                else
                {
                    string errorMsg;
                    FormatDataTruncError( col->GetName(), dataRowIdx + 1, errorMsg);
                    LOG(WARNING) << "Convert data warning: " << errorMsg;
                    colDataSize = col->GetLength();
                }
            }
            if ( aries::EncodeType::DICT == col->encode_type )
            {
                auto& colBuff = dictColumnBuffs[ dictColIdx++ ];
                colBuff->pushData( dataRowIdx, colData, colDataSize );
            }
            else
            {
                auto& charColBuff = charColumnBuffs[ charColIdx++ ];
                charColBuff->pushData( dataRowIdx, colData, colDataSize );
            }
        }
        else
        {
            fixedSizeColumnBuffs[ fixedSizeColIdx++ ]->pushData( dataRowIdx, colData, colDataSize );
        }
    }
    // check column count
    if ( readColCnt > 0 && readColCnt < colCnt)
    {
        colData = "";
        colDataSize = 1; // empty value
        for (size_t colIdx = readColCnt; colIdx < colCnt; ++colIdx)
        {
            auto& col = cols[ colIdx ];
            if ( strTypeFlags[ colIdx ] )
            {
                if ( aries::EncodeType::DICT == col->encode_type )
                {
                    auto& colBuff = dictColumnBuffs[ dictColIdx++ ];
                    colBuff->pushData( dataRowIdx, colData, colDataSize );
                }
                else
                {
                    auto& charColBuff = charColumnBuffs[ charColIdx++ ];
                    charColBuff->pushData( dataRowIdx, colData, colDataSize );
                }
            }
            else
                fixedSizeColumnBuffs[fixedSizeColIdx++]->pushData( dataRowIdx, colData, colDataSize );
        }
    }
    return readColCnt;
}
// static int readLine( READ_INFO &read_info,
//                      int colCnt,
//                      size_t enclosed_length,
//                      vector<string>& line )
// {
//     line.clear();
//     int readColCnt = 0;
//     string field;
// 
//     // read a line
//     for (; readColCnt < colCnt; ++readColCnt)
//     {
//         uint length;
//         uchar *pos;
// 
//         if (read_info.read_field())
//             break;
// 
//         pos = read_info.row_start;
//         length = (uint) (read_info.row_end - pos);
// 
//         // null
//         if ((!read_info.enclosed &&
//              (enclosed_length && length == 4 &&
//               !memcmp(pos, STRING_WITH_LEN("NULL")))) ||
//             (length == 1 && read_info.found_null)) {
//             field.assign("", 0);
//         } else {
//             field.assign((char *) pos, length);
//         }
//         line.emplace_back(field);
//     }
//     return readColCnt;
// }
static int readLineForPreScan( READ_INFO &read_info,
                               int colCnt,
                               size_t enclosed_length,
                               const vector<ColumnEntryPtr>& cols,
                               vector< int8_t >& isStrType,
                               std::vector<size_t>& columnMaxSizes )
{
    int readColCnt = 0;

    // read a line
    for (; readColCnt < colCnt; ++readColCnt)
    {
        uint length;
        uchar *pos;

        if (read_info.read_field())
            break;

        pos = read_info.row_start;
        length = (uint) (read_info.row_end - pos);

        // null
        if ( ( !read_info.enclosed &&
               (enclosed_length && length == 4 &&
               !memcmp(pos, STRING_WITH_LEN("NULL"))) ) ||
            ( length == 1 && read_info.found_null ) ) {
            length = 0;
        }
        if ( isStrType[ readColCnt ] )
        {
            CheckCharLen( cols[ readColCnt ]->GetName(), length );
            if ( length >  columnMaxSizes[ readColCnt ] )
                columnMaxSizes[ readColCnt ] = length;
        }
    }
    return readColCnt;
}

int64_t preScan2( THD *thd,
                  const DatabaseEntrySPtr& dbEntry,
                  const TableEntrySPtr& tableEntry,
                  const string& csvFilePath,
                  READ_INFO &read_info,
                  const string& enclosed,
                  const vector<ColumnEntryPtr>& cols,
                  uint64_t& lineIdx )
{
    size_t enclosed_length = enclosed.length();
    int colCnt = cols.size();

    // vector<string> line;
    int colIdx;
    uint64_t readLineCnt = 0;
    int readColCnt = 0;
    ColumnEntryPtr colEntry;
#ifdef ARIES_PROFILE
    aries::CPU_Timer tTotal, tScan;
    tTotal.begin();
#endif
    int64_t scanTime = 0, totalTime = 0;
    float s;

    vector< int8_t > isStrType;

    std::vector<size_t> columnMaxSizes; // include null byte
    columnMaxSizes.reserve( colCnt );

    for ( colIdx = 0; colIdx < colCnt; ++colIdx )
    {
        colEntry = cols[ colIdx ];
        if ( ColumnEntry::IsStringType( colEntry->GetType() ) )
        {
            columnMaxSizes.emplace_back( 1 );
            isStrType.emplace_back( 1 );
        }
        else
        {
            isStrType.emplace_back( 0 );
            size_t itemStoreSize = colEntry->GetItemStoreSize();
            columnMaxSizes.emplace_back( itemStoreSize );
        }
    }

    // read until end of file
    for (;; ++lineIdx)
    {
        if ( IsThdKilled( thd ) )
            goto interrupted;

        // read a line
#ifdef ARIES_PROFILE
        tScan.begin();
#endif
        readColCnt = readLineForPreScan( read_info,
                                         colCnt,
                                         enclosed_length,
                                         cols,
                                         isStrType,
                                         columnMaxSizes );
#ifdef ARIES_PROFILE
        scanTime += tScan.end();
#endif

        /* Have not read any field, thus input file is simply ended */
        /*
         * mysql 5.7.26
         * 1. for empty lines:
         * ERROR 1366 (HY000): Incorrect integer value: '' for column 'f1' at row 3
         * 2. for lines with all spaces:
         * ERROR 1366 (HY000): Incorrect integer value: '  ' for column 'f1' at row 2
         */
        if (!readColCnt)
            break;
        // got a line
        // check column count
        if ( readColCnt < colCnt ) {
            // string tmpLine = "line data: ";
            // for (const auto &f : line) {
            //     tmpLine.append(f).append("||");
            // }
            // LOG(INFO) << tmpLine;
            // cleanup generated column files
            // ARIES_EXCEPTION( ER_WARN_TOO_FEW_RECORDS, lineIdx + 1 );
        } else if ( readColCnt > colCnt ) {
            // string tmpLine = "line data: ";
            // for (const auto &f : line) {
            //     tmpLine.append(f).append("||");
            // }
            // LOG(INFO) << tmpLine;
            // cleanup generated column files
            ARIES_EXCEPTION( ER_WARN_TOO_MANY_RECORDS, lineIdx + 1 );
        }

        ++readLineCnt;

        /*
          We don't need to reset auto-increment field since we are restoring
          its default value at the beginning of each loop iteration.
        */
        if (read_info.next_line())            // Skip to next line
            break;
        if (read_info.line_cuted)
        {
            // mysql 5.7.26:
            // in strict mode, return error
            // thd->cuted_fields++;			/* To long row */
            // push_warning_printf(thd, Sql_condition::SL_WARNING,
            //                     ER_WARN_TOO_MANY_RECORDS, ER(ER_WARN_TOO_MANY_RECORDS),
            //                     thd->get_stmt_da()->current_row_for_condition());
            // string tmpLine = "line data: ";
            // for (const auto &f : line) {
            //     tmpLine.append(f).append("||");
            // }
            // LOG(INFO) << tmpLine;
            ARIES_EXCEPTION( ER_WARN_TOO_MANY_RECORDS, lineIdx + 1 );
        }
    }
    for ( colIdx = 0; colIdx < colCnt; ++colIdx )
    {
        colEntry = cols[ colIdx ];
        if ( isStrType[ colIdx ] )
        {
            size_t schemaLen = colEntry->GetLength();
            size_t maxItemLen = columnMaxSizes[ colIdx ];
            if ( maxItemLen < schemaLen && EncodeType::DICT != colEntry->encode_type )
            {
                LOG(INFO) << "column [" << colIdx
                          << "]: max item size is less than schema defined: " << schemaLen
                          << ", it will be adjusted to " << maxItemLen
                          << " and save "
                          << (schemaLen - maxItemLen) * readLineCnt << " bytes!";
                colEntry->FixLength( maxItemLen );
            }
        }
    }
#ifdef ARIES_PROFILE
    totalTime += tTotal.end();
    s = ( totalTime + 0.0 ) / 1000 / 1000;
    LOG( INFO ) << "Import " <<  csvFilePath << " time ( pre scan total ): " << s << " s, "
                << s / 60 << " m";
    s = ( scanTime + 0.0 ) / 1000 / 1000;
    LOG( INFO ) << "--\ttime ( scan ): " << s << " s, "
                << s / 60 << " m";
#endif
    return readLineCnt;

interrupted:
    SendKillMessage();
    return -1;
}

class ConvertResult2
{
public:
    ConvertResult2()
            : errorCode( 0 )
    {}
    int errorCode;
    string errorMsg;
    vector< FixedSizeColumnBuffPtr > results;
};
using ConvertResult2Ptr = shared_ptr<ConvertResult2>;
ConvertResult2Ptr
ConvertColumns( const vector< FixedSizeColumnBuffPtr >& fixedSizeColumnBuffs,
                size_t startColIdx,
                size_t jobCount,
                int64_t startLineIdx )
{
    ConvertResult2Ptr result = make_shared<ConvertResult2>();
    int colIdx;
    size_t itemCount;
    FixedSizeColumnBuffPtr colBuff;
    ColumnEntryPtr colEntry;
    int8_t containNull;
    string defValue;
    shared_ptr<string> defValuePtr;

    // const int tmpColBuffSize = ARIES_MAX_CHAR_WIDTH + 1;
    // uchar tmpColBuff[ tmpColBuffSize ] = {0}; // enough for all data types
    for ( size_t idx = 0; idx < jobCount; ++idx )
    {
        // convert a column
        colIdx = startColIdx + idx;
        colBuff = fixedSizeColumnBuffs[ colIdx ];
        colEntry = colBuff->getColumnEntry();
        containNull = colEntry->IsAllowNull() ? 1 : 0;
        defValuePtr = colEntry->GetDefault();
        defValue = defValuePtr ? *defValuePtr : "";

        itemCount = colBuff->getItemCount();
        for ( size_t tmpLineIdx = 0; tmpLineIdx < itemCount; ++tmpLineIdx )
        {
            // covert an item of a column
            auto colData = colBuff->getOriginalItem( tmpLineIdx );
            result->errorCode = ToColumnValue( colEntry,
                                               containNull,
                                               defValue,
                                               colData,
                                               startLineIdx + tmpLineIdx,
                                               ( uchar* )colBuff->getResultItem( tmpLineIdx ),
                                               colBuff->getResultItemSize(),
                                               result->errorMsg );
            if ( 0 != result->errorCode )
                return result;
            // colBuff->pushResult( (char*)tmpColBuff );
        }
        result->results.push_back( colBuff );
    }
    return result;
}

struct ImportCsvPartionContext
{
    // 每个列对应一个 vector item
    vector< ColumnBlockFileSPtr > m_partitionColumnBlockFiles;
    // 每个列对应一个 vector item
    vector< shared_ptr<WRITE_BUFF_INFO> > m_partitionColumnWriteBuffs;

    size_t m_rowCount = 0;
};
using ImportCsvPartionContextSPtr = std::shared_ptr< ImportCsvPartionContext >;

struct ImportCsvContext
{
    aries_engine::AriesTransactionPtr m_tx;
    TableEntrySPtr m_tableEntry;
    vector< ColumnEntryPtr > m_cols;
    vector<int8_t> m_containNulls;
    size_t m_totalBlockCount = 0;
    size_t m_newBlockCount = 0;
    size_t m_nextBlockIndex = 0;
    // 每个partition对应一个ImportCsvPartionContext 
    vector< ImportCsvPartionContextSPtr > m_partitionContext;
    // [ partitionIndex, { blockIndices } ]
    map< uint32_t, set< int > > m_partitionBlockIndices;

    bool m_lastLines = false;

    size_t GetNextBlockIndex()
    {
        ++m_newBlockCount;
        ++m_totalBlockCount;
        return m_nextBlockIndex++;
    }
};

void
prepareImportInfos( ImportCsvContext &context,
                    const DatabaseEntrySPtr& dbEntry,
                    const TableEntrySPtr& tableEntry,
                    const uint64_t totalLineCnt,
                    int batchWriteSize,
                    const string& outputDir,
                    std::vector<string>& defaultValues, // OUT
                    vector< int8_t >& strTypeFlags, // OUT
                    std::vector<ColumnBlockFileSPtr>& blockFiles, // OUT
                    vector< shared_ptr<WRITE_BUFF_INFO> >& writeBuffs, // OUT
                    vector< ColumnBuffPtr >& allColumnBuffs,  // OUT
                    vector< FixedSizeColumnBuffPtr >& fixedSizeColumnBuffs,  // OUT
                    vector< CharColumnBuffPtr >& charColumnBuffs,  // OUT
                    vector< DictEncodedColumnBuffSPtr >& dictColumnBuffs, // OUT
                    // unordered_map< int64_t, DictEncodedColumnBuffSPtr >& dictMap, // OUT
                    uint32_t &startBlockIndex,
                    uint32_t &buffLineCnt )
{
    char errbuf[MYSYS_STRERROR_SIZE] = {0};
    int colCnt = context.m_cols.size();
    int colIdx;
    ColumnEntryPtr colEntry;
    size_t blockSize = 0;
    int writeBuffSize = batchWriteSize;
    auto initTable = 
        AriesInitialTableManager::GetInstance().getTable( dbEntry->GetName(),
                                                          tableEntry->GetName() );
    context.m_totalBlockCount = initTable->GetBlockCount();
    startBlockIndex  = initTable->GetBlockCount() - 1;
    auto blockRowCount = initTable->GetBlockRowCount( startBlockIndex );
    bool appendToExising = false;
    buffLineCnt = 0;

    if ( blockRowCount == 0 )
        --context.m_totalBlockCount;

    // last block is not full, start with appending to the last block
    if ( blockRowCount < ARIES_BLOCK_FILE_ROW_COUNT )
    {
        appendToExising  = true;
        buffLineCnt = blockRowCount;
    }
    else
    {
        startBlockIndex += 1;
    }

    for ( colIdx = 0; colIdx < colCnt; ++colIdx )
    {
        colEntry = context.m_cols[ colIdx ];

        auto filePtr = std::make_shared< ColumnBlockFile >( outputDir,
                                                            initTable,
                                                            tableEntry->GetName(),
                                                            colIdx,
                                                            colEntry,
                                                            EncodeType::DICT == colEntry->encode_type,
                                                            startBlockIndex,
                                                            appendToExising );
        if ( -1 == filePtr->Open() )
        {
            set_my_errno( errno );

            boost::filesystem::remove_all( outputDir );

            ARIES_EXCEPTION( EE_CANTCREATEFILE, filePtr->m_filePath.data(), my_errno(),
                             strerror_r( my_errno(), errbuf, sizeof(errbuf) ) );
        }
        blockFiles.emplace_back( filePtr );

        if ( 0 == writeBuffSize )
        {
            if ( 0 == blockSize )
            {
                blockSize = block_size( filePtr->m_fd );
            }
            writeBuffSize = 16 * blockSize;
            LOG( INFO ) << "Write batch size: " << writeBuffSize;
        }
        writeBuffs.push_back( make_shared<WRITE_BUFF_INFO>( writeBuffSize ) );

        int8_t containNull = colEntry->IsAllowNull() ? 1 : 0;
        auto defValuePtr = colEntry->GetDefault();
        context.m_containNulls.emplace_back( containNull );
        defaultValues.emplace_back( defValuePtr ? *defValuePtr : "" );

        if ( ColumnEntry::IsStringType( colEntry->GetType() ) )
        {
            strTypeFlags.emplace_back( 1 );
            if ( EncodeType::DICT == colEntry->encode_type )
            {
                auto colBuff = make_shared< DictEncodedColumnBuff >( colEntry );
                dictColumnBuffs.emplace_back( colBuff );
                // dictMap[ colEntry->GetDictId() ] = colBuff;
                allColumnBuffs.emplace_back( colBuff );
            }
            else
            {
                auto colBuff = make_shared< CharColumnBuff >( colEntry );
                charColumnBuffs.emplace_back( colBuff );
                allColumnBuffs.emplace_back( colBuff );
            }
        }
        else
        {
            strTypeFlags.emplace_back( 0 );
            auto colBuff = make_shared< FixedSizeColumnBuff >( colEntry );
            fixedSizeColumnBuffs.emplace_back( colBuff );

            allColumnBuffs.emplace_back( colBuff );
        }
    }
}

void
prepareImportInfosPartitioned( ImportCsvContext &context,
                               const DatabaseEntrySPtr& dbEntry,
                               const TableEntrySPtr& tableEntry,
                               const uint64_t totalLineCnt,
                               int batchWriteSize,
                               const string& outputDir,
                               std::vector<string>& defaultValues, // OUT
                               vector< int8_t >& strTypeFlags, // OUT
                               std::vector<ColumnBlockFileSPtr>& blockFiles, // OUT
                               vector< shared_ptr<WRITE_BUFF_INFO> >& writeBuffs, // OUT
                               vector< ColumnBuffPtr >& allColumnBuffs,  // OUT
                               vector< FixedSizeColumnBuffPtr >& fixedSizeColumnBuffs,  // OUT
                               vector< CharColumnBuffPtr >& charColumnBuffs,  // OUT
                               vector< DictEncodedColumnBuffSPtr >& dictColumnBuffs, // OUT
                               // unordered_map< int64_t, DictEncodedColumnBuffSPtr >& dictMap, // OUT
                               uint32_t &startBlockIndex,
                               uint32_t &buffLineCnt )
{
    char errbuf[MYSYS_STRERROR_SIZE] = {0};
    int colCnt = context.m_cols.size();
    int colIdx;
    ColumnEntryPtr colEntry;
    size_t blockSize = 0;
    int writeBuffSize = batchWriteSize;
    auto initTable = 
        AriesInitialTableManager::GetInstance().getTable( dbEntry->GetName(),
                                                          tableEntry->GetName() );
    
    context.m_totalBlockCount = initTable->GetBlockCount();
    startBlockIndex  = initTable->GetBlockCount() - 1;
    auto blockRowCount = initTable->GetBlockRowCount( startBlockIndex );
    if ( blockRowCount > 0 )
        ++startBlockIndex;
    else
        --context.m_totalBlockCount;
    context.m_nextBlockIndex = startBlockIndex;

    buffLineCnt = 0;

    for ( colIdx = 0; colIdx < colCnt; ++colIdx )
    {
        colEntry = context.m_cols[ colIdx ];

        int8_t containNull = colEntry->IsAllowNull() ? 1 : 0;
        auto defValuePtr = colEntry->GetDefault();
        context.m_containNulls.emplace_back( containNull );
        defaultValues.emplace_back( defValuePtr ? *defValuePtr : "" );

        if ( ColumnEntry::IsStringType( colEntry->GetType() ) )
        {
            strTypeFlags.emplace_back( 1 );
            if ( EncodeType::DICT == colEntry->encode_type )
            {
                auto colBuff = make_shared< DictEncodedColumnBuff >( colEntry );
                dictColumnBuffs.emplace_back( colBuff );
                // dictMap[ colEntry->GetDictId() ] = colBuff;
                allColumnBuffs.emplace_back( colBuff );
            }
            else
            {
                auto colBuff = make_shared< CharColumnBuff >( colEntry );
                charColumnBuffs.emplace_back( colBuff );
                allColumnBuffs.emplace_back( colBuff );
            }
        }
        else
        {
            strTypeFlags.emplace_back( 0 );
            auto colBuff = make_shared< FixedSizeColumnBuff >( colEntry );
            fixedSizeColumnBuffs.emplace_back( colBuff );

            allColumnBuffs.emplace_back( colBuff );
        }
    }

    uint32_t partitionCount = tableEntry->GetPartitionCount();
    for ( size_t i = 0; i < partitionCount; ++i )
    {
        context.m_partitionContext.emplace_back( std::make_shared< ImportCsvPartionContext >() );
    }

    for ( uint32_t partitionIndex = 0; partitionIndex < partitionCount; ++ partitionIndex )
    {
        // 同一个partition中所有列的block index保持一致
        auto blockIndex = context.GetNextBlockIndex();
        context.m_partitionBlockIndices[ partitionIndex ].emplace( blockIndex );

        for ( colIdx = 0; colIdx < colCnt; ++colIdx )
        {
            colEntry = context.m_cols[ colIdx ];

            auto filePtr = std::make_shared< ColumnBlockFile >( outputDir,
                                                                initTable,
                                                                tableEntry->GetName(),
                                                                colIdx,
                                                                colEntry,
                                                                EncodeType::DICT == colEntry->encode_type,
                                                                blockIndex );
            if ( -1 == filePtr->Open() )
            {
                set_my_errno( errno );

                boost::filesystem::remove_all( outputDir );

                ARIES_EXCEPTION( EE_CANTCREATEFILE, filePtr->m_filePath.data(), my_errno(),
                                 strerror_r( my_errno(), errbuf, sizeof(errbuf) ) );
            }
            if ( 0 == writeBuffSize )
            {
                if ( 0 == blockSize )
                {
                    blockSize = block_size( filePtr->m_fd );
                }
                writeBuffSize = 16 * blockSize;
                LOG( INFO ) << "Write batch size: " << writeBuffSize;
            }

            context.m_partitionContext[ partitionIndex ]->m_partitionColumnBlockFiles.emplace_back( filePtr );
            context.m_partitionContext[ partitionIndex ]->m_partitionColumnWriteBuffs.emplace_back( make_shared<WRITE_BUFF_INFO>( writeBuffSize ) );
        }
    }
}

string formatStatistics( int64_t readLineCnt,
                         int64_t readAndScanTime,
                         int64_t convertDataTime,
                         int64_t writeTime,
                         int64_t pkCheckTime,
                         int64_t fkCheckTime,
                         int64_t backupTime,
                         int64_t moveTime,
                         int64_t otherTime )

{
    string str;
    str.append("--\ttotal line: ").append( std::to_string( readLineCnt ) ).append( "\n" );
    float s = ( readAndScanTime + 0.0 ) / 1000 / 1000;
    str.append( "--\ttime ( read and scan ): " ).append( std::to_string( s ) ).append( " s, " ).append( std::to_string( s / 60 ) ).append( " m\n" );
    s = ( convertDataTime + 0.0 ) / 1000 / 1000;
    str.append( "--\ttime ( convert data ): " ).append( std::to_string( s ) ).append( " s, " ).append( std::to_string( s / 60 ) ).append( " m\n" );
    s = ( writeTime + 0.0 ) / 1000 / 1000;
    str.append( "--\ttime ( write data ): " ).append( std::to_string( s ) ).append( " s, " ).append( std::to_string( s / 60 ) ).append( " m\n" );
    s = ( pkCheckTime + 0.0 ) / 1000 / 1000;
    str.append( "--\ttime ( primary key check ): " ).append( std::to_string( s ) ).append( " s, " ).append( std::to_string( s / 60 ) ).append( " m\n" );
    s = ( fkCheckTime + 0.0 ) / 1000 / 1000;
    str.append( "--\ttime ( foreign key check ): " ).append( std::to_string( s ) ).append( " s, " ).append( std::to_string( s / 60 ) ).append( " m\n" );
    s = ( backupTime + 0.0 ) / 1000 / 1000;
    str.append( "--\ttime ( backup ): " ).append( std::to_string( s ) ).append( " s, " ).append( std::to_string( s / 60 ) ).append( " m\n" );
    s = ( moveTime + 0.0 ) / 1000 / 1000;
    str.append( "--\ttime ( move ): " ).append( std::to_string( s ) ).append( " s, " ).append( std::to_string( s / 60 ) ).append( " m\n" );
    s = ( otherTime + 0.0 ) / 1000 / 1000;
    str.append( "--\ttime ( other ): " ).append( std::to_string( s ) ).append( " s, " ).append( std::to_string( s / 60 ) ).append( " m\n" );
    return str;
}

void checkKeys( const DatabaseEntrySPtr& dbEntry,
                const TableEntrySPtr& tableEntry,
                vector< std::pair< string, aries_engine::AriesTableKeysSPtr > >& uniqKeyIndice,
                const vector< vector< int > >& uniqKeyColIndice,
                vector< aries_engine::TupleDataSPtr >& uniqKeyColValueBuffs,
                const bool foreign_key_checks,
                vector< std::pair< string, aries_engine::AriesTableKeysSPtr > >& fkIndice,
                const vector< vector< int > >& fkColIndice,
                vector< aries_engine::TupleDataSPtr >& fkColValueBuffs,
                vector< ColumnBuffPtr > allColumnBuffs,
                uint64_t startLineIdx, // index of the last read line, 0 based
                uint64_t endlLineIdx, // index of the last read line, 0 based
                int64_t& pkCheckTime, int64_t& fkCheckTime )
{
#ifdef ARIES_PROFILE
    aries::CPU_Timer t;
#endif
    size_t ukCount = uniqKeyIndice.size();
    size_t fkCount = fkColIndice.size();

    auto& cols = tableEntry->GetColumns();

    // 检查每一行是否有重复key
    for ( uint64_t tmpLineIdx = startLineIdx; tmpLineIdx <= endlLineIdx; ++tmpLineIdx )
    {
        // 检查所有的unique key
#ifdef ARIES_PROFILE
        t.begin();
#endif
        for ( size_t keyIdx = 0; keyIdx < ukCount; ++ keyIdx )
        {
            // 取出组成key的列的值，在索引中进行查找
            auto& keyColIndice = uniqKeyColIndice[ keyIdx ];

            auto tupleData = uniqKeyColValueBuffs[ keyIdx ];
            for ( auto keyColIdx : keyColIndice )
            {
                auto& colEntry = cols[ keyColIdx ];

                auto dataBuffer = tupleData->data[ keyColIdx ];
                memset( dataBuffer->GetData(), 0, dataBuffer->GetTotalBytes() );

                auto itemData = allColumnBuffs[ keyColIdx ]->getResultItem( tmpLineIdx - startLineIdx );
                if ( ColumnType::DECIMAL == colEntry->GetType() )
                {
                    aries_acc::Decimal dec( ( aries_acc::CompactDecimal* ) itemData,
                                            colEntry->numeric_precision,
                                            colEntry->numeric_scale );
                    memcpy( dataBuffer->GetData(),
                            &dec,
                            sizeof( aries_acc::Decimal ) );
                }
                else
                {
                    memcpy( dataBuffer->GetData(),
                            itemData,
                            colEntry->GetItemStoreSize() );
                }
            }
            auto key = aries_engine::AriesMvccTable::MakeIndexKey( keyColIndice, tupleData, 0 );
            auto insertResult =
                uniqKeyIndice[ keyIdx ].second->InsertKey( key,
                                                        static_cast< aries_acc::RowPos >( -( tmpLineIdx + 1 ) ) );
            if ( !insertResult )
            {
                string keyStrValues;
                for ( auto keyColIdx : keyColIndice )
                    keyStrValues.append( allColumnBuffs[ keyColIdx ]->getOriginalItem( tmpLineIdx - startLineIdx ) ).append( "-" );

                keyStrValues.erase( keyStrValues.size() - 1 );
                auto msg = format_err_msg( ER( ER_DUP_ENTRY_WITH_KEY_NAME ),
                                           keyStrValues.data(),
                                           uniqKeyIndice[ keyIdx ].first.data() );
                ARIES_EXCEPTION_SIMPLE( ER_DUP_ENTRY, msg.data() );
            }
        }
#ifdef ARIES_PROFILE
        pkCheckTime += t.end();
#endif
        if ( !foreign_key_checks )
            continue;
        // 检查所有的foreign key
#ifdef ARIES_PROFILE
        t.begin();
#endif
        for ( size_t keyIdx = 0; keyIdx < fkCount; ++ keyIdx )
        {
            // 取出组成foreign key的列的值，在parent table 中进行查找
            auto& keyColIndice = fkColIndice[ keyIdx ];

            auto tupleData = fkColValueBuffs[ keyIdx ];
            for ( auto keyColIdx : keyColIndice )
            {
                auto& colEntry = cols[ keyColIdx ];

                auto dataBuffer = tupleData->data[ keyColIdx ];
                memset( dataBuffer->GetData(), 0, dataBuffer->GetTotalBytes() );

                auto itemData = allColumnBuffs[ keyColIdx ]->getResultItem( tmpLineIdx - startLineIdx );
                if ( ColumnType::DECIMAL == colEntry->GetType() )
                {
                    aries_acc::Decimal dec( ( aries_acc::CompactDecimal* ) itemData,
                                            colEntry->numeric_precision,
                                            colEntry->numeric_scale );
                    memcpy( dataBuffer->GetData(),
                            &dec,
                            sizeof( aries_acc::Decimal ) );
                }
                else
                {
                    memcpy( dataBuffer->GetData(),
                            itemData,
                            colEntry->GetItemStoreSize() );
                }
            }
            auto key = aries_engine::AriesMvccTable::MakeIndexKey( keyColIndice, tupleData, 0 );
            std::pair< void*, bool > insertResult;
            bool keyExists = fkIndice[ keyIdx ].second->FindKey( key ).first;
            if ( !keyExists )
            {
                const auto& fks = tableEntry->GetForeignKeys();
                auto& fk = fks[ keyIdx ];
                string msg( "Cannot add or update a child row: a foreign key constraint fails (" );
                msg.append( "`" ).append( dbEntry->GetName() ).append( "`." ).append( "`" ).append( tableEntry->GetName() ).append( "`, " );
                msg.append( "CONSTRAINT `").append( fk->name ).append( "` FOREIGN KEY (");
                for ( auto& fkColName : fk->keys )
                {
                    msg.append( "`" ).append( fkColName ).append( "`," );
                }
                msg.erase( msg.size() - 1 );
                msg.append( ") REFERENCES `" ).append( fk->referencedTable ).append( "` (" );
                for ( auto& pkColName : fk->referencedKeys )
                {
                    msg.append( "`" ).append( pkColName ).append( "`," );
                }
                msg.erase( msg.size() - 1 );
                msg.append( ")" );
                ARIES_EXCEPTION_SIMPLE( ER_NO_REFERENCED_ROW_2, msg.data() );
            }
        }
#ifdef ARIES_PROFILE
        fkCheckTime += t.end();
#endif
    }
}

void processCharBuffColumns( vector< CharColumnBuffPtr >& charColumnBuffs,
                             std::vector<ColumnBlockFileSPtr>& blockFiles,
                             vector< shared_ptr<WRITE_BUFF_INFO> >& writeBuffs )
{
    for ( auto& charColBuff : charColumnBuffs )
    {
        auto colEntry = charColBuff->getColumnEntry();
        int colIdx = colEntry->GetColumnIndex();

        LOG( INFO ) << "Writing char column " << colIdx
                    << ", " << colEntry->GetName() << ", size: " << charColBuff->getResultDataSize();
        
        if ( !batchWrite( blockFiles[ colIdx ]->m_fd,
                          writeBuffs[ colIdx ],
                          ( uchar* )charColBuff->getResultData(),
                          charColBuff->getResultDataSize() ) )
        {
            set_my_errno( errno );
            char errbuf[MYSYS_STRERROR_SIZE] = {0};
            ARIES_EXCEPTION(EE_WRITE, blockFiles[colIdx]->m_filePath.data(), my_errno(),
                            strerror_r(my_errno(), errbuf, sizeof(errbuf)));
        }
        charColBuff->clear();
        LOG( INFO ) << "Writing char column " << colIdx
                    << ", " << colEntry->GetName() << " DONE";
    }
}

void processDictEncodedColumns( aries_engine::AriesTransactionPtr& tx,
                                vector< DictEncodedColumnBuffSPtr >& dictColumnBuffs,
                                std::vector<ColumnBlockFileSPtr>& blockFiles,
                                vector< shared_ptr<WRITE_BUFF_INFO> >& writeBuffs )
{
    for ( auto& colBuff : dictColumnBuffs )
    {
        auto colEntry = colBuff->getColumnEntry();
        int colIdx = colEntry->GetColumnIndex();

        LOG( INFO ) << "Writing dict encoded char column " << colIdx
                    << ", " << colEntry->GetName() << ", size: " << colBuff->getResultDataSize();
        if ( !batchWrite( blockFiles[ colIdx ]->m_fd,
                          writeBuffs[ colIdx ],
                          ( uchar* )colBuff->getResultData(),
                          colBuff->getResultDataSize() ) )
        {
            set_my_errno( errno );
            char errbuf[MYSYS_STRERROR_SIZE] = {0};
            ARIES_EXCEPTION(EE_WRITE, blockFiles[colIdx]->m_filePath.data(), my_errno(),
                            strerror_r(my_errno(), errbuf, sizeof(errbuf)));
        }

        if ( !AriesDictManager::GetInstance().AddDictTuple( tx,
                                                            colEntry->GetDict(),
                                                            colBuff->GetNewDictIndices() ) )
            ARIES_EXCEPTION( ER_OUT_OF_RESOURCES );

        colBuff->clear();
        LOG( INFO ) << "Writing dict encoded char column " << colIdx
                    << ", " << colEntry->GetName() << " DONE";
    }
}

/*
void processDicts( const string& outputDir,
                   const string& tableName,
                   unordered_map< int64_t, DictEncodedColumnBuffSPtr >& dictMap,
                   vector< shared_ptr<WRITE_BUFF_INFO> >& writeBuffs,
                   vector< string >& dictFileNames )
{
    for ( auto it : dictMap )
    {
        auto colBuff = it.second;
        auto colEntry = colBuff->getColumnEntry();
        int colIdx = colEntry->GetColumnIndex();

        LOG( INFO ) << "Writing dict of column " << colIdx
                    << ", " << colEntry->GetName() << ", size: " << colBuff->getDictDataSize();
        auto dictFileName = AriesDictManager::GetInstance().GetDictFileName( colEntry->GetDictId(), colEntry->GetDictName() );
        string dictFilePath = outputDir + "/" + dictFileName;

        // write dict data
        int dictFd = open( dictFilePath.data(),
                           O_CREAT | O_RDWR | O_EXCL, S_IRUSR | S_IWUSR );
        if ( -1 == dictFd )
        {
            set_my_errno(errno);
            char errbuf[ MYSYS_STRERROR_SIZE ];
            ARIES_EXCEPTION( EE_CANTCREATEFILE, dictFilePath.data(), my_errno(),
                             strerror_r( my_errno(), errbuf, sizeof(errbuf) ) );
        }
        auto fileHelper = std::make_shared< fd_helper >( dictFd );
        AriesInitialTable::WriteBlockFileHeader( dictFd,
                                                 dictFilePath,
                                                 colBuff->getDictItemCount(),
                                                 colEntry->IsAllowNull(),
                                                 colEntry->GetItemStoreSize(),
                                                 false );
        if ( !batchWrite( dictFd,
                          writeBuffs[ colIdx ],
                          ( uchar* )colBuff->getDictData(),
                          colBuff->getDictDataSize() ) )
        {
            set_my_errno( errno );
            char errbuf[MYSYS_STRERROR_SIZE] = {0};
            ARIES_EXCEPTION( EE_WRITE, dictFilePath.data(), my_errno(),
                             strerror_r(my_errno(), errbuf, sizeof(errbuf)) );
        }

        LOG( INFO ) << "Writing dict of column " << colIdx
                    << ", " << colEntry->GetName() << " DONE";
        dictFileNames.emplace_back( dictFileName );
    }
}
*/
void WriteDataBlockFileHeaders( const vector< ColumnEntryPtr >& cols,
                                const std::vector<ColumnBlockFileSPtr>& blockFiles,
                                const uint32_t rowCnt )
{
    size_t colCnt = cols.size();
    for ( size_t colIdx = 0; colIdx < colCnt; ++colIdx )
    {
        AriesInitialTable::WriteColumnBlockFileHeader( cols[ colIdx ], blockFiles[ colIdx ]->m_fd,
                                                       blockFiles[ colIdx ]->m_filePath,
                                                       rowCnt,
                                                       true );
    }
}

void BatchWriteColumnBuffPartitioned(
    ImportCsvContext &context,
    vector< FixedSizeColumnBuffPtr > &fixedSizeColumnBuffs,
    vector< CharColumnBuffPtr >& charColumnBuffs,
    vector< DictEncodedColumnBuffSPtr >& dictColumnBuffs,
    const vector< uint64_t > &partitionDistRowIndices )
{
    auto rowCount = partitionDistRowIndices.size();

    bool firstRound = true;
    vector< ColumnEntryPtr > cols;

    for ( size_t rowIndex = 0; rowIndex < rowCount; ++rowIndex )
    {
        auto partitionIndex = partitionDistRowIndices[ rowIndex ];
        auto columnPartitionContext = context.m_partitionContext[ partitionIndex ];
        ColumnBlockFileSPtr colBlockFile;
        ColumnEntryPtr colEntry;
        size_t blockRowCount = 0;

        for ( auto &buff : fixedSizeColumnBuffs )
        {
            colEntry = buff->getColumnEntry();
            auto colIdx = colEntry->GetColumnIndex();

            colBlockFile = columnPartitionContext->m_partitionColumnBlockFiles[ colIdx ];

            if ( !batchWrite( colBlockFile->m_fd,
                              columnPartitionContext->m_partitionColumnWriteBuffs[ colIdx ],
                              (uchar*)buff->getResultItem( rowIndex ),
                              buff->getResultItemSize(),
                              false ) )
            {
                char errbuf[MYSYS_STRERROR_SIZE] = {0};
                ARIES_EXCEPTION(EE_WRITE, colBlockFile->m_filePath.data(), my_errno(),
                                strerror_r(my_errno(), errbuf, sizeof(errbuf)));
            }
            blockRowCount = colBlockFile->IncRowCount();

            if ( firstRound )
                cols.emplace_back( colEntry );
        }

        for ( auto &buff : charColumnBuffs )
        {
            colEntry = buff->getColumnEntry();
            auto colIdx = colEntry->GetColumnIndex();

            colBlockFile = columnPartitionContext->m_partitionColumnBlockFiles[ colIdx ];

            if ( !batchWrite( colBlockFile->m_fd,
                              columnPartitionContext->m_partitionColumnWriteBuffs[ colIdx ],
                              (uchar*)buff->getResultItem( rowIndex ),
                              buff->getResultItemSize(),
                              false ) )
            {
                char errbuf[MYSYS_STRERROR_SIZE] = {0};
                ARIES_EXCEPTION(EE_WRITE, colBlockFile->m_filePath.data(), my_errno(),
                                strerror_r(my_errno(), errbuf, sizeof(errbuf)));
            }
            blockRowCount = colBlockFile->IncRowCount();

            if ( firstRound )
                cols.emplace_back( colEntry );
        }

        for ( auto &buff : dictColumnBuffs )
        {
            colEntry = buff->getColumnEntry();
            auto colIdx = colEntry->GetColumnIndex();

            colBlockFile = columnPartitionContext->m_partitionColumnBlockFiles[ colIdx ];

            if ( !batchWrite( colBlockFile->m_fd,
                              columnPartitionContext->m_partitionColumnWriteBuffs[ colIdx ],
                              (uchar*)buff->getResultItem( rowIndex ),
                              colEntry->GetDictIndexItemSize(),
                              false ) )
            {
                char errbuf[MYSYS_STRERROR_SIZE] = {0};
                ARIES_EXCEPTION(EE_WRITE, colBlockFile->m_filePath.data(), my_errno(),
                                strerror_r(my_errno(), errbuf, sizeof(errbuf)));
            }
            blockRowCount = colBlockFile->IncRowCount();

            if ( firstRound )
                cols.emplace_back( colEntry );
        }
        ++columnPartitionContext->m_rowCount;

        // block is full
        if ( ARIES_BLOCK_FILE_ROW_COUNT == blockRowCount )
        {
            vector< ColumnBlockFileSPtr > blockFiles;

            for ( size_t i = 0; i < cols.size(); ++i )
            {
                auto colEntry = cols[ i ];
                auto colIdx = colEntry->GetColumnIndex();
                auto colBlockFile = columnPartitionContext->m_partitionColumnBlockFiles[ colIdx ];

                if ( !flushWriteBuff(
                        colBlockFile->m_fd,
                        columnPartitionContext->m_partitionColumnWriteBuffs[ colIdx ] ) )
                {
                    char errbuf[MYSYS_STRERROR_SIZE] = {0};
                    ARIES_EXCEPTION(EE_WRITE, colBlockFile->m_filePath.data(), my_errno(),
                                    strerror_r(my_errno(), errbuf, sizeof(errbuf)));
                }

                blockFiles.emplace_back( colBlockFile );
            }

            WriteDataBlockFileHeaders( cols,
                                       blockFiles,
                                       ARIES_BLOCK_FILE_ROW_COUNT );

            if ( rowIndex != rowCount - 1 )
            {
                auto nextBlockIndex = context.GetNextBlockIndex();
                for ( auto blockFile : blockFiles )
                    blockFile->Reset( nextBlockIndex );
                context.m_partitionBlockIndices[ partitionIndex ].emplace( nextBlockIndex );
            }
        }

        firstRound = false;
    }

    for ( auto &buff : fixedSizeColumnBuffs )
    {
        buff->clear();
    }

    for ( auto &buff : charColumnBuffs )
    {
        buff->clear();
    }

    for ( auto &buff : dictColumnBuffs )
    {
        auto colEntry = buff->getColumnEntry();
        if ( !AriesDictManager::GetInstance().AddDictTuple( context.m_tx,
                                                            colEntry->GetDict(),
                                                            buff->GetNewDictIndices() ) )
            ARIES_EXCEPTION( ER_OUT_OF_RESOURCES );
        buff->clear();
    }
}

bool PartitionColumnBuff( ImportCsvContext &context,
                          FixedSizeColumnBuffPtr dataBuff,
                          vector< uint64_t > &partitionDistRowIndices )
{
    auto rowCount = dataBuff->getItemCount();
    auto partitionDefs = context.m_tableEntry->GetPartitions();

    auto resultBuffer = dataBuff->getResultBuffer();
    auto valueType = resultBuffer->GetDataType().DataType.ValueType;
    bool nullable = resultBuffer->GetDataType().isNullable();
    for ( size_t rowIndex = 0; rowIndex < rowCount; ++rowIndex )
    {
        uint32_t dstPartitionIndex = UINT32_MAX;
        aries_acc::AriesDate date;
        aries_acc::AriesDatetime dateTime;
        string valueStr;

        bool dstPartitionFound = false;
        switch ( valueType )
        {
            case AriesValueType::DATE:
            {
                if ( nullable )
                {
                    if ( resultBuffer->isDateDataNull( rowIndex ) )
                    {
                        dstPartitionFound = true;
                        dstPartitionIndex = 0;
                    }
                    else
                    {
                        date = resultBuffer->GetNullableDate( rowIndex )->value;
                    }
                }
                else
                {
                    date = *( resultBuffer->GetDate( rowIndex ) );
                }

                break;
            }
            case AriesValueType::DATETIME:
            {
                if ( nullable )
                {
                    if ( resultBuffer->isDatetimeDataNull( rowIndex ) )
                    {
                        dstPartitionFound = true;
                        dstPartitionIndex = 0;
                    }
                    else
                    {
                        dateTime = resultBuffer->GetNullableDatetime( rowIndex )->value;
                    }
                }
                else
                {
                    dateTime = *( resultBuffer->GetDatetime( rowIndex ) );
                }
                break;
            }
            default:
            {
                ThrowNotSupportedException( "partition on non date and datetime column" );
                break;
            }
        }
        if ( dstPartitionFound )
        {
            partitionDistRowIndices.push_back( dstPartitionIndex );
            continue;
        }

        for ( auto &partDef : partitionDefs )
        {
            switch ( valueType )
            {
                case AriesValueType::DATE:
                {
                    if ( date.toTimestamp() < partDef->m_value || partDef->m_isMaxValue )
                    {
                        dstPartitionFound = true;
                        dstPartitionIndex = partDef->m_partOrdPos - 1;
                    }
                    break;
                }
                case AriesValueType::DATETIME:
                {
                    if ( dateTime.toTimestamp() < partDef->m_value || partDef->m_isMaxValue )
                    {
                        dstPartitionFound = true;
                        dstPartitionIndex = partDef->m_partOrdPos - 1;
                    }
                    break;
                }
                default:
                {
                    break;
                }
            }
            if ( dstPartitionFound )
            {
                break;
            }
        }

        if ( !dstPartitionFound )
        {
            valueStr = resultBuffer->GetDateAsString( rowIndex );
            ARIES_EXCEPTION( ER_NO_PARTITION_FOR_GIVEN_VALUE, valueStr.data() );
        }

        partitionDistRowIndices.push_back( dstPartitionIndex );
    }
    return true;
}

/**
 * @brief buffLineCnt should equal to ARIES_BLOCK_FILE_ROW_COUNT, or less for the last block
 */
void processBuffColumns( aries_engine::AriesTransactionPtr& tx,
                         ImportCsvContext &context,
                         const DatabaseEntrySPtr& dbEntry,
                         const TableEntrySPtr& tableEntry,
                         vector< ColumnBuffPtr > allColumnBuffs,
                         vector< FixedSizeColumnBuffPtr >& fixedSizeColumnBuffs,
                         vector< CharColumnBuffPtr >& charColumnBuffs,
                         vector< DictEncodedColumnBuffSPtr >& dictColumnBuffs,
                         uint32_t buffLineCnt,
                         std::vector<ColumnBlockFileSPtr>& blockFiles,
                         vector< shared_ptr<WRITE_BUFF_INFO> >& writeBuffs,
                         uint64_t endLineIdx, // index of the last read line, 0 based
                         size_t threadCnt,
                         const vector<size_t>& threadsJobCnt,
                         const vector<size_t>& threadsJobStartIdx,
                         int64_t& convertDataTime,
                         int64_t& writeTime,
                         int64_t& pkCheckTime, int64_t& fkCheckTime )
{
    int colIdx;
    uint64_t startLineIdx = endLineIdx - buffLineCnt + 1;
    LOG( INFO ) << "Converting lines [ " << startLineIdx << ", " << endLineIdx << " ]";

    bool isPartitioned = tableEntry->IsPartitioned();
    auto partitionColumnIndex = tableEntry->GetPartitionColumnIndex();

    vector< future< ConvertResult2Ptr > > workThreads;
#ifdef ARIES_PROFILE
    aries::CPU_Timer t;
    t.begin();
#endif
    for( size_t threadIdx = 0; threadIdx < threadCnt; ++threadIdx )
    {
        workThreads.push_back(std::async(std::launch::async, [=] {
            return ConvertColumns( fixedSizeColumnBuffs,
                                   threadsJobStartIdx[ threadIdx ],
                                   threadsJobCnt[ threadIdx ],
                                   startLineIdx );
        }));
    }

    for( auto& thrd : workThreads )
        thrd.wait();
#ifdef ARIES_PROFILE
    convertDataTime += t.end();
#endif

    vector< ConvertResult2Ptr > convertResults;
    for( auto& thrd : workThreads )
    {
        auto convertResult = thrd.get();
        if ( 0 != convertResult->errorCode )
        {
            ARIES_EXCEPTION_SIMPLE( convertResult->errorCode,
                                    convertResult->errorMsg );
            break;
        }
        convertResults.emplace_back( convertResult );
    }
    workThreads.clear();

/*
    if ( key_checks && ( uniqKeyIndice.size() > 0 || fkIndice.size() > 0 ) )
        checkKeys( dbEntry, tableEntry,
                   uniqKeyIndice, uniqKeyColIndice, uniqKeyColValueBuffs,
                   foreign_key_checks, fkIndice, fkColIndice, fkColValueBuffs,
                   allColumnBuffs,
                   startLineIdx,
                   endLineIdx,
                   pkCheckTime, fkCheckTime );
*/
    if ( isPartitioned )
    {
        bool partitionColumnFound = false;
        vector< uint64_t > partitionDistRowIndices;

        for( auto& convertResult : convertResults )
        {
            for ( size_t resultIdx = 0;
                  resultIdx < convertResult->results.size();
                  ++resultIdx )
            {
                auto colEntry = convertResult->results[ resultIdx ]->getColumnEntry();
                colIdx = colEntry->GetColumnIndex();
                if ( colIdx == partitionColumnIndex )
                {
                    partitionColumnFound = true;
                    PartitionColumnBuff( context,
                                         convertResult->results[ resultIdx ],
                                         partitionDistRowIndices );
                    break;
                }
            }
            if ( partitionColumnFound )
                break;
        }

        assert( partitionColumnFound );

        vector< FixedSizeColumnBuffPtr > convertedFixedSizeColumnBuffs;

        for( auto& convertResult : convertResults )
        {
            convertedFixedSizeColumnBuffs.insert( convertedFixedSizeColumnBuffs.end(),
                                                  convertResult->results.begin(),
                                                  convertResult->results.end() );
        }
        BatchWriteColumnBuffPartitioned( context,
                                         convertedFixedSizeColumnBuffs,
                                         charColumnBuffs,
                                         dictColumnBuffs,
                                         partitionDistRowIndices );
        for( auto& convertResult : convertResults )
        {
            for ( size_t resultIdx = 0;
                  resultIdx < convertResult->results.size();
                  ++resultIdx )
            {
                convertResult->results[ resultIdx ]->clear();
            }
            convertResult->results.clear();
        }
    }
    else
    {
        for( auto& convertResult : convertResults )
        {
            for ( size_t resultIdx = 0;
                  resultIdx < convertResult->results.size();
                  ++resultIdx )
            {
                const char* data = convertResult->results[ resultIdx ]->getResultData();
                size_t size = convertResult->results[ resultIdx ]->getResultDataSize();
                auto colEntry = convertResult->results[ resultIdx ]->getColumnEntry();
                colIdx = colEntry->GetColumnIndex();
                LOG( INFO ) << "Writing lines [ " << startLineIdx << ", " << endLineIdx << " ], column " << colIdx
                            << ", " << colEntry->GetName() << ", size " << size;
    #ifdef ARIES_PROFILE
                t.begin();
    #endif

                if ( !batchWrite( blockFiles[ colIdx ]->m_fd,
                                  writeBuffs[ colIdx ],
                                  (uchar*)data,
                                  size ) )
                {
                    char errbuf[MYSYS_STRERROR_SIZE] = {0};
                    ARIES_EXCEPTION(EE_WRITE, blockFiles[colIdx]->m_filePath.data(), my_errno(),
                                    strerror_r(my_errno(), errbuf, sizeof(errbuf)));
                }
                convertResult->results[ resultIdx ]->clear();

    #ifdef ARIES_PROFILE
                writeTime += t.end();
    #endif
            }
        }
        processCharBuffColumns( charColumnBuffs,
                                blockFiles,
                                writeBuffs );

        processDictEncodedColumns( tx,
                                   dictColumnBuffs,
                                   blockFiles,
                                   writeBuffs );
        WriteDataBlockFileHeaders( context.m_cols,
                                   blockFiles,
                                   buffLineCnt );

        for ( size_t colIdx = 0; colIdx < context.m_cols.size(); ++colIdx )
        {
            auto& filePtr = blockFiles[ colIdx ];
            // next block
            if ( -1 == filePtr->Reset( filePtr->m_blockIdx + 1 ) )
            {
                set_my_errno( errno );

                char errbuf[MYSYS_STRERROR_SIZE] = {0};
                ARIES_EXCEPTION( EE_CANTCREATEFILE, filePtr->m_filePath.data(), my_errno(),
                                 strerror_r( my_errno(), errbuf, sizeof(errbuf) ) );
            }
        }
    }
}

#define MOVE_RESULT_FILES \
do { \
    vector< string > outputFileNames = listFiles( tempDir ); \
    for ( string fileName : outputFileNames ) \
    { \
        string oldPath = tempDir + "/" + fileName; \
        string newPath = tableDataDir + "/" + fileName; \
        if ( IsCurrentThdKilled() ) \
            goto interrupted; \
        LOG(INFO) << "Moving data file from " << oldPath << " to " << newPath; \
        if ( 0 != rename( oldPath.data(), newPath.data() ) ) \
        { \
            set_my_errno(errno); \
            restoreDataFiles( backedUpFiles ); \
            boost::filesystem::remove_all( tempDir ); \
            ARIES_EXCEPTION( EE_WRITE, newPath.data(), my_errno(), \
                             strerror_r( my_errno(), errbuf, sizeof(errbuf) ) ); \
        } \
      }  \
} while ( 0 )

static int64_t
importWriteBatchPreScanMultiThreads( THD *thd,
                                     aries_engine::AriesTransactionPtr& tx,
                                     const DatabaseEntrySPtr& dbEntry,
                                     const TableEntrySPtr& tableEntry,
                                     const string& csvFilePath,
                                     READ_INFO &read_info,
                                     int batchWriteSize,
                                     const string& enclosed,
                                     const vector<ColumnEntryPtr>& cols,
                                     uint64_t& currentLineIdx,
                                     uint64_t totalLineCnt )
{
    LOG( INFO ) << "Import mode: pre scan, multiple threads convert, write in batch";
    // bool foreign_key_checks = thd->variables.foreign_key_checks;
    uint64_t startLineIdx = currentLineIdx;
    string tableDataDir = Configuartion::GetInstance().GetDataDirectory(dbEntry->GetName(), tableEntry->GetName());
    tableDataDir.append("/");
    size_t enclosed_length = enclosed.length();
    size_t colCnt = cols.size();

    string dictDir = Configuartion::GetInstance().GetDictDataDirectory();
    string tempDir = MakeTmpDir();
    string tempDictDir = tempDir + "/dict";
    boost::filesystem::create_directories( tempDictDir );

    string tableBackupDir, bkDirSuffix;
    map<string, string> backedUpFiles;

    vector<ColumnBlockFileSPtr> blockFiles;
    vector<string> defaultValues;
    vector< int8_t > strTypeFlags;

    uint32_t buffLineCnt = 0;

    vector< ColumnBuffPtr > allColumnBuffs;
    vector< FixedSizeColumnBuffPtr > fixedSizeColumnBuffs;
    vector< CharColumnBuffPtr > charColumnBuffs;
    vector< DictEncodedColumnBuffSPtr > dictColumnBuffs;
    // unordered_map< int64_t, DictEncodedColumnBuffSPtr > dictMap;

    uint64_t readLineCnt = 0;
    uint32_t startBlockIndex;
    uint32_t extraLineCnt = 0;
    size_t readColCnt = 0;
    ColumnEntryPtr colEntry;

    vector< shared_ptr<WRITE_BUFF_INFO> > writeBuffs;

    int errorCode;
    char errbuf[MYSYS_STRERROR_SIZE] = {0};

    int64_t readAndScanTime = 0;
    int64_t writeTime = 0;
    int64_t backupTime = 0;
    int64_t moveTime = 0;
    int64_t otherTime = 0;
    int64_t threadsTime = 0;
    int64_t convertDataTime = 0;
    int64_t pkCheckTime = 0;
    int64_t fkCheckTime = 0;
    string stats;

#ifdef ARIES_PROFILE
    aries::CPU_Timer t;
#endif
    float s;
#ifdef ARIES_PROFILE
    t.begin();
#endif

    ImportCsvContext context;
    context.m_tableEntry = tableEntry;
    context.m_tx = tx;
    context.m_cols = tableEntry->GetColumns();

    if ( !tableEntry->IsPartitioned() )
    {
        prepareImportInfos( context,
                            dbEntry, tableEntry, totalLineCnt - currentLineIdx,
                            batchWriteSize,
                            tempDir,
                            defaultValues,
                            strTypeFlags,
                            blockFiles, writeBuffs,
                            allColumnBuffs,
                            fixedSizeColumnBuffs,
                            charColumnBuffs,
                            dictColumnBuffs,
                            // dictMap,
                            startBlockIndex,
                            buffLineCnt );
    }
    else
    {
        prepareImportInfosPartitioned( context,
                                       dbEntry, tableEntry, totalLineCnt - currentLineIdx,
                                       batchWriteSize,
                                       tempDir,
                                       defaultValues,
                                       strTypeFlags,
                                       blockFiles, writeBuffs,
                                       allColumnBuffs,
                                       fixedSizeColumnBuffs,
                                       charColumnBuffs,
                                       dictColumnBuffs,
                                       // dictMap,
                                       startBlockIndex,
                                       buffLineCnt );
    }
    extraLineCnt = buffLineCnt; 

#ifdef ARIES_PROFILE
    otherTime += t.end();
#endif
    // my_bool bCheckKeys = get_sys_var_value< my_bool, my_bool >(
    //                         "primary_key_checks",
    //                         OPT_SESSION );
    bool primary_key_checks = thd->variables.primary_key_checks;

    vector<size_t> threadsJobCnt;
    vector<size_t> threadsJobStartIdx;
    size_t fixedSizeColCnt = fixedSizeColumnBuffs.size();
    size_t threadCnt = getConcurrency( fixedSizeColCnt, threadsJobCnt, threadsJobStartIdx );
    LOG(INFO) << "Load data concurrency count: " << threadCnt;

    auto mvccTable = AriesMvccTableManager::GetInstance().getMvccTable( dbEntry->GetName(), tableEntry->GetName() );
    AriesMvccTableSPtr incrMvccTable;
    AriesTableKeysSPtr oldPrimaryKey;
    AriesTableKeysSPtr incrPrimaryKey;
    string newInitTableMetaFile;

    // vector< string > dictFileNames;

    // read until end of file
    try
    {
        for ( ; ; ++currentLineIdx ) {
            if ( IsThdKilled( thd ) )
                goto interrupted;

            // read a line
#ifdef ARIES_PROFILE
            t.begin();
#endif
            readColCnt = readLine2( read_info, readLineCnt,
                                    cols, strTypeFlags, context.m_containNulls,
                                    enclosed_length,
                                    fixedSizeColumnBuffs,
                                    charColumnBuffs,
                                    dictColumnBuffs/*,
                                    readAndScanTime,
                                    scanVectorOpTIme,
                                    otherTime*/ );
#ifdef ARIES_PROFILE
            readAndScanTime += t.end();
#endif
            /* Have not read any field, thus input file is simply ended */
            /*
             * mysql 5.7.26
             * 1. for empty lines:
             * ERROR 1366 (HY000): Incorrect integer value: '' for column 'f1' at row 3
             * 2. for lines with all spaces:
             * ERROR 1366 (HY000): Incorrect integer value: '  ' for column 'f1' at row 2
             */
#ifdef ARIES_PROFILE
            t.begin();
#endif
            if (!readColCnt)
                break;
            // got a line
            if (readColCnt < colCnt)
            {
                // string tmpLine = "line data: ";
                // for (const auto &f : line) {
                //     tmpLine.append(f).append("||");
                // }
                // LOG(INFO) << tmpLine;
                // ARIES_EXCEPTION( ER_WARN_TOO_FEW_RECORDS, lineIdx + 1 );
                string tmpErrMsg = format_mysql_err_msg(ER_WARN_TOO_FEW_RECORDS, currentLineIdx + 1);
                LOG(INFO) << tmpErrMsg;
            }

            ++readLineCnt;
            ++buffLineCnt;
#ifdef ARIES_PROFILE
            otherTime += t.end();
#endif
            // process only fixed size columns
            // process block file row count at a time
            if ( buffLineCnt == allColumnBuffs[ 0 ]->getCapacity() )
            {
                processBuffColumns( tx, context, dbEntry, tableEntry,
                                    allColumnBuffs,
                                    fixedSizeColumnBuffs,
                                    charColumnBuffs,
                                    dictColumnBuffs,
                                    buffLineCnt,
                                    blockFiles,
                                    writeBuffs,
                                    currentLineIdx,
                                    threadCnt,
                                    threadsJobCnt,
                                    threadsJobStartIdx,
                                    convertDataTime,
                                    writeTime,
                                    pkCheckTime, fkCheckTime );

                buffLineCnt = 0;
            }
            /*
              We don't need to reset auto-increment field since we are restoring
              its default value at the beginning of each loop iteration.
            */
            if (read_info.next_line())            // Skip to next line
                break;
            if (read_info.line_cuted)
            {
                // mysql 5.7.26:
                // in strict mode, return error
                // thd->cuted_fields++;			/* To long row */
                // push_warning_printf(thd, Sql_condition::SL_WARNING,
                //                     ER_WARN_TOO_MANY_RECORDS, ER(ER_WARN_TOO_MANY_RECORDS),
                //                     thd->get_stmt_da()->current_row_for_condition());
                // string tmpLine = "line data: ";
                // for (const auto &f : line) {
                //     tmpLine.append(f).append("||");
                // }
                // LOG(INFO) << tmpLine;
                // cleanup generated column files
                ARIES_EXCEPTION( ER_WARN_TOO_MANY_RECORDS, currentLineIdx + 1 );
            }
        }

        LOG( INFO ) << "Scan finished";
        if ( buffLineCnt > 0 )
        {
            context.m_lastLines = true;
            processBuffColumns( tx, context, dbEntry, tableEntry,
                                allColumnBuffs,
                                fixedSizeColumnBuffs,
                                charColumnBuffs,
                                dictColumnBuffs,
                                buffLineCnt,
                                blockFiles,
                                writeBuffs,
                                startLineIdx + readLineCnt - 1,
                                threadCnt,
                                threadsJobCnt,
                                threadsJobStartIdx,
                                convertDataTime,
                                writeTime,
                                pkCheckTime, fkCheckTime );

        }
        buffLineCnt = 0;

        // handle last block
        if ( tableEntry->IsPartitioned() )
        {
            for ( size_t i = 0; i < tableEntry->GetPartitionCount(); ++i )
            {
                auto &ctx = context.m_partitionContext[ i ];
                for ( size_t j = 0; j < cols.size(); ++j )
                {
                    auto &filePtr = ctx->m_partitionColumnBlockFiles[ j ];
                    if ( !flushWriteBuff(
                            filePtr->m_fd,
                            ctx->m_partitionColumnWriteBuffs[ j ] ) )
                    {
                        char errbuf[MYSYS_STRERROR_SIZE] = {0};
                        ARIES_EXCEPTION(EE_WRITE, filePtr->m_filePath.data(), my_errno(),
                                        strerror_r(my_errno(), errbuf, sizeof(errbuf)));
                    }

                    // write partition meta file
                    AriesInitialTable::WriteColumnBlockFileHeader(
                        cols[ j ],
                        filePtr->m_fd,
                        filePtr->m_filePath,
                        filePtr->m_rowCount,
                        true );
                    filePtr->Close();
                }
            }
        }
        else
        {
            for ( size_t colIdx = 0; colIdx < colCnt; ++colIdx )
            {
                if ( blockFiles[ colIdx ] )
                {
                    blockFiles[ colIdx ]->Close();
                }
            }
            blockFiles.clear();
        }

        // write dicts
        // processDicts( tempDictDir, tableEntry->GetName(),
        //               dictMap,
        //               writeBuffs,
        //               dictFileNames );

        {
            auto initTableMetaFile = tempDir + "/" + ARIES_INIT_TABLE_META_FILE_NAME;
            if ( tableEntry->IsPartitioned() )
                AriesInitialTable::WriteMetaFile( initTableMetaFile, readLineCnt + extraLineCnt, context.m_newBlockCount );
            else
                AriesInitialTable::WriteMetaFile( initTableMetaFile, readLineCnt + extraLineCnt );
        }
    }
    catch ( AriesException& e)
    {
        boost::filesystem::remove_all( tempDir );
        throw;
    }
    catch ( std::bad_alloc& e)
    {
        boost::filesystem::remove_all( tempDir );
        ARIES_EXCEPTION( ER_OUT_OF_RESOURCES );
    }

    if ( 0 == readLineCnt )
    {
        LOG(INFO) << "Empty data file";
        boost::filesystem::remove_all( tempDir );
        return 0;
    }

    // check keys
    if ( primary_key_checks )
    {
        LOG( INFO ) << "Checking primary keys";
        mvccTable->CreatePrimaryKeyIndexIfNotExists();
        incrMvccTable = make_shared< AriesMvccTable >( dbEntry->GetName(), tableEntry->GetName(), tempDir );
        incrMvccTable->CreateIncrementPrimaryKeyIndexIfNotExists( startBlockIndex, extraLineCnt );
        oldPrimaryKey = mvccTable->GetPrimaryKey();
        incrPrimaryKey = incrMvccTable->GetPrimaryKey();
        if ( oldPrimaryKey && incrPrimaryKey )
        {
            if ( !oldPrimaryKey->Merge( incrPrimaryKey ) )
            {
                boost::filesystem::remove_all( tempDir );
                ARIES_EXCEPTION_SIMPLE( ER_DUP_ENTRY, "Duplicate entry for key" );
            }
        }
    }

#ifdef ARIES_PROFILE
    t.begin();
#endif

    errorCode = backupTable( dbEntry, tableEntry, tableBackupDir, bkDirSuffix, backedUpFiles );

#ifdef ARIES_PROFILE
    backupTime += t.end();
#endif
    if ( -2 == errorCode )
        goto interrupted;
    if ( -1 == errorCode )
    {
        restoreDataFiles( backedUpFiles );
        boost::filesystem::remove_all( tempDir );
        ARIES_EXCEPTION(  EE_WRITE, tableBackupDir.data(), my_errno(),
                          strerror_r( my_errno(), errbuf, sizeof(errbuf) ) );
    }
#ifdef ARIES_PROFILE
    t.begin();
#endif
    boost::filesystem::create_directory( tableDataDir );

    /*
    for ( auto& dictFile : dictFileNames )
    {
        if ( IsCurrentThdKilled() )
            goto interrupted;
        auto srcPath = tempDictDir + "/" + dictFile;
        auto dstPath = dictDir + "/" + dictFile;
        LOG(INFO) << "Moving data file from " << srcPath << " to " << dstPath;
        if ( 0 != rename( srcPath.data(), dstPath.data() ) )
        {
            set_my_errno(errno);
            restoreDataFiles( backedUpFiles );
            boost::filesystem::remove_all( tempDir );
            ARIES_EXCEPTION( EE_WRITE, dstPath.data(), my_errno(),
                             strerror_r( my_errno(), errbuf, sizeof(errbuf) ) );
        }
    }
    */

    MOVE_RESULT_FILES;

    newInitTableMetaFile = AriesInitialTable::GetMetaFilePath( dbEntry->GetName(), tableEntry->GetName() );
    if ( tableEntry->IsPartitioned() )
        AriesInitialTable::WriteMetaFile( newInitTableMetaFile,
                                          mvccTable->GetInitialTable()->GetTotalRowCount() + readLineCnt,
                                          context.m_totalBlockCount );
    else
        AriesInitialTable::WriteMetaFile( newInitTableMetaFile,
                                          mvccTable->GetInitialTable()->GetTotalRowCount() + readLineCnt );

    if ( tableEntry->IsPartitioned() )
    {
        AriesInitialTable initTable( dbEntry->GetName(), tableEntry->GetName() );
        auto &partitionMetaInfos = initTable.GetPartitionMetaInfo();
        auto partitionCount = partitionMetaInfos.size();
        for ( size_t i = 0; i < partitionCount; ++i )
        {
            auto &ctx = context.m_partitionContext[ i ];
            auto &partitionMetaInfo = partitionMetaInfos[ i ];
            auto &newBlockIndices = context.m_partitionBlockIndices[ i ];
            partitionMetaInfo.RowCount += ctx->m_rowCount;
            partitionMetaInfo.BlockCount += newBlockIndices.size();
            partitionMetaInfo.BlocksID.insert( partitionMetaInfo.BlocksID.end(),
                                               newBlockIndices.begin(),
                                               newBlockIndices.end() );
            initTable.WritePartitionMetaFile( i, partitionMetaInfo );
        }
    }

#ifdef ARIES_PROFILE
    moveTime = t.end();
#endif

    LOG(INFO) << "Moving data files done";

#ifdef ARIES_PROFILE
    t.begin();
#endif

    boost::filesystem::remove_all( tempDir );

#ifdef ARIES_PROFILE
    otherTime += t.end();
#endif

    if ( !tableBackupDir.empty() )
    {
        pthread_t tid;
        auto removeBkInfo = new REMOVE_OLD_BACKUP_INFO( dbEntry->GetName(), tableEntry->GetName(), bkDirSuffix );
        pthread_create( &tid, &connection_attr, removeOlderBackups, (void*)removeBkInfo );
    }

#ifdef ARIES_PROFILE
    stats = formatStatistics( readLineCnt,
                              readAndScanTime,
                              convertDataTime,
                              writeTime,
                              pkCheckTime,
                              fkCheckTime,
                              backupTime,
                              moveTime,
                              otherTime);
    // s = ( scanVectorOpTIme + 0.0 ) / 1000 / 1000;
    // stats.append( "--\ttime ( scan vector op ): " ).append( std::to_string( s ) ).append( " s, " ).append( std::to_string( s / 60 ) ).append( " m\n" );
    s = ( threadsTime + 0.0 ) / 1000 / 1000;
    stats.append( "--\ttime ( create threads ): " ).append( std::to_string( s ) ).append( " s, " ).append( std::to_string( s / 60 ) ).append( " m" );
    LOG( INFO ) << "Import " << csvFilePath << " statistics:\n" << stats;
#endif
    return readLineCnt;

interrupted:
    // restore original data files
    restoreDataFiles( backedUpFiles );

    // cleanup generated column files
    boost::filesystem::remove_all( tempDir );
    SendKillMessage();

    return -1;
}

// static int64_t
// importWriteBatchPreScannedMultiThreads( THD *thd,
//                                         const DatabaseEntrySPtr& dbEntry,
//                                         const TableEntrySPtr& tableEntry,
//                                         const string& csvFilePath,
//                                         READ_INFO &read_info,
//                                         int batchWriteSize,
//                                         const string& enclosed,
//                                         const vector<ColumnEntryPtr>& cols,
//                                         uint64_t& lineIdx,
//                                         uint64_t lineCount,
//                                         const std::vector<size_t>& itemStoreSizes )
// {
//     LOG( INFO ) << "Import mode: pre scanned, multiple threads convert, write in batch";
//     string tableDataDir = Configuartion::GetInstance().GetDataDirectory(dbEntry->GetName(), tableEntry->GetName());
//     tableDataDir.append("/");
//     size_t enclosed_length = enclosed.length();
//     int colCnt = cols.size();
// 
//     string tempDir = MakeTmpDir();
//     string tableBackupDir, bkDirSuffix;
//     std::map<string, string> backedUpFiles;
// 
//     std::vector<ColumnBlockFileSPtr> blockFiles;
//     std::vector<int8_t> containNulls;
//     std::vector<string> defaultValues;
//     std::vector<string> oldDataFiles;
// 
//     int buffLineCnt = 0;
//     vector< shared_ptr< vector<string> > > buffColumns;
//     vector<string> line;
//     int fd;
//     int colIdx;
//     uint64_t readLineCnt = 0;
//     int readColCnt = 0;
//     ColumnEntryPtr colEntry;
// 
//     size_t writtenSize = 0;
//     vector< shared_ptr<WRITE_BUFF_INFO> > writeBuffs;
// 
//     int errorCode;
//     char errbuf[MYSYS_STRERROR_SIZE] = {0};
// 
//     int64_t readAndScanTime = 0;
//     int64_t writeTime = 0;
//     int64_t backupTime = 0;
//     int64_t moveTime = 0;
//     int64_t otherTime = 0;
//     int64_t vectorOpTime = 0;
//     int64_t convertDataTime = 0;
//     int64_t micros = 0;
//     string stats;
// #ifdef ARIES_PROFILE
//     aries::CPU_Timer t;
// #endif
//     float s;
// 
//     unsigned int threadIdx = 0;
//     vector< future< ConvertResultPtr > > workThreads;
//     vector<size_t> threadsJobCnt;
//     vector<size_t> threadsJobStartIdx;
//     size_t threadCnt = getConcurrency( colCnt, threadsJobCnt, threadsJobStartIdx );
// 
//     LOG(INFO) << "Load data concurrency count: " << threadCnt;
// #ifdef ARIES_PROFILE
//     t.begin();
// #endif
//     prepareImportInfosPreScanned( dbEntry, tableEntry, cols,
//                                   itemStoreSizes, lineCount, batchWriteSize,
//                                   tempDir, containNulls, defaultValues,
//                                   blockFiles, writeBuffs );
// 
//     for (colIdx = 0; colIdx < colCnt; ++colIdx)
//     {
//         shared_ptr< vector<string> > v = make_shared< vector< string > >();
//         v->reserve( ARIES_BLOCK_FILE_ROW_COUNT );
//         buffColumns.emplace_back( v );
//     }
// #ifdef ARIES_PROFILE
//     otherTime += t.end();
// #endif
//     // read until end of file
//     for (;; ++lineIdx) {
//         if (IsCurrentThdKilled())
//             goto interrupted;
// 
//         // read a line
// #ifdef ARIES_PROFILE
//         t.begin();
// #endif
//         readColCnt = readLine(read_info, colCnt, enclosed_length, line);
// #ifdef ARIES_PROFILE
//         readAndScanTime += t.end();
// #endif
// 
//         /* Have not read any field, thus input file is simply ended */
//         /*
//          * mysql 5.7.26
//          * 1. for empty lines:
//          * ERROR 1366 (HY000): Incorrect integer value: '' for column 'f1' at row 3
//          * 2. for lines with all spaces:
//          * ERROR 1366 (HY000): Incorrect integer value: '  ' for column 'f1' at row 2
//          */
// #ifdef ARIES_PROFILE
//         t.begin();
// #endif
//         if (!readColCnt)
//             break;
//         // got a line
//         // check column count
//         if ( readColCnt < colCnt ) {
//             string tmpLine = "line data: ";
//             for (const auto &f : line) {
//                 tmpLine.append(f).append("||");
//             }
//             LOG(INFO) << tmpLine;
//             // cleanup generated column files
//             boost::filesystem::remove_all( tempDir );
//             ARIES_EXCEPTION( ER_WARN_TOO_FEW_RECORDS, lineIdx + 1 );
//         } else if ( readColCnt > colCnt ) {
//             string tmpLine = "line data: ";
//             for (const auto &f : line) {
//                 tmpLine.append(f).append("||");
//             }
//             LOG(INFO) << tmpLine;
//             // cleanup generated column files
//             boost::filesystem::remove_all( tempDir );
//             ARIES_EXCEPTION( ER_WARN_TOO_MANY_RECORDS, lineIdx + 1 );
//         }
// 
//         ++readLineCnt;
// #ifdef ARIES_PROFILE
//         otherTime += t.end();
// #endif
//         ++buffLineCnt;
// #ifdef ARIES_PROFILE
//         t.begin();
// #endif
//         for ( colIdx = 0; colIdx < colCnt; ++ colIdx )
//         {
//             buffColumns[ colIdx ]->emplace_back( line[ colIdx ] );
//         }
// #ifdef ARIES_PROFILE
//         vectorOpTime += t.end();
// #endif
//         if ( buffLineCnt == ARIES_BLOCK_FILE_ROW_COUNT )
//         {
// #ifdef ARIES_PROFILE
//             t.begin();
// #endif
//             for( threadIdx = 0; threadIdx < threadCnt; ++threadIdx )
//             {
//                 workThreads.push_back(std::async(std::launch::async, [=] {
//                     return ConvertColumns( buffColumns, cols, itemStoreSizes,
//                                            threadsJobStartIdx[ threadIdx ],
//                                            threadsJobCnt[ threadIdx ],
//                                            lineIdx - buffLineCnt + 1 );
//                 }));
//             }
// #ifdef ARIES_PROFILE
//             micros = t.end();
// #endif
//             otherTime += micros;
// #ifdef ARIES_PROFILE
//             t.begin();
// #endif
//             for( auto& thrd : workThreads )
//                 thrd.wait();
// #ifdef ARIES_PROFILE
//             convertDataTime += t.end();
// #endif
//             for( auto& thrd : workThreads )
//             {
//                 auto convertResult = thrd.get();
//                 if ( convertResult->errorResult )
//                 {
//                     boost::filesystem::remove_all(tempDir);
//                     ARIES_EXCEPTION_SIMPLE( convertResult->errorResult->GetErrorCode(),
//                                             convertResult->errorResult->GetErrorMsg().data() );
//                     break;
//                 }
//                 for ( int resultIdx = 0;
//                       resultIdx < convertResult->results.size();
//                       ++resultIdx )
//                 {
//                     auto& converter = convertResult->results[ resultIdx ];
//                     const char* data = converter->GetResult();
//                     size_t size = converter->GetResultSize();
//                     colIdx = converter->GetColumnEntry()->GetColumnIndex();
// #ifdef ARIES_PROFILE
//                     t.begin();
// #endif
//                     if ( !batchWrite( blockFiles[ colIdx ]->m_fd,
//                                       writeBuffs[ colIdx ],
//                                       (uchar*)data, size ) )
//                     {
//                         boost::filesystem::remove_all(tempDir);
//                         ARIES_EXCEPTION(EE_WRITE, blockFiles[ colIdx ]->m_filePath.data(), my_errno(),
//                                         strerror_r(my_errno(), errbuf, sizeof(errbuf)));
//                     }
// #ifdef ARIES_PROFILE
//                     writeTime += t.end();
// #endif
//                 }
//             }
//             workThreads.clear();
//             for ( colIdx = 0; colIdx < colCnt; ++ colIdx )
//             {
//                 buffColumns[ colIdx ]->clear();
//             }
//             buffLineCnt = 0;
//         }
//         /*
//           We don't need to reset auto-increment field since we are restoring
//           its default value at the beginning of each loop iteration.
//         */
//         if (read_info.next_line())            // Skip to next line
//             break;
//         if (read_info.line_cuted)
//         {
//             // mysql 5.7.26:
//             // in strict mode, return error
//             // thd->cuted_fields++;			/* To long row */
//             // push_warning_printf(thd, Sql_condition::SL_WARNING,
//             //                     ER_WARN_TOO_MANY_RECORDS, ER(ER_WARN_TOO_MANY_RECORDS),
//             //                     thd->get_stmt_da()->current_row_for_condition());
//             string tmpLine = "line data: ";
//             for (const auto &f : line) {
//                 tmpLine.append(f).append("||");
//             }
//             LOG(INFO) << tmpLine;
//             // cleanup generated column files
//             boost::filesystem::remove_all( tempDir );
//             ARIES_EXCEPTION( ER_WARN_TOO_MANY_RECORDS, lineIdx + 1 );
//         }
//     }
//     if ( 0 == readLineCnt )
//     {
//         LOG(INFO) << "Empty data file";
//         boost::filesystem::remove_all( tempDir );
//         return 0;
//     }
//     if ( buffLineCnt > 0 )
//     {
//         for( threadIdx = 0; threadIdx < threadCnt; ++threadIdx )
//         {
//             workThreads.push_back(std::async(std::launch::async, [=] {
//                 return ConvertColumns( buffColumns, cols, itemStoreSizes,
//                                        threadsJobStartIdx[ threadIdx ],
//                                        threadsJobCnt[ threadIdx ],
//                                        lineIdx - buffLineCnt + 1 );
//             }));
//         }
//         for( auto& thrd : workThreads )
//             thrd.wait();
//         for( auto& thrd : workThreads )
//         {
//             auto convertResult = thrd.get();
//             if ( convertResult->errorResult )
//             {
//                 boost::filesystem::remove_all(tempDir);
//                 ARIES_EXCEPTION_SIMPLE( convertResult->errorResult->GetErrorCode(),
//                                         convertResult->errorResult->GetErrorMsg().data() );
//                 break;
//             }
//             for ( int resultIdx = 0; resultIdx < convertResult->results.size(); ++resultIdx )
//             {
//                 auto& converter = convertResult->results[ resultIdx ];
//                 const char* data = converter->GetResult();
//                 size_t size = converter->GetResultSize();
//                 colIdx = converter->GetColumnEntry()->GetColumnIndex();
// #ifdef ARIES_PROFILE
//                 t.begin();
// #endif
//                 if ( !batchWrite( blockFiles[ colIdx ]->m_fd,
//                                   writeBuffs[ colIdx ],
//                                   (uchar*)data, size ) )
//                 {
//                     boost::filesystem::remove_all(tempDir);
//                     ARIES_EXCEPTION(EE_WRITE, blockFiles[ colIdx ]->m_filePath.data(), my_errno(),
//                                     strerror_r(my_errno(), errbuf, sizeof(errbuf)));
//                 }
// #ifdef ARIES_PROFILE
//                 writeTime += t.end();
// #endif
//             }
//         }
//         workThreads.clear();
//         for ( colIdx = 0; colIdx < colCnt; ++ colIdx )
//         {
//             buffColumns[ colIdx ]->clear();
//         }
//     }
// #ifdef ARIES_PROFILE
//     t.begin();
// #endif
//     for ( colIdx = 0; colIdx < colCnt; ++colIdx )
//     {
//         if (writeBuffs[colIdx]->empty())
//             continue;
//         writtenSize = my_write(blockFiles[colIdx]->m_fd,
//                                writeBuffs[colIdx]->get(),
//                                writeBuffs[colIdx]->getDataSize(),
//                                MYF( MY_FNABP ));
//         if ( 0 != writtenSize )
//         {
//             // cleanup generated column files
//             boost::filesystem::remove_all(tempDir);
//             ARIES_EXCEPTION(EE_WRITE, blockFiles[ colIdx ]->m_filePath.data(), my_errno(),
//                             strerror_r(my_errno(), errbuf, sizeof(errbuf)));
//         }
//         writeBuffs[colIdx]->clear();
//     }
// #ifdef ARIES_PROFILE
//     writeTime += t.end();
//     t.begin();
// #endif
//     errorCode = backupTable( dbEntry, tableEntry, tableBackupDir, bkDirSuffix, backedUpFiles );
// #ifdef ARIES_PROFILE
//     backupTime += t.end();
// #endif
//     if ( -2 == errorCode )
//         goto interrupted;
//     if ( -1 == errorCode )
//     {
//         restoreDataFiles( backedUpFiles );
//         boost::filesystem::remove_all( tempDir );
//         ARIES_EXCEPTION(  EE_WRITE, tableBackupDir.data(), my_errno(),
//                           strerror_r( my_errno(), errbuf, sizeof(errbuf) ) );
//     }
// #ifdef ARIES_PROFILE
//     t.begin();
// #endif
//     boost::filesystem::create_directory( tableDataDir );
//     MOVE_RESULT_FILES;
// #ifdef ARIES_PROFILE
//     moveTime = t.end();
// #endif
//     LOG(INFO) << "Moving data files done";
// #ifdef ARIES_PROFILE
//     t.begin();
// #endif
//     boost::filesystem::remove_all( tempDir );
// #ifdef ARIES_PROFILE
//     otherTime += t.end();
// #endif
//     if ( !tableBackupDir.empty() )
//     {
//         pthread_t tid;
//         auto removeBkInfo = new REMOVE_OLD_BACKUP_INFO( dbEntry->GetName(), tableEntry->GetName(), bkDirSuffix );
//         pthread_create( &tid, &connection_attr, removeOlderBackups, (void*)removeBkInfo );
//     }
// #ifdef ARIES_PROFILE
//     stats = formatStatistics( readLineCnt,
//                               readAndScanTime,
//                               convertDataTime,
//                               writeTime,
//                               0, 0,
//                               backupTime,
//                               moveTime,
//                               otherTime);
//     s = ( vectorOpTime + 0.0 ) / 1000 / 1000;
//     stats.append( "--\ttime ( vector op ): " ).append( std::to_string( s ) ).append( " s, " ).append( std::to_string( s / 60 ) ).append( " m" );
//     LOG( INFO ) << "Import " << csvFilePath << " statistics:\n" << stats;
// #endif
//     return readLineCnt;
// 
// interrupted:
//     // restore original data files
//     restoreDataFiles( backedUpFiles );
// 
//     // cleanup generated column files
//     boost::filesystem::remove_all( tempDir );
//     SendKillMessage();
// 
//     return -1;
// }

// static int64_t
// importWriteBatchPreScannedSingleThread( THD *thd,
//                                         const DatabaseEntrySPtr& dbEntry,
//                                         const TableEntrySPtr& tableEntry,
//                                         const string& csvFilePath,
//                                         READ_INFO &read_info,
//                                         int batchWriteSize,
//                                         const string& enclosed,
//                                         const vector<ColumnEntryPtr>& cols,
//                                         uint64_t& lineIdx,
//                                         uint64_t lineCount,
//                                         const std::vector<size_t>& itemStoreSizes ) {
//     LOG( INFO ) << "Import mode: pre scanned, single thread convert, write in batch";
//     string tableDataDir = Configuartion::GetInstance().GetDataDirectory( dbEntry->GetName(), tableEntry->GetName() );
//     tableDataDir.append( "/" );
//     size_t enclosed_length = enclosed.length();
//     int colCnt = cols.size();
// 
//     string tempDir = MakeTmpDir();
//     string tableBackupDir, bkDirSuffix;
//     std::map<string, string> backedUpFiles;
// 
//     std::vector<ColumnBlockFileSPtr> blockFiles;
//     std::vector<int8_t> containNulls;
//     std::vector<string> defaultValues;
//     std::vector<string> oldDataFiles;
// 
//     vector<string> line;
//     int fd;
//     int colIdx;
//     uint64_t readLineCnt = 0;
//     int readColCnt = 0;
//     ColumnEntryPtr colEntry;
// 
//     size_t writtenSize = 0;
//     vector< shared_ptr<WRITE_BUFF_INFO> > writeBuffs;
//     const int tmpColBuffSize = ARIES_MAX_CHAR_WIDTH + 1;
//     uchar tmpColBuff[ tmpColBuffSize ] = {0}; // enough for all data types
// 
//     int errorCode;
//     string errorMsg;
//     char errbuf[MYSYS_STRERROR_SIZE] = {0};
// 
//     int64_t readAndScanTime = 0;
//     int64_t writeTime = 0;
//     int64_t backupTime = 0;
//     int64_t moveTime = 0;
//     int64_t otherTime = 0;
//     int64_t convertDataTime = 0;
//     string stats;
// #ifdef ARIES_PROFILE
//     aries::CPU_Timer t, tConvertData;
//     t.begin();
// #endif
// 
//     prepareImportInfosPreScanned( dbEntry, tableEntry, cols,
//                                   itemStoreSizes, lineCount, batchWriteSize,
//                                   tempDir, containNulls, defaultValues,
//                                   blockFiles, writeBuffs );
// #ifdef ARIES_PROFILE
//     otherTime += t.end();
// #endif
//     // read until end of file
//     for (;; ++lineIdx)
//     {
//         if (IsCurrentThdKilled())
//             goto interrupted;
// 
//         // read a line
// #ifdef ARIES_PROFILE
//         t.begin();
// #endif
//         readColCnt = readLine( read_info, colCnt, enclosed_length, line );
// #ifdef ARIES_PROFILE
//         readAndScanTime += t.end();
// #endif
//         /* Have not read any field, thus input file is simply ended */
//         /*
//          * mysql 5.7.26
//          * 1. for empty lines:
//          * ERROR 1366 (HY000): Incorrect integer value: '' for column 'f1' at row 3
//          * 2. for lines with all spaces:
//          * ERROR 1366 (HY000): Incorrect integer value: '  ' for column 'f1' at row 2
//          */
// #ifdef ARIES_PROFILE
//         t.begin();
// #endif
//         if (!readColCnt)
//             break;
//         // got a line
//         // check column count
//         if ( readColCnt < colCnt ) {
//             string tmpLine = "line data: ";
//             for (const auto &f : line) {
//                 tmpLine.append(f).append("||");
//             }
//             LOG(INFO) << tmpLine;
//             // cleanup generated column files
//             boost::filesystem::remove_all( tempDir );
//             ARIES_EXCEPTION( ER_WARN_TOO_FEW_RECORDS, lineIdx + 1 );
//         } else if ( readColCnt > colCnt ) {
//             string tmpLine = "line data: ";
//             for (const auto &f : line) {
//                 tmpLine.append(f).append("||");
//             }
//             LOG(INFO) << tmpLine;
//             // cleanup generated column files
//             boost::filesystem::remove_all( tempDir );
//             ARIES_EXCEPTION( ER_WARN_TOO_MANY_RECORDS, lineIdx + 1 );
//         }
// 
//         ++readLineCnt;
// #ifdef ARIES_PROFILE
//         otherTime += t.end();
// #endif
//         for ( colIdx = 0; colIdx < colCnt; ++colIdx )
//         {
// #ifdef ARIES_PROFILE
//             t.begin();
// #endif
//             colEntry = cols[ colIdx ];
//             int8_t containNull = containNulls[ colIdx ];
//             std::string& defValue = defaultValues[ colIdx ];
//             string& colStr = line[ colIdx ];
// 
//             ARIES_ASSERT(itemStoreSizes[colIdx] <= tmpColBuffSize,
//                          "Column size of " + colEntry->GetName() + " too big: " + std::to_string( itemStoreSizes[ colIdx ] ) );
// #ifdef ARIES_PROFILE
//             tConvertData.begin();
// #endif
//             errorCode = ToColumnValue( colEntry, containNull, defValue, colStr,
//                                        lineIdx, tmpColBuff, tmpColBuffSize, errorMsg );
//             if ( 0 != errorCode )
//             {
//                 boost::filesystem::remove_all(tempDir);
//                 ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg );
//             }
// #ifdef ARIES_PROFILE
//             convertDataTime += tConvertData.end();
// #endif
// 
//             // batch write column
// #ifdef ARIES_PROFILE
//             t.begin();
// #endif
//             if ( !batchWrite( blockFiles[colIdx]->m_fd,
//                               writeBuffs[ colIdx ],
//                               tmpColBuff,
//                               itemStoreSizes[ colIdx ] ) )
//             {
//                 // cleanup generated column files
//                 boost::filesystem::remove_all(tempDir);
//                 ARIES_EXCEPTION(EE_WRITE, blockFiles[colIdx]->m_filePath.data(), my_errno(),
//                                 strerror_r(my_errno(), errbuf, sizeof(errbuf)));
//             }
// #ifdef ARIES_PROFILE
//             writeTime += t.end();
// #endif
//         }
// 
//         /*
//           We don't need to reset auto-increment field since we are restoring
//           its default value at the beginning of each loop iteration.
//         */
// #ifdef ARIES_PROFILE
//         t.begin();
// #endif
//         if (read_info.next_line())            // Skip to next line
//             break;
//         if (read_info.line_cuted)
//         {
//             // mysql 5.7.26:
//             // in strict mode, return error
//             // thd->cuted_fields++;			/* To long row */
//             // push_warning_printf(thd, Sql_condition::SL_WARNING,
//             //                     ER_WARN_TOO_MANY_RECORDS, ER(ER_WARN_TOO_MANY_RECORDS),
//             //                     thd->get_stmt_da()->current_row_for_condition());
//             string tmpLine = "line data: ";
//             for (const auto &f : line) {
//                 tmpLine.append(f).append("||");
//             }
//             LOG(INFO) << tmpLine;
//             // cleanup generated column files
//             boost::filesystem::remove_all( tempDir );
//             ARIES_EXCEPTION( ER_WARN_TOO_MANY_RECORDS, lineIdx + 1 );
//         }
// #ifdef ARIES_PROFILE
//         otherTime += t.end();
// #endif
//     }
// 
//     if ( 0 == readLineCnt )
//     {
//         LOG(INFO) << "Empty data file";
//         boost::filesystem::remove_all( tempDir );
//         return 0;
//     }
// #ifdef ARIES_PROFILE
//     t.begin();
// #endif
//     for ( colIdx = 0; colIdx < colCnt; ++colIdx )
//     {
//         if (writeBuffs[colIdx]->empty())
//             continue;
//         writtenSize = my_write(blockFiles[colIdx]->m_fd,
//                                writeBuffs[colIdx]->get(),
//                                writeBuffs[colIdx]->getDataSize(),
//                                MYF( MY_FNABP ));
//         if ( 0 != writtenSize )
//         {
//             // cleanup generated column files
//             boost::filesystem::remove_all(tempDir);
//             ARIES_EXCEPTION(EE_WRITE, blockFiles[colIdx]->m_filePath.data(), my_errno(),
//                             strerror_r(my_errno(), errbuf, sizeof(errbuf)));
//         }
//         writeBuffs[colIdx]->clear();
//     }
// #ifdef ARIES_PROFILE
//     writeTime += t.end();
//     t.begin();
// #endif
// 
//     errorCode = backupTable( dbEntry, tableEntry, tableBackupDir, bkDirSuffix, backedUpFiles );
// #ifdef ARIES_PROFILE
//     backupTime += t.end();
// #endif
//     if ( -2 == errorCode )
//         goto interrupted;
//     if ( -1 == errorCode )
//     {
//         restoreDataFiles( backedUpFiles );
//         boost::filesystem::remove_all( tempDir );
//         ARIES_EXCEPTION(  EE_WRITE, tableBackupDir.data(), my_errno(),
//                           strerror_r( my_errno(), errbuf, sizeof(errbuf) ) );
//     }
// #ifdef ARIES_PROFILE
//     t.begin();
// #endif
//     boost::filesystem::create_directory( tableDataDir );
//     MOVE_RESULT_FILES;
// #ifdef ARIES_PROFILE
//     moveTime = t.end();
// #endif
//     LOG(INFO) << "Moving data files done";
// #ifdef ARIES_PROFILE
//     t.begin();
// #endif
//     boost::filesystem::remove_all( tempDir );
// #ifdef ARIES_PROFILE
//     otherTime += t.end();
// #endif
//     if ( !tableBackupDir.empty() )
//     {
//         pthread_t tid;
//         auto removeBkInfo = new REMOVE_OLD_BACKUP_INFO( dbEntry->GetName(), tableEntry->GetName(), bkDirSuffix );
//         pthread_create( &tid, &connection_attr, removeOlderBackups, (void*)removeBkInfo );
//     }
//     stats = formatStatistics( readLineCnt,
//                               readAndScanTime,
//                               convertDataTime,
//                               writeTime,
//                               0, 0,
//                               backupTime,
//                               moveTime,
//                               otherTime);
//     LOG( INFO ) << "Import " << csvFilePath << " statistics:\n" << stats;
//     return readLineCnt;
// 
// interrupted:
//     // restore original data files
//     restoreDataFiles( backedUpFiles );
// 
//     // cleanup generated column files
//     boost::filesystem::remove_all( tempDir );
//     SendKillMessage();
// 
//     return -1;
// }

//static int64_t
//importWriteBatchNoPreScanSingleThread( THD *thd,
//                                       const DatabaseEntrySPtr& dbEntry,
//                                       const TableEntrySPtr& tableEntry,
//                                       const string& csvFilePath,
//                                       READ_INFO &read_info,
//                                       int batchWriteSize,
//                                       const string& enclosed,
//                                       const vector<ColumnEntryPtr>& cols,
//                                       uint64_t& lineIdx ) {
//    LOG( INFO ) << "Import mode: no pre scan, single thread convert, write in batch";
//    string tableDataDir = Configuartion::GetInstance().GetDataDirectory( dbEntry->GetName(), tableEntry->GetName() );
//    tableDataDir.append( "/" );
//    size_t enclosed_length = enclosed.length();
//    int colCnt = cols.size();
//
//    string tempDir = MakeTmpDir();
//    string tableBackupDir, bkDirSuffix;
//    std::map<string, string> backedUpFiles;
//
//    std::vector<ColumnBlockFileSPtr> blockFiles;
//    std::vector<size_t> itemStoreSizes;
//    std::vector<int8_t> containNulls;
//    std::vector<string> defaultValues;
//    std::vector<size_t> columnMaxSizes;
//    std::vector<string> oldDataFiles;
//
//    vector<string> line;
//    int fd;
//    int colIdx;
//    uint64_t readLineCnt = 0;
//    int readColCnt = 0;
//    ColumnEntryPtr colEntry;
//
//    uchar header[ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE] = { 0 };
//    BlockFileHeader headerInfo;
//    size_t blockSize = 0;
//    int writeBuffSize = batchWriteSize;
//    size_t writtenSize = 0;
//    int appendedSize = 0;
//    vector< shared_ptr<WRITE_BUFF_INFO> > writeBuffs;
//    const int tmpColBuffSize = ARIES_MAX_CHAR_WIDTH + 1;
//    uchar tmpColBuff[ tmpColBuffSize ] = {0}; // enough for all data types
//
//    int errorCode;
//    string errorMsg;
//    char errbuf[MYSYS_STRERROR_SIZE] = {0};
//
//    int64_t readAndScanTime = 0;
//    int64_t writeTime = 0;
//    int64_t convertDataTime = 0;
//    int64_t strColRereadTime = 0;
//    int64_t strColRewriteTime = 0;
//    int64_t adjustHeaderTime = 0;
//    int64_t backupTime = 0;
//    int64_t moveTime = 0;
//    int64_t otherTime = 0;
// #ifdef ARIES_PROFILE
//    aries::CPU_Timer t;
// #endif
//    float s;
// #ifdef ARIES_PROFILE
//    t.begin();
// #endif
//    columnMaxSizes.assign( colCnt, 1 );
//    prepareImportInfosNotPreScanned( dbEntry, tableEntry, cols,
//                                     batchWriteSize,
//                                     tempDir,
//                                     // itemStoreSizes,
//                                     containNulls, defaultValues,
//                                     blockFiles, writeBuffs );
// #ifdef ARIES_PROFILE
//    otherTime += t.end();
// #endif
//
//    // read until end of file
//    for (;; ++lineIdx)
//    {
//        if (IsCurrentThdKilled())
//            goto interrupted;
//
//        // read a line
// #ifdef ARIES_PROFILE
//        t.begin();
// #endif
//        readColCnt = readLine( read_info, colCnt, enclosed_length, line );
// #ifdef ARIES_PROFILE
//        readAndScanTime += t.end();
// #endif
//        /* Have not read any field, thus input file is simply ended */
//        /*
//         * mysql 5.7.26
//         * 1. for empty lines:
//         * ERROR 1366 (HY000): Incorrect integer value: '' for column 'f1' at row 3
//         * 2. for lines with all spaces:
//         * ERROR 1366 (HY000): Incorrect integer value: '  ' for column 'f1' at row 2
//         */
// #ifdef ARIES_PROFILE
//        t.begin();
// #endif
//        if (!readColCnt)
//            break;
//        // got a line
//        // check column count
//        if ( readColCnt < colCnt ) {
//            string tmpLine = "line data: ";
//            for (const auto &f : line) {
//                tmpLine.append(f).append("||");
//            }
//            LOG(INFO) << tmpLine;
//            // cleanup generated column files
//            boost::filesystem::remove_all( tempDir );
//            ARIES_EXCEPTION( ER_WARN_TOO_FEW_RECORDS, lineIdx + 1 );
//        } else if ( readColCnt > colCnt ) {
//            string tmpLine = "line data: ";
//            for (const auto &f : line) {
//                tmpLine.append(f).append("||");
//            }
//            LOG(INFO) << tmpLine;
//            // cleanup generated column files
//            boost::filesystem::remove_all( tempDir );
//            ARIES_EXCEPTION( ER_WARN_TOO_MANY_RECORDS, lineIdx + 1 );
//        }
//
//        ++readLineCnt;
// #ifdef ARIES_PROFILE
//        otherTime += t.end();
// #endif
//        for ( colIdx = 0; colIdx < colCnt; ++colIdx )
//        {
//            colEntry = cols[ colIdx ];
//            int8_t containNull = containNulls[ colIdx ];
//            std::string& defValue = defaultValues[ colIdx ];
//            string& colStr = line[ colIdx ];
//
//            ARIES_ASSERT(itemStoreSizes[colIdx] <= tmpColBuffSize,
//                         "Column size of " + colEntry->GetName() + " too big: " + std::to_string( itemStoreSizes[ colIdx ] ) );
//            errorCode = ToColumnValue( colEntry, containNull, defValue, colStr,
//                                       lineIdx, tmpColBuff, tmpColBuffSize, errorMsg );
//            if ( 0 != errorCode )
//            {
//                boost::filesystem::remove_all(tempDir);
//                ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg );
//            }
//            if ( ColumnEntry::IsStringType( colEntry->GetType() ) )
//            {
//                CheckCharLen(cols[colIdx], colStr.length());
//                if ( colStr.length() >  columnMaxSizes[ colIdx ] )
//                    columnMaxSizes[ colIdx ] = colStr.length();
//            }
//
//            // batch write column
//            appendedSize = writeBuffs[colIdx]->append( tmpColBuff, itemStoreSizes[colIdx] );
// #ifdef ARIES_PROFILE
//            t.begin();
// #endif
//            while ( appendedSize < itemStoreSizes[ colIdx ] )
//            {
//                if ( writeBuffs[ colIdx ]->isFull() )
//                {
//                    writtenSize = my_write(blockFiles[colIdx]->m_fd,
//                                           writeBuffs[colIdx]->get(),
//                                           writeBuffs[colIdx]->getDataSize(),
//                                           MYF(0));
//                    if (writtenSize != writeBuffs[colIdx]->getDataSize())
//                    {
//                        // cleanup generated column files
//                        boost::filesystem::remove_all(tempDir);
//                        ARIES_EXCEPTION(EE_WRITE, blockFiles[colIdx]->m_filePath.data(), my_errno(),
//                                        strerror_r(my_errno(), errbuf, sizeof(errbuf)));
//                    }
//                    writeBuffs[ colIdx ]->clear();
//                }
//                int leftSize = itemStoreSizes[colIdx] - appendedSize;
//                if ( leftSize > 0 )
//                {
//                    appendedSize += writeBuffs[colIdx]->append(tmpColBuff + appendedSize, leftSize );
//                }
//            }
//            if ( writeBuffs[ colIdx ]->isFull() )
//            {
//                writtenSize = my_write(blockFiles[colIdx]->m_fd,
//                                       writeBuffs[colIdx]->get(),
//                                       writeBuffs[colIdx]->getDataSize(),
//                                       MYF(0));
//                if (writtenSize != writeBuffs[colIdx]->getDataSize())
//                {
//                    // cleanup generated column files
//                    boost::filesystem::remove_all(tempDir);
//                    ARIES_EXCEPTION(EE_WRITE, blockFiles[colIdx]->m_filePath.data(), my_errno(),
//                                    strerror_r(my_errno(), errbuf, sizeof(errbuf)));
//                }
//                writeBuffs[ colIdx ]->clear();
//            }
// #ifdef ARIES_PROFILE
//            writeTime += t.end();
// #endif
//        }
//
//        /*
//          We don't need to reset auto-increment field since we are restoring
//          its default value at the beginning of each loop iteration.
//        */
//        if (read_info.next_line())            // Skip to next line
//            break;
//        if (read_info.line_cuted)
//        {
//            // mysql 5.7.26:
//            // in strict mode, return error
//            // thd->cuted_fields++;			/* To long row */
//            // push_warning_printf(thd, Sql_condition::SL_WARNING,
//            //                     ER_WARN_TOO_MANY_RECORDS, ER(ER_WARN_TOO_MANY_RECORDS),
//            //                     thd->get_stmt_da()->current_row_for_condition());
//            string tmpLine = "line data: ";
//            for (const auto &f : line) {
//                tmpLine.append(f).append("||");
//            }
//            LOG(INFO) << tmpLine;
//            // cleanup generated column files
//            boost::filesystem::remove_all( tempDir );
//            ARIES_EXCEPTION( ER_WARN_TOO_MANY_RECORDS, lineIdx + 1 );
//        }
//    }
//
//    if ( 0 == readLineCnt )
//    {
//        LOG(INFO) << "Empty data file";
//        boost::filesystem::remove_all( tempDir );
//        return 0;
//    }
//
//    LOG( INFO ) << "scan finish";
//
//    for ( colIdx = 0; colIdx < colCnt; ++colIdx )
//    {
//        if (writeBuffs[colIdx]->empty())
//            continue;
//        writtenSize = my_write(blockFiles[colIdx]->m_fd,
//                               writeBuffs[colIdx]->get(),
//                               writeBuffs[colIdx]->getDataSize(),
//                               MYF(0));
//        if (writtenSize != writeBuffs[colIdx]->getDataSize())
//        {
//            // cleanup generated column files
//            boost::filesystem::remove_all(tempDir);
//            ARIES_EXCEPTION(EE_WRITE, blockFiles[colIdx]->m_filePath.data(), my_errno(),
//                            strerror_r(my_errno(), errbuf, sizeof(errbuf)));
//        }
//        writeBuffs[colIdx]->clear();
//    }
//
//    for ( colIdx = 0; colIdx < colCnt; ++colIdx )
//    {
//        colEntry = cols[ colIdx ];
//        bool handled = false;
//        if ( ColumnEntry::IsStringType( colEntry->GetType() ) )
//        {
//            size_t maxItemLen = columnMaxSizes[ colIdx ];
//            size_t schemaLen = colEntry->GetLength();
//            if ( maxItemLen < schemaLen )
//            {
//                handled = true;
//                itemStoreSizes[ colIdx ] = maxItemLen + containNulls[colIdx];
//                LOG(INFO) << "column [" << colIdx
//                          << "]: max item size is less than schema defined: " << schemaLen
//                          << ", it will be adjusted to " << maxItemLen
//                          << " and save "
//                          << ( schemaLen - maxItemLen ) * readLineCnt << " bytes!";
//
//                string newPath = blockFiles[ colIdx ]->m_filePath + ".new";
//                int newfd = open( newPath.data(), O_CREAT | O_RDWR | O_EXCL, S_IRUSR | S_IWUSR );
//                if ( -1 == newfd )
//                {
//                    set_my_errno(errno);
//                    // cleanup generated column files
//                    boost::filesystem::remove_all( tempDir );
//                    ARIES_EXCEPTION( EE_CANTCREATEFILE, newPath.data(), my_errno(),
//                                     strerror_r( my_errno(), errbuf, sizeof(errbuf) ) );
//                }
//                auto newFile = std::make_shared<fd_helper>( newfd );
//
//                headerInfo.rows = readLineCnt;
//                headerInfo.containNull = containNulls[ colIdx ];
//                headerInfo.itemLen = itemStoreSizes[ colIdx ];
//                memcpy( header, &headerInfo, sizeof( BlockFileHeader ) );
//                if ( ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE != my_write( newfd, header, ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE, MYF( 0 ) ) )
//                {
//                    // cleanup generated column files
//                    boost::filesystem::remove_all( tempDir );
//                    ARIES_EXCEPTION( EE_WRITE, newPath.data(), my_errno(),
//                                     strerror_r( my_errno(), errbuf, sizeof(errbuf) ) );
//                }
//
//                // std::shared_ptr<uchar> itemBuff( new uchar[ itemStoreSizes[ colIdx ] ] );
//                int64_t dataSize = my_seek( blockFiles[ colIdx ]->m_fd, 0, SEEK_END, MYF(0) ) - ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE;
//                my_seek( blockFiles[ colIdx ]->m_fd, ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE, SEEK_SET, MYF(0) );
//                if ( -1 == dataSize )
//                {
//                    set_my_errno(errno);
//                    // cleanup generated column files
//                    boost::filesystem::remove_all( tempDir );
//                    ARIES_EXCEPTION( EE_STAT, newPath.data(), my_errno(),
//                                     strerror_r( my_errno(), errbuf, sizeof(errbuf) ) );
//                }
//                int64_t readSize = 0;
//                int64_t totalReadSize = 0;
//                while ( totalReadSize < dataSize )
//                {
// #ifdef ARIES_PROFILE
//                    t.begin();
// #endif
//                    readSize = my_read( blockFiles[ colIdx ]->m_fd, tmpColBuff, tmpColBuffSize, MYF(0) );
//                    totalReadSize += readSize;
// #ifdef ARIES_PROFILE
//                    strColRereadTime += t.end();
//                    t.begin();
// #endif
//                    appendedSize = writeBuffs[colIdx]->append( tmpColBuff, readSize );
//                    while ( appendedSize < readSize )
//                    {
//                        if ( writeBuffs[ colIdx ]->isFull() )
//                        {
//                            writtenSize = my_write(newfd,
//                                                   writeBuffs[colIdx]->get(),
//                                                   writeBuffs[colIdx]->getDataSize(),
//                                                   MYF(0));
//                            if (writtenSize != writeBuffs[colIdx]->getDataSize())
//                            {
//                                // cleanup generated column files
//                                boost::filesystem::remove_all(tempDir);
//                                ARIES_EXCEPTION(EE_WRITE, blockFiles[colIdx]->m_filePath.data(), my_errno(),
//                                                strerror_r(my_errno(), errbuf, sizeof(errbuf)));
//                            }
//                            writeBuffs[ colIdx ]->clear();
//                        }
//                        int leftSize = readSize - appendedSize;
//                        if ( leftSize > 0 )
//                        {
//                            appendedSize += writeBuffs[colIdx]->append(tmpColBuff + appendedSize, leftSize );
//                        }
//                    }
//                    if ( writeBuffs[ colIdx ]->isFull() )
//                    {
//                        writtenSize = my_write(newfd,
//                                               writeBuffs[colIdx]->get(),
//                                               writeBuffs[colIdx]->getDataSize(),
//                                               MYF(0));
//                        if (writtenSize != writeBuffs[colIdx]->getDataSize())
//                        {
//                            // cleanup generated column files
//                            boost::filesystem::remove_all(tempDir);
//                            ARIES_EXCEPTION(EE_WRITE, newPath.data(), my_errno(),
//                                            strerror_r(my_errno(), errbuf, sizeof(errbuf)));
//                        }
//                        writeBuffs[ colIdx ]->clear();
//                    }
// #ifdef ARIES_PROFILE
//                    strColRewriteTime += t.end();
// #endif
//                }
//                if ( !writeBuffs[colIdx]->empty() )
//                {
//                    writtenSize = my_write(newfd,
//                                           writeBuffs[colIdx]->get(),
//                                           writeBuffs[colIdx]->getDataSize(),
//                                           MYF(0));
//                    if (writtenSize != writeBuffs[colIdx]->getDataSize())
//                    {
//                        // cleanup generated column files
//                        boost::filesystem::remove_all(tempDir);
//                        ARIES_EXCEPTION(EE_WRITE, newPath.data(), my_errno(),
//                                        strerror_r(my_errno(), errbuf, sizeof(errbuf)));
//                    }
//                    writeBuffs[colIdx]->clear();
//                }
//                blockFiles[ colIdx ]->m_filePath = newPath;
//            }
//        }
//        if ( !handled )
//        {
// #ifdef ARIES_PROFILE
//            t.begin();
// #endif
//            void* mapAddr = mmap( NULL, ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, blockFiles[ colIdx ]->m_fd, 0 );
//            if ( MAP_FAILED == mapAddr )
//            {
//                set_my_errno(errno);
//                // cleanup generated column files
//                boost::filesystem::remove_all( tempDir );
//                ARIES_EXCEPTION( EE_WRITE, blockFiles[ colIdx ]->m_filePath.data(), my_errno(),
//                                 strerror_r( my_errno(), errbuf, sizeof(errbuf) ) );
//            }
//            headerInfo.rows = readLineCnt;
//            headerInfo.containNull = containNulls[ colIdx ];
//            headerInfo.itemLen = itemStoreSizes[ colIdx ];
//            memcpy( (uchar*)mapAddr, &headerInfo, sizeof( BlockFileHeader ) );
//
//            int unmapRet = munmap( mapAddr, ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE );
//            if ( 0 != unmapRet )
//            {
//                set_my_errno(errno);
//                // cleanup generated column files
//                boost::filesystem::remove_all( tempDir );
//                ARIES_EXCEPTION(  EE_WRITE, blockFiles[ colIdx ]->m_filePath.data(), my_errno(),
//                                  strerror_r( my_errno(), errbuf, sizeof(errbuf) ) );
//            }
// #ifdef ARIES_PROFILE
//            adjustHeaderTime += t.end();
// #endif
//        }
//        blockFiles[ colIdx ] = nullptr;
//    }
// #ifdef ARIES_PROFILE
//    t.begin();
// #endif
//    errorCode = backupTable( dbEntry, tableEntry, tableBackupDir, bkDirSuffix, backedUpFiles );
// #ifdef ARIES_PROFILE
//    backupTime += t.end();
// #endif
//    if ( -2 == errorCode )
//        goto interrupted;
//    if ( -1 == errorCode )
//    {
//        restoreDataFiles( backedUpFiles );
//        boost::filesystem::remove_all( tempDir );
//        ARIES_EXCEPTION(  EE_WRITE, tableBackupDir.data(), my_errno(),
//                          strerror_r( my_errno(), errbuf, sizeof(errbuf) ) );
//    }
// #ifdef ARIES_PROFILE
//    t.begin();
// #endif
//    boost::filesystem::create_directory( tableDataDir );
//    MOVE_RESULT_FILES;
// #ifdef ARIES_PROFILE
//    moveTime = t.end();
// #endif
//    LOG(INFO) << "Moving data files done";
//
//    boost::filesystem::remove_all( tempDir );
//
//    if ( !tableBackupDir.empty() )
//    {
//        pthread_t tid;
//        auto removeBkInfo = new REMOVE_OLD_BACKUP_INFO( dbEntry->GetName(), tableEntry->GetName(), bkDirSuffix );
//        pthread_create( &tid, &connection_attr, removeOlderBackups, (void*)removeBkInfo );
//    }
//    s = ( readAndScanTime + 0.0 ) / 1000 / 1000;
//    LOG( INFO ) << "Import " <<  csvFilePath << " time ( read and scan ): " << s << " s, "
//               << s / 60 << " m";
//    s = ( writeTime + 0.0 ) / 1000 / 1000;
//    LOG( INFO ) << "--\ttime ( write ): " << s << " s, "
//                << s / 60 << " m";
//    s = ( strColRereadTime + 0.0 ) / 1000 / 1000;
//    LOG( INFO ) << "--\ttime ( string reread ): " << s << " s, "
//                << s / 60 << " m";
//    s = ( strColRewriteTime + 0.0 ) / 1000 / 1000;
//    LOG( INFO ) << "--\ttime ( string rewrite ): " << s << " s, "
//                << s / 60 << " m";
//    s = ( adjustHeaderTime + 0.0 ) / 1000 / 1000;
//    LOG( INFO ) << "--\ttime ( adjust header ): " << s << " s, "
//                << s / 60 << " m";
//    s = ( backupTime + 0.0 ) / 1000 / 1000;
//    LOG( INFO ) << "--\ttime ( backup ): " << s << " s, "
//                << s / 60 << " m";
//    s = ( moveTime + 0.0 ) / 1000 / 1000;
//    LOG( INFO ) << "--\ttime ( move ): " << s << " s, "
//                << s / 60 << " m";
//    return readLineCnt;
//
//interrupted:
//    // restore original data files
//    restoreDataFiles( backedUpFiles );
//
//    // cleanup generated column files
//    boost::filesystem::remove_all( tempDir );
//    SendKillMessage();
//
//    return -1;
//}

// static void
// read_sep_field(THD *thd, READ_INFO &read_info,
//                const string& enclosed,
//                const vector<ColumnEntryPtr>& cols,
//                uint64_t& lineIdx,
//                vector<vector<string>>& lines)
// {
//     size_t enclosed_length;
// 
//     enclosed_length=enclosed.length();
//     int colCnt = cols.size();
// 
//     for (;; ++lineIdx)
//     {
//         if (thd->killed)
//         {
//             thd->send_kill_message();
//         }
// 
//         /*
//           Check whether default values of the fields not specified in column list
//           are correct or not.
//         */
//         // if (validate_default_values_of_unset_fields(thd, table))
//         // {
//         //     read_info.error= true;
//         //     break;
//         // }
// 
//         vector<string> line;
//         int readCnt = 0;
//         string field;
//         for ( ; readCnt < colCnt; ++readCnt )
//         {
//             uint length;
//             uchar *pos;
// 
//             if (read_info.read_field())
//                 break;
// 
//             pos=read_info.row_start;
//             length=(uint) (read_info.row_end-pos);
// 
//             // null
//             if ((!read_info.enclosed &&
//                  (enclosed_length && length == 4 &&
//                   !memcmp(pos, STRING_WITH_LEN("NULL")))) ||
//                 (length == 1 && read_info.found_null))
//             {
//                 field.assign( "", 0 );
//             }
//             else
//             {
//                 field.assign( (char*)pos, length );
//             }
//             line.emplace_back( field );
//         }
// 
//         /* Have not read any field, thus input file is simply ended */
//         if ( !readCnt )
//             break;
//         // check column count
//         size_t csvColumnCount = line.size();
//         if ( csvColumnCount < cols.size() )
//         {
//             string tmpLine = "line data: ";
//             for ( const auto& f : line )
//             {
//                 tmpLine.append( f ).append( "||" );
//             }
//             LOG(INFO) << tmpLine;
//             ARIES_EXCEPTION( ER_WARN_TOO_FEW_RECORDS, lineIdx + 1 );
//         }
//         else if ( csvColumnCount > cols.size() )
//         {
//             string tmpLine = "line data: ";
//             for ( const auto& f : line )
//             {
//                 tmpLine.append( f ).append( "||" );
//             }
//             LOG(INFO) << tmpLine;
//             ARIES_EXCEPTION( ER_WARN_TOO_MANY_RECORDS, lineIdx + 1 );
//         }
//         lines.emplace_back( line );
// 
//         /*
//           We don't need to reset auto-increment field since we are restoring
//           its default value at the beginning of each loop iteration.
//         */
//         if (read_info.next_line())			// Skip to next line
//             break;
//         if (read_info.line_cuted)
//         {
//             // mysql 5.7.26:
//             // in strict mode, return error
//             // thd->cuted_fields++;			/* To long row */
//             // push_warning_printf(thd, Sql_condition::SL_WARNING,
//             //                     ER_WARN_TOO_MANY_RECORDS, ER(ER_WARN_TOO_MANY_RECORDS),
//             //                     thd->get_stmt_da()->current_row_for_condition());
//             string tmpLine = "line data: ";
//             for ( const auto& f : line )
//             {
//                 tmpLine.append( f ).append( "||" );
//             }
//             LOG(INFO) << tmpLine;
//             ARIES_EXCEPTION( ER_WARN_TOO_MANY_RECORDS,  lineIdx + 1 );
//         }
//         // continue_loop:;
//     }
// }

// static int64_t
// importWriteAll( THD *thd,
//                 const DatabaseEntrySPtr& dbEntry,
//                 const TableEntrySPtr& tableEntry,
//                 const string& csvFilePath,
//                 READ_INFO &read_info,
//                 const string& enclosed,
//                 const vector<ColumnEntryPtr>& cols,
//                 uint64_t& lineIdx )
// {
//     LOG( INFO ) << "Import mode: single thread, write all";
//     vector<vector<string>> lines; // buffer that holds a batch of rows in splitted text
// 
//     int64_t readAndScanTime = 0;
//     int64_t writeTime = 0;
//     int64_t backupTime = 0;
//     int64_t moveTime = 0;
// #ifdef ARIES_PROFILE
//     aries::CPU_Timer t;
// #endif
//     float s;
// #ifdef ARIES_PROFILE
//     t.begin();
// #endif
//     read_sep_field(thd, read_info,
//                    enclosed, cols,
//                    lineIdx, lines);
// #ifdef ARIES_PROFILE
//     readAndScanTime = t.end();
// #endif
//     if ( !lines.size() )
//     {
//         return 0;
//     }
// 
//     int errorCode;
//     string tableDataDir = Configuartion::GetInstance().GetDataDirectory( dbEntry->GetName(),
//                                                                   tableEntry->GetName() );
//     tableDataDir.append( "/" );
// 
//     string tempDir = MakeTmpDir();
//     string tableBackupDir, bkDirSuffix;
//     std::map<string, string> backedUpFiles;
// 
//     std::vector<string> outputFileNames;
//     std::vector<string> outputFilePaths;
//     size_t numValues = lines.size();
// #ifdef ARIES_PROFILE
//     t.begin();
// #endif
//     try
//     {
//         for ( const auto& colEntry: cols )
//         {
//             uint64_t colIndex = colEntry->GetColumnIndex();
//             size_t itemLen = 0;
//             int containNull = colEntry->IsAllowNull() ? 1 : 0;
//             auto defValuePtr = colEntry->GetDefault();
//             std::string defValue = defValuePtr ? * defValuePtr : "";
//             std::string outputFileName = tableEntry->GetName() + std::to_string( colIndex );
//             std::string outputFilePath = tempDir + "/" + outputFileName;
//             if ( IsCurrentThdKilled() )
//                 goto interrupted;
//             outputFileNames.emplace_back( outputFileName );
//             outputFilePaths.emplace_back( outputFilePath );
//             LOG(INFO) << "Writting column " << outputFilePath;
// 
//             switch ( colEntry->GetType() )
//             {
//                 case schema::ColumnType::BOOL:
//                 case schema::ColumnType::TINY_INT: // 1 byte
//                     itemLen = sizeof( int8_t ) + containNull;
//                     writeAriesColumnInt8( colEntry, lines, numValues, colIndex, itemLen, containNull, defValue, outputFilePath );
//                     break;
//                 case schema::ColumnType::SMALL_INT:  // 2 bytes
//                     itemLen = sizeof( int16_t ) + containNull;
//                     writeAriesColumnInt16( colEntry, lines, numValues, colIndex, itemLen, containNull, defValue, outputFilePath );
//                     break;
//                 case schema::ColumnType::INT: // 4 bytes
//                     itemLen = sizeof( int32_t ) + containNull;
//                     writeAriesColumnInt32( colEntry, lines, numValues, colIndex, itemLen, containNull, defValue, outputFilePath );
//                     break;
//                 case schema::ColumnType::LONG_INT: // 8 bytes
//                     itemLen = sizeof( int64_t ) + containNull;
//                     writeAriesColumnInt64( colEntry, lines, numValues, colIndex, itemLen, containNull, defValue, outputFilePath );
//                     break;
//                 case schema::ColumnType::DECIMAL:
//                     itemLen = GetDecimalRealBytes( colEntry->numeric_precision, colEntry->numeric_scale ) + containNull;
//                     writeAriesColumnCompactDecimal( colEntry, lines, numValues, colIndex, colEntry->numeric_precision, colEntry->numeric_scale, itemLen, containNull, defValue, outputFilePath );
//                     break;
//                 case schema::ColumnType::FLOAT:
//                     itemLen = sizeof( float ) + containNull;
//                     writeAriesColumnFloat( colEntry, lines, numValues, colIndex, itemLen, containNull, defValue, outputFilePath );
//                     break;
//                 case schema::ColumnType::DOUBLE:
//                     itemLen = sizeof( double ) + containNull;
//                     writeAriesColumnDouble( colEntry, lines, numValues, colIndex, itemLen, containNull, defValue, outputFilePath );
//                     break;
//                 case schema::ColumnType::DATE:
//                     itemLen = sizeof( aries_acc::AriesDate ) + containNull;
//                     writeAriesColumnDate( colEntry, lines, numValues, colIndex, itemLen, containNull, defValue, outputFilePath );
//                     break;
//                 case schema::ColumnType::DATE_TIME:
//                     itemLen = sizeof( aries_acc::AriesDatetime ) + containNull;
//                     writeAriesColumnDatetime( colEntry, lines, numValues, colIndex, itemLen, containNull, defValue, outputFilePath );
//                     break;
//                 case schema::ColumnType::TIMESTAMP:
//                     itemLen = sizeof( aries_acc::AriesTimestamp ) + containNull;
//                     writeAriesColumnTimestamp( colEntry, lines, numValues, colIndex, itemLen, containNull, defValue, outputFilePath );
//                     break;
//                 case schema::ColumnType::TEXT:
//                 case schema::ColumnType::VARBINARY:
//                 case schema::ColumnType::BINARY:
//                     itemLen = colEntry->GetLength() + containNull;
//                     writeAriesColumnChar( colEntry, lines, numValues, colIndex, itemLen, containNull, defValue, outputFilePath );
//                     break;
//                 case schema::ColumnType::YEAR:
//                     itemLen = sizeof( aries_acc::AriesYear ) + containNull;
//                     writeAriesColumnYear( colEntry, lines, numValues, colIndex, itemLen, containNull, defValue, outputFilePath );
//                     break;
//                 case schema::ColumnType::TIME:
//                     itemLen = sizeof( aries_acc::AriesTime ) + containNull;
//                     writeAriesColumnTime( colEntry, lines, numValues, colIndex, itemLen, containNull, defValue, outputFilePath );
//                     break;
//                 case schema::ColumnType::LIST:
//                 case schema::ColumnType::UNKNOWN:
//                 {
//                     string msg = "load data for column type " + std::to_string((int) colEntry->GetType());
//                     ARIES_EXCEPTION( ER_UNKNOWN_ERROR,  msg.data() );
//                     break;
//                 }
//             }
//         }
//     }
//     catch (const std::exception& e) {
//         boost::filesystem::remove_all( tempDir );
//         throw;
//     }
//     catch (...) {
//         boost::filesystem::remove_all( tempDir );
//         ARIES_EXCEPTION_SIMPLE( ER_UNKNOWN_ERROR, "Failed to import data");
//     }
// #ifdef ARIES_PROFILE
//     writeTime = t.end();
// #endif
//     char errbuf[MYSYS_STRERROR_SIZE];
//     // backup data files
// #ifdef ARIES_PROFILE
//     t.begin();
// #endif
//     errorCode = backupTable( dbEntry, tableEntry, tableBackupDir, bkDirSuffix, backedUpFiles );
// #ifdef ARIES_PROFILE
//     backupTime = t.end();
// #endif
//     if ( -2 == errorCode )
//         goto interrupted;
//     if ( -1 == errorCode )
//     {
//         restoreDataFiles( backedUpFiles );
//         boost::filesystem::remove_all( tempDir );
//         ARIES_EXCEPTION(  EE_WRITE, tableBackupDir.data(), my_errno(),
//                           strerror_r( my_errno(), errbuf, sizeof(errbuf) ) );
//     }
// #ifdef ARIES_PROFILE
//     t.begin();
// #endif
//     boost::filesystem::create_directory( tableDataDir );
//     MOVE_RESULT_FILES;
// #ifdef ARIES_PROFILE
//     moveTime = t.end();
// #endif
//     LOG(INFO) << "Moving data files done";
// 
//     boost::filesystem::remove_all( tempDir );
// 
//     if ( !tableBackupDir.empty() )
//     {
//         pthread_t tid;
//         auto removeBkInfo = new REMOVE_OLD_BACKUP_INFO( dbEntry->GetName(), tableEntry->GetName(), bkDirSuffix );
//         pthread_create( &tid, &connection_attr, removeOlderBackups, (void*)removeBkInfo );
//     }
//     s = ( readAndScanTime + 0.0 ) / 1000 / 1000;
//     LOG( INFO ) << "Import " <<  csvFilePath << " time ( read and scan ): " << s << " s, "
//                 << s / 60 << " m";
//     s = ( writeTime + 0.0 ) / 1000 / 1000;
//     LOG( INFO ) << "--\ttime ( write ): " << s << " s, "
//                 << s / 60 << " m";
//     s = ( backupTime + 0.0 ) / 1000 / 1000;
//     LOG( INFO ) << "--\ttime ( backup ): " << s << " s, "
//                 << s / 60 << " m";
//     s = ( moveTime + 0.0 ) / 1000 / 1000;
//     LOG( INFO ) << "--\ttime ( move ): " << s << " s, "
//                 << s / 60 << " m";
//     return numValues;
// 
// interrupted:
//     // restore original data files
//     restoreDataFiles( backedUpFiles );
// 
//     // cleanup generated column files
//     boost::filesystem::remove_all( tempDir );
//     SendKillMessage();
//     return 0;
// }

/**
 * reuturn value:
 *  -1: error
 * >=0: imported lines
 */
int64_t importCsvFile( aries_engine::AriesTransactionPtr& tx,
                       const DatabaseEntrySPtr& dbEntry, const TableEntrySPtr& tableEntry,
                       const std::string& csvFilePath,
                       uint64_t& skipLines,
                       const std::string& fieldSeperator,
                       const bool escapeGiven,
                       const std::string& escapeChar,
                       bool optEnclosed,
                       const std::string& encloseChar,
                       const std::string& lineSeperator,
                       const std::string& lineStart )
{
    int64_t preSanTime = 0, importTime = 0;
#ifdef ARIES_PROFILE
    aries::CPU_Timer tTotal, tPreScan, tImport;
    tTotal.begin();
#endif
    THD* thd = current_thd;

    const int escape_char= (escapeChar.length() && (escapeGiven ||
                                                  !(thd->variables.sql_mode & MODE_NO_BACKSLASH_ESCAPES)))
                           ? escapeChar[0] : INT_MAX;

    /* Report problems with non-ascii separators */
    // if (!escaped->is_ascii() || !enclosed->is_ascii() ||
    //     !field_term->is_ascii() ||
    //     !ex->line.line_term->is_ascii() || !ex->line.line_start->is_ascii())
    // {
    //     push_warning(thd, Sql_condition::SL_WARNING,
    //                  WARN_NON_ASCII_SEPARATOR_NOT_IMPLEMENTED,
    //                  ER(WARN_NON_ASCII_SEPARATOR_NOT_IMPLEMENTED));
    // }
    bool read_file_from_client = false;
    int file = open( csvFilePath.data(), O_RDONLY );
    if ( -1 == file )
    {
        set_my_errno(errno);
        char errbuf[MYSYS_STRERROR_SIZE];
        ARIES_EXCEPTION( EE_CANTCREATEFILE, csvFilePath.data(), my_errno(),
                         strerror_r( my_errno(), errbuf, sizeof(errbuf) ) );
    }
    int64_t fileLen = filesize( file );
    LOG( INFO ) << "File size: " << fileLen;
    auto fileHelper = std::make_shared< fd_helper >( file );

    uint tot_length= 1024 * 1024; // big enough for one line
    auto& cols = tableEntry->GetColumns();
    READ_INFO read_info(file, tot_length,
                        thd->variables.collation_database,
                        fieldSeperator, lineStart, lineSeperator,
                        encloseChar,
                        escape_char, read_file_from_client, false);
    if (read_info.error)
    {
        // Can't allocate buffers
        fileHelper = nullptr;
        ARIES_EXCEPTION_SIMPLE( EE_OUTOFMEMORY, "Out of memory" );
    }

    uint64_t totalLineCount = 0, skipped = 0, currentLineIdx = 0; // 当前被处理的行号，从0开始

    /* Skip lines if there is a line terminator */
    if ( lineSeperator.length() )
    {
        while ( skipped < skipLines )
        {
            if ( IsCurrentThdKilled() )
                SendKillMessage();
            if ( read_info.next_line() )
                break;
            ++skipped;
            ++currentLineIdx;
        }
    }
    uint64_t tmpLineIdx = currentLineIdx;

    char* optmzCharColumn = getenv("RATEUP_OPTIMIZE_CHAR_COLUMN");
    if ( optmzCharColumn && 0 == memcmp( "true", optmzCharColumn, 3) )
    {

    #ifdef ARIES_PROFILE
        tPreScan.begin();
    #endif

        totalLineCount = preScan2( thd, dbEntry, tableEntry,
                                   csvFilePath,
                                   read_info,
                                   encloseChar,
                                   cols,
                                   tmpLineIdx );
    #ifdef ARIES_PROFILE
        preSanTime = tPreScan.end();
    #endif
    }

    READ_INFO read_info2(file, tot_length,
                         thd->variables.collation_database,
                         fieldSeperator, lineStart, lineSeperator,
                         encloseChar,
                         escape_char, read_file_from_client, false);
    if ( lineSeperator.length() )
    {
        uint64_t skipped2 = 0;
        while ( skipped2 < skipLines )
        {
            if ( IsCurrentThdKilled() )
                SendKillMessage();
            if ( read_info2.next_line() )
                break;
            ++skipped2;
        }
    }

    // char* writeMod = getenv("RATEUP_WRITE_MODE");
    // if ( writeMod && 0 == memcmp( "all", writeMod, 3) )
    // {
    //     numValues = importWriteAll( thd, dbEntry, tableEntry, csvFilePath,
    //                                 read_info,
    //                                 encloseChar, cols, lineIdx );

    // }
    // else
    // {
    //     int batchWriteSize = 0;
    //     char* blockWriteSizeStr = getenv("RATEUP_BATCH_WRITE_SIZE");
    //     if ( blockWriteSizeStr && blockWriteSizeStr[0] != '\0' )
    //     {
    //         char* tail;
    //         batchWriteSize = std::strtol( blockWriteSizeStr, &tail, 10 );
    //         if ( *tail != '\0' )
    //         {
    //             batchWriteSize = 0;
    //         }
    //         if ( batchWriteSize > MAX_BATCH_WRITE_BUFF_SIZE )
    //             batchWriteSize = MAX_BATCH_WRITE_BUFF_SIZE;
    //         else if ( batchWriteSize < MIN_BATCH_WRITE_BUFF_SIZE )
    //             batchWriteSize = MIN_BATCH_WRITE_BUFF_SIZE;
    //         LOG( INFO ) << "Write batch size (env): " << batchWriteSize;
    //     }
    //     char* multiThreadWrite = getenv("RATEUP_MULTI_THREAD_WRITE");
    //     char* doPreScan = getenv("RATEUP_PRE_SCAN");
    // #ifdef ARIES_PROFILE
    //     tImport.begin();
    // #endif
    //     if ( doPreScan && doPreScan[0] == '1' )
    //     {
    //         uint64_t tmpLineIdx = lineIdx;

    //         std::vector<size_t> columnMaxSizes;
    //         tPreScan.begin();
    //         numValues = preScan( thd, dbEntry, tableEntry,
    //                              csvFilePath,
    //                              read_info,
    //                              encloseChar,
    //                              cols,
    //                              tmpLineIdx,
    //                              columnMaxSizes );
    //         preSanTime = tPreScan.end();

    //         READ_INFO read_info2(file, tot_length,
    //                              thd->variables.collation_database,
    //                              fieldSeperator, lineStart, lineSeperator,
    //                              encloseChar,
    //                              escape_char, read_file_from_client, false);
    //         if ( lineSeperator.length() )
    //         {
    //             uint64_t skipped2 = 0;
    //             while ( skipped2 < skipLines )
    //             {
    //                 if ( IsCurrentThdKilled() )
    //                     SendKillMessage();
    //                 if ( read_info2.next_line() )
    //                     break;
    //                 ++skipped2;
    //             }
    //         }
    //         if ( multiThreadWrite && multiThreadWrite[0] == '0' )
    //             importWriteBatchPreScannedSingleThread( thd, dbEntry, tableEntry, csvFilePath,
    //                                                     read_info2,
    //                                                     batchWriteSize,
    //                                                     encloseChar, cols, lineIdx,
    //                                                     numValues, columnMaxSizes );
    //         else
    //             importWriteBatchPreScannedMultiThreads( thd, dbEntry, tableEntry, csvFilePath,
    //                                                     read_info2,
    //                                                     batchWriteSize,
    //                                                     encloseChar, cols, lineIdx,
    //                                                     numValues, columnMaxSizes );
    //     }
    //     else
    //     {
    //         if ( multiThreadWrite && multiThreadWrite[0] == '0' )
    //             numValues = importWriteBatchNoPreScanSingleThread( thd, dbEntry, tableEntry, csvFilePath,
    //                                                                read_info,
    //                                                                batchWriteSize,
    //                                                                encloseChar, cols, lineIdx );
    //         else
    //             numValues = importWriteBatchNoPreScanMultiThreads( thd, dbEntry, tableEntry, csvFilePath,
    //                                                                read_info,
    //                                                                batchWriteSize,
    //                                                                encloseChar, cols, lineIdx );

    //     }
// #ifdef ARIES_PROFILE
    //     importTime = tImport.end();
// #endif
    // }
    int batchWriteSize = 0;
    char* blockWriteSizeStr = getenv("RATEUP_BATCH_WRITE_SIZE");
    if ( blockWriteSizeStr && blockWriteSizeStr[0] != '\0' )
    {
        char* tail;
        batchWriteSize = std::strtol( blockWriteSizeStr, &tail, 10 );
        if ( *tail != '\0' )
        {
            batchWriteSize = 0;
        }
        if ( batchWriteSize > MAX_BATCH_WRITE_BUFF_SIZE )
            batchWriteSize = MAX_BATCH_WRITE_BUFF_SIZE;
        else if ( batchWriteSize < MIN_BATCH_WRITE_BUFF_SIZE )
            batchWriteSize = MIN_BATCH_WRITE_BUFF_SIZE;
        LOG( INFO ) << "Write batch size (env): " << batchWriteSize;
    }
#ifdef ARIES_PROFILE
    tImport.begin();
#endif
    totalLineCount = importWriteBatchPreScanMultiThreads(
        thd, tx, dbEntry, tableEntry, csvFilePath,
        read_info2,
        batchWriteSize,
        encloseChar, cols, currentLineIdx,
        totalLineCount );
#ifdef ARIES_PROFILE
    importTime = tImport.end();
#endif
    // if ( ImportMode::REPLACE == importMode )
    // {
        // clear table
        AriesMvccTableManager::GetInstance().deleteCache( dbEntry->GetName(),
                                                          tableEntry->GetName() );
        AriesInitialTableManager::GetInstance().removeTable( dbEntry->GetName(),
                                                             tableEntry->GetName() );
        AriesMvccTableManager::GetInstance().resetInitTableOfMvccTable( dbEntry->GetName(),
                                                                        tableEntry->GetName() );
        // AriesXLogManager::GetInstance().AddTruncateEvent( tableEntry->GetId() );
        // AriesMvccTableManager::GetInstance().getMvccTable( dbEntry->GetName(),
        //                                                    tableEntry->GetName() )->RebuildPrimaryKeyIndex();

    // }
    // else
    // {
    //     auto mvccTable = AriesMvccTableManager::GetInstance().getTable( dbEntry->GetName(),
    //                                                                     tableEntry->GetName() );
    //     mvccTable->GetInitialTable()->Clear();
    // }

    skipLines = skipped;
#ifdef ARIES_PROFILE
    float s = ( tTotal.end() + 0.0 ) / 1000 / 1000;
    LOG( INFO ) << "Import " << csvFilePath << " time ( total ): " << s << "s, "
                << s / 60 << "m";
    s = ( preSanTime + 0.0 ) / 1000 / 1000;
    LOG( INFO ) << "--\ttime ( pre scan ): " << s << " s, "
                << s / 60 << " m";
    s = ( importTime + 0.0 ) / 1000 / 1000;
    LOG( INFO ) << "--\ttime ( import ): " << s << " s, "
                << s / 60 << " m";
#endif
    return totalLineCount;
}

int64_t importCsvFile( aries_engine::AriesTransactionPtr& tx,
                       const std::string& dbName, const std::string& tableName,
                       const std::string& csvFilePath,
                       uint64_t& skipLines,
                       const std::string& fieldSeperator,
                       const bool escapeGiven,
                       const std::string& escapeChar,
                       bool optEnclosed,
                       const std::string& encloseChar,
                       const std::string& lineSeperator,
                       const std::string& lineStart )
{
    auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(dbName);
    if ( !dbEntry )
    {
        ARIES_EXCEPTION( ER_BAD_DB_ERROR, dbName.data() );
    }
    auto tableEntry = dbEntry->GetTableByName(  tableName );
    if ( !tableEntry )
    {
        ARIES_EXCEPTION( ER_BAD_TABLE_ERROR, tableName.data() );
    }

    return importCsvFile( tx, dbEntry, tableEntry,
                          csvFilePath,
                          skipLines,
                          fieldSeperator,
                          escapeGiven,
                          escapeChar,
                          optEnclosed,
                          encloseChar,
                          lineSeperator,
                          lineStart);
}
