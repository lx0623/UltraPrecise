/*
 * DataConvertor.h
 *
 *  Created on: Jan 13, 2020
 *      Author: lichi
 */

#ifndef DATACONVERTOR_H_
#define DATACONVERTOR_H_
#include <vector>
#include <future>
#include <string>
#include <fstream>
#include <limits>
#include "csv.hpp"
#include "../../../src/schema/ColumnEntry.h"
#include "aries_types.hxx"
#include "AriesAssert.h"
#include "AriesDatetimeTrans.h"
using namespace aries::schema;
using namespace aries_acc;
using namespace csv;
using namespace std;
static constexpr int64_t FILE_HEADER_SIZE_IN_BYTES = 4096;

struct ARIES_PACKED HeaderInfo
{
    int32_t version;
    uint64_t numValues;
    int8_t containNull;
    int16_t itemLen;
};

enum class DataConverterStatus
{
    DATA_OK, DATA_EMPTY, DATA_TRUNCATED, DATA_OUT_OF_RANGE, DATA_INVALID, DATA_TOO_BIG
};

struct Int8Converter
{
    int8_t operator()( const string& val, DataConverterStatus& status )
    {
        int8_t result = 0;
        status = DataConverterStatus::DATA_OK;
        if( !val.empty() )
        {
            char* pEnd;
            int64_t value = strtoll( val.c_str(), &pEnd, 10 );
            if( *pEnd == 0 )
            {
                if( value >= numeric_limits< signed char >::min() && value <= numeric_limits< signed char >::max() )
                    result = value;
                else
                    status = DataConverterStatus::DATA_OUT_OF_RANGE;
            }
            else if( pEnd == val.c_str() )
                status = DataConverterStatus::DATA_INVALID;
            else
                status = DataConverterStatus::DATA_TRUNCATED;
        }
        else
            status = DataConverterStatus::DATA_EMPTY;
        return result;
    }
};
using BoolConverter = Int8Converter;

struct Int16Converter
{
    int16_t operator()( const string& val, DataConverterStatus& status )
    {
        int16_t result = 0;
        status = DataConverterStatus::DATA_OK;
        if( !val.empty() )
        {
            char* pEnd;
            int64_t value = strtoll( val.c_str(), &pEnd, 10 );
            if( *pEnd == 0 )
            {
                if( value >= numeric_limits< short >::min() && value <= numeric_limits< short >::max() )
                    result = value;
                else
                    status = DataConverterStatus::DATA_OUT_OF_RANGE;
            }
            else if( pEnd == val.c_str() )
                status = DataConverterStatus::DATA_INVALID;
            else
                status = DataConverterStatus::DATA_TRUNCATED;
        }
        else
            status = DataConverterStatus::DATA_EMPTY;
        return result;
    }
};

struct Int32Converter
{
    int32_t operator()( const string& val, DataConverterStatus& status )
    {
        int32_t result = 0;
        status = DataConverterStatus::DATA_OK;
        if( !val.empty() )
        {
            char* pEnd;
            int64_t value = strtoll( val.c_str(), &pEnd, 10 );
            if( *pEnd == 0 )
            {
                if( value >= numeric_limits< int >::min() && value <= numeric_limits< int >::max() )
                    result = value;
                else
                    status = DataConverterStatus::DATA_OUT_OF_RANGE;
            }
            else if( pEnd == val.c_str() )
                status = DataConverterStatus::DATA_INVALID;
            else
                status = DataConverterStatus::DATA_TRUNCATED;
        }
        else
            status = DataConverterStatus::DATA_EMPTY;
        return result;
    }
};

struct Int64Converter
{
    int64_t operator()( const string& val, DataConverterStatus& status )
    {
        int64_t result = 0;
        status = DataConverterStatus::DATA_OK;
        if( !val.empty() )
        {
            char* pEnd;
            int64_t value = strtoll( val.c_str(), &pEnd, 10 );
            if( *pEnd == 0 )
            {
                if( errno != ERANGE )
                    result = value;
                else
                    status = DataConverterStatus::DATA_OUT_OF_RANGE;
            }
            else if( pEnd == val.c_str() )
                status = DataConverterStatus::DATA_INVALID;
            else
                status = DataConverterStatus::DATA_TRUNCATED;
        }
        else
            status = DataConverterStatus::DATA_EMPTY;
        return result;
    }
};

struct FloatConverter
{
    float operator()( const string& val, DataConverterStatus& status )
    {
        return val.empty() ? 0.0f : std::stof( val );
    }
};

struct DoubleConverter
{
    double operator()( const string& val, DataConverterStatus& status )
    {
        return val.empty() ? 0.0 : std::stod( val );
    }
};

struct AriesDateConverter
{
    AriesDate operator()( const string& val, DataConverterStatus& status )
    {
        return val.empty() ? AriesDate() : AriesDatetimeTrans::GetInstance().ToAriesDate( val );
    }
};

struct AriesDatetimeConverter
{
    AriesDatetime operator()( const string& val, DataConverterStatus& status )
    {
        return val.empty() ? AriesDatetime() : AriesDatetimeTrans::GetInstance().ToAriesDatetime( val );
    }
};

struct AriesTimeConverter
{
    AriesTime operator()( const string& val, DataConverterStatus& status )
    {
        return val.empty() ? AriesTime() : AriesDatetimeTrans::GetInstance().ToAriesTime( val );
    }
};

struct AriesTimestampConverter
{
    AriesTimestamp operator()( const string& val, DataConverterStatus& status )
    {
        return val.empty() ? AriesTimestamp() : AriesDatetimeTrans::GetInstance().ToAriesTimestamp( val );
    }
};

struct AriesYearConverter
{
    AriesYear operator()( const string& val, DataConverterStatus& status )
    {
        return val.empty() ? AriesYear() : AriesDatetimeTrans::GetInstance().ToAriesYear( val );
    }
};

struct DataConverter
{
    DataConverter( int columnIndex, bool bHasNull, const string& filePath )
            : m_index( columnIndex ), m_bHasNull( bHasNull ), m_totalTupleNum( 0 )
    {
        m_file.open( filePath, ios::binary | ios::trunc );
        m_file.seekp( FILE_HEADER_SIZE_IN_BYTES );
    }
    virtual bool Convert( const vector< CSVRow >& rows ) = 0;
    virtual ~DataConverter()
    {
    }
    bool IsNull( const string_view& val ) const
    {
        return val == "\\N";
    }
    virtual bool PostProcess()
    {
        return true;
    }
protected:
    int m_index;            //column index, 0 based
    bool m_bHasNull;
    ofstream m_file;
    size_t m_totalTupleNum; // total row count
};

template< typename type_t, typename converter_t >
struct SimpleConvertor: public DataConverter
{
    SimpleConvertor( int columnIndex, bool bHasNull, const string& filePath, converter_t converter )
            : DataConverter( columnIndex, bHasNull, filePath ), m_converter( converter )
    {
    }
    virtual ~SimpleConvertor()
    {
    }
    virtual bool Convert( const vector< CSVRow >& rows )
    {
        bool bRet = true;
        if( m_file.is_open() )
        {
            if( m_bHasNull )
            {
                m_buffer.resize( rows.size() * sizeof(nullable_type< type_t > ) );
                ConvertNullableData( rows );
            }
            else
            {
                m_buffer.resize( rows.size() * sizeof(type_t) );
                ConvertData( rows );
            }
            m_file.write( m_buffer.data(), m_buffer.size() );
            m_totalTupleNum += rows.size();
        }
        else
            bRet = false;
        return bRet;
    }

    virtual bool PostProcess()
    {
        if( m_file.is_open() )
        {
            m_file.flush();
            m_file.seekp( 0 );
            HeaderInfo header;
            header.version = 1;
            header.itemLen = sizeof(int) + m_bHasNull;
            header.containNull = m_bHasNull;
            header.numValues = m_totalTupleNum;
            m_file.write( ( char* )&header, sizeof(HeaderInfo) );
            m_file.close();
            return true;
        }
        else
            return false;
    }
private:
    bool ConvertData( const vector< CSVRow >& rows )
    {
        int index = m_index;
        type_t* outData = ( type_t* )m_buffer.data();
        DataConverterStatus status;
        for( const auto& row : rows )
        {
            const auto& strVal = row.get_string_view( index );
            if( IsNull( strVal ) )
                *outData = type_t();
            else
                *outData = m_converter(
                { strVal.data(), strVal.size() }, status );
            ++outData;
        }
        return true;
    }

    bool ConvertNullableData( const vector< CSVRow >& rows )
    {
        int index = m_index;
        nullable_type< type_t >* outData = ( nullable_type< type_t >* )m_buffer.data();
        DataConverterStatus status;
        for( const auto& row : rows )
        {
            const auto& strVal = row.get_string_view( index );
            if( IsNull( strVal ) )
                outData->flag = 0;
            else
            {
                outData->flag = 1;
                outData->value = m_converter(
                { strVal.data(), strVal.size() }, status );
            }
            ++outData;
        }
        return true;
    }
private:
    vector< char > m_buffer;
    converter_t m_converter;
};

struct StringConvertor: public DataConverter
{
    StringConvertor( int columnIndex, bool bHasNull, const string& filePath )
            : DataConverter( columnIndex, bHasNull, filePath ), m_maxLen( 0 )
    {
        m_indices.push_back( 0 );
    }
    virtual ~StringConvertor()
    {
    }
    virtual bool Convert( const vector< CSVRow >& rows )
    {
        int index = m_index;
        int maxLen = m_maxLen;
        if( index > 0 )
        {
            for( const auto& row : rows )
            {
                const auto& strVal = row.get_string_view( index );
                m_buffer.append( strVal.data(), strVal.size() );
                m_indices.push_back( m_buffer.size() );
                maxLen = std::max( maxLen, ( int )strVal.size() );
            }
        }
        else
        {
            for( const auto& row : rows )
            {
                const auto& strVal = row.get_string_view( index );
                m_buffer.append( strVal.data(), strVal.size() );
                m_indices.push_back( m_buffer.size() );
                maxLen = std::max( maxLen, ( int )strVal.size() );
            }
        }
        m_maxLen = maxLen;
        m_totalTupleNum += rows.size();
        return true;
    }

    virtual bool PostProcess()
    {
        if( m_file.is_open() )
        {
            m_fill.resize( m_maxLen );
            memset( ( void* )m_fill.data(), 0, m_maxLen );
            char buf[1024 * 64];
            m_file.rdbuf()->pubsetbuf( buf, sizeof( buf ) );

            if( m_bHasNull )
                return PostProcessNullableString();
            else
                return PostProcessString();
        }
        else
            return false;
    }

private:
    bool PostProcessString()
    {
        int maxLen = m_maxLen;
        m_file.seekp( 0 );
        HeaderInfo header;
        header.version = 1;
        header.itemLen = maxLen + m_bHasNull;
        header.containNull = m_bHasNull;
        header.numValues = m_totalTupleNum;
        m_file.write( ( char* )&header, sizeof(HeaderInfo) );
        m_file.seekp( FILE_HEADER_SIZE_IN_BYTES );

        const char* data = m_buffer.data();
        int strLen = 0;
        for( int i = 0; i < m_indices.size() - 1; ++i )
        {
            strLen = m_indices[i + 1] - m_indices[i];
            if( !IsNull( csv::string_view( data, strLen ) ) )
            {
                m_file.write( data, strLen );
                m_file.write( m_fill.data(), maxLen - strLen );
            }
            else
            {
                m_file.write( m_fill.data(), maxLen );
            }
            data += strLen;
        }
        m_file.close();
        return true;
    }

    bool PostProcessNullableString()
    {
        int maxLen = m_maxLen;
        const char* data = m_buffer.data();
        int strLen = 0;
        for( int i = 0; i < m_indices.size() - 1; ++i )
        {
            strLen = m_indices[i + 1] - m_indices[i];
            if( !IsNull( csv::string_view( data, strLen ) ) )
            {
                m_file.write( data, strLen );
                m_file.write( m_fill.data(), maxLen - strLen );
            }
            else
                m_file.write( m_fill.data(), maxLen );
            data += strLen;
        }
        m_file.close();
        return true;
    }

private:
    string m_buffer;
    vector< long > m_indices;
    int m_maxLen;
    string m_fill;
};

struct DecimalConvertor: public DataConverter
{
    DecimalConvertor( int columnIndex, bool bHasNull, uint32_t precision, uint32_t scale, const string& filePath )
            : DataConverter( columnIndex, bHasNull, filePath ), m_precision( precision ), m_scale( scale )
    {
    }

    virtual ~DecimalConvertor()
    {
    }

    virtual bool Convert( const vector< CSVRow >& rows )
    {
        bool bRet = true;
        if( m_file.is_open() )
        {
            if( m_bHasNull )
            {
                m_buffer.resize( rows.size() * sizeof(nullable_type< Decimal > ) );
                ConvertToNullableDecimal( rows );
            }
            else
            {
                m_buffer.resize( rows.size() * sizeof(Decimal) );
                ConvertToDecimal( rows );
            }
            m_totalTupleNum += rows.size();
            m_file.write( m_buffer.data(), m_buffer.size() );
        }
        else
            bRet = false;
        return bRet;
    }

    virtual bool PostProcess()
    {
        if( m_file.is_open() )
        {
            m_file.flush();
            m_file.seekp( 0 );
            HeaderInfo header;
            header.version = 1;
            header.itemLen = sizeof(Decimal) + m_bHasNull;
            header.containNull = m_bHasNull;
            header.numValues = m_totalTupleNum;
            m_file.write( ( char* )&header, sizeof(HeaderInfo) );
            m_file.close();
            return true;
        }
        else
            return false;
    }

private:
    bool ConvertToDecimal( const vector< CSVRow >& rows )
    {
        int index = m_index;
        uint32_t prec = m_precision;
        uint32_t scale = m_scale;
        Decimal* outData = ( Decimal* )m_buffer.data();
        for( const auto& row : rows )
        {
            const auto& strVal = row.get_string_view( index );
            if( IsNull( strVal ) )
                new ( outData ) Decimal( prec, scale );
            else
                new ( outData ) Decimal( prec, scale, string
                { strVal.data(), strVal.size() }.c_str() );
            ++outData;
        }
        return true;
    }

    bool ConvertToNullableDecimal( const vector< CSVRow >& rows )
    {
        int index = m_index;
        uint32_t prec = m_precision;
        uint32_t scale = m_scale;
        nullable_type< Decimal >* outData = ( nullable_type< Decimal >* )m_buffer.data();
        for( const auto& row : rows )
        {
            const auto& strVal = row.get_string_view( index );
            if( IsNull( strVal ) )
                outData->flag = 0;
            else
            {
                outData->flag = 1;
                new ( &outData->value ) Decimal( prec, scale, string
                { strVal.data(), strVal.size() }.c_str() );
            }
            ++outData;
        }
        return true;
    }

private:
    vector< char > m_buffer;
    uint32_t m_precision;
    uint32_t m_scale;
};

struct CsvFileImporterConfig
{
    string m_csvFilePath;
    vector< ColumnEntryPtr > m_cols;
    vector< string > m_outputFilePaths;
    char m_delimiter;
    char m_quote = '"';
    bool m_bHasHeader = false;
    static constexpr int BLOCK_SIZE = 5000000;
};

struct CsvFileImporter
{
    static void ImportCsvFile( const CsvFileImporterConfig& config )
    {
        assert( !config.m_csvFilePath.empty() && !config.m_cols.empty() && config.m_cols.size() == config.m_outputFilePaths.size() );

        shared_ptr< vector< shared_ptr< DataConverter > > > convertors = make_shared< vector< shared_ptr< DataConverter > > >();
        int index = 0;
        for( const auto& col : config.m_cols )
        {
            switch( col->GetType() )
            {
                case ColumnType::BOOL:
                case ColumnType::TINY_INT:
                {
                    convertors->emplace_back(
                            make_shared< SimpleConvertor< int8_t, Int8Converter > >( index, col->IsAllowNull(), config.m_outputFilePaths[index],
                                    Int8Converter() ) );
                    break;
                }
                case ColumnType::SMALL_INT:
                {
                    convertors->emplace_back(
                            make_shared< SimpleConvertor< int16_t, Int16Converter > >( index, col->IsAllowNull(), config.m_outputFilePaths[index],
                                    Int16Converter() ) );
                    break;
                }
                case ColumnType::INT:
                {
                    convertors->emplace_back(
                            make_shared< SimpleConvertor< int32_t, Int32Converter > >( index, col->IsAllowNull(), config.m_outputFilePaths[index],
                                    Int32Converter() ) );
                    break;
                }
                case ColumnType::LONG_INT:
                {
                    convertors->emplace_back(
                            make_shared< SimpleConvertor< int64_t, Int64Converter > >( index, col->IsAllowNull(), config.m_outputFilePaths[index],
                                    Int64Converter() ) );
                    break;
                }
                case ColumnType::FLOAT:
                {
                    convertors->emplace_back(
                            make_shared< SimpleConvertor< float, FloatConverter > >( index, col->IsAllowNull(), config.m_outputFilePaths[index],
                                    FloatConverter() ) );
                    break;
                }
                case ColumnType::DOUBLE:
                {
                    convertors->emplace_back(
                            make_shared< SimpleConvertor< double, DoubleConverter > >( index, col->IsAllowNull(), config.m_outputFilePaths[index],
                                    DoubleConverter() ) );
                    break;
                }
                case ColumnType::DECIMAL:
                {
                    convertors->emplace_back(
                            make_shared< DecimalConvertor >( index, col->IsAllowNull(), col->numeric_precision, col->numeric_scale,
                                    config.m_outputFilePaths[index] ) );
                    break;
                }
                case ColumnType::TEXT:
                case ColumnType::VARBINARY:
                case ColumnType::BINARY:
                {
                    convertors->emplace_back( make_shared< StringConvertor >( index, col->IsAllowNull(), config.m_outputFilePaths[index] ) );
                    break;
                }
                case ColumnType::DATE:
                {
                    convertors->emplace_back(
                            make_shared< SimpleConvertor< AriesDate, AriesDateConverter > >( index, col->IsAllowNull(),
                                    config.m_outputFilePaths[index], AriesDateConverter() ) );
                    break;
                }
                case ColumnType::DATE_TIME:
                {
                    convertors->emplace_back(
                            make_shared< SimpleConvertor< AriesDatetime, AriesDatetimeConverter > >( index, col->IsAllowNull(),
                                    config.m_outputFilePaths[index], AriesDatetimeConverter() ) );
                    break;
                }
                case ColumnType::TIME:
                {
                    convertors->emplace_back(
                            make_shared< SimpleConvertor< AriesTime, AriesTimeConverter > >( index, col->IsAllowNull(),
                                    config.m_outputFilePaths[index], AriesTimeConverter() ) );
                    break;
                }
                case ColumnType::TIMESTAMP:
                {
                    convertors->emplace_back(
                            make_shared< SimpleConvertor< AriesTimestamp, AriesTimestampConverter > >( index, col->IsAllowNull(),
                                    config.m_outputFilePaths[index], AriesTimestampConverter() ) );
                    break;
                }
                case ColumnType::YEAR:
                {
                    convertors->emplace_back(
                            make_shared< SimpleConvertor< AriesYear, AriesYearConverter > >( index, col->IsAllowNull(),
                                    config.m_outputFilePaths[index], AriesYearConverter() ) );
                    break;
                }
                default:
                    ARIES_EXCEPTION_SIMPLE( ER_UNKNOWN_ERROR, "invalid column type " + std::to_string( ( int ) col->GetType() ) );
                    break;
            }
        }

        shared_ptr< vector< CSVRow > > allRows = make_shared< vector< CSVRow > >();
        CSVFormat format;
        format.delimiter( config.m_delimiter );
        if( !config.m_bHasHeader )
        {
            vector< string > colNames;
            for( int i = 0; i < config.m_cols.size(); ++i )
                colNames.push_back( std::to_string( i ) );
            format.column_names( colNames );
        }
        CSVReader reader( config.m_csvFilePath, format );
        CSVRow row;
        int count = config.BLOCK_SIZE;
        int num = 0;
        bool bContinue = true;
        do
        {
            while( count-- && ( bContinue = reader.read_row( row ) ) )
            {
                ++num;
                allRows->emplace_back( std::move( row ) );
            }
            vector< future< void > > workThreads;
            for( auto& conv : *convertors )
            {
                workThreads.push_back( std::async( std::launch::async, [&]
                {   conv->Convert( *allRows.get() );} ) );
            }
            for( auto& t : workThreads )
                t.wait();
            count = config.BLOCK_SIZE;
            allRows->clear();
        } while( bContinue );

        vector< future< void > > workThreads;
        for( auto& conv : *convertors )
        {
            workThreads.push_back( std::async( std::launch::async, [&]
            {   conv->PostProcess();} ) );
        }
        for( auto& t : workThreads )
            t.wait();
    }
};

#endif /* DATACONVERTOR_H_ */
