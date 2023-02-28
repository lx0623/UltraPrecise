#include "AriesSqlFunctions.hxx"

using namespace aries;

// this file is to implement the functions or operators which will be called in dynamic kernel

BEGIN_ARIES_ACC_NAMESPACE

    template< typename type_t, bool hasNull = false >
    struct InComparator: public CallableComparator
    {
        ARIES_HOST_DEVICE InComparator( const void* data, int count )
                : CallableComparator( data, count )
        {
        }

        ARIES_HOST_DEVICE virtual bool Compare( const void* val ) const
        {
            return binary_find( ( const type_t* )Data, Count, *( type_t* )val );
        }
    };

    template< typename type_t >
    struct InComparator< type_t, true > : public CallableComparator
    {
        ARIES_HOST_DEVICE InComparator( const void* data, int count )
                : CallableComparator( data, count )
        {
        }

        ARIES_HOST_DEVICE virtual bool Compare( const void* val ) const
        {
            return binary_find( ( const type_t* )Data, Count, *( nullable_type< type_t >* )val );
        }
    };

    template< >
    struct InComparator< char, false > : public CallableComparator
    {
        ARIES_HOST_DEVICE InComparator( const void* data, size_t len, int count )
                : CallableComparator( data, count ), Len( len )
        {
        }

        ARIES_HOST_DEVICE virtual bool Compare( const void* val ) const
        {
            return binary_find< false >( ( const char* )Data, Len, Count, ( const char* )val );
        }

    private:
        size_t Len;
    };

    template< >
    struct InComparator< char, true > : public CallableComparator
    {
        ARIES_HOST_DEVICE InComparator( const void* data, size_t len, int count )
                : CallableComparator( data, count ), Len( len )
        {
        }

        ARIES_HOST_DEVICE virtual bool Compare( const void* val ) const
        {
            return binary_find< true >( ( const char* )Data, Len, Count, ( const char* )val );
        }

    private:
        size_t Len;
    };

    template< typename type_t, bool hasNull = false >
    struct NotInComparator: public CallableComparator
    {
        ARIES_HOST_DEVICE NotInComparator( const void* data, int count )
                : CallableComparator( data, count )
        {
        }

        ARIES_HOST_DEVICE virtual bool Compare( const void* val ) const
        {
            if ( Count == 0 )
            {
                return false;
            }
            return !binary_find( ( const type_t* )Data, Count, *( type_t* )val );
        }
    };

    template< typename type_t >
    struct NotInComparator< type_t, true > : public CallableComparator
    {
        ARIES_HOST_DEVICE NotInComparator( const void* data, int count )
                : CallableComparator( data, count )
        {
        }

        ARIES_HOST_DEVICE virtual bool Compare( const void* val ) const
        {
            if ( Count == 0 )
            {
                return false;
            }
            return !binary_find( ( const type_t* )Data, Count, *( nullable_type< type_t >* )val );
        }
    };

    template< >
    struct NotInComparator< char, false > : public CallableComparator
    {
        ARIES_HOST_DEVICE NotInComparator( const void* data, size_t len, int count )
                : CallableComparator( data, count ), Len( len )
        {
        }

        ARIES_HOST_DEVICE virtual bool Compare( const void* val ) const
        {
            if ( Count == 0 )
            {
                return false;
            }
            return !binary_find< false >( ( const char* )Data, Len, Count, ( const char* )val );
        }

    private:
        size_t Len;
    };

    template< >
    struct NotInComparator< char, true > : public CallableComparator
    {
        ARIES_HOST_DEVICE NotInComparator( const void* data, size_t len, int count )
                : CallableComparator( data, count ), Len( len )
        {
        }

        ARIES_HOST_DEVICE virtual bool Compare( const void* val ) const
        {
            if ( Count == 0 )
            {
                return false;
            }
            return !binary_find< true >( ( const char* )Data, Len, Count, ( const char* )val );
        }

    private:
        size_t Len;
    };

    __device__ CallableComparator* create_in_comparator( const AriesKernelParamInfo& param )
    {
        CallableComparator* comparator = nullptr;
        switch( param.Type )
        {
            case AriesValueType::CHAR:
            {
                if( !param.HasNull )
                {
                    if( param.Len > 1 )
                        comparator = new InComparator< char, false >( param.Data, param.Len, param.Count );
                    else
                        comparator = new InComparator< int8_t, false >( param.Data, param.Count );
                }
                else
                {
                    if( param.Len > 1 )
                        comparator = new InComparator< char, true >( param.Data, param.Len, param.Count );
                    else
                        comparator = new InComparator< int8_t, true >( param.Data, param.Count );
                }
                break;
            }
            case AriesValueType::BOOL:
            case AriesValueType::INT8:
            {
                if( !param.HasNull )
                    comparator = new InComparator< int8_t, false >( param.Data, param.Count );
                else
                    comparator = new InComparator< int8_t, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::INT16:
            {
                if( !param.HasNull )
                    comparator = new InComparator< int16_t, false >( param.Data, param.Count );
                else
                    comparator = new InComparator< int16_t, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::INT32:
            {
                if( !param.HasNull )
                    comparator = new InComparator< int32_t, false >( param.Data, param.Count );
                else
                    comparator = new InComparator< int32_t, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::INT64:
            {
                if( !param.HasNull )
                    comparator = new InComparator< int64_t, false >( param.Data, param.Count );
                else
                    comparator = new InComparator< int64_t, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::UINT8:
            {
                if( !param.HasNull )
                    comparator = new InComparator< uint8_t, false >( param.Data, param.Count );
                else
                    comparator = new InComparator< uint8_t, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::UINT16:
            {
                if( !param.HasNull )
                    comparator = new InComparator< uint16_t, false >( param.Data, param.Count );
                else
                    comparator = new InComparator< uint16_t, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::UINT32:
            {
                if( !param.HasNull )
                    comparator = new InComparator< uint32_t, false >( param.Data, param.Count );
                else
                    comparator = new InComparator< uint32_t, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::UINT64:
            {
                if( !param.HasNull )
                    comparator = new InComparator< uint64_t, false >( param.Data, param.Count );
                else
                    comparator = new InComparator< uint64_t, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::FLOAT:
            {
                if( !param.HasNull )
                    comparator = new InComparator< float, false >( param.Data, param.Count );
                else
                    comparator = new InComparator< float, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::DOUBLE:
            {
                if( !param.HasNull )
                    comparator = new InComparator< double, false >( param.Data, param.Count );
                else
                    comparator = new InComparator< double, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::DECIMAL:
            {
                if( !param.HasNull )
                    comparator = new InComparator< Decimal, false >( param.Data, param.Count );
                else
                    comparator = new InComparator< Decimal, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::DATE:
            {
                if( !param.HasNull )
                    comparator = new InComparator< AriesDate, false >( param.Data, param.Count );
                else
                    comparator = new InComparator< AriesDate, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::DATETIME:
            {
                if( !param.HasNull )
                    comparator = new InComparator< AriesDatetime, false >( param.Data, param.Count );
                else
                    comparator = new InComparator< AriesDatetime, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::TIMESTAMP:
            {
                if( !param.HasNull )
                    comparator = new InComparator< AriesTimestamp, false >( param.Data, param.Count );
                else
                    comparator = new InComparator< AriesTimestamp, true >( param.Data, param.Count );
                break;
            }
            default:
                //FIXME need support all data types.
                break;
        }
        return comparator;
    }

    __device__ CallableComparator* create_notin_comparator( const AriesKernelParamInfo& param )
    {
        CallableComparator* comparator = nullptr;
        switch( param.Type )
        {
            case AriesValueType::CHAR:
            {
                if( !param.HasNull )
                {
                    if( param.Len > 1 )
                        comparator = new NotInComparator< char, false >( param.Data, param.Len, param.Count );
                    else
                        comparator = new NotInComparator< int8_t, false >( param.Data, param.Count );
                }
                else
                {
                    if( param.Len > 1 )
                        comparator = new NotInComparator< char, true >( param.Data, param.Len, param.Count );
                    else
                        comparator = new NotInComparator< int8_t, true >( param.Data, param.Count );
                }
                break;
            }
            case AriesValueType::BOOL:
            case AriesValueType::INT8:
            {
                if( !param.HasNull )
                    comparator = new NotInComparator< int8_t, false >( param.Data, param.Count );
                else
                    comparator = new NotInComparator< int8_t, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::INT16:
            {
                if( !param.HasNull )
                    comparator = new NotInComparator< int16_t, false >( param.Data, param.Count );
                else
                    comparator = new NotInComparator< int16_t, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::INT32:
            {
                if( !param.HasNull )
                    comparator = new NotInComparator< int32_t, false >( param.Data, param.Count );
                else
                    comparator = new NotInComparator< int32_t, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::INT64:
            {
                if( !param.HasNull )
                    comparator = new NotInComparator< int64_t, false >( param.Data, param.Count );
                else
                    comparator = new NotInComparator< int64_t, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::UINT8:
            {
                if( !param.HasNull )
                    comparator = new NotInComparator< uint8_t, false >( param.Data, param.Count );
                else
                    comparator = new NotInComparator< uint8_t, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::UINT16:
            {
                if( !param.HasNull )
                    comparator = new NotInComparator< uint16_t, false >( param.Data, param.Count );
                else
                    comparator = new NotInComparator< uint16_t, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::UINT32:
            {
                if( !param.HasNull )
                    comparator = new NotInComparator< uint32_t, false >( param.Data, param.Count );
                else
                    comparator = new NotInComparator< uint32_t, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::UINT64:
            {
                if( !param.HasNull )
                    comparator = new NotInComparator< uint64_t, false >( param.Data, param.Count );
                else
                    comparator = new NotInComparator< uint64_t, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::FLOAT:
            {
                if( !param.HasNull )
                    comparator = new NotInComparator< float, false >( param.Data, param.Count );
                else
                    comparator = new NotInComparator< float, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::DOUBLE:
            {
                if( !param.HasNull )
                    comparator = new NotInComparator< double, false >( param.Data, param.Count );
                else
                    comparator = new NotInComparator< double, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::DECIMAL:
            {
                if( !param.HasNull )
                    comparator = new NotInComparator< Decimal, false >( param.Data, param.Count );
                else
                    comparator = new NotInComparator< Decimal, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::DATE:
            {
                if( !param.HasNull )
                    comparator = new NotInComparator< AriesDate, false >( param.Data, param.Count );
                else
                    comparator = new NotInComparator< AriesDate, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::DATETIME:
            {
                if( !param.HasNull )
                    comparator = new NotInComparator< AriesDatetime, false >( param.Data, param.Count );
                else
                    comparator = new NotInComparator< AriesDatetime, true >( param.Data, param.Count );
                break;
            }
            case AriesValueType::TIMESTAMP:
            {
                if( !param.HasNull )
                    comparator = new NotInComparator< AriesTimestamp, false >( param.Data, param.Count );
                else
                    comparator = new NotInComparator< AriesTimestamp, true >( param.Data, param.Count );
                break;
            }
            default:
                //FIXME need support all data types.
                break;
        }
        return comparator;
    }

    extern "C" __global__ void KernelCreateInComparator( const AriesKernelParamInfo* info, size_t count, CallableComparator** output )
    {
        if( threadIdx.x == 0 && blockIdx.x == 0 )
        {
            for( int i = 0; i < count; ++i )
            {
                const auto& param = info[i];
                switch( param.OpType )
                {
                    case AriesComparisonOpType::IN:
                        output[i] = create_in_comparator( param );
                        break;
                    case AriesComparisonOpType::NOTIN:
                        output[i] = create_notin_comparator( param );
                        break;
                    default:
                        break;
                }
            }
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE int get_utf8_string_len( const char* buf, int len )
    {
        int total_character_length = 0;
        for( int i = 0; i < len; )
        {
            unsigned char c = ( unsigned char )buf[ i ];
            if( c == 0 )
                break;
            total_character_length += 1;
            if( c < 0x80 )
                i += 1;
            else if( c < 0xe0 )
                i += 2;
            else/* if (c < 0xf0)*/
                i += 3;
        }
        return total_character_length;
    }

    ARIES_HOST_DEVICE_NO_INLINE const char* get_utf8_char_pos( const char* buf, int index )
    {
        int i = 0;
        while( --index > 0 )
        {
        	unsigned char c = ( unsigned char )buf[ i ];
            if( c == 0 )
                break;
            if( c < 0x80 )
                i += 1;
            else if( c < 0xe0 )
                i += 2;
            else/* if (c < 0xf0)*/
                i += 3;
        }
        return buf + i;
    }

    ARIES_HOST_DEVICE_NO_INLINE void copy_utf8_char( char* output, const char* input, int count )
    {
        int i = 0;
        while( count-- > 0 )
        {
        	unsigned char c = ( unsigned char )input[ i ];
            if( c < 0x80 )
            {
                output[ i ] = input[ i ];
                i += 1;
            }
            else if( c < 0xe0 )
            {
                output[ i ] = input[ i ];
                output[ i + 1 ] = input[ i + 1 ];
                i += 2;
            }
            else/* if (c < 0xf0)*/
            {
                output[ i ] = input[ i ];
                output[ i + 1 ] = input[ i + 1 ];
                output[ i + 2 ] = input[ i + 2 ];
                i += 3;
            }
        }
    }


END_ARIES_ACC_NAMESPACE
