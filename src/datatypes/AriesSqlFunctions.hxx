#ifndef ARIESSQLFUNCTIONS_HXX_
#define ARIESSQLFUNCTIONS_HXX_
//#pragma once
#include "AriesTimeCalc.hxx"
#include "AriesCastFunctions.hxx"
#include "functions.hxx"
#include "AriesTruncateFunctions.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    template< typename type_t >
    ARIES_HOST_DEVICE type_t abs( type_t value )
    {
        return value < 0 ? -value : value;
    }

    template< typename type_t >
    struct AbsWrapper
    {
        ARIES_HOST_DEVICE
        type_t operator()( type_t t ) const
        {
            return abs( t );
        }
    };

    template< typename type_t >
    struct DateWrapper
    {
        ARIES_HOST_DEVICE
        auto operator()( type_t t ) const
        {
            return DATE( t );
        }
    };

    template< typename type_t, typename type_u >
    struct DateSubWrapper
    {
        ARIES_HOST_DEVICE
        auto operator()( type_t t, type_u u ) const
        {
            return DATE_SUB( t, u );
        }
    };

    template< typename type_t, typename type_u >
    struct DateAddWrapper
    {
        ARIES_HOST_DEVICE
        auto operator()( type_t t, type_u u ) const
        {
            return DATE_ADD( t, u );
        }
    };

    template< typename type_t, typename type_u >
    struct UnixTimestampWrapper
    {
        ARIES_HOST_DEVICE
        auto operator()( type_t t, type_u u ) const
        {
            return UNIX_TIMESTAMP( t, u );
        }
    };

    template< typename type_t, typename type_u >
    struct DateDiffWrapper
    {
        ARIES_HOST_DEVICE
        int32_t operator()( type_t t, type_u u ) const
        {
            return DATEDIFF( t, u );
        }
    };

    template< typename type_t, typename type_u >
    struct TimeDiffWrapper
    {
        ARIES_HOST_DEVICE
        auto operator()( type_t t, type_u u ) const
        {
            return TIMEDIFF( t, u );
        }
    };

    template< typename type_t, typename type_u >
    struct ExtractWrapper
    {
        ARIES_HOST_DEVICE
        auto operator()( type_t t, type_u u ) const
        {
            return EXTRACT( t, u );
        }
    };

    template< typename type_t >
    struct MonthWrapper
    {
        ARIES_HOST_DEVICE
        auto operator()( type_t t ) const
        {
            return MONTH( t );
        }
    };

    template< typename type_t >
    struct CastAsDoubleWrapper
    {
        ARIES_HOST_DEVICE
        auto operator()( type_t t ) const
        {
            return cast_as_double( t );
        }
    };

    template< typename type_t >
    struct CastAsIntWrapper
    {
        ARIES_HOST_DEVICE
        int operator()( type_t t ) const
        {
            return cast_as_signed( t );
        }
    };

    template< typename type_t >
    struct CastAsLongWrapper
    {
        ARIES_HOST_DEVICE
        int64_t operator()( type_t t ) const
        {
            return cast_as_long( t );
        }
    };

    template< typename type_t >
    struct TruncateWrapper
    {
        ARIES_HOST_DEVICE
        type_t operator()( type_t t, int p ) const
        {
            return truncate( t, p );
        }
    };

    // for single column functions
    template< typename output_t, typename func_t, typename type_t, typename ... args_t >
    ARIES_HOST_DEVICE output_t sql_function_wrapper( func_t f, type_t param, args_t ... args )
    {
        return f( param, args... );
    }

    template< typename output_t, typename func_t, typename type_t, template< typename > class type_nullable, typename ... args_t >
    ARIES_HOST_DEVICE output_t sql_function_wrapper( func_t f, type_nullable< type_t > param, args_t ... args )
    {
        return output_t( param.flag, f( param.value, args... ) );
    }

    // for two columns functions
    template< typename output_t, typename func_t, typename type_t, typename type_u, template< typename > class type_nullable, typename ... args_t >
    ARIES_HOST_DEVICE output_t sql_function_wrapper( func_t f, type_nullable< type_t > left, type_nullable< type_u > right, args_t ... args )
    {
        return output_t( left.flag && right.flag, f( left.value, right.value, args... ) );
    }

    template< typename output_t, typename func_t, typename type_t, typename type_u, template< typename > class type_nullable, typename ... args_t >
    ARIES_HOST_DEVICE output_t sql_function_wrapper( func_t f, type_t left, type_nullable< type_u > right, args_t ... args )
    {
        return output_t( right.flag, f( left, right.value, args... ) );
    }

    struct AriesKernelParamInfo
    {
        void* Data;
        size_t Count;
        size_t Len;
        aries::AriesValueType Type;
        bool  HasNull;
        aries::AriesComparisonOpType OpType;
    };

    // ARIES_HOST_DEVICE int8_t* AriesColumnDataIterator::operator[]( int i ) const
    // {
    //     if( m_indices != nullptr )
    //     {
    //         int indicesBlockIndex = aries_acc::binary_search<aries_acc::bounds_upper>(
    //                                     m_indiceBlockSizePrefixSum,
    //                                     m_indiceBlockCount, i ) - 1;
    //         int offset = i - m_indiceBlockSizePrefixSum[ indicesBlockIndex ];
    //         int pos = m_indices[ indicesBlockIndex ][ offset ];
    //         if( pos != -1 )
    //         {
    //             int dataBlockIndex = aries_acc::binary_search<aries_acc::bounds_upper>( m_dataBlockSizePrefixSum, m_dataBlockCount, pos ) - 1;
    //             return m_data[ dataBlockIndex ] + ( pos - m_dataBlockSizePrefixSum[ dataBlockIndex ] ) * m_perItemSize;
    //         }
    //         else
    //             return m_nullData;
    //     }
    //     else
    //     {
    //         int dataBlockIndex = aries_acc::binary_search<aries_acc::bounds_upper>( m_dataBlockSizePrefixSum, m_dataBlockCount, i ) - 1;
    //         return m_data[ dataBlockIndex ] + ( i - m_dataBlockSizePrefixSum[ dataBlockIndex ] ) * m_perItemSize;
    //     }

    //     // if( m_indices != nullptr )
    //     // {
    //     //     int pos = m_indices[ index ];
    //     //     if( pos != -1 )
    //     //         return m_data + pos * m_perItemSize;
    //     //     else
    //     //         return m_nullData;
    //     // }
    //     // else
    //     //     return m_data + index * m_perItemSize;
    // }

    extern "C"  __global__ void KernelCreateInComparator( const AriesKernelParamInfo* info, size_t count, CallableComparator** output );

    ARIES_HOST_DEVICE_NO_INLINE int get_utf8_string_len( const char* buf, int len );
    ARIES_HOST_DEVICE_NO_INLINE const char* get_utf8_char_pos( const char* buf, int index );
    ARIES_HOST_DEVICE_NO_INLINE void copy_utf8_char( char* output, const char* input, int count );

END_ARIES_ACC_NAMESPACE

#endif //ARIESSQLFUNCTIONS_HXX_
