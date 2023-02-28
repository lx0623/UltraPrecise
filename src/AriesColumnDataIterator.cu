#include "AriesColumnDataIterator.hxx"
#include "datatypes/functions.hxx"
#include "CudaAcc/algorithm/cta_search.hxx"
BEGIN_ARIES_ACC_NAMESPACE

template< typename index_type_t >
ARIES_HOST_DEVICE
int8_t* DoGetData( int i,
                   int8_t** data,
                   int64_t* dataBlockSizePrefixSum,
                   int32_t dataBlockCount,
                   index_type_t** indices,
                   int64_t* indiceBlockSizePrefixSum,
                   int32_t indiceBlockCount,
                   size_t perItemSize,
                   int8_t* nullData )
{
    int indicesBlockIndex = aries_acc::binary_search<aries_acc::bounds_upper>(
                                indiceBlockSizePrefixSum,
                                indiceBlockCount, i ) - 1;
    int offset = i - indiceBlockSizePrefixSum[ indicesBlockIndex ];
    index_type_t pos = indices[ indicesBlockIndex ][ offset ];
    if( pos != -1 )
    {
        int dataBlockIndex = aries_acc::binary_search<aries_acc::bounds_upper>( dataBlockSizePrefixSum, dataBlockCount, pos ) - 1;
        return data[ dataBlockIndex ] + ( pos - dataBlockSizePrefixSum[ dataBlockIndex ] ) * perItemSize;
    }
    else
        return nullData;
}
template< typename index_type_t >
ARIES_HOST_DEVICE
int8_t* DoGetData( int i,
                   int8_t** data,
                   int64_t* dataBlockSizePrefixSum,
                   int32_t dataBlockCount,
                   nullable_type< index_type_t >** indices,
                   int64_t* indiceBlockSizePrefixSum,
                   int32_t indiceBlockCount,
                   size_t perItemSize,
                   int8_t* nullData )
{
    int indicesBlockIndex = aries_acc::binary_search<aries_acc::bounds_upper>(
                                indiceBlockSizePrefixSum,
                                indiceBlockCount, i ) - 1;
    int offset = i - indiceBlockSizePrefixSum[ indicesBlockIndex ];
    nullable_type< index_type_t > nullablePos = indices[ indicesBlockIndex ][ offset ];
    if( 0 != nullablePos.flag )
    {
        index_type_t pos = nullablePos.value;
        int dataBlockIndex = aries_acc::binary_search<aries_acc::bounds_upper>( dataBlockSizePrefixSum, dataBlockCount, pos ) - 1;
        return data[ dataBlockIndex ] + ( pos - dataBlockSizePrefixSum[ dataBlockIndex ] ) * perItemSize;
    }
    else
        return nullData;
}

ARIES_HOST_DEVICE_NO_INLINE int8_t* AriesColumnDataIterator::operator[]( int i ) const
    {
        int8_t* result = nullptr;
        if ( m_indices )
        {
            switch ( m_indiceValueType.DataType.ValueType )
            {
            case aries::AriesValueType::INT8:
                if ( m_indiceValueType.HasNull )
                    result = DoGetData( i,
                               m_data, m_dataBlockSizePrefixSum, m_dataBlockCount,
                               ( nullable_type< int8_t >** )m_indices, m_indiceBlockSizePrefixSum, m_indiceBlockCount,
                               m_perItemSize,
                               m_nullData );
                else
                    result = DoGetData( i,
                               m_data, m_dataBlockSizePrefixSum, m_dataBlockCount,
                               m_indices, m_indiceBlockSizePrefixSum, m_indiceBlockCount,
                               m_perItemSize,
                               m_nullData );
                break;
            case aries::AriesValueType::INT16:
                if ( m_indiceValueType.HasNull )
                    result = DoGetData( i,
                               m_data, m_dataBlockSizePrefixSum, m_dataBlockCount,
                               ( nullable_type< int16_t >** )m_indices, m_indiceBlockSizePrefixSum, m_indiceBlockCount,
                               m_perItemSize,
                               m_nullData );
                else
                    result = DoGetData( i,
                               m_data, m_dataBlockSizePrefixSum, m_dataBlockCount,
                               ( int16_t** )m_indices, m_indiceBlockSizePrefixSum, m_indiceBlockCount,
                               m_perItemSize,
                               m_nullData );
                break;
            case aries::AriesValueType::INT32:
                if ( m_indiceValueType.HasNull )
                    result = DoGetData( i,
                               m_data, m_dataBlockSizePrefixSum, m_dataBlockCount,
                               ( nullable_type< int32_t >** )m_indices, m_indiceBlockSizePrefixSum, m_indiceBlockCount,
                               m_perItemSize,
                               m_nullData );
                else
                    result = DoGetData( i,
                               m_data, m_dataBlockSizePrefixSum, m_dataBlockCount,
                               ( int32_t** )m_indices, m_indiceBlockSizePrefixSum, m_indiceBlockCount,
                               m_perItemSize,
                               m_nullData );
                break;
            }
        }
        else
        {
            int dataBlockIndex = aries_acc::binary_search<aries_acc::bounds_upper>( m_dataBlockSizePrefixSum, m_dataBlockCount, i ) - 1;
            result = m_data[ dataBlockIndex ] + ( i - m_dataBlockSizePrefixSum[ dataBlockIndex ] ) * m_perItemSize;
        }
        return result;
    }

END_ARIES_ACC_NAMESPACE
