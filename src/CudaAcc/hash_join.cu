#include "hash_join.hxx"

#include <cuda.h>
#include "AriesEngineUtil.h"
#include "datatypes/aries_types.hxx"
#include "AriesEngine/AriesUtil.h"
#include "utils/murmur_hash.h"

#define HASH_SEED1 0
#define HASH_SEED2 3

#define NOT_FOUND 0xFFFFFFFF
// #define EMPTY_VALUE 0xFFFFFFFF
#define EMPTY_VALUE_LONG 0xFFFFFFFFFFFFFFFF
#define EMPTY_VALUE_SHORT 0xFFFF
#define RETRY_LIMIT 10
#define INVALID_HASH_ID -1

BEGIN_ARIES_ACC_NAMESPACE

static __managed__ __device__ unsigned long long EMPTY_VALUE = EMPTY_VALUE_LONG;

#define GET_EMPTY_VALUE( value_type ) ( *( ( value_type * )( &EMPTY_VALUE ) ) )

#define THREAD_COUNT_PER_BLOCK 256
#define GET_BLOCK_SIZE( count ) div_up( size_t( count ), size_t( THREAD_COUNT_PER_BLOCK ) )

#define PRINT_CUDA_ERROR()                                                                  \
do                                                                                          \
{                                                                                           \
    auto err = cudaGetLastError();                                                       \
    if ( err != cudaSuccess )                                                               \
    {                                                                                       \
        string msg;                                                                         \
        msg.append( __FILE__ ).append(":").append( std::to_string( __LINE__ ) ).append( " cuda error: " ).append( cudaGetErrorString( err ) ); \
        ARIES_EXCEPTION_SIMPLE( ER_UNKNOWN_ERROR, msg.data() );                             \
    }                                                                                       \
} while ( 0 )


template< typename type_t >
__device__ uint32_t insert_key( const type_t* key, size_t len, type_t* table, int table_size )
{
    uint32_t hash = murmur_hash( ( const void* )key, len, HASH_SEED1 ) % table_size;
    type_t* addr = table + hash;

    type_t old = atomicCAS( addr, GET_EMPTY_VALUE( type_t ), *key );
    int retry = 0;
    while ( old != GET_EMPTY_VALUE( type_t ) && retry < RETRY_LIMIT )
    {
        uint32_t hash2 = murmur_hash( ( const void* )key, len, HASH_SEED2 );
        hash = ( hash + hash2 ) % table_size;
        addr = table + hash;
        old = atomicCAS( addr, GET_EMPTY_VALUE( type_t ), *key );
        retry++;
    }

    return old == GET_EMPTY_VALUE( type_t ) ? hash : NOT_FOUND;
}

__device__ void cuda_memcpy( int8_t* dst, int8_t* src, int length )
{
    int i = 0;
    while ( i < length )
    {
        dst[ i ] = src[ i ];
        i++;
    }
}

__device__ int cuda_memcmp( int8_t* left, int8_t* right, int length, bool aligned = true )
{
    int byte_count = length;

    int off = 0;
    if ( aligned )
    {
        int long_count = length / 8;
        int int_count = ( length - ( long_count * 8 ) ) / 4;
        byte_count = length - long_count * 8 - int_count * 4;
        for ( int i = 0; i < long_count; i ++ )
        {
            long left_value =  *( ( long* )( left + off ) );
            long right_value = *( ( long* )( right + off ) );
            if ( left_value > right_value )
            {
                return 1;
            }
            else if ( left_value < right_value )
            {
                return -1;
            }
            off += 8;
        }

        for ( int i = 0; i < int_count; i++ )
        {
            int left_value =  *( ( int* )( left + off ) );
            int right_value = *( ( int* )( right + off ) );

            if ( left_value > right_value )
            {
                return 1;
            }
            else if ( left_value < right_value )
            {
                return -1;
            }
            off += 4;
        }
    }
    for ( int i = 0; i < byte_count; i++ )
    {
        int8_t left_value =  *( left + off );
        int8_t right_value = *( right + off );

        if ( left_value > right_value )
        {
            return 1;
        }
        else if ( left_value < right_value )
        {
            return -1;
        }
        off += 1;
    }

    return 0;
}

__device__ uint2 insert_key( ColumnDataIterator* inputs, int inputs_count, int index, flag_type_t* flags_table, int table_size )
{
    uint32_t hash = 0;
    flag_type_t old = 0;
    flag_type_t emptyValue  = GET_EMPTY_VALUE( flag_type_t );

    int retry = 0;
    do
    {
        auto seed = retry == 0 ? HASH_SEED1 : HASH_SEED2;
        for ( int i = 0; i < inputs_count; i++ )
        {
            const auto& input = inputs[ i ];
            int8_t* data = input.GetData( index );
            int perItemSize = input.m_perItemSize;
            if( input.m_hasNull )
            {
                if( *data == 0 )
                    return make_uint2( 0, NOT_FOUND );
                ++data;
                --perItemSize;
            }
            auto hash2 = murmur_hash( data, perItemSize, seed, false );
            hash = ( hash + hash2 ) % table_size;

            auto* addr = flags_table + hash;

            old = atomicCAS( addr, emptyValue, 0x00 );

            if ( old == emptyValue )
            {
                break;
            }
        }
        retry ++;
    } while ( old != emptyValue && retry < RETRY_LIMIT );

    if ( old == emptyValue )
    {
        return make_uint2( 1, hash );
    }

    return make_uint2( 1, NOT_FOUND );
}

__device__
uint32_t search_key( const AriesHashTableMultiKeys* table,
                     ColumnDataIterator* inputs,
                     int inputs_count,
                     int index )
{
    uint32_t hash = 0;
    int retry = 0;
    do
    {
        auto seed = retry == 0 ? HASH_SEED1 : HASH_SEED2;
        for ( int i = 0; i < inputs_count; i++ )
        {
            const auto& input = inputs[ i ];
            uint32_t hash2;
            if( input.m_hasNull )
            {
                int8_t* pValue = ( int8_t* )( input.GetData( index ) );
                if( *pValue )
                    hash2 = murmur_hash( pValue + 1, input.m_perItemSize - 1, seed, false );
                else
                    return NOT_FOUND;
            }
            else 
                hash2 = murmur_hash( ( const void* )input.GetData( index ), input.m_perItemSize, seed, false );
            hash = ( hash + hash2 ) % table->table_size;

            auto* addr = ( flag_type_t* )( table->flags_table ) + hash;

            if ( *addr == GET_EMPTY_VALUE( flag_type_t ) )
            {
                return NOT_FOUND;
            }

            bool match = true;
            for ( int j = 0; j < inputs_count; j++ )
            {
                const auto& input = inputs[ j ];
                if( input.m_hasNull )
                {
                    int8_t* pValue = ( int8_t* )( input.GetData( index ) );
                    if( *pValue )
                    {
                        if ( 0 != cuda_memcmp( pValue + 1, table->keys_array[ j ] + ( hash * table->keys_length[ j ] ), table->keys_length[ j ], false ) )
                        {
                            match = false;
                            break;
                        }
                    }
                    else
                        return NOT_FOUND;
                }
                else 
                {
                    if ( 0 != cuda_memcmp( input.GetData( index ), table->keys_array[ j ] + ( hash * table->keys_length[ j ] ), table->keys_length[ j ], false ) )
                    {
                        match = false;
                        break;
                    }
                }
            }

            if ( match )
            {
                return hash;
            }
        }
        retry ++;
    } while ( retry < RETRY_LIMIT );

    return NOT_FOUND;
}


template< typename type_a, typename type_b >
__device__ uint32_t search_key( const type_a& key, int len, type_b* table, int table_size )
{
    uint32_t hash = murmur_hash( ( void* )&key, sizeof( type_b ), HASH_SEED1 ) % table_size;
    type_b value = *( table + hash );

    if ( value == key )
    {
        return hash;
    }
    else if ( value == GET_EMPTY_VALUE( type_b ) )
    {
        return NOT_FOUND;
    }

    int retry_count = 0;
    uint32_t hash2 = murmur_hash( ( void* )&key, sizeof( type_b ), HASH_SEED2 );
    do
    {
        hash = ( hash + hash2 ) % table_size;
        value = *( table + hash );
    } while ( retry_count++ < RETRY_LIMIT && value != GET_EMPTY_VALUE( type_b ) && key != value );

    if ( value == key )
    {
        return hash;
    }
    else
    {
        return NOT_FOUND;
    }
}

template< typename type_t >
__global__ void init_hash_table( type_t* table, size_t count )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for ( int i = tid; i < count; i += stride )
    {
        table[ i ] = GET_EMPTY_VALUE( type_t );
    }
}

__global__ void init_hash_table( unsigned short* table, size_t count )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for ( int i = tid; i < count; i += stride )
    {
       table[ i ] = EMPTY_VALUE_SHORT;
    }
}

template< typename type_t >
__global__ void init_flags( type_t* data, size_t count, type_t value )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for ( int i = tid; i < count; i += stride )
    {
       data[ i ] = value;
    }
}


template< typename type_a, typename type_b, typename index_type_t >
void
__global__ do_join( type_a* keys_table,
                    int32_t* ids_table,
                    int32_t table_size,
                    type_a* bad_values_array,
                    int32_t* bad_ids,
                    int32_t bad_count,
                    type_b** inputs,
                    int64_t* block_size_prefix_sum, 
                    size_t block_count,
                    index_type_t* indices,
                    size_t count,
                    int* left_indices,
                    int* right_indices,
                    int32_t* left_output, 
                    int32_t* right_output, 
                    int32_t* output_count )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for ( int i = tid; i < count; i += stride )
    {
        int pos = indices != nullptr ? indices[ i ] : i;
        if( pos == NULL_INDEX )
            continue;
        int blockIndex = binary_search<bounds_upper>( block_size_prefix_sum, block_count, pos ) - 1;
        type_b* addr = inputs[ blockIndex ] + pos - block_size_prefix_sum[ blockIndex ];

        uint32_t hash = NOT_FOUND;
        if( *addr != GET_EMPTY_VALUE( type_b ) )
            hash = search_key( *addr, sizeof( type_b ), keys_table, table_size );

        int left_id = EMPTY_VALUE;
        if ( hash != NOT_FOUND )
        {
            left_id = ids_table[ hash ];
        }
        else
        {
            for ( int j = 0; j < bad_count; j ++ )
            {
                if ( bad_values_array[ j ] == *addr )
                {
                    left_id = bad_ids[ j ];
                    break;
                }
            }
        }

        if ( left_id != EMPTY_VALUE )
        {
            int off = atomicAdd( output_count, 1 );
            right_output[ off ] = right_indices ? right_indices[ i ] : i;
            left_output[ off ] = left_indices ? left_indices[ left_id ] : left_id;
        }
    }
}

template< typename type_a, typename type_b, typename index_type_t >
static void
__global__ do_left_join( type_a* keys_table,
                        int32_t* ids_table,
                        int32_t table_size,
                        type_a* bad_values_array,
                        int32_t* bad_ids,
                        int32_t bad_count,
                        type_b** inputs,
                        int64_t* block_size_prefix_sum, 
                        size_t block_count,
                        index_type_t* indices,
                        size_t count,
                        int32_t* left_output, 
                        int32_t* right_output, 
                        int32_t* output_count,
                        int* left_matched_count )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for ( int i = tid; i < count; i += stride )
    {
        int pos = indices != nullptr ? indices[ i ] : i;
        if( pos == NULL_INDEX )
            continue;
        int blockIndex = binary_search<bounds_upper>( block_size_prefix_sum, block_count, pos ) - 1;
        type_b* addr = inputs[ blockIndex ] + pos - block_size_prefix_sum[ blockIndex ];

        uint32_t hash = NOT_FOUND;
        if( *addr != GET_EMPTY_VALUE( type_b ) )
            hash = search_key( *addr, sizeof( type_b ), keys_table, table_size );

        int left_id = EMPTY_VALUE;
        if ( hash != NOT_FOUND )
        {
            left_id = ids_table[ hash ];
        }
        else
        {
            for ( int j = 0; j < bad_count; j ++ )
            {
                if ( bad_values_array[ j ] == *addr )
                {
                    left_id = bad_ids[ j ];
                    break;
                }
            }
        }

        if ( left_id != EMPTY_VALUE )
        {
            int off = atomicAdd( output_count, 1 );
            right_output[ off ] = i;
            left_output[ off ] = left_id;
            atomicAdd( left_matched_count + left_id, 1 );
        }
    }
}

template< typename type_a, typename type_b, typename index_type_t >
static void
__global__ do_left_join_with_nullable( type_a* keys_table,
                                        int32_t* ids_table,
                                        int32_t table_size,
                                        type_a* bad_values_array,
                                        int32_t* bad_ids,
                                        int32_t bad_count,
                                        type_b** inputs,
                                        int64_t* block_size_prefix_sum, 
                                        size_t block_count,
                                        index_type_t* indices,
                                        size_t count,
                                        int32_t* left_output, 
                                        int32_t* right_output, 
                                        int32_t* output_count,
                                        int* left_matched_count )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for( int i = tid; i < count; i += stride )
    {
        int pos = indices != nullptr ? indices[ i ] : i;
        if( pos == NULL_INDEX )
            continue;
        int blockIndex = binary_search<bounds_upper>( block_size_prefix_sum, block_count, pos ) - 1;
        nullable_type< type_b > value = *( ( nullable_type< type_b >* )inputs[ blockIndex ] + pos - block_size_prefix_sum[ blockIndex ] );
        if( value.is_null() )
        {
            continue;
        }

        uint32_t hash = NOT_FOUND;
        if( value.value != GET_EMPTY_VALUE( type_b ) )
            hash = search_key( value.value, sizeof( type_b ), keys_table, table_size );

        int left_id = EMPTY_VALUE;
        if( hash != NOT_FOUND )
        {
            left_id = ids_table[ hash ];
        }
        else
        {
            for( int j = 0; j < bad_count; j ++ )
            {
                if( bad_values_array[ j ] == value.value )
                {
                    left_id = bad_ids[ j ];
                    break;
                }
            }
        }

        if( left_id != EMPTY_VALUE )
        {
            int off = atomicAdd( output_count, 1 );
            right_output[ off ] = i;
            left_output[ off ] = left_id;
            atomicAdd( left_matched_count + left_id, 1 );
        }
    }
}

template< typename type_a, typename type_b, typename index_type_t >
void __global__ do_simple_half_join_left_as_hash( type_a* keys_table,
        int32_t* ids_table,
        int32_t table_size,
        type_a* bad_values_array,
        int32_t* bad_ids,
        int32_t bad_count,
        type_b** inputs,
        int64_t* block_size_prefix_sum,
        size_t block_count,
        index_type_t* indices,
        size_t count,
        bool is_semi_join,
        int32_t* output_flags )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for( int i = tid; i < count; i += stride )
    {
        int pos = indices != nullptr ? indices[i] : i;
        if( pos == NULL_INDEX )
            continue;
        int blockIndex = binary_search< bounds_upper >( block_size_prefix_sum, block_count, pos ) - 1;
        type_b* addr = inputs[blockIndex] + pos - block_size_prefix_sum[blockIndex];

        uint32_t hash = NOT_FOUND;
        if( *addr != GET_EMPTY_VALUE( type_b ) )
            hash = search_key( *addr, sizeof(type_b), keys_table, table_size );

        int left_id = EMPTY_VALUE;
        if( hash != NOT_FOUND )
        {
            left_id = ids_table[hash];
        }
        else
        {
            for( int j = 0; j < bad_count; j++ )
            {
                if( bad_values_array[j] == *addr )
                {
                    left_id = bad_ids[j];
                    break;
                }
            }
        }
        
        if( left_id != EMPTY_VALUE )
        {
            if( is_semi_join )
                atomicExch( output_flags + left_id, 1 );
            else
                atomicExch( output_flags + left_id, 0 );
        }
    }
}

template< typename type_a, typename type_b, typename index_type_t >
void __global__ do_simple_half_join_left_as_hash_with_nullable( type_a* keys_table,
        int32_t* ids_table,
        int32_t table_size,
        type_a* bad_values_array,
        int32_t* bad_ids,
        int32_t bad_count,
        type_b** inputs,
        int64_t* block_size_prefix_sum,
        size_t block_count,
        index_type_t* indices,
        size_t count,
        bool is_semi_join,
        int32_t* has_null_flag,
        int32_t* output_flags )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for( int i = tid; i < count; i += stride )
    {
        int left_id = EMPTY_VALUE;
        int pos = indices != nullptr ? indices[i] : i;
        if( pos != NULL_INDEX )
        {
            int blockIndex = binary_search< bounds_upper >( block_size_prefix_sum, block_count, pos ) - 1;
            nullable_type< type_b >* addr = ( nullable_type< type_b >* )inputs[blockIndex] + pos - block_size_prefix_sum[blockIndex];

            if ( !addr->is_null() )
            {
                uint32_t hash = NOT_FOUND;
                if( addr->value != GET_EMPTY_VALUE( type_b ) )
                    hash = search_key( addr->value, sizeof(type_b), keys_table, table_size );
    
                if( hash != NOT_FOUND )
                {
                    left_id = ids_table[hash];
                }
                else
                {
                    for( int j = 0; j < bad_count; j++ )
                    {
                        if( bad_values_array[j] == addr->value )
                        {
                            left_id = bad_ids[j];
                            break;
                        }
                    }
                }
            }
            else
                atomicExch( has_null_flag, 1 );
        }
        else
            atomicExch( has_null_flag, 1 );


        if( left_id != EMPTY_VALUE )
        {
            if( is_semi_join )
                atomicExch( output_flags + left_id, 1 );
            else
                atomicExch( output_flags + left_id, 0 );
        }
    }
}

template< typename type_a, typename type_b, typename index_type_t >
void __global__ do_simple_half_join_right_as_hash( type_a* keys_table,
                        int32_t* ids_table,
                        int32_t table_size,
                        type_a* bad_values_array,
                        int32_t* bad_ids,
                        int32_t bad_count,
                        type_b** inputs,
                        int64_t* block_size_prefix_sum,
                        size_t block_count,
                        index_type_t* indices,
                        size_t count,
                        bool is_semi_join,
                        bool use_indices_for_output,
                        int32_t* output_count,
                        int32_t* output_flags )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for( int i = tid; i < count; i += stride )
    {
        int left_id = EMPTY_VALUE;
        int pos = indices != nullptr ? indices[i] : i;
        if( pos != NULL_INDEX )
        {
            int blockIndex = binary_search< bounds_upper >( block_size_prefix_sum, block_count, pos ) - 1;
            type_b* addr = inputs[blockIndex] + pos - block_size_prefix_sum[blockIndex];
    
            uint32_t hash = NOT_FOUND;
            if( *addr != GET_EMPTY_VALUE( type_b ) )
                hash = search_key( *addr, sizeof(type_b), keys_table, table_size );

            if( hash != NOT_FOUND )
            {
                left_id = ids_table[hash];
            }
            else
            {
                for( int j = 0; j < bad_count; j++ )
                {
                    if( bad_values_array[j] == *addr )
                    {
                        left_id = bad_ids[j];
                        break;
                    }
                }
            }
        }

        bool found = is_semi_join ? left_id != EMPTY_VALUE : left_id == EMPTY_VALUE;
        if( found )
        {
            if( use_indices_for_output )
            {
                int offset = atomicAdd( output_count, 1 );
                output_flags[offset] = pos;
            }
            else
                output_flags[i] = 1;
        }
    }
}

template< typename type_a, typename type_b, typename index_type_t >
void __global__ do_simple_half_join_right_as_hash_with_nullable( type_a* keys_table,
                        int32_t* ids_table,
                        int32_t table_size,
                        type_a* bad_values_array,
                        int32_t* bad_ids,
                        int32_t bad_count,
                        type_b** inputs,
                        int64_t* block_size_prefix_sum,
                        size_t block_count,
                        index_type_t* indices,
                        size_t count,
                        bool is_semi_join,
                        bool isNotIn,
                        bool use_indices_for_output,
                        int32_t* output_count,
                        int32_t* output_flags )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for( int i = tid; i < count; i += stride )
    {
        int left_id = EMPTY_VALUE;
        int pos = indices != nullptr ? indices[i] : i;
        if( pos != NULL_INDEX )
        {
            int blockIndex = binary_search< bounds_upper >( block_size_prefix_sum, block_count, pos ) - 1;
            nullable_type< type_b >* addr = ( nullable_type< type_b >* )inputs[blockIndex] + pos - block_size_prefix_sum[blockIndex];
            
            if( !addr->is_null() )
            {
                uint32_t hash = NOT_FOUND;
                if( addr->value != GET_EMPTY_VALUE( type_b ) )
                    hash = search_key( addr->value, sizeof(type_b), keys_table, table_size );

                if( hash != NOT_FOUND )
                {
                    left_id = ids_table[hash];
                }
                else
                {
                    for( int j = 0; j < bad_count; j++ )
                    {
                        if( bad_values_array[j] == addr->value )
                        {
                            left_id = bad_ids[j];
                            break;
                        }
                    }
                }
            }
            else
            {
                if( isNotIn )
                    continue;
            }
        }
        else
        {
            if( isNotIn )
                continue;
        }

        bool found = is_semi_join ? left_id != EMPTY_VALUE : left_id == EMPTY_VALUE;
        if( found )
        {
            if( use_indices_for_output )
            {
                int offset = atomicAdd( output_count, 1 );
                output_flags[offset] = pos;
            }
            else
                output_flags[i] = 1;
        }
    }
}

template< typename type_a, typename type_b, typename index_type_t >
void
__global__ do_join_with_nullable( type_a* keys_table,
                    int32_t* ids_table,
                    int32_t table_size,
                    type_a* bad_values_array,
                    int32_t* bad_ids,
                    int32_t bad_count,
                    HashIdType null_value_index,
                    type_b** inputs,
                    int64_t* block_size_prefix_sum, 
                    size_t block_count,
                    index_type_t* indices,
                    size_t count,
                    int* left_indices,
                    int* right_indices,
                    int32_t* left_output, 
                    int32_t* right_output, 
                    int32_t* output_count )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for ( int i = tid; i < count; i += stride )
    {
        int pos = indices != nullptr ? indices[ i ] : i;
        if( pos == NULL_INDEX )
            continue;
        int blockIndex = binary_search<bounds_upper>( block_size_prefix_sum, block_count, pos ) - 1;
        nullable_type< type_b > value = *( ( nullable_type< type_b >* )inputs[ blockIndex ] + pos - block_size_prefix_sum[ blockIndex ] );

        if ( value.is_null() )
        {
            continue;
        }

        uint32_t hash = NOT_FOUND;
        if( value.value != GET_EMPTY_VALUE( type_b ) )
            hash = search_key( value.value, sizeof(type_b), keys_table, table_size );

        int left_id = EMPTY_VALUE;
        if ( hash != NOT_FOUND )
        {
            left_id = ids_table[ hash ];
        }
        else
        {
            for ( int j = 0; j < bad_count; j ++ )
            {
                if ( bad_values_array[ j ] == value.value )
                {
                    left_id = bad_ids[ j ];
                    break;
                }
            }
        }

        if ( left_id != EMPTY_VALUE )
        {
            int off = atomicAdd( output_count, 1 );
            right_output[ off ] = right_indices ? right_indices[ i ] : i;
            left_output[ off ] = left_indices ? left_indices[ left_id ] : left_id;
        }
    }
}

__device__ HashIdType
search_key( AriesHashTableMultiKeys* table, ColumnDataIterator* inputs, int index )
{
    auto keys_count = table->count;

    // if ( keys_count > 1 )
    {
        auto hash = search_key( table, inputs, keys_count, index );
        if ( hash == NOT_FOUND )
        {
            for ( int i = 0; i < table->bad_count; i++ )
            {
                bool found = true;
                for ( int j = 0; j < keys_count; j++ )
                {
                    const auto& input = inputs[ j ];
                    if( input.m_hasNull )
                    {
                        int8_t* pValue = ( int8_t* )( input.GetData( index ) );
                        if( *pValue )
                        {
                            if ( 0 != cuda_memcmp( pValue + 1, ( int8_t* )( table->bad_values_array[ j ] ) + i * table->keys_length[ j ], table->keys_length[ j ], false ) )
                            {
                                found = false;
                                break;
                            }
                        }
                        else
                            return INVALID_HASH_ID;
                    }
                    else 
                    {
                        if ( 0 != cuda_memcmp( input.GetData( index ), ( int8_t* )( table->bad_values_array[ j ] ) + i * table->keys_length[ j ], table->keys_length[ j ], false ) )
                        {
                            found = false;
                            break;
                        }
                    }
                }

                if ( found )
                {
                    return table->bad_ids[ i ];
                }
            }
        }
        else
        {
            return table->ids[ hash ];
        }
    }
    #if 0
    else
    {
        uint32_t hash = NOT_FOUND;
        bool hasNull = inputs[ 0 ].m_hasNull;
        if ( table->keys_length[ 0 ] == 4 )
        {
            int value;
            if( hasNull )
            {
                nullable_type< int >* pValue = ( nullable_type< int >* )( inputs[ 0 ].GetData( index ) );
                if( pValue->flag )
                    value = pValue->value;
                else
                    return INVALID_HASH_ID;
            }
            else 
                value = *( int* )( inputs[ 0 ].GetData( index ) );
            hash = search_key( value, 4, ( int* )( table->keys_array[ 0 ] ), table->table_size );

            if ( hash != NOT_FOUND )
            {
                return table->ids[ hash ];
            }
            else
            {
                for ( int i = 0; i < table->bad_count; i++ )
                {
                    int key_value = *( int* )( table->bad_values_array[ 0 ] + i * sizeof( int ) );
                    if ( key_value == value )
                    {
                        return table->bad_ids[ i ];
                    }
                }
            }
        }
        else
        {
            unsigned long long value;
            if( hasNull )
            {
                nullable_type< unsigned long long >* pValue = ( nullable_type< unsigned long long >* )( inputs[ 0 ].GetData( index ) );
                if( pValue->flag )
                    value = pValue->value;
                else
                    return INVALID_HASH_ID;
            }
            else
                value = *( unsigned long long* )( inputs[ 0 ].GetData( index ) );
            hash = search_key( value, 8, ( unsigned long long* )( table->keys_array[ 0 ] ), table->table_size );

            if ( hash != NOT_FOUND )
            {
                return table->ids[ hash ];
            }
            else
            {
                for ( int i = 0; i < table->bad_count; i++ )
                {
                    unsigned long long key_value = *( unsigned long long* )( table->bad_values_array[ 0 ] + i * sizeof( int ) );
                    if ( key_value == value )
                    {
                        return table->bad_ids[ i ];
                    }
                }
            }
        }
    }
    #endif

    return INVALID_HASH_ID;
}


template< typename type_a, typename type_b >
__device__ HashIdType 
search_key( type_a value,
            type_b* table_values,
            HashIdType* table_ids,
            int table_size,
            type_b* bad_values_array,
            HashIdType* bad_ids,
            int bad_count )
{
    uint32_t hash = murmur_hash( ( void* )&value, sizeof( type_b ), HASH_SEED1 ) % table_size;
    type_b hash_value = *( table_values + hash );

    if ( value == hash_value )
    {
        return table_ids[ hash ];
    }
    else if ( hash_value == GET_EMPTY_VALUE( type_b ) )
    {
        return INVALID_HASH_ID;
    }

    int8_t retry_count = 0;
    uint32_t hash2 = murmur_hash( ( void* )&value, sizeof( type_b ), HASH_SEED2 );
    do
    {
        hash = ( hash + hash2 ) % table_size;
        hash_value = *( table_values + hash );
    } while ( retry_count++ < RETRY_LIMIT && hash_value != GET_EMPTY_VALUE( type_b ) && hash_value != value );

    if ( value == hash_value )
    {
        return table_ids[ hash ];
    }
    else
    {
        for ( int j = 0; j < bad_count; j ++ )
        {
            if  (bad_values_array[ j ] == value )
            {
                return bad_ids[ j ];
            }
        }

        return INVALID_HASH_ID;
    }
}

template< typename type_t, typename index_type_t > void
__global__ build_hash_table_with_nullable( type_t** inputs,
                             int64_t* block_size_prefix_sum,
                             size_t block_count,
                             index_type_t* indices,
                             size_t count,
                             type_t* keys_table,
                             HashIdType* ids_table,
                             int32_t table_size,
                             type_t* bad_values_array,
                             HashIdType* bad_ids,
                             int32_t bad_capacity,
                             int32_t* bad_count,
                             HashIdType* null_value_index,
                             context_t& context )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    type_t tmp_value;

    for ( int i = tid; i < count; i += stride )
    {
        int pos = indices == nullptr ? i : indices[ i ];
        int blockIndex = binary_search<bounds_upper>( block_size_prefix_sum, block_count, pos ) - 1;
        const int8_t* flag = ( int8_t* )( inputs[ blockIndex ] ) + ( pos - block_size_prefix_sum[ blockIndex ] ) * ( sizeof( type_t ) + 1 );
        if ( *flag == 0 )
        {
            *null_value_index = i;
            continue;
        }

        /**
         * 似乎不能将一个“未对齐”的地址强制转换为 int，会报 misaligned address 错误，
         * 所以这里先声明一个 tmp_value，然后将 内存内容拷贝到 tmp_value 中。
         */
        memcpy( &tmp_value, flag + 1, sizeof( type_t ) );
        const type_t* addr = &tmp_value;
        uint32_t hash = NOT_FOUND;
        if( *addr != GET_EMPTY_VALUE( type_t ) )
            hash = insert_key( addr, sizeof( type_t ), keys_table, table_size );
        if ( hash != NOT_FOUND )
        {
            ids_table[ hash ] = i;
        }
        else
        {
            int old = atomicAdd( bad_count, 1 );
            bad_values_array[ old ] = *addr;
            bad_ids[ old ] = i;
        }
    }
}

template< typename type_t, typename index_type_t > void
__global__ build_hash_table( type_t** inputs,
                             int64_t* block_size_prefix_sum,
                             size_t block_count,
                             index_type_t* indices,
                             size_t count,
                             type_t* keys_table,
                             HashIdType* ids_table,
                             int32_t table_size,
                             type_t* bad_values_array,
                             HashIdType* bad_ids,
                             int32_t bad_capacity,
                             int32_t* bad_count,
                             context_t& context )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for ( int i = tid; i < count; i += stride )
    {
        int pos = indices == nullptr ? i : indices[ i ];
        int blockIndex = binary_search<bounds_upper>( block_size_prefix_sum, block_count, pos ) - 1;
        const type_t* addr = inputs[ blockIndex ] + pos - block_size_prefix_sum[ blockIndex ];
        uint32_t hash = NOT_FOUND;
        if( *addr != GET_EMPTY_VALUE( type_t ) )
            hash = insert_key( addr, sizeof( type_t ), keys_table, table_size );
        if ( hash != NOT_FOUND )
        {
            ids_table[ hash ] = i;
        }
        else
        {
            int old = atomicAdd( bad_count, 1 );
            if( old < bad_capacity )
            {
                bad_values_array[ old ] = *addr;
                bad_ids[ old ] = i;
            }
        }
    }
}

template< typename type_t >
HashTable< type_t > build_hash_table( type_t** inputs,
                                      int64_t* block_size_prefix_sum,
                                      size_t block_count,
                                      bool nullable,
                                      int8_t* indices,
                                      AriesValueType indiceValueType,
                                      size_t count,
                                      context_t& context )
{
    int32_t table_size = HASH_SCALE_FACTOR * count;
    mem_t< type_t > keys_table( table_size );
    mem_t< HashIdType > ids_table( table_size );

    auto block_size = div_up( table_size, 256 );
    int bad_capacity = max( 100, ( int )( HASH_BAD_SCALE_FACTOR * count ) );
    managed_mem_t< int32_t > bad_count( 1, context );
    managed_mem_t< HashIdType > null_value_index( 1, context );
    *bad_count.data() = 0;
    do
    {
        init_hash_table<<< block_size, 256 >>>( keys_table.data(), table_size );
        cudaDeviceSynchronize();
        if( bad_capacity < *bad_count.data() )
            bad_capacity = ( int )( *bad_count.data() * HASH_SCALE_FACTOR );
        mem_t< type_t > bad_values_array( bad_capacity );
        mem_t< HashIdType > bad_ids( bad_capacity );
        
        *( null_value_index.data() ) = -1;
        *bad_count.data() = 0;
    
        if ( indices )
        {
            switch ( indiceValueType )
            {
            case aries::AriesValueType::INT8:
                if ( nullable )
                {
                    build_hash_table_with_nullable<<< block_size, 256 >>>( inputs,
                                                         block_size_prefix_sum,
                                                         block_count,
                                                         indices,
                                                         count,
                                                         keys_table.data(),
                                                         ids_table.data(),
                                                         table_size,
                                                         bad_values_array.data(),
                                                         bad_ids.data(),
                                                         bad_capacity,
                                                         bad_count.data(),
                                                         null_value_index.data(),
                                                         context );
                }
                else
                {
                    build_hash_table<<< block_size, 256 >>>( inputs,
                                                         block_size_prefix_sum,
                                                         block_count,
                                                         indices,
                                                         count,
                                                         keys_table.data(),
                                                         ids_table.data(),
                                                         table_size,
                                                         bad_values_array.data(),
                                                         bad_ids.data(),
                                                         bad_capacity,
                                                         bad_count.data(),
                                                         context );
                }
                break;
            case aries::AriesValueType::INT16:
                if ( nullable )
                {
                    build_hash_table_with_nullable<<< block_size, 256 >>>( inputs,
                                                         block_size_prefix_sum,
                                                         block_count,
                                                         ( int16_t* )indices,
                                                         count,
                                                         keys_table.data(),
                                                         ids_table.data(),
                                                         table_size,
                                                         bad_values_array.data(),
                                                         bad_ids.data(),
                                                         bad_capacity,
                                                         bad_count.data(),
                                                         null_value_index.data(),
                                                         context );
                }
                else
                {
                    build_hash_table<<< block_size, 256 >>>( inputs,
                                                         block_size_prefix_sum,
                                                         block_count,
                                                         ( int16_t* )indices,
                                                         count,
                                                         keys_table.data(),
                                                         ids_table.data(),
                                                         table_size,
                                                         bad_values_array.data(),
                                                         bad_ids.data(),
                                                         bad_capacity,
                                                         bad_count.data(),
                                                         context );
                }
                break;
            case aries::AriesValueType::INT32:
                if ( nullable )
                {
                    build_hash_table_with_nullable<<< block_size, 256 >>>( inputs,
                                                         block_size_prefix_sum,
                                                         block_count,
                                                         ( int32_t* )indices,
                                                         count,
                                                         keys_table.data(),
                                                         ids_table.data(),
                                                         table_size,
                                                         bad_values_array.data(),
                                                         bad_ids.data(),
                                                         bad_capacity,
                                                         bad_count.data(),
                                                         null_value_index.data(),
                                                         context );
                }
                else
                {
                    build_hash_table<<< block_size, 256 >>>( inputs,
                                                         block_size_prefix_sum,
                                                         block_count,
                                                         ( int32_t* )indices,
                                                         count,
                                                         keys_table.data(),
                                                         ids_table.data(),
                                                         table_size,
                                                         bad_values_array.data(),
                                                         bad_ids.data(),
                                                         bad_capacity,
                                                         bad_count.data(),
                                                         context );
                }
                break;
            }
        }
        else if ( nullable )
        {
            build_hash_table_with_nullable<<< block_size, 256 >>>( inputs,
                                                     block_size_prefix_sum, 
                                                     block_count,
                                                     indices,
                                                     count,
                                                     keys_table.data(), 
                                                     ids_table.data(), 
                                                     table_size, 
                                                     bad_values_array.data(),
                                                     bad_ids.data(), 
                                                     bad_capacity, 
                                                     bad_count.data(),
                                                     null_value_index.data(),
                                                     context );
        }
        else
        {
            build_hash_table<<< block_size, 256 >>>( inputs,
                                                     block_size_prefix_sum, 
                                                     block_count,
                                                     indices,
                                                     count,
                                                     keys_table.data(), 
                                                     ids_table.data(), 
                                                     table_size, 
                                                     bad_values_array.data(),
                                                     bad_ids.data(), 
                                                     bad_capacity, 
                                                     bad_count.data(),
                                                     context );
        }
        cudaDeviceSynchronize();
        PRINT_CUDA_ERROR();

        if( *bad_count.data() <= bad_capacity )
            return { keys_table.release_data(), ids_table.release_data(), table_size, bad_values_array.release_data(), bad_ids.release_data(), *bad_count.data(), *null_value_index.data() };
    } while( bad_capacity < *bad_count.data() );
    ARIES_ASSERT( 0, "build hash table error" );
    return HashTable< type_t >();
}

__global__ void build_hash_table( ColumnDataIterator* inputs,
                                  flag_type_t* flags_table,
                                  int input_count,
                                  size_t total_row_count,
                                  int8_t** keys_array,
                                  HashIdType* ids_table,
                                  int32_t table_size,
                                  int8_t** bad_values_array,
                                  HashIdType* bad_ids,
                                  int32_t bad_capacity,
                                  int32_t* bad_count )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for ( int i = tid; i < total_row_count; i += stride )
    {
        auto ret = insert_key( inputs, input_count, i, flags_table, table_size );
        auto hash = ret.y;
        if ( hash == NOT_FOUND )
        {
            //仅处理非null值
            if( ret.x == 1 )
            {
                int old = atomicAdd( bad_count, 1 );
                if( old < bad_capacity )
                {
                    for ( int j = 0; j < input_count; j++ )
                    {
                        const auto& input = inputs[ j ];
                        int8_t* data = input.GetData( i );
                        int perItemSize = input.m_perItemSize;
                        if( input.m_hasNull )
                        {
                            ++data;
                            --perItemSize;
                        }
                        cuda_memcpy( bad_values_array[ j ] + old * perItemSize, data, perItemSize );
                    }
                    bad_ids[ old ] = i;
                }
            }
        }
        else
        {
            ids_table[ hash ] = i;
            for ( int j = 0; j < input_count; j++ )
            {
                const auto& input = inputs[ j ];
                int8_t* data = input.GetData( i );
                int perItemSize = input.m_perItemSize;
                if( input.m_hasNull )
                {
                    ++data;
                    --perItemSize;
                }
                cuda_memcpy( keys_array[ j ] + hash * perItemSize, data, perItemSize );
            }
        }
    }
}

AriesHashTableUPtr build_hash_table( ColumnDataIterator* input, size_t total_row_count, AriesValueType value_type, context_t& context )
{
    AriesHashTableUPtr hash_table = std::make_unique< AriesHashTable >();
    switch ( value_type )
    {
        case AriesValueType::INT8:
        case AriesValueType::INT16:
        case AriesValueType::INT32:
        case AriesValueType::UINT8:
        case AriesValueType::UINT16:
        case AriesValueType::UINT32:
        {
            auto table = build_hash_table( ( int32_t** ) input->m_data,
                              input->m_blockSizePrefixSum,
                              input->m_blockCount,
                              input->m_hasNull,
                              input->m_indices,
                              input->m_indiceValueType,
                              total_row_count,
                              context );
            hash_table->Keys = table.keys;
            hash_table->Ids = table.ids;
            hash_table->TableSize = table.table_size;
            hash_table->BadValues = table.bad_values;
            hash_table->BadIds = table.bad_ids;
            hash_table->BadCount = table.bad_count;
            hash_table->NullValueIndex = table.null_value_index;
            break;
        }

        case AriesValueType::INT64:
        case AriesValueType::UINT64:
        {
            auto table = build_hash_table( ( unsigned long long** ) input->m_data,
                    input->m_blockSizePrefixSum,
                    input->m_blockCount,
                    input->m_hasNull,
                    input->m_indices,
                    input->m_indiceValueType,
                    total_row_count,
                    context );
            hash_table->Keys = table.keys;
            hash_table->Ids = table.ids;
            hash_table->TableSize = table.table_size;
            hash_table->BadValues = table.bad_values;
            hash_table->BadIds = table.bad_ids;
            hash_table->BadCount = table.bad_count;
            hash_table->NullValueIndex = table.null_value_index;
            break;
        }

        default:
            return nullptr;
    }
    hash_table->ValueType = value_type;
    hash_table->HashRowCount = total_row_count;
    return hash_table;
}

AriesHashTableMultiKeysUPtr build_hash_table( ColumnDataIterator* inputs, int input_count, size_t total_row_count, context_t& context )
{
    size_t table_size = HASH_SCALE_FACTOR * total_row_count;
    int bad_capacity = max( 100, ( int )( HASH_BAD_SCALE_FACTOR * total_row_count ) );

    AriesManagedArray< int8_t* > keys_array( input_count );
    AriesManagedArray< int8_t* > bad_values_array( input_count );
    AriesManagedArray< int > keys_length( input_count );
    mem_t< flag_type_t > flags_table( table_size );
    mem_t< HashIdType > ids_table( table_size );
    managed_mem_t< int > bad_count( 1, context );
    *bad_count.data() = 0;
    auto block_size = div_up( table_size, 256ul );

    for ( int i = 0; i < input_count; i++ )
    {
        const auto& input = inputs[ i ];
        int perItemSize = input.m_perItemSize;
        if( input.m_hasNull )
            --perItemSize;
        mem_t< int8_t > keys( perItemSize * table_size );
        keys_array[ i ] = keys.release_data();
        keys_length[ i ] = perItemSize;
    }
    keys_array.PrefetchToGpu();
    keys_length.PrefetchToGpu();

    do
    {
        init_hash_table<<< block_size, 256 >>>( flags_table.data(), table_size );
        cudaDeviceSynchronize();
        if( bad_capacity < *bad_count.data() )
            bad_capacity = ( int )( *bad_count.data() * HASH_SCALE_FACTOR );
    
        for ( int i = 0; i < input_count; i++ )
        {
            const auto& input = inputs[ i ];
            int perItemSize = input.m_perItemSize;
            if( input.m_hasNull )
                --perItemSize;
            mem_t< int8_t > values( perItemSize * bad_capacity );
            bad_values_array[ i ] = values.release_data();
        }
        mem_t< HashIdType > bad_ids( bad_capacity );
        
        *bad_count.data() = 0;
        bad_values_array.PrefetchToGpu();
        
        block_size = div_up( total_row_count, size_t( 256 ) );
        build_hash_table<<< block_size, 256 >>>( inputs,
                                                 flags_table.data(),
                                                 input_count,
                                                 total_row_count,
                                                 keys_array.GetData(),
                                                 ids_table.data(),
                                                 table_size,
                                                 bad_values_array.GetData(),
                                                 bad_ids.data(),
                                                 bad_capacity,
                                                 bad_count.data() );
        cudaDeviceSynchronize();
        PRINT_CUDA_ERROR();
        LOG( INFO ) << "bad count: " << *bad_count.data();
    
        if( *bad_count.data() <= bad_capacity )
        {
            auto hash_table = std::make_unique< AriesHashTableMultiKeys >();
    
            hash_table->table_size = table_size;
            hash_table->keys_length = keys_length.ReleaseData();
            hash_table->count = input_count;
            hash_table->ids = ids_table.release_data();
            hash_table->keys_array = keys_array.ReleaseData();
            hash_table->bad_values_array = bad_values_array.ReleaseData();
            hash_table->bad_ids = bad_ids.release_data();
            hash_table->bad_count = *bad_count.data();
            hash_table->flags_table = flags_table.release_data();
            return hash_table;
        }
        else
        {
            for ( int i = 0; i < input_count; i++ )
            {
                cudaFree( bad_values_array[ i ] );
            }
        }
    } while( bad_capacity < *bad_count.data() );
    ARIES_ASSERT( 0, "build hash table error" );
    return nullptr;
}

template< typename type_a, typename type_b>
JoinPair hash_inner_join( const HashTable< type_a >& hash_table, 
                          type_b ** inputs,
                          bool nullable,
                          int64_t* block_size_prefix_sum,
                          size_t block_count,
                          int8_t* indices,
                          AriesValueType indiceValueType,
                          size_t count,
                          const AriesIndicesArraySPtr& hash_table_indices,
                          const AriesIndicesArraySPtr& input_table_indices,
                          context_t& context )
{
    managed_mem_t< int32_t > result_count( 1, context );
    mem_t< int32_t > left_output( count );
    mem_t< int32_t > right_output( count );

    int block_size = div_up( count, ( size_t )256 );

    *result_count.data() = 0;
    int* left_indices = hash_table_indices ? hash_table_indices->PrefetchToGpu(), hash_table_indices->GetData() : nullptr;
    int* right_indices = input_table_indices ? input_table_indices->PrefetchToGpu(), input_table_indices->GetData() : nullptr;

    if ( nullable )
    {
        if ( indices )
        {
            switch ( indiceValueType )
            {
            case AriesValueType::INT8:
                do_join_with_nullable<<< block_size, 256 >>>( hash_table.keys,
                    hash_table.ids,
                    hash_table.table_size,
                    hash_table.bad_values,
                    hash_table.bad_ids,
                    hash_table.bad_count,
                    hash_table.null_value_index,
                    inputs,
                    block_size_prefix_sum,
                    block_count,
                    indices,
                    count,
                    left_indices,
                    right_indices,
                    left_output.data(),
                    right_output.data(),
                    result_count.data() );
                break;
            case AriesValueType::INT16:
                do_join_with_nullable<<< block_size, 256 >>>( hash_table.keys,
                    hash_table.ids,
                    hash_table.table_size,
                    hash_table.bad_values,
                    hash_table.bad_ids,
                    hash_table.bad_count,
                    hash_table.null_value_index,
                    inputs,
                    block_size_prefix_sum,
                    block_count,
                    ( int16_t* )indices,
                    count,
                    left_indices,
                    right_indices,
                    left_output.data(),
                    right_output.data(),
                    result_count.data() );
                break;
            case AriesValueType::INT32:
                do_join_with_nullable<<< block_size, 256 >>>( hash_table.keys,
                    hash_table.ids,
                    hash_table.table_size,
                    hash_table.bad_values,
                    hash_table.bad_ids,
                    hash_table.bad_count,
                    hash_table.null_value_index,
                    inputs,
                    block_size_prefix_sum,
                    block_count,
                    ( int32_t* )indices,
                    count,
                    left_indices,
                    right_indices,
                    left_output.data(),
                    right_output.data(),
                    result_count.data() );
                break;
            }
        }
        else
        {
            do_join_with_nullable<<< block_size, 256 >>>( hash_table.keys,
                hash_table.ids,
                hash_table.table_size,
                hash_table.bad_values,
                hash_table.bad_ids,
                hash_table.bad_count,
                hash_table.null_value_index,
                inputs,
                block_size_prefix_sum,
                block_count,
                indices,
                count,
                left_indices,
                right_indices,
                left_output.data(),
                right_output.data(),
                result_count.data() );
        }
    }
    else
    {
        if ( indices )
        {
            switch ( indiceValueType )
            {
            case AriesValueType::INT8:
                do_join<<< block_size, 256 >>>( hash_table.keys,
                     hash_table.ids,
                     hash_table.table_size,
                     hash_table.bad_values,
                     hash_table.bad_ids,
                     hash_table.bad_count,
                     inputs,
                     block_size_prefix_sum,
                     block_count,
                     indices,
                     count,
                     left_indices,
                     right_indices,
                     left_output.data(),
                     right_output.data(),
                     result_count.data() );
                break;
            case AriesValueType::INT16:
                do_join<<< block_size, 256 >>>( hash_table.keys,
                     hash_table.ids,
                     hash_table.table_size,
                     hash_table.bad_values,
                     hash_table.bad_ids,
                     hash_table.bad_count,
                     inputs,
                     block_size_prefix_sum,
                     block_count,
                     ( int16_t* )indices,
                     count,
                     left_indices,
                     right_indices,
                     left_output.data(),
                     right_output.data(),
                     result_count.data() );
                break;
            case AriesValueType::INT32:
                do_join<<< block_size, 256 >>>( hash_table.keys,
                     hash_table.ids,
                     hash_table.table_size,
                     hash_table.bad_values,
                     hash_table.bad_ids,
                     hash_table.bad_count,
                     inputs,
                     block_size_prefix_sum,
                     block_count,
                     ( int32_t* )indices,
                     count,
                     left_indices,
                     right_indices,
                     left_output.data(),
                     right_output.data(),
                     result_count.data() );
                break;
            }
        }
        else
            do_join<<< block_size, 256 >>>( hash_table.keys,
                 hash_table.ids, 
                 hash_table.table_size, 
                 hash_table.bad_values,
                 hash_table.bad_ids, 
                 hash_table.bad_count,
                 inputs,
                 block_size_prefix_sum, 
                 block_count,
                 indices, 
                 count,
                 left_indices,
                 right_indices,
                 left_output.data(),
                 right_output.data(),
                 result_count.data() );
    }

    cudaDeviceSynchronize();
    PRINT_CUDA_ERROR();
    auto join_count = *result_count.data();

    JoinPair result;
    result.JoinCount = join_count;
    if ( join_count > 0 )
    {
        result.LeftIndices = std::make_shared< AriesInt32Array >();
        result.LeftIndices->AttachBuffer( left_output.release_data(), join_count );
        result.RightIndices = std::make_shared< AriesInt32Array >();
        result.RightIndices->AttachBuffer( right_output.release_data(), join_count );
    }
    return result;
}

template< typename type_a >
JoinPair hash_inner_join( const HashTable< type_a >& hash_table,
                          ColumnDataIterator* input,
                          size_t count,
                          AriesColumnType column_type,
                          const AriesIndicesArraySPtr& hash_table_indices,
                          const AriesIndicesArraySPtr& input_table_indices,
                          context_t& context )
{
    switch ( column_type.DataType.ValueType )
    {
        case AriesValueType::INT8:
        case AriesValueType::INT16:
        case AriesValueType::INT32:
        case AriesValueType::UINT8:
        case AriesValueType::UINT16:
        case AriesValueType::UINT32:
        {
            return hash_inner_join( hash_table,
                            ( int32_t** )input->m_data,
                            column_type.HasNull,
                            input->m_blockSizePrefixSum,
                            input->m_blockCount,
                            input->m_indices,
                            input->m_indiceValueType,
                            count,
                            hash_table_indices,
                            input_table_indices,
                            context );
        }
        case AriesValueType::INT64:
        case AriesValueType::UINT64:
        {
            return hash_inner_join( hash_table,
                     ( unsigned long long** )input->m_data,
                     column_type.HasNull,
                     input->m_blockSizePrefixSum,
                     input->m_blockCount,
                     input->m_indices,
                     input->m_indiceValueType,
                     count,
                     hash_table_indices,
                     input_table_indices,
                     context );
        }
        default:
            assert(0);
            break;
    }
    throw AriesException( ER_UNKNOWN_ERROR, "unsupported value type with hash join" );
}

template< typename type_a, typename type_b >
JoinPair hash_left_join(  const HashTable< type_a >& hash_table, 
                          type_b ** inputs,
                          bool nullable,
                          int64_t* block_size_prefix_sum,
                          size_t block_count,
                          int8_t* indices,
                          AriesValueType indiceValueType,
                          size_t left_count,
                          size_t right_count,
                          int* left_matched_count,
                          context_t& context )
{
    auto max_join_count = left_count + right_count;
    managed_mem_t< int32_t > result_count( 1, context );
    mem_t< int32_t > left_output( max_join_count );
    mem_t< int32_t > right_output( max_join_count );

    int block_size = GET_BLOCK_SIZE( left_count );

    init_flags<<< block_size, THREAD_COUNT_PER_BLOCK >>>( left_matched_count, left_count, 0x00 );
    cudaDeviceSynchronize();

    PRINT_CUDA_ERROR();

    if ( right_count == 0 )
    {
        JoinPair result;
        result.JoinCount = 0;
        result.LeftIndices = std::make_shared< AriesInt32Array >();
        result.LeftIndices->AttachBuffer( left_output.release_data(), 0 );
        result.RightIndices = std::make_shared< AriesInt32Array >();
        result.RightIndices->AttachBuffer( right_output.release_data(), 0 );
        return result;
    }

    block_size = GET_BLOCK_SIZE( right_count );

    *result_count.data() = 0;

    if ( nullable )
    {
        if ( indices )
        {
            switch ( indiceValueType )
            {
            case AriesValueType::INT8:
                do_left_join_with_nullable<<< block_size, THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                     hash_table.ids, 
                     hash_table.table_size, 
                     hash_table.bad_values,
                     hash_table.bad_ids, 
                     hash_table.bad_count,
                     inputs,
                     block_size_prefix_sum, 
                     block_count,
                     indices,
                     right_count,
                     left_output.data(),
                     right_output.data(),
                     result_count.data(),
                     left_matched_count );
                break;
        
            case AriesValueType::INT16:
                do_left_join_with_nullable<<< block_size, THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                     hash_table.ids, 
                     hash_table.table_size, 
                     hash_table.bad_values,
                     hash_table.bad_ids, 
                     hash_table.bad_count,
                     inputs,
                     block_size_prefix_sum, 
                     block_count,
                     ( int16_t* )indices,
                     right_count,
                     left_output.data(),
                     right_output.data(),
                     result_count.data(),
                     left_matched_count );
                break;
            case AriesValueType::INT32:
                do_left_join_with_nullable<<< block_size, THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                     hash_table.ids, 
                     hash_table.table_size, 
                     hash_table.bad_values,
                     hash_table.bad_ids, 
                     hash_table.bad_count,
                     inputs,
                     block_size_prefix_sum, 
                     block_count,
                     ( int32_t* )indices,
                     right_count,
                     left_output.data(),
                     right_output.data(),
                     result_count.data(),
                     left_matched_count );
                break;
            }
        }
        else
            do_left_join_with_nullable<<< block_size, THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                 hash_table.ids, 
                 hash_table.table_size, 
                 hash_table.bad_values,
                 hash_table.bad_ids, 
                 hash_table.bad_count,
                 inputs,
                 block_size_prefix_sum, 
                 block_count,
                 indices,
                 right_count,
                 left_output.data(),
                 right_output.data(),
                 result_count.data(),
                 left_matched_count );      
    }
    else
    {
        if ( indices )
        {
            switch ( indiceValueType )
            {
            case AriesValueType::INT8:
                do_left_join<<< block_size, THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                     hash_table.ids, 
                     hash_table.table_size, 
                     hash_table.bad_values,
                     hash_table.bad_ids, 
                     hash_table.bad_count,
                     inputs,
                     block_size_prefix_sum, 
                     block_count,
                     indices,
                     right_count,
                     left_output.data(),
                     right_output.data(),
                     result_count.data(),
                     left_matched_count );
                break;
        
            case AriesValueType::INT16:
                do_left_join<<< block_size, THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                     hash_table.ids, 
                     hash_table.table_size, 
                     hash_table.bad_values,
                     hash_table.bad_ids, 
                     hash_table.bad_count,
                     inputs,
                     block_size_prefix_sum, 
                     block_count,
                     ( int16_t* )indices,
                     right_count,
                     left_output.data(),
                     right_output.data(),
                     result_count.data(),
                     left_matched_count );
                break;
            case AriesValueType::INT32:
                do_left_join<<< block_size, THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                     hash_table.ids, 
                     hash_table.table_size, 
                     hash_table.bad_values,
                     hash_table.bad_ids, 
                     hash_table.bad_count,
                     inputs,
                     block_size_prefix_sum, 
                     block_count,
                     ( int32_t* )indices,
                     right_count,
                     left_output.data(),
                     right_output.data(),
                     result_count.data(),
                     left_matched_count );
                break;
            }
        }
        else
            do_left_join<<< block_size, THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                 hash_table.ids, 
                 hash_table.table_size, 
                 hash_table.bad_values,
                 hash_table.bad_ids, 
                 hash_table.bad_count,
                 inputs,
                 block_size_prefix_sum, 
                 block_count,
                 indices,
                 right_count,
                 left_output.data(),
                 right_output.data(),
                 result_count.data(),
                 left_matched_count );
    }

    cudaDeviceSynchronize();
    PRINT_CUDA_ERROR();

    auto join_count = *result_count.data();

    JoinPair result;
    result.JoinCount = join_count;
    result.LeftIndices = std::make_shared< AriesInt32Array >();
    result.LeftIndices->AttachBuffer( left_output.release_data(), join_count );
    result.RightIndices = std::make_shared< AriesInt32Array >();
    result.RightIndices->AttachBuffer( right_output.release_data(), join_count );
    return result;
}

template< typename type_a >
JoinPair hash_left_join(  const HashTable< type_a >& hash_table,
                          ColumnDataIterator* input,
                          AriesColumnType column_type,
                          size_t left_count,
                          size_t right_count,
                          int* left_matched_count,
                          context_t& context )
{
    switch ( column_type.DataType.ValueType )
    {
        case AriesValueType::INT8:
        case AriesValueType::INT16:
        case AriesValueType::INT32:
        case AriesValueType::UINT8:
        case AriesValueType::UINT16:
        case AriesValueType::UINT32:
        {
            return hash_left_join( hash_table,
                                   ( int32_t** )input->m_data,
                                   column_type.HasNull,
                                   input->m_blockSizePrefixSum,
                                   input->m_blockCount,
                                   input->m_indices,
                                   input->m_indiceValueType,
                                   left_count,
                                   right_count,
                                   left_matched_count,
                                   context
                                 );
        }
        case AriesValueType::INT64:
        case AriesValueType::UINT64:
        {
            return hash_left_join( hash_table,
                                   ( unsigned long long** )input->m_data,
                                   column_type.HasNull,
                                   input->m_blockSizePrefixSum,
                                   input->m_blockCount,
                                   input->m_indices,
                                   input->m_indiceValueType,
                                   left_count,
                                   right_count,
                                   left_matched_count,
                                   context
                                 );
        }
        default:
            assert( 0 );
            break;
    }
    throw AriesException( ER_UNKNOWN_ERROR, "unsupported value type with hash join" );
}

__global__ void do_join( AriesHashTableMultiKeys* table,
                         ColumnDataIterator* inputs_columns,
                         int inputs_count,
                         size_t total_row_count,
                         int* left_indices,
                         int* right_indices,
                         HashIdType* output_left_ids,
                         HashIdType* output_right_ids,
                         int* result_count )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for ( int i = tid; i < total_row_count; i += stride )
    {
        auto id = search_key( table, inputs_columns, i );

        if ( id != INVALID_HASH_ID )
        {
            auto old = atomicAdd( result_count, 1 );
            output_left_ids[ old ] = left_indices ? left_indices[ id ] : id;
            output_right_ids[ old ] = right_indices ? right_indices[ i ] : i;
        }
    }
}

AriesJoinResult
hash_inner_join( const AriesHashTableMultiKeys& hash_table, 
                 ColumnDataIterator* inputs,
                 int inputs_count,
                 size_t total_row_count,
                 const AriesIndicesArraySPtr& hash_table_indices,
                 const AriesIndicesArraySPtr& input_table_indices,
                 context_t& context )
{
    mem_t< HashIdType > output_left_ids( total_row_count );
    mem_t< HashIdType > output_right_ids( total_row_count );
    managed_mem_t< AriesHashTableMultiKeys > hash_table_new( 1, context );
    int* left_indices = hash_table_indices ? hash_table_indices->GetData() : nullptr;
    int* right_indices = input_table_indices ? input_table_indices->GetData() : nullptr;

    auto* hash_table_ptr = hash_table_new.data();
    *hash_table_ptr = hash_table;

    managed_mem_t< int > result_count( 1, context );
    *result_count.data() = 0;

    hash_table_new.PrefetchToGpu();
    auto block_size = div_up( total_row_count, size_t( 256 ) );
    do_join<<< block_size, 256 >>>( hash_table_ptr, inputs, inputs_count, total_row_count, left_indices, right_indices, output_left_ids.data(), output_right_ids.data(), result_count.data() );

    cudaDeviceSynchronize();
    PRINT_CUDA_ERROR();

    JoinPair result;
    result.JoinCount = *result_count.data();

    if ( result.JoinCount > 0 )
    {
        result.LeftIndices = std::make_shared< AriesInt32Array >();
        result.LeftIndices->AttachBuffer( output_left_ids.release_data(), result.JoinCount );
        result.RightIndices = std::make_shared< AriesInt32Array >();
        result.RightIndices->AttachBuffer( output_right_ids.release_data() , result.JoinCount );
    }
    return result;
}

__global__ void do_join( AriesHashTableMultiKeys* tables,
                         AriesHashJoinDataWrapper* datas,
                         int hash_table_count,
                         size_t total_row_count,
                         int** left_indices,
                         int* right_indices,
                         HashIdType** output_left_idss,
                         HashIdType* output_right_ids,
                         int* result_count )
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ HashIdType s[];
    HashIdType* tmp_ids = s + threadIdx.x * hash_table_count;

    for ( int i = tid; i < total_row_count; i += stride )
    {
        bool found = true;
        for ( int j = 0; j < hash_table_count; j++ )
        {
            auto id = search_key( tables + j, ( ColumnDataIterator* )( datas[ j ].Inputs ), i );
            if ( id == INVALID_HASH_ID )
            {
                found = false;
                break;
            }

            tmp_ids[ j ] = id;
        }

        if ( found )
        {
            auto old = atomicAdd( result_count, 1 );
            for ( int j = 0; j < hash_table_count; j++ )
            {
                auto& left_ind = left_indices[ j ];
                output_left_idss[ j ][ old ] = left_ind ? left_ind[ tmp_ids[ j ] ] : tmp_ids[ j ];
            }

            output_right_ids[ old ] = right_indices ? right_indices[ i ] : i;
        }
    }
}

AriesStarJoinResult
hash_inner_join( const std::vector< AriesHashTableWrapper >& hash_tables,
                 const std::vector< AriesHashJoinDataWrapper >& datas,
                 size_t total_row_count,
                 const std::vector< AriesIndicesArraySPtr >& hash_table_indices,
                 const AriesIndicesArraySPtr& input_table_indices,
                 context_t& context )
{
    AriesManagedArray< AriesHashTableMultiKeys > tables( hash_tables.size() );
    int input_count = hash_tables.size();

    int index = 0;
    for ( const auto& table : hash_tables )
    {
        if ( table.Type == HashTableType::SingleKey )
        {
            auto* t = ( AriesHashTable* )( table.Ptr );

            managed_mem_t< int8_t* > keys_addrs( 1, context );
            managed_mem_t< int8_t* > bad_values_addrs( 1, context );
            managed_mem_t< int > length_addrs( 1, context );

            tables[ index ].keys_array = keys_addrs.release_data();
            tables[ index ].bad_values_array = bad_values_addrs.release_data();
            tables[ index ].keys_length = length_addrs.release_data();

            tables[ index ].table_size = t->TableSize;
            tables[ index ].count =  1;

            tables[ index ].keys_array[ 0 ] = ( int8_t* )( t->Keys );
            tables[ index ].ids = t->Ids;

            tables[ index ].bad_values_array[ 0 ] = ( int8_t* )( t->BadValues );
            tables[ index ].bad_ids = t->BadIds;
            tables[ index ].bad_count = t->BadCount;

            tables[ index ].flags_table = nullptr;

            switch ( t->ValueType )
            {
                case AriesValueType::INT8:
                case AriesValueType::INT16:
                case AriesValueType::INT32:
                case AriesValueType::UINT8:
                case AriesValueType::UINT16:
                case AriesValueType::UINT32:
                    tables[ index ].keys_length[ 0 ] = 4;
                    break;
                default:
                    tables[ index ].keys_length[ 0 ] = 8;
                    break;
            }
        }
        else
        {
            auto* multi_table = ( AriesHashTableMultiKeys* )( table.Ptr );

            tables[ index ].table_size = multi_table->table_size;
            tables[ index ].count =  multi_table->count;

            tables[ index ].keys_array = multi_table->keys_array;
            tables[ index ].ids = multi_table->ids;

            tables[ index ].bad_values_array = multi_table->bad_values_array;
            tables[ index ].bad_ids = multi_table->bad_ids;

            tables[ index ].keys_length = multi_table->keys_length;
            tables[ index ].flags_table = ( flag_type_t* )( multi_table->flags_table );
            tables[ index ].bad_count = multi_table->bad_count;
        }
        index++;
    }

    AriesManagedArray< AriesHashJoinDataWrapper > join_datas( datas.size() );
    for ( int i = 0; i < datas.size(); i++ )
    {
        join_datas[ i ].Count = datas[ i ].Count;
        join_datas[ i ].Inputs = datas[ i ].Inputs;
    }

    AriesManagedArray< HashIdType* > output_left_idss( hash_tables.size() );
    mem_t< HashIdType > output_right_ids( total_row_count );
    mem_t< HashIdType > tmp_left_ids( hash_tables.size() );
    managed_mem_t< int > result_count( 1, context );
    *result_count.data() = 0;

    for ( int i = 0; i < hash_tables.size(); i++ )
    {
        mem_t< HashIdType > ids( total_row_count );
        output_left_idss[ i ] = ids.release_data();
    }

    AriesManagedArray< int* > left_indices( hash_table_indices.size() );
    index = 0;
    for( const auto& ind : hash_table_indices )
        left_indices[ index++ ] = ind ? ind->GetData() : nullptr;
    int* right_indices = input_table_indices ? input_table_indices->GetData() : nullptr;

    auto block_size = div_up( total_row_count, ( size_t )256 );

    int shared_size = input_count * sizeof( HashIdType ) * 256;
    tables.PrefetchToGpu();
    join_datas.PrefetchToGpu();
    output_left_idss.PrefetchToGpu();
    left_indices.PrefetchToGpu();
    do_join<<< block_size, 256, shared_size >>>( tables.GetData(),
                             join_datas.GetData(),
                             input_count,
                             total_row_count,
                             left_indices.GetData(),
                             right_indices,
                             output_left_idss.GetData(),
                             output_right_ids.data(),
                             result_count.data() );

    cudaDeviceSynchronize();
    PRINT_CUDA_ERROR();

    for ( int i = 0; i < hash_tables.size(); i++ )
    {
        auto& table = tables[ i ];
        if ( hash_tables[ i ].Type == HashTableType::SingleKey )
        {
            context.free( table.bad_values_array );
            context.free( table.keys_array );
            context.free( table.keys_length );
        }
    }

    AriesStarJoinResult result;
    result.JoinCount = *result_count.data();
    if ( result.JoinCount > 0 )
    {
        result.FactIds = std::make_shared< AriesArray< HashIdType > >();
        result.FactIds->AttachBuffer( output_right_ids.release_data(), result.JoinCount );

        for ( int i = 0; i < input_count; i++ )
        {
            auto ids = std::make_shared< AriesArray< HashIdType > >();
            ids->AttachBuffer( output_left_idss[ i ], result.JoinCount );
            result.DimensionIds.emplace_back( ids );
        }
    }
    else
    {
        for ( int i = 0; i < input_count; i++ )
        {
            context.free( output_left_idss[ i ] );
        }
    }
    return result;
}

template< typename type_a, typename type_b >
AriesInt32ArraySPtr simple_half_join_left_as_hash( const HashTable< type_a >& hash_table,
                                        type_b ** inputs,
                                        bool nullable,
                                        int64_t* block_size_prefix_sum,
                                        size_t block_count,
                                        int8_t* indices,
                                        AriesValueType indiceValueType,
                                        size_t input_row_count,
                                        size_t output_count,
                                        bool is_semi_join,
                                        bool isNotIn,
                                        context_t& context )
{
    AriesInt32ArraySPtr result = std::make_shared< AriesInt32Array >( output_count );
    int32_t* output_flags = result->GetData();
    if( is_semi_join )
        init_value( output_flags, output_count, 0, context );
    else
        init_value( output_flags, output_count, 1, context );
    if( nullable )
    {
        AriesManagedArray< int32_t > hasNullFlag(1);
        int32_t* has_null_flag = hasNullFlag.GetData();
        *has_null_flag = 0;
        if ( indices )
        {
            switch ( indiceValueType )
            {
            case AriesValueType::INT8:
                do_simple_half_join_left_as_hash_with_nullable<<< GET_BLOCK_SIZE( input_row_count ), THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                     hash_table.ids,
                     hash_table.table_size,
                     hash_table.bad_values,
                     hash_table.bad_ids,
                     hash_table.bad_count,
                     inputs,
                     block_size_prefix_sum,
                     block_count,
                     indices,
                     input_row_count,
                     is_semi_join,
                     has_null_flag,
                     output_flags );
                break;

            case AriesValueType::INT16:
                do_simple_half_join_left_as_hash_with_nullable<<< GET_BLOCK_SIZE( input_row_count ), THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                     hash_table.ids,
                     hash_table.table_size,
                     hash_table.bad_values,
                     hash_table.bad_ids,
                     hash_table.bad_count,
                     inputs,
                     block_size_prefix_sum,
                     block_count,
                     ( int16_t* )indices,
                     input_row_count,
                     is_semi_join,
                     has_null_flag,
                     output_flags );
                break;

            case AriesValueType::INT32:
                do_simple_half_join_left_as_hash_with_nullable<<< GET_BLOCK_SIZE( input_row_count ), THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                     hash_table.ids,
                     hash_table.table_size,
                     hash_table.bad_values,
                     hash_table.bad_ids,
                     hash_table.bad_count,
                     inputs,
                     block_size_prefix_sum,
                     block_count,
                     ( int32_t* )indices,
                     input_row_count,
                     is_semi_join,
                     has_null_flag,
                     output_flags );
                break;
            }
        }
        else
            do_simple_half_join_left_as_hash_with_nullable<<< GET_BLOCK_SIZE( input_row_count ), THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                 hash_table.ids,
                 hash_table.table_size,
                 hash_table.bad_values,
                 hash_table.bad_ids,
                 hash_table.bad_count,
                 inputs,
                 block_size_prefix_sum,
                 block_count,
                 indices,
                 input_row_count,
                 is_semi_join,
                 has_null_flag,
                 output_flags );
        context.synchronize();
        if( *has_null_flag && isNotIn )
            init_value( output_flags, output_count, 0, context );
    }
    else
    {
        if ( indices )
        {
            switch ( indiceValueType )
            {
            case AriesValueType::INT8:
                do_simple_half_join_left_as_hash<<< GET_BLOCK_SIZE( input_row_count ), THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                     hash_table.ids,
                     hash_table.table_size,
                     hash_table.bad_values,
                     hash_table.bad_ids,
                     hash_table.bad_count,
                     inputs,
                     block_size_prefix_sum,
                     block_count,
                     indices,
                     input_row_count,
                     is_semi_join,
                     output_flags );
                break;

            case AriesValueType::INT16:
                do_simple_half_join_left_as_hash<<< GET_BLOCK_SIZE( input_row_count ), THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                     hash_table.ids,
                     hash_table.table_size,
                     hash_table.bad_values,
                     hash_table.bad_ids,
                     hash_table.bad_count,
                     inputs,
                     block_size_prefix_sum,
                     block_count,
                     ( int16_t* )indices,
                     input_row_count,
                     is_semi_join,
                     output_flags );
                break;

            case AriesValueType::INT32:
                do_simple_half_join_left_as_hash<<< GET_BLOCK_SIZE( input_row_count ), THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                     hash_table.ids,
                     hash_table.table_size,
                     hash_table.bad_values,
                     hash_table.bad_ids,
                     hash_table.bad_count,
                     inputs,
                     block_size_prefix_sum,
                     block_count,
                     ( int32_t* )indices,
                     input_row_count,
                     is_semi_join,
                     output_flags );
                break;
            }
        }
        else
            do_simple_half_join_left_as_hash<<< GET_BLOCK_SIZE( input_row_count ), THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                 hash_table.ids,
                 hash_table.table_size,
                 hash_table.bad_values,
                 hash_table.bad_ids,
                 hash_table.bad_count,
                 inputs,
                 block_size_prefix_sum,
                 block_count,
                 indices,
                 input_row_count,
                 is_semi_join,
                 output_flags );
    }
    cudaDeviceSynchronize();
    PRINT_CUDA_ERROR();
    if( isNotIn )
    {
        if( hash_table.null_value_index != -1 )
            result->SetValue( 0, hash_table.null_value_index );
    }

    return result;
}

template< typename type_a >
AriesInt32ArraySPtr simple_half_join_left_as_hash( const HashTable< type_a >& hash_table,
                          ColumnDataIterator* input,
                          AriesColumnType column_type,
                          size_t input_row_count,
                          size_t output_count,
                          bool is_semi_join,
                          bool isNotIn,
                          context_t& context )
{
    switch ( column_type.DataType.ValueType )
    {
        case AriesValueType::INT8:
        case AriesValueType::INT16:
        case AriesValueType::INT32:
        case AriesValueType::UINT8:
        case AriesValueType::UINT16:
        case AriesValueType::UINT32:
        {
            return simple_half_join_left_as_hash( hash_table,
                                   ( int32_t** )input->m_data,
                                   column_type.HasNull,
                                   input->m_blockSizePrefixSum,
                                   input->m_blockCount,
                                   input->m_indices,
                                   input->m_indiceValueType,
                                   input_row_count,
                                   output_count,
                                   is_semi_join,
                                   isNotIn,
                                   context
                                 );
        }
        case AriesValueType::INT64:
        case AriesValueType::UINT64:
        {
            return simple_half_join_left_as_hash( hash_table,
                                   ( unsigned long long** )input->m_data,
                                   column_type.HasNull,
                                   input->m_blockSizePrefixSum,
                                   input->m_blockCount,
                                   input->m_indices,
                                   input->m_indiceValueType,
                                   input_row_count,
                                   output_count,
                                   is_semi_join,
                                   isNotIn,
                                   context
                                 );
        }
        default:
            assert( 0 );
            break;
    }
    throw AriesException( ER_UNKNOWN_ERROR, "unsupported value type with hash join" );
}

template< typename type_a, typename type_b >
AriesInt32ArraySPtr simple_half_join_right_as_hash( const HashTable< type_a >& hash_table,
                                        type_b ** inputs,
                                        bool nullable,
                                        int64_t* block_size_prefix_sum,
                                        size_t block_count,
                                        int8_t* indices,
                                        AriesValueType indiceValueType,
                                        size_t input_row_count,
                                        bool is_semi_join,
                                        bool isNotIn,
                                        bool use_indices_for_output,
                                        context_t& context )
{
    AriesInt32ArraySPtr result = std::make_shared< AriesInt32Array >( input_row_count, true );
    if( isNotIn && hash_table.null_value_index != -1 )
        return result;
    int32_t* output_flags = result->GetData();

    managed_mem_t< int > result_count( 1, context );
    int* output_count = result_count.data();
    *output_count = 0;

    if( nullable )
    {
        if ( indices )
        {
            switch ( indiceValueType )
            {
            case AriesValueType::INT8:
                do_simple_half_join_right_as_hash_with_nullable<<< GET_BLOCK_SIZE( input_row_count ), THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                        hash_table.ids,
                        hash_table.table_size,
                        hash_table.bad_values,
                        hash_table.bad_ids,
                        hash_table.bad_count,
                        inputs,
                        block_size_prefix_sum,
                        block_count,
                        indices,
                        input_row_count,
                        is_semi_join,
                        isNotIn,
                        use_indices_for_output,
                        output_count,
                        output_flags );
                break;

            case AriesValueType::INT16:
                do_simple_half_join_right_as_hash_with_nullable<<< GET_BLOCK_SIZE( input_row_count ), THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                        hash_table.ids,
                        hash_table.table_size,
                        hash_table.bad_values,
                        hash_table.bad_ids,
                        hash_table.bad_count,
                        inputs,
                        block_size_prefix_sum,
                        block_count,
                        ( int16_t* )indices,
                        input_row_count,
                        is_semi_join,
                        isNotIn,
                        use_indices_for_output,
                        output_count,
                        output_flags );
                break;
            case AriesValueType::INT32:
                do_simple_half_join_right_as_hash_with_nullable<<< GET_BLOCK_SIZE( input_row_count ), THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                        hash_table.ids,
                        hash_table.table_size,
                        hash_table.bad_values,
                        hash_table.bad_ids,
                        hash_table.bad_count,
                        inputs,
                        block_size_prefix_sum,
                        block_count,
                        ( int32_t* )indices,
                        input_row_count,
                        is_semi_join,
                        isNotIn,
                        use_indices_for_output,
                        output_count,
                        output_flags );
                break;
            }
        }
        else
            do_simple_half_join_right_as_hash_with_nullable<<< GET_BLOCK_SIZE( input_row_count ), THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                    hash_table.ids,
                    hash_table.table_size,
                    hash_table.bad_values,
                    hash_table.bad_ids,
                    hash_table.bad_count,
                    inputs,
                    block_size_prefix_sum,
                    block_count,
                    indices,
                    input_row_count,
                    is_semi_join,
                    isNotIn,
                    use_indices_for_output,
                    output_count,
                    output_flags );
    }
    else
    {
        if ( indices )
        {
            switch ( indiceValueType )
            {
            case AriesValueType::INT8:
                do_simple_half_join_right_as_hash<<< GET_BLOCK_SIZE( input_row_count ), THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                        hash_table.ids,
                        hash_table.table_size,
                        hash_table.bad_values,
                        hash_table.bad_ids,
                        hash_table.bad_count,
                        inputs,
                        block_size_prefix_sum,
                        block_count,
                        indices,
                        input_row_count,
                        is_semi_join,
                        use_indices_for_output,
                        output_count,
                        output_flags );
                break;

            case AriesValueType::INT16:
                do_simple_half_join_right_as_hash<<< GET_BLOCK_SIZE( input_row_count ), THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                        hash_table.ids,
                        hash_table.table_size,
                        hash_table.bad_values,
                        hash_table.bad_ids,
                        hash_table.bad_count,
                        inputs,
                        block_size_prefix_sum,
                        block_count,
                        ( int16_t* )indices,
                        input_row_count,
                        is_semi_join,
                        use_indices_for_output,
                        output_count,
                        output_flags );
                break;
            case AriesValueType::INT32:
                do_simple_half_join_right_as_hash<<< GET_BLOCK_SIZE( input_row_count ), THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                        hash_table.ids,
                        hash_table.table_size,
                        hash_table.bad_values,
                        hash_table.bad_ids,
                        hash_table.bad_count,
                        inputs,
                        block_size_prefix_sum,
                        block_count,
                        ( int32_t* )indices,
                        input_row_count,
                        is_semi_join,
                        use_indices_for_output,
                        output_count,
                        output_flags );
                break;
            }
        }
        else
            do_simple_half_join_right_as_hash<<< GET_BLOCK_SIZE( input_row_count ), THREAD_COUNT_PER_BLOCK >>>( hash_table.keys,
                    hash_table.ids,
                    hash_table.table_size,
                    hash_table.bad_values,
                    hash_table.bad_ids,
                    hash_table.bad_count,
                    inputs,
                    block_size_prefix_sum,
                    block_count,
                    indices,
                    input_row_count,
                    is_semi_join,
                    use_indices_for_output,
                    output_count,
                    output_flags );
    }

    cudaDeviceSynchronize();
    PRINT_CUDA_ERROR();
    if( use_indices_for_output )
        result->SetItemCount( *output_count );
    return result;
}

template< typename type_a >
AriesInt32ArraySPtr simple_half_join_right_as_hash( const HashTable< type_a >& hash_table,
                                                    ColumnDataIterator* input,
                                                    AriesColumnType column_type,
                                                    size_t input_row_count,
                                                    bool is_semi_join,
                                                    bool isNotIn,
                                                    bool use_indices_for_output,
                                                    context_t& context )
{
    switch ( column_type.DataType.ValueType )
    {
        case AriesValueType::INT8:
        case AriesValueType::INT16:
        case AriesValueType::INT32:
        case AriesValueType::UINT8:
        case AriesValueType::UINT16:
        case AriesValueType::UINT32:
        {
            return simple_half_join_right_as_hash( hash_table,
                                   ( int32_t** )input->m_data,
                                   column_type.HasNull,
                                   input->m_blockSizePrefixSum,
                                   input->m_blockCount,
                                   input->m_indices,
                                   input->m_indiceValueType,
                                   input_row_count,
                                   is_semi_join,
                                   isNotIn,
                                   use_indices_for_output,
                                   context
                                 );
        }
        case AriesValueType::INT64:
        case AriesValueType::UINT64:
        {
            return simple_half_join_right_as_hash( hash_table,
                                   ( unsigned long long** )input->m_data,
                                   column_type.HasNull,
                                   input->m_blockSizePrefixSum,
                                   input->m_blockCount,
                                   input->m_indices,
                                   input->m_indiceValueType,
                                   input_row_count,
                                   is_semi_join,
                                   isNotIn,
                                   use_indices_for_output,
                                   context
                                 );
        }
        default:
            assert( 0 );
            break;
    }
    throw AriesException( ER_UNKNOWN_ERROR, "unsupported value type with hash join" );
}

template
JoinPair hash_inner_join( const HashTable< unsigned long long >& hash_table,
                          ColumnDataIterator* input,
                          size_t count,
                          AriesColumnType column_type,
                          const AriesIndicesArraySPtr& hash_table_indices,
                          const AriesIndicesArraySPtr& input_table_indices,
                          context_t& context );

template
JoinPair hash_inner_join( const HashTable< int32_t >& hash_table,
                          ColumnDataIterator* input,
                          size_t count,
                          AriesColumnType column_type,
                          const AriesIndicesArraySPtr& hash_table_indices,
                          const AriesIndicesArraySPtr& input_table_indices,
                          context_t& context );

template
JoinPair hash_inner_join( const HashTable< unsigned long long >&,
                          unsigned long long ** inputs,
                          bool nullable,
                          int64_t* block_size_prefix_sum,
                          size_t block_count,
                          int8_t* indices,
                          AriesValueType indiceValueType,
                          size_t count,
                          const AriesIndicesArraySPtr& hash_table_indices,
                          const AriesIndicesArraySPtr& input_table_indices,
                          context_t& context);

template
JoinPair hash_inner_join( const HashTable< int32_t >&,
                          int32_t ** inputs,
                          bool nullable,
                          int64_t* block_size_prefix_sum,
                          size_t block_count,
                          int8_t* indices,
                          AriesValueType indiceValueType,
                          size_t count,
                          const AriesIndicesArraySPtr& hash_table_indices,
                          const AriesIndicesArraySPtr& input_table_indices,
                          context_t& context);

template
JoinPair hash_inner_join( const HashTable< unsigned long long >&,
                          int32_t ** inputs,
                          bool nullable,
                          int64_t* block_size_prefix_sum,
                          size_t block_count,
                          int8_t* indices,
                          AriesValueType indiceValueType,
                          size_t count,
                          const AriesIndicesArraySPtr& hash_table_indices,
                          const AriesIndicesArraySPtr& input_table_indices,
                          context_t& context);


template
JoinPair hash_inner_join( const HashTable< int32_t >&,
                          unsigned long long ** inputs,
                          bool nullable,
                          int64_t* block_size_prefix_sum,
                          size_t block_count,
                          int8_t* indices,
                          AriesValueType indiceValueType,
                          size_t count,
                          const AriesIndicesArraySPtr& hash_table_indices,
                          const AriesIndicesArraySPtr& input_table_indices,
                          context_t& context);

template
JoinPair hash_left_join(  const HashTable< int32_t >& hash_table, 
                          int32_t ** inputs,
                          bool nullable,
                          int64_t* block_size_prefix_sum,
                          size_t block_count,
                          int8_t* indices,
                          AriesValueType indiceValueType,
                          size_t left_count,
                          size_t right_count,
                          int* left_matched_count,
                          context_t& context );

template
JoinPair hash_left_join(  const HashTable< int32_t >& hash_table, 
                          unsigned long long ** inputs,
                          bool nullable,
                          int64_t* block_size_prefix_sum,
                          size_t block_count,
                          int8_t* indices,
                          AriesValueType indiceValueType,
                          size_t left_count,
                          size_t right_count,
                          int* left_matched_count,
                          context_t& context );

template
JoinPair hash_left_join(  const HashTable< unsigned long long >& hash_table, 
                          int32_t ** inputs,
                          bool nullable,
                          int64_t* block_size_prefix_sum,
                          size_t block_count,
                          int8_t* indices,
                          AriesValueType indiceValueType,
                          size_t left_count,
                          size_t right_count,
                          int* left_matched_count,
                          context_t& context );

template
JoinPair hash_left_join(  const HashTable< unsigned long long >& hash_table, 
                          unsigned long long ** inputs,
                          bool nullable,
                          int64_t* block_size_prefix_sum,
                          size_t block_count,
                          int8_t* indices,
                          AriesValueType indiceValueType,
                          size_t left_count,
                          size_t right_count,
                          int* left_matched_count,
                          context_t& context );

template
JoinPair hash_left_join(  const HashTable< int32_t >& hash_table,
                          ColumnDataIterator* input,
                          AriesColumnType column_type,
                          size_t left_count,
                          size_t right_count,
                          int* left_matched_count,
                          context_t& context );
template
JoinPair hash_left_join(  const HashTable< unsigned long long >& hash_table,
                          ColumnDataIterator* input,
                          AriesColumnType column_type,
                          size_t left_count,
                          size_t right_count,
                          int* left_matched_count,
                          context_t& context );

template
AriesInt32ArraySPtr simple_half_join_left_as_hash( const HashTable< int32_t >& hash_table,
                          ColumnDataIterator* input,
                          AriesColumnType column_type,
                          size_t input_row_count,
                          size_t output_count,
                          bool is_semi_join,
                          bool isNotIn,
                          context_t& context );

template
AriesInt32ArraySPtr simple_half_join_left_as_hash( const HashTable< unsigned long long >& hash_table,
                          ColumnDataIterator* input,
                          AriesColumnType column_type,
                          size_t input_row_count,
                          size_t output_count,
                          bool is_semi_join,
                          bool isNotIn,
                          context_t& context );

template
AriesInt32ArraySPtr simple_half_join_right_as_hash( const HashTable< int32_t >& hash_table,
                                                    ColumnDataIterator* input,
                                                    AriesColumnType column_type,
                                                    size_t input_row_count,
                                                    bool is_semi_join,
                                                    bool isNotIn,
                                                    bool use_indices_for_output,
                                                    context_t& context );

template
AriesInt32ArraySPtr simple_half_join_right_as_hash( const HashTable< unsigned long long >& hash_table,
                                                    ColumnDataIterator* input,
                                                    AriesColumnType column_type,
                                                    size_t input_row_count,
                                                    bool is_semi_join,
                                                    bool isNotIn,
                                                    bool use_indices_for_output,
                                                    context_t& context );

template 
HashTable< int32_t > build_hash_table( int32_t** inputs,
                                      int64_t* block_size_prefix_sum,
                                      size_t block_count,
                                      bool nullable,
                                      int8_t* indices,
                                      AriesValueType indiceValueType,
                                      size_t count,
                                      context_t& context );

template 
HashTable< unsigned long long > build_hash_table( unsigned long long** inputs,
                                      int64_t* block_size_prefix_sum,
                                      size_t block_count,
                                      bool nullable,
                                      int8_t* indices,
                                      AriesValueType indiceValueType,
                                      size_t count,
                                      context_t& context );

template< typename index_type_t >
ARIES_HOST_DEVICE
int8_t* DoGetData( int itemIdx,
                   int8_t** data,
                   int64_t* blockSizePrefixSum,
                   int blockCount,
                   index_type_t* indices,
                   int perItemSize,
                   int8_t* nullData )
{
    int64_t pos = itemIdx;
    if( indices != nullptr )
        pos = indices[ itemIdx ];
    if( pos != -1 )
    {
        int blockIndex = binary_search< bounds_upper >( blockSizePrefixSum, blockCount, pos ) - 1;
        return data[ blockIndex ] + ( pos - blockSizePrefixSum[ blockIndex ] ) * perItemSize;
    }
    else
        return nullData;

} 
ARIES_HOST_DEVICE int8_t* ColumnDataIterator::GetData( int index ) const
{
    int8_t* result = nullptr;
    if ( nullptr == m_indices )
    {
        int pos = index;
        int blockIndex = binary_search< bounds_upper >( m_blockSizePrefixSum, m_blockCount, pos ) - 1;
        result = m_data[ blockIndex ] + ( pos - m_blockSizePrefixSum[ blockIndex ] ) * m_perItemSize;
    }
    else
    {
        switch ( m_indiceValueType )
        {
        case AriesValueType::INT8:
            result = DoGetData( index, m_data, m_blockSizePrefixSum, m_blockCount, m_indices, m_perItemSize, m_nullData );
            break;
        case AriesValueType::INT16:
            result = DoGetData( index, m_data, m_blockSizePrefixSum, m_blockCount, ( int16_t* )m_indices, m_perItemSize, m_nullData );
            break;
        case AriesValueType::INT32:
            result = DoGetData( index, m_data, m_blockSizePrefixSum, m_blockCount, ( int32_t* )m_indices, m_perItemSize, m_nullData );
            break;
        }
    }
    return result;
    /*
    int pos = index;
    if( m_indices != nullptr )
        pos = m_indices[index];
    if( pos != -1 )
    {
        int blockIndex = binary_search< bounds_upper >( m_blockSizePrefixSum, m_blockCount, pos ) - 1;
        return m_data[ blockIndex ] + ( pos - m_blockSizePrefixSum[ blockIndex ] ) * m_perItemSize;
    }
    else
        return m_nullData;
    */
}

END_ARIES_ACC_NAMESPACE
