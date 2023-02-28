/*
 * kernel_util.hxx
 *
 *  Created on: Jun 19, 2019
 *      Author: lichi
 */

#ifndef KERNEL_UTIL_HXX_
#define KERNEL_UTIL_HXX_

#include "transform.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    template< typename launch_arg_t = empty_t, typename type_t >
    void flip_flags( type_t* data, size_t count, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            data[index] = !data[index];
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename type_u = type_t >
    void merge_flags_and( type_t* in_out_data, type_u* in_data, size_t count, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            in_out_data[index] = in_out_data[index] && in_data[index];
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename type_u = type_t >
    void merge_flags_or( type_t* in_out_data, type_u* in_data, size_t count, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            in_out_data[index] = in_out_data[index] || in_data[index];
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename type_u >
    void gather_filtered_data( const type_t *data, size_t count, const type_u* flags, const type_u* psum, type_t *outData, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            if( flags[ index ] )
            outData[ psum[ index ] ] = data[ index ];
            //memcpy( outData + psum[ index ], data + index, sizeof( type_t ) );
            };
        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename type_u >
    void scatter_associated_data( const type_t *data, size_t count, const type_u* flags, const type_u* psum, type_t *outData, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            if( flags[ index ] )
            outData[ index ] = data[ psum[ index ] ];
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename type_u >
    void scatter_associated_data2( type_t *data, size_t count, const type_u* flags, const type_u* psum, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            if( data[ index ] )
            data[ index ] = flags[ psum[ index ] ];
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
    }

//    template< typename launch_arg_t = empty_t, typename type_t, typename type_u >
//    void scatter_flag_data( const type_t *data, const type_t *associated, size_t count, const type_u* flags, const type_u* psum, type_t *outData,
//            context_t& context )
//    {
//        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
//        auto k = [=] ARIES_DEVICE(int index)
//        {
////            if( flags[ index ] )
////            outData[ associated[ index ] ] = data[ psum[ index ] ];
//            if( flags[ index ] )
//            outData[ associated[ index ] ] = 1;
//        };
//        transform< launch_t >( k, count, context );
//        context.synchronize();
//    }

    template< typename launch_arg_t = empty_t, typename type_t, typename type_u >
    void scatter_flag_data( const type_t *data, size_t count, const type_u* flags, type_t *outData, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            if( flags[ index ] )
            outData[ data[ index ] ] = 1;
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t >
    void convert_seg_data( const type_t *segData, size_t segCount, type_t *outData, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            outData[ segData[ index ] ] = 1;
        };
        transform< launch_t >( k, segCount, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t >
    int32_t find_max_load_flag_index( const type_t *flag, size_t flagCount, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        managed_mem_t< int32_t > maxIndex( 1, context );
        int32_t *pMax = maxIndex.data();
        *pMax = 0;
        auto k = [=] ARIES_DEVICE(int index)
        {
            if( flag[ index ] )
            atomicMax( pMax, index );
        };
        transform< launch_t >( k, flagCount, context );
        context.synchronize();
        return *pMax;
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename type_u >
    void gather_filtered_data( const char *data, size_t len, size_t count, const type_t* flags, const type_u* psum, char *outData, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            if( flags[ index ] )
            memcpy( outData + psum[ index ] * len, data + index * len, len );
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t >
    void gather_filtered_data( const DataBlockInfo* data_input, int32_t block_count, size_t count, const type_t* flags, const type_t* psum,
            int8_t** data_output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            type_t idx = flags[ index ];
            if( idx )
            {
                for( int32_t i = 0; i < block_count; ++i )
                {
                    DataBlockInfo info = data_input[i];
                    memcpy( data_output[i] + psum[ index ] * (size_t)info.ElementSize, info.Data + index * (size_t)info.ElementSize, (size_t)info.ElementSize );
                }
            }
        };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t >
    void gather_filtered_data_ex( const SimpleDataBlockInfo* data_input, int32_t block_count, size_t count, const type_t* indices,
            int8_t** data_output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            type_t idx = indices[ index ];
            for( int32_t i = 0; i < block_count; ++i )
            {
                SimpleDataBlockInfo info = data_input[i];
                memcpy( data_output[i] + index * (size_t)info.ElementSize, info.Data + idx * (size_t)info.ElementSize, (size_t)info.ElementSize );
            }
        };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename type_u >
    void gather_joined_data( const type_t *data, size_t count, const type_u* indices, type_t *outData, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            auto i = indices[ index ];
            if( i != -1 )
            outData[ index ] = data[ i ];
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t >
    void gather_joined_data( const char *data, size_t len, size_t count, const type_t* indices, char *outData, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            auto i = indices[ index ];
            if( i != -1 )
            memcpy( outData + index * len, data + i * len, len );
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t >
    void gather_joined_data( const DataBlockInfo* data_input, int32_t block_count, size_t count, const type_t* indices, int8_t** data_output,
            context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            type_t idx = indices[ index ];
            if( idx != -1 )
            {
                int8_t* pOutData;
                for( int32_t i = 0; i < block_count; ++i )
                {
                    DataBlockInfo info = data_input[i];
                    pOutData = data_output[i] + index * ( (size_t)info.ElementSize + info.Offset );
                    *pOutData = 1;
                    memcpy( pOutData + info.Offset, info.Data + idx * (size_t)info.ElementSize, (size_t)info.ElementSize );
                }
            }
            else
            {
                for( int32_t i = 0; i < block_count; ++i )
                *( data_output[i] + index * ( (size_t)data_input[i].ElementSize + data_input[i].Offset ) ) = 0;
            }
        };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t >
    void init_sequence( type_t* data, size_t count, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            data[index] = index;
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t >
    void init_sequence_begin_with( type_t* data, size_t count, type_t value, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            data[index] = value + index;
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename type_u >
    void init_value( type_t* data, size_t count, type_u value, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            data[index] = value;
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t>
    void init_value( aries_acc::Decimal* data, size_t count, aries_acc::Decimal value, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        aries_acc::Decimal* value_gpu;
        value_gpu = (aries_acc::Decimal*)context.alloc(sizeof(aries_acc::Decimal));
        cudaMemcpy(value_gpu, &value, sizeof(aries_acc::Decimal), cudaMemcpyHostToDevice);
        auto k = [=] ARIES_DEVICE(int index)
        {
            data[index] = *value_gpu;
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
        context.free(value_gpu);
    }

    template< typename launch_arg_t = empty_t>
    void init_value( char* data, size_t len, size_t count, aries_acc::Decimal value, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        char* value_gpu;
        value_gpu = (char *)context.alloc(len);
        cudaMemcpy(value_gpu, &value, len, cudaMemcpyHostToDevice);
        auto k = [=] ARIES_DEVICE(int index)
        {
            memcpy( data + index * len, value_gpu, len );
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
        context.free(value_gpu);
    }

    template< typename launch_arg_t = empty_t >
    void init_value( char* data, size_t len, size_t count, const char* value, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            memcpy( data + index * len, value, len );
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t >
    void make_column_nullable( DataBlockInfo block, size_t tupleNum, int8_t* data_output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 16 > > >::type_t launch_t;
        size_t elementSize = block.ElementSize;
        const int8_t* data = block.Data;
        auto k = [=] ARIES_DEVICE(int index)
        {
            int8_t* dst = data_output + index * ( elementSize + 1 );
            *dst = 1;
            memcpy( dst + 1, data + index * elementSize, elementSize );
        };

        transform< launch_t >( k, tupleNum, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t >
    void make_column_nullable( const DataBlockInfo* data_input, int32_t block_count, size_t count, const int8_t* flags, int8_t** data_output,
            context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 16 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            int8_t* dst;
            for( int32_t i = 0; i < block_count; ++i )
            {
                DataBlockInfo info = data_input[i];
                dst = data_output[i] + index * ( (size_t)info.ElementSize + 1 );
                *dst = flags[ index ];
                memcpy( dst + 1, info.Data + index * (size_t)info.ElementSize, (size_t)info.ElementSize );
            }
        };

        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t >
    void combine_keys( const DataBlockInfo* data_input, size_t block_count, const type_t* associated, const type_t* indices, size_t indices_count,
            int8_t* data_output, size_t combined_size, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;

        auto k = [=] ARIES_DEVICE(int index)
        {
            type_t idx = associated[ indices[ index ] ];

            size_t offset = 0;
            int8_t* pOutput = data_output + index * combined_size;
            for( int32_t i = 0; i < block_count; ++i )
            {
                DataBlockInfo info = data_input[i];
                int8_t* pInput = info.Data + idx * (size_t)info.ElementSize;
                if( info.Offset && !*pInput ) // info.Offset: 1 means has null, 0 means not null
                memset( pOutput + offset, 0, info.ElementSize );// if pInput is a null value, we use 0 to fill the memory
                else
                memcpy( pOutput + offset, pInput, info.ElementSize );
                offset += info.ElementSize;
            }
        };

        transform< launch_t >( k, indices_count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t >
    void gather_group_data( const DataBlockInfo* data_input, int32_t block_count, const type_t* associated, const type_t* indices,
            size_t indices_count, int8_t** data_output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            type_t idx = associated[ indices[ index ] ];
            for( int32_t i = 0; i < block_count; ++i )
            {
                DataBlockInfo info = data_input[i];
                memcpy( data_output[i] + index * (size_t)info.ElementSize, info.Data + idx * (size_t)info.ElementSize, (size_t)info.ElementSize );
            }
        };

        transform< launch_t >( k, indices_count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename type_u, typename type_r >
    void do_div( type_t* dividend, type_u* divisor, type_r* result, size_t count, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            type_r r(dividend[ index ]);
            r /= divisor[ index ];
            result[index] = r;
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename output_t >
    void convert_ariesbool_to_numeric( const AriesBool* data, size_t count, output_t* output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            output[index] = (output_t)data[index];
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename output_t >
    void convert_ariesbool_to_buf( const AriesBool* data, size_t count, output_t* output, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            AriesBool val = data[index];
            if( val.is_unknown() )
            {
                int8_t* flag = (int8_t*)( output + index );
                *flag = 0;
            }
            else
            {
                output[index] = (output_t)val.is_true();
            }
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t >
    void extend_data_size( const int8_t *data, size_t dataLen, size_t count, int8_t *outData, size_t outlen, size_t offset, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(int index)
        {
            *( outData + index * outlen ) = 1;
            memcpy( outData + index * outlen + offset, data + index * dataLen, dataLen );
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename output_t >
    void get_filtered_index(type_t *associated, size_t count, output_t *outIndex, output_t *selected_count, context_t& context ) {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(size_t index)
        {
            if (associated[index]) {
                outIndex[atomicAdd(selected_count, 1)] = index;
            }
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
    }

    template< typename launch_arg_t = empty_t, typename type_t, typename output_t >
    void gather_filtered_index( const type_t *associated, const output_t *psum, size_t count, output_t *outIndex, context_t& context )
    {
        typedef typename conditional_typedef_t< launch_arg_t, launch_box_t< arch_20_cta< 128, 3 >, arch_35_cta< 128, 6 >, arch_52_cta< 256, 15 > > >::type_t launch_t;
        auto k = [=] ARIES_DEVICE(size_t index)
        {
            if(associated[ index ] )
            {
                outIndex[ psum[ index ] ] = index;
            }
        };
        transform< launch_t >( k, count, context );
        context.synchronize();
    }

END_ARIES_ACC_NAMESPACE

#endif /* KERNEL_UTIL_HXX_ */
