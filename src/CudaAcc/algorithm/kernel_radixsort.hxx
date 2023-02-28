/*
 * kernel_radixsort.hxx
 *
 *  Created on: Jul 17, 2019
 *      Author: lichi
 */

#ifndef KERNEL_RADIXSORT_HXX_
#define KERNEL_RADIXSORT_HXX_
#include "cub/cub.cuh"
#include "kernel_util.hxx"
#include "kernel_scan.hxx"
#include "kernel_shuffle.hxx"
#include "kernel_mergesort.hxx"
#include "AriesDeviceProperty.h"

#include <glog/logging.h>

BEGIN_ARIES_ACC_NAMESPACE

    template< typename val_t, typename output_t = char >
    void getCharNfromCol( const char* colData, size_t len, const val_t * associated, int tupleNum, int round, output_t *charInString,
            context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            charInString[ index ] = *( colData + associated[ index ] * len + round );
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, tupleNum, context );
        context.synchronize();
    }

    template< typename val_t >
    void getCharAsUint32fromCol( const char* colData, size_t len, const val_t * associated, int tupleNum, int offset, uint32_t *charInString,
            context_t& context )
    {
        assert( len - offset > 0 );
        size_t copySize = std::min( size_t( len - offset ), sizeof(uint32_t) );
        auto k = [=] ARIES_DEVICE(int index)
        {
            uint32_t temp = 0;
            memcpy( &temp, colData + len * associated[ index ] + offset, copySize );
            charInString[ index ] = __byte_perm (temp, 0, 0x0123);
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, tupleNum, context );
        context.synchronize();
    }

    template< typename type_t, template< typename > class type_nullable, typename val_t >
    void getValuefromNullable( const type_nullable< type_t >* keys, int tupleNum, const val_t& associated, type_t *output, context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            output[ index ] = keys[ associated[ index ] ].value;
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, tupleNum, context );
        context.synchronize();
    }

    template< typename type_t, template< typename > class type_nullable, typename output_t = int8_t >
    void getflagfromNullable( const type_nullable< type_t >* keys, int tupleNum, output_t *output, context_t& context )
    {
        auto k = [=] ARIES_DEVICE(int index)
        {
            output[ index ] = keys[ index ].flag;
        };
        transform< launch_box_t< arch_52_cta< 256, 15 > > >( k, tupleNum, context );
        context.synchronize();
    }

    template< typename key_t, typename val_t >
    void radix_sort( const key_t* keys, const val_t* values, key_t* sorted_keys, val_t* sorted_values, int count, context_t& context,
            bool bAsc = true )
    {
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;

        if( bAsc )
        {
            cub::DeviceRadixSort::SortPairs( d_temp_storage, temp_storage_bytes, keys, sorted_keys, values, sorted_values, count );
            mem_t< int8_t > temp( temp_storage_bytes );
            d_temp_storage = temp.data();
            cub::DeviceRadixSort::SortPairs( d_temp_storage, temp_storage_bytes, keys, sorted_keys, values, sorted_values, count );
        }
        else
        {
            cub::DeviceRadixSort::SortPairsDescending( d_temp_storage, temp_storage_bytes, keys, sorted_keys, values, sorted_values, count );
            mem_t< int8_t > temp( temp_storage_bytes );
            d_temp_storage = temp.data();
            cub::DeviceRadixSort::SortPairsDescending( d_temp_storage, temp_storage_bytes, keys, sorted_keys, values, sorted_values, count );
        }
    }

    template< typename key_t, typename val_t >
    void radix_sort_ex( const key_t* keys, const val_t* values, key_t* sorted_keys, val_t* sorted_values, int count, context_t& context, bool bAsc = true )
    {
        if ( 0 == count )
            return;
        int device_id = context.active_device_id();
        size_t mem_cost = ( sizeof( key_t ) + sizeof( val_t ) ) * count * 3.5;
        size_t mem_available = AriesDeviceProperty::GetInstance().GetMemoryCapacity() * 0.9;
        size_t block_count = 1;
        size_t round_count = 1;
        while( mem_cost > mem_available )
        {
            mem_cost >>= 1;
            block_count <<= 1;
            ++round_count;
        }

        size_t block_size = count / block_count;
        size_t tail_size = count % block_count;

        //分块排序
        size_t block_offset = 0;
        const key_t* input_keys = nullptr;
        const val_t* input_vals = nullptr;
        key_t* output_keys = nullptr;
        val_t* output_vals = nullptr;
        LOG(INFO) << "radix_sort_ex block_count" << block_count;
        for (size_t i = 0; i < block_count; ++i)
        {
            size_t item_count = ( i == 0 ? block_size + tail_size : block_size );
            input_keys = keys + block_offset;
            input_vals = values + block_offset;
            output_keys = sorted_keys + block_offset;
            output_vals = sorted_values + block_offset;
            cudaMemPrefetchAsync( input_keys, sizeof( key_t ) * item_count, device_id );
            cudaMemPrefetchAsync( input_vals, sizeof( val_t ) * item_count, device_id );

            cudaMemPrefetchAsync( output_keys, sizeof( key_t ) * item_count, device_id );
            cudaMemPrefetchAsync( output_vals, sizeof( val_t ) * item_count, device_id );

            radix_sort( input_keys, input_vals, output_keys, output_vals, item_count, context, bAsc );
            context.synchronize();
            block_offset += item_count;
        }

        if( block_count > 1 )
        {
            //merge
            managed_mem_t< key_t > keys( count, context, false );
            key_t* keys_output = keys.data();
            managed_mem_t< val_t > vals( count, context, false );
            val_t* vals_output = vals.data();

            for( int round = 1; round < round_count; ++round )
            {
                size_t input_offset = 0;
                size_t output_offset = 0;
                for( int i = 0; i < ( block_count >> round ); ++i )
                {
                    size_t second_sorted_block_size = block_size * ( 1 << ( round - 1 ) );
                    size_t first_sorted_block_size = second_sorted_block_size + ( i == 0 ? tail_size : 0 );
                    
                    mem_t< int > partitions;
                    if( bAsc )
                        partitions = std::move( merge_path_partitions< bounds_lower >( sorted_keys + input_offset, first_sorted_block_size, 
                                                                                    sorted_keys + input_offset + first_sorted_block_size, second_sorted_block_size, 
                                                                                    block_size, less_t< key_t >(), context ) );
                    else 
                        partitions = std::move( merge_path_partitions< bounds_lower >( sorted_keys + input_offset, first_sorted_block_size, 
                                                                                    sorted_keys + input_offset + first_sorted_block_size, second_sorted_block_size, 
                                                                                    block_size, greater_t< key_t >(), context ) );

                    managed_mem_t< merge_range_t > merge_ranges( partitions.size() - 1, context, false ); 
                    merge_range_t *range = merge_ranges.data();
                    for( int p = 0; p < partitions.size() - 1; ++p )
                    {
                        merge_range_t r = compute_merge_range( first_sorted_block_size, second_sorted_block_size, p, block_size, 
                                                            partitions.get_value( p ), partitions.get_value( p + 1 ) );
                        range[ p ] = r;
                    }

                    for( int index = 0; index < merge_ranges.size(); ++index )
                    {
                        merge_range_t r = range[ index ];
                        if( r.a_count() > 0 )
                            cudaMemPrefetchAsync( sorted_keys + input_offset + r.a_begin, sizeof( key_t ) * r.a_count(), device_id );
                        if( r.b_count() > 0 )
                            cudaMemPrefetchAsync( sorted_keys + input_offset + first_sorted_block_size + r.b_begin, sizeof( key_t ) * r.b_count(), device_id );

                        if( r.a_count() > 0 )
                            cudaMemPrefetchAsync( sorted_values + input_offset + r.a_begin, sizeof( val_t ) * r.a_count(), device_id );
                        if( r.b_count() > 0 )
                            cudaMemPrefetchAsync( sorted_values + input_offset + first_sorted_block_size + r.b_begin, sizeof( val_t ) * r.b_count(), device_id );
                        
                        
                        cudaMemPrefetchAsync( keys_output + output_offset, sizeof( key_t ) * r.total(), device_id );
                        cudaMemPrefetchAsync( vals_output + output_offset, sizeof( val_t ) * r.total(), device_id );
                        if( bAsc )
                            merge( sorted_keys + input_offset + r.a_begin, 
                                sorted_values + input_offset + r.a_begin, r.a_count(), 
                                sorted_keys + input_offset + first_sorted_block_size + r.b_begin, 
                                sorted_values + input_offset + first_sorted_block_size + r.b_begin, r.b_count(),
                                keys_output + output_offset, vals_output + output_offset, less_t< key_t >(), context );
                        else 
                            merge( sorted_keys + input_offset + r.a_begin, 
                                sorted_values + input_offset + r.a_begin, r.a_count(), 
                                sorted_keys + input_offset + first_sorted_block_size + r.b_begin, 
                                sorted_values + input_offset + first_sorted_block_size + r.b_begin, r.b_count(),
                                keys_output + output_offset, vals_output + output_offset, greater_t< key_t >(), context );
                        context.synchronize();

                        output_offset += r.total();
                    }
                    
                    input_offset += first_sorted_block_size + second_sorted_block_size;
                }
                std::swap( sorted_keys, keys_output );
                std::swap( sorted_values, vals_output );
            }
            if( round_count % 2 == 0 )
            {
                cudaMemcpy( keys_output, sorted_keys, count * sizeof( key_t ), cudaMemcpyDefault );
                keys.free();
                cudaMemcpy( vals_output, sorted_values, count * sizeof( val_t ), cudaMemcpyDefault );
                context.synchronize();
            }
        }
    }

    template< typename type_t, typename val_t, typename enable_if< is_arithmetic< type_t >::value, type_t >::type* = nullptr >
    void radix_sort( type_t* keys, val_t* associated, int count, context_t& context, bool bAsc = true, bool bShuffle = false )
    {
        if( count == 0 )
            return;
        mem_t< type_t > keys_output( count );
        mem_t< val_t > associated_output( count );
        radix_sort_ex( keys, associated, keys_output.data(), associated_output.data(), count, context, bAsc );
        cudaMemcpy( associated, associated_output.data(), count * sizeof(val_t), cudaMemcpyDefault );
        if( bShuffle )
            cudaMemcpy( keys, keys_output.data(), count * sizeof(type_t), cudaMemcpyDefault );
    }

    template< typename type_t, template< typename > class type_nullable, typename val_t,
            typename enable_if< is_arithmetic< type_t >::value, type_t >::type* = nullptr >
    void radix_sort( type_nullable< type_t >* keys, val_t* output_associated, int tupleNum, context_t& context, bool bAsc = true, bool nullSmaller =
            true, bool bShuffle = false )
    {
        if( tupleNum == 0 )
            return;
        mem_t< val_t > associatedData( tupleNum );
        val_t* associated = associatedData.data();
        init_sequence( associated, tupleNum, context );

        mem_t< int32_t > flags( tupleNum );
        int32_t* flagData = flags.data();
        getflagfromNullable( keys, tupleNum, flagData, context );

        mem_t< int32_t > outPrefixSum( tupleNum );
        managed_mem_t< int32_t > outPrefixSumCount( 1, context );
        scan( flagData, tupleNum, outPrefixSum.data(), plus_t< int32_t >(), outPrefixSumCount.data(), context );
        context.synchronize();
        int32_t notNullTotal = outPrefixSumCount.data()[0];
        ARIES_ASSERT( tupleNum >= notNullTotal,
                "sort nullable data error, tupleNum: " + std::to_string( tupleNum ) + ", notNullTotal: " + std::to_string( notNullTotal ) );
        int32_t nullCount = tupleNum - notNullTotal;

        mem_t< val_t > sortedAssociatedData( tupleNum );
        val_t* sorted_associated = sortedAssociatedData.data();
        if( nullCount > 0 )
        {
            if( ( nullSmaller && bAsc ) || ( !nullSmaller && !bAsc ) )
                gather_filtered_data( associated, tupleNum, flagData, outPrefixSum.data(), sorted_associated + nullCount, context );
            else
                gather_filtered_data( associated, tupleNum, flagData, outPrefixSum.data(), sorted_associated, context );

            flip_flags( flagData, tupleNum, context );
            scan( flagData, tupleNum, outPrefixSum.data(), plus_t< int32_t >(), outPrefixSum.data() + tupleNum, context );
            if( ( nullSmaller && bAsc ) || ( !nullSmaller && !bAsc ) )
                gather_filtered_data( associated, tupleNum, flagData, outPrefixSum.data(), sorted_associated, context );
            else
                gather_filtered_data( associated, tupleNum, flagData, outPrefixSum.data(), sorted_associated + notNullTotal, context );

            cudaMemcpy( associated, sorted_associated, tupleNum * sizeof(val_t), cudaMemcpyDefault );
            if( ( nullSmaller && bAsc ) || ( !nullSmaller && !bAsc ) )
                associated += nullCount;
        }

        mem_t< type_t > vals_input( notNullTotal );
        getValuefromNullable( keys, notNullTotal, associated, vals_input.data(), context );
        radix_sort( vals_input.data(), associated, notNullTotal, context, bAsc, true );

        if( ( nullSmaller && bAsc ) || ( !nullSmaller && !bAsc ) )
            associated -= nullCount;

        shuffle_by_index( output_associated, tupleNum, associated, sorted_associated, context );
        cudaMemcpy( output_associated, sorted_associated, tupleNum * sizeof(val_t), cudaMemcpyDefault );
        if( bShuffle )
        {
            mem_t< type_nullable< type_t > > keys_temp( tupleNum );
            type_nullable< type_t >* keys_temp_data = keys_temp.data();
            cudaMemset( keys_temp_data, 0, tupleNum * sizeof(type_nullable< type_t > ) );
            if( ( nullSmaller && bAsc ) || ( !nullSmaller && !bAsc ) )
                keys_temp_data += nullCount;
            DataBlockInfo block;
            block.Data = ( int8_t* )vals_input.data();
            block.ElementSize = sizeof(type_t);
            make_column_nullable( block, notNullTotal, ( int8_t* )keys_temp_data, context );
            if( ( nullSmaller && bAsc ) || ( !nullSmaller && !bAsc ) )
                keys_temp_data -= nullCount;
            cudaMemcpy( keys, keys_temp_data, tupleNum * sizeof(type_nullable< type_t > ), cudaMemcpyDefault );
        }
    }

    template< typename val_t >
    void radix_sort( char* keys, size_t len, val_t* output_associated, int tupleNum, context_t& context, bool bAsc = true, bool bShuffle = false )
    {
        if( tupleNum == 0 )
            return;
        mem_t< uint32_t > charData( tupleNum );
        uint32_t *charInString = charData.data();

        mem_t< uint32_t > sortedCharData( tupleNum );
        uint32_t *sortedCharInString = sortedCharData.data();

        mem_t< val_t > associatedData( tupleNum );
        val_t* associated = associatedData.data();
        init_sequence( associated, tupleNum, context );

        mem_t< val_t > sortedAssociatedData( tupleNum );
        val_t* sorted_associated = sortedAssociatedData.data();

        uint32_t loopCount = div_up( len, sizeof(uint32_t) );
        uint32_t offset = ( loopCount - 1 ) * sizeof(uint32_t);

        for( uint32_t i = 0; i < loopCount; ++i )
        {
            getCharAsUint32fromCol( keys, len, associated, tupleNum, offset, charInString, context );
            radix_sort_ex( charInString, associated, sortedCharInString, sorted_associated, tupleNum, context, bAsc );
            std::swap( associated, sorted_associated );
            offset -= sizeof(uint32_t);
        }

        shuffle_by_index( output_associated, tupleNum, associated, sorted_associated, context );
        cudaMemcpy( output_associated, sorted_associated, tupleNum * sizeof(val_t), cudaMemcpyDefault );
        if( bShuffle )
        {
            mem_t< char > keys_temp( len * tupleNum );
            char* keys_temp_data = keys_temp.data();
            shuffle_by_index( keys, len, tupleNum, associated, keys_temp_data, context );
            cudaMemcpy( keys, keys_temp_data, len * tupleNum, cudaMemcpyDefault );
        }
    }

    template< typename val_t >
    void radix_sort_has_null( char* keys, size_t len, val_t* output_associated, int tupleNum, context_t& context, bool bAsc = true, bool nullSmaller =
            true, bool bShuffle = false )
    {
        if( tupleNum == 0 )
            return;
        mem_t< val_t > associatedData( tupleNum ); // alloc more space for the last memcpy
        val_t* associated = associatedData.data();
        init_sequence( associated, tupleNum, context );

        mem_t< int32_t > flags( tupleNum );
        int32_t* flagInString = flags.data();
        getCharNfromCol( keys, len, associated, tupleNum, 0, flagInString, context );

        mem_t< int32_t > outPrefixSum( tupleNum );
        managed_mem_t< int32_t > outPrefixSumCount( 1, context );
        scan( flagInString, tupleNum, outPrefixSum.data(), plus_t< int32_t >(), outPrefixSumCount.data(), context );
        context.synchronize();
        int32_t notNullTotal = outPrefixSumCount.data()[0];

        ARIES_ASSERT( tupleNum >= notNullTotal,
                "sort nullable data error, tupleNum: " + std::to_string( tupleNum ) + ", notNullTotal: " + std::to_string( notNullTotal ) );
        int32_t nullCount = tupleNum - notNullTotal;
        if( nullCount > 0 )
        {
            mem_t< val_t > associatedData( tupleNum );
            if( ( nullSmaller && bAsc ) || ( !nullSmaller && !bAsc ) )
                gather_filtered_data( associated, tupleNum, flagInString, outPrefixSum.data(), associatedData.data() + nullCount, context );
            else
                gather_filtered_data( associated, tupleNum, flagInString, outPrefixSum.data(), associatedData.data(), context );

            flip_flags( flagInString, tupleNum, context );
            scan( flagInString, tupleNum, outPrefixSum.data(), plus_t< int32_t >(), outPrefixSum.data() + tupleNum, context );
            context.synchronize();
            if( ( nullSmaller && bAsc ) || ( !nullSmaller && !bAsc ) )
                gather_filtered_data( associated, tupleNum, flagInString, outPrefixSum.data(), associatedData.data(), context );
            else
                gather_filtered_data( associated, tupleNum, flagInString, outPrefixSum.data(), associatedData.data() + notNullTotal, context );
            cudaMemcpy( associated, associatedData.data(), tupleNum * sizeof(val_t), cudaMemcpyDefault );
            if( ( nullSmaller && bAsc ) || ( !nullSmaller && !bAsc ) )
                associated += nullCount;
        }

        flags.free();
        outPrefixSum.free();

        mem_t< uint32_t > charData( notNullTotal );
        uint32_t *charInString = charData.data();

        mem_t< uint32_t > sortedCharData( notNullTotal );
        uint32_t *sortedCharInString = sortedCharData.data();

        mem_t< val_t > sortedAssociatedData( tupleNum );
        val_t* sorted_associated = sortedAssociatedData.data();

        uint32_t loopCount = div_up( len - 1, sizeof(uint32_t) );
        uint32_t offset = ( loopCount - 1 ) * sizeof(uint32_t) + 1;

        for( uint32_t i = 0; i < loopCount; ++i )
        {
            getCharAsUint32fromCol( keys, len, associated, notNullTotal, offset, charInString, context );
            radix_sort_ex( charInString, associated, sortedCharInString, sorted_associated, notNullTotal, context, bAsc );
            std::swap( associated, sorted_associated );
            offset -= sizeof(uint32_t);
        }

        if( loopCount % 2 )
        {
            std::swap( associated, sorted_associated );
            cudaMemcpy( associated, sorted_associated, notNullTotal * sizeof(val_t), cudaMemcpyDefault );
        }

        if( ( nullSmaller && bAsc ) || ( !nullSmaller && !bAsc ) )
            associated -= nullCount;

        shuffle_by_index( output_associated, tupleNum, associated, sorted_associated, context );
        cudaMemcpy( output_associated, sorted_associated, tupleNum * sizeof(val_t), cudaMemcpyDefault );
        if( bShuffle )
        {
            mem_t< char > keys_temp( len * tupleNum );
            char* keys_temp_data = keys_temp.data();
            shuffle_by_index( keys, len, tupleNum, associated, keys_temp_data, context );
            cudaMemcpy( keys, keys_temp_data, len * tupleNum, cudaMemcpyDefault );
        }
    }

END_ARIES_ACC_NAMESPACE

#endif /* KERNEL_RADIXSORT_HXX_ */
