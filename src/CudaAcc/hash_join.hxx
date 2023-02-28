#ifndef __HASH_JOIN_HXX__
#define __HASH_JOIN_HXX__

#include <vector>
#include "CudaAcc/AriesEngineException.h"
#include "AriesEngineDef.h"
#include "algorithm/context.hxx"
BEGIN_ARIES_ACC_NAMESPACE

#define HASH_SCALE_FACTOR 2.0
#define HASH_BAD_SCALE_FACTOR 0.05

#ifdef LOWER_API_LEVEL
typedef int flag_type_t;
#define FLAG_EMTPY_VALUE 0xFFFFFFFF
#else
typedef unsigned short flag_type_t;
#define FLAG_EMTPY_VALUE 0xFFFF
#endif

template< typename type_t >
HashTable< type_t > build_hash_table( type_t** inputs,
                                      int64_t* block_size_prefix_sum,
                                      size_t block_count,
                                      bool nullable,
                                      int8_t* indices,
                                      AriesValueType indiceValueType,
                                      size_t count,
                                      context_t& context );

AriesHashTableMultiKeysUPtr build_hash_table( ColumnDataIterator* inputs, int input_count, size_t total_row_count, context_t& context );

AriesHashTableUPtr build_hash_table( ColumnDataIterator* input, size_t total_row_count, AriesValueType value_type, context_t& context );

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
                          context_t& context );

template< typename type_a >
JoinPair hash_inner_join( const HashTable< type_a >& hash_table,
                          ColumnDataIterator* input,
                          size_t count,
                          AriesColumnType column_type,
                          const AriesIndicesArraySPtr& hash_table_indices,
                          const AriesIndicesArraySPtr& input_table_indices,
                          context_t& context );

/**
 * star join
 */

AriesStarJoinResult
hash_inner_join( const std::vector< AriesHashTableWrapper >& hash_tables,
                 const std::vector< AriesHashJoinDataWrapper >& datas,
                 size_t total_row_count,
                 const std::vector< AriesIndicesArraySPtr >& hash_table_indices,
                 const AriesIndicesArraySPtr& input_table_indices,
                 context_t& context );

AriesJoinResult
hash_inner_join( const AriesHashTableMultiKeys& hash_tables, 
                 ColumnDataIterator* inputs,
                 int inputs_count,
                 size_t total_row_count,
                 const AriesIndicesArraySPtr& hash_table_indices,
                 const AriesIndicesArraySPtr& input_table_indices,
                 context_t& context );

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
                          context_t& context );

template< typename type_a >
JoinPair hash_left_join(  const HashTable< type_a >& hash_table,
                          ColumnDataIterator* input,
                          AriesColumnType column_type,
                          size_t left_count,
                          size_t right_count,
                          int* left_matched_count,
                          context_t& context );

template< typename type_a, typename type_b >
AriesInt32ArraySPtr simple_half_join_left_as_hash( const HashTable< type_a >& hash_table,
                                        type_b ** inputs,
                                        int64_t* block_size_prefix_sum,
                                        size_t block_count,
                                        int8_t* indices,
                                        AriesValueType indiceValueType,
                                        size_t input_row_count,
                                        size_t output_count,
                                        bool is_semi_join,
                                        bool isNotIn,
                                        context_t& context );

template< typename type_a >
AriesInt32ArraySPtr simple_half_join_left_as_hash( const HashTable< type_a >& hash_table,
                          ColumnDataIterator* input,
                          AriesColumnType column_type,
                          size_t input_row_count,
                          size_t output_count,
                          bool is_semi_join,
                          bool isNotIn,
                          context_t& context );

template< typename type_a, typename type_b >
AriesInt32ArraySPtr simple_half_join_right_as_hash( const HashTable< type_a >& hash_table,
                                        type_b ** inputs,
                                        int64_t* block_size_prefix_sum,
                                        size_t block_count,
                                        int8_t* indices,
                                        AriesValueType indiceValueType,
                                        size_t input_row_count,
                                        bool is_semi_join,
                                        bool isNotIn,
                                        bool use_indices_for_output,
                                        context_t& context );

template< typename type_a >
AriesInt32ArraySPtr simple_half_join_right_as_hash( const HashTable< type_a >& hash_table,
                          ColumnDataIterator* input,
                          AriesColumnType column_type,
                          size_t input_row_count,
                          bool is_semi_join,
                          bool isNotIn,
                          bool use_indices_for_output,
                          context_t& context );

END_ARIES_ACC_NAMESPACE

#endif
