// //
// // Created by david on 2020-6-10.
// //

// #include <gtest/gtest.h>

// // #include "CudaAcc/AriesDecimalAlgorithm.h"
// #include "CudaAcc/AriesSqlOperator_helper.h"

// BEGIN_ARIES_ACC_NAMESPACE

// using Decimal = aries_acc::Decimal;

// AriesDataBufferSPtr GenerateSourceData( uint16_t precision, uint16_t scale, bool hasNull, bool isCompact, size_t count, Decimal *init = nullptr)
// {
//     AriesColumnType type;
//     if ( isCompact )
//     {
//         type = AriesColumnType{ { AriesValueType::COMPACT_DECIMAL, precision, scale }, hasNull };
//     }
//     else
//     {
//         type = AriesColumnType{ { AriesValueType::DECIMAL, precision, scale }, hasNull };
//     }
//     auto len = type.GetDataTypeSize();
//     AriesDataBufferSPtr result = std::make_shared< AriesDataBuffer >( type, count );
//     int8_t *p = result->GetData();
//     int dataLen = len - ( hasNull ? 1 : 0 );
//     for ( size_t i = 0; i < count; ++i )
//     {
//         Decimal d( precision, scale );
//         if ( init == nullptr)
//         {
//             int startIndex = NUM_TOTAL_DIG - GetDecimalValidElementsCount( precision, scale );
//             int scalePartIndex = startIndex + NEEDELEMENTS( precision - scale );
//             int modValue = 10;
//             // for intg part
//             for ( int j = startIndex; j < scalePartIndex; ++j )
//             {
//                 modValue *= 10;
//                 d.values[j] = ( ( i + j ) * j ) % modValue;
//             }
//             // for frac part
//             int lastIntDigits = scale % DIG_PER_INT32;
//             int lastModValue = 1;
//             for ( int j = 0; j < lastIntDigits; ++j )
//             {
//                 lastModValue *= 10;
//             }
//             int lastValueBase = PER_DEC_MAX_SCALE / lastModValue;
//             modValue = PER_DEC_MAX_SCALE;
//             for ( size_t j = scalePartIndex; j < NUM_TOTAL_DIG; ++j )
//             {
//                 if ( j == NUM_TOTAL_DIG - 1 )
//                 {
//                     d.values[j] = ( ( i + j * j + 1 ) % lastModValue ) * lastValueBase;
//                 }
//                 else
//                 {
//                     d.values[j] = ( i + j ) % PER_DEC_MAX_SCALE;
//                 }
//             }
//         }
//         else
//         {
//             d = *init;
//         }

//         int8_t *t = p + len * i;
//         if ( hasNull )
//         {
//             *t++ = 1;
//         }
//         if ( isCompact )
//         {
//             d.ToCompactDecimal( (char *) t, dataLen );
//         }
//         else
//         {
//             memcpy( t, &d, dataLen );
//         }
//     }

//     return result;
// }

// AriesInt32ArraySPtr GenerateAssociateArray(size_t count)
// {
//     AriesInt32ArraySPtr associate = std::make_shared< AriesInt32Array >(count);
//     InitSequenceValue( associate );
//     return associate;
// }

// AriesInt32ArraySPtr GenerateIndicesArray( size_t count, size_t groupCount, AriesInt32ArraySPtr &groups )
// {
//     AriesInt32ArraySPtr indices = std::make_shared< AriesInt32Array >( count );
//     groups = std::make_shared< AriesInt32Array >( groupCount );
//     vector< int > hostIndices;
//     hostIndices.resize( count );
//     vector< int > hostGroups;
//     hostGroups.resize( groupCount );
//     auto gp = hostGroups.data();
//     auto ip = hostIndices.data();
//     size_t countOfGroup = count / groupCount;
//     ARIES_ASSERT( countOfGroup != 0, "group count is more than count" );
//     size_t endOfCurrentIndex = countOfGroup;
//     size_t groupId = 0;
//     gp[groupId] = 0;
//     for ( size_t i = 0; i < count; ++i )
//     {
//         if ( i == endOfCurrentIndex && groupId < groupCount - 1 )
//         {
//             ++groupId;
//             gp[groupId] = i;
//             endOfCurrentIndex += countOfGroup;
//         }
//         ip[i] = groupId;
//     }
//     indices->CopyFromHostMem( ip, indices->GetTotalBytes() );
//     groups->CopyFromHostMem( gp, groups->GetTotalBytes() );
//     return indices;
// }

// TEST( UT_TestAriesDecimalAlgorithm, sum )
// {
//     AriesDecimalAlgorithm::GetInstance().SetGpuMemorySize( (size_t) 2 * 1024 * 1024 * 1024 );
//     int count = 100000000;
//     int groupCount = 100;
//     AriesAggFunctionType aggType = AriesAggFunctionType::SUM;
//     SumStrategy strategy = SumStrategy::SOLE_TEMP_SUM;
//     auto srcCol = GenerateSourceData( 15, 2, false, false, count );
//     auto associate = GenerateAssociateArray( count );
//     AriesInt32ArraySPtr groups;
//     auto groupFlags = GenerateIndicesArray( count, groupCount, groups );
//     auto res = AriesDecimalAlgorithm::GetInstance().AggregateColumn( srcCol, aggType, associate, groupFlags, groups, strategy );
//     for ( int i = 0; i < groupCount; i++ )
//     {
//         ASSERT_EQ( res->GetDecimalAsString( i ), "49000000499995000.00" );
//     }
//     strategy = SumStrategy::NO_TEMP_SUM;
//     res = AriesDecimalAlgorithm::GetInstance().AggregateColumn( srcCol, aggType, associate, groupFlags, groups, strategy );
//     for ( int i = 0; i < groupCount; i++ )
//     {
//         ASSERT_EQ( res->GetDecimalAsString( i ), "49000000499995000.00" );
//     }
//     strategy = SumStrategy::SHARE_TEMP_SUM;
//     res = AriesDecimalAlgorithm::GetInstance().AggregateColumn( srcCol, aggType, associate, groupFlags, groups, strategy );
//     for ( int i = 0; i < groupCount; i++ )
//     {
//         ASSERT_EQ( res->GetDecimalAsString( i ), "49000000499995000.00" );
//     }

//     srcCol = GenerateSourceData( 15, 2, true, false, count );
//     strategy = SumStrategy::SOLE_TEMP_SUM;
//     res = AriesDecimalAlgorithm::GetInstance().AggregateColumn( srcCol, aggType, associate, groupFlags, groups, strategy );
//     for ( int i = 0; i < groupCount; i++ )
//     {
//         ASSERT_EQ( res->GetDecimalAsString( i ), "49000000499995000.00" );
//     }

//     srcCol = GenerateSourceData( 15, 2, false, true, count );
//     res = AriesDecimalAlgorithm::GetInstance().AggregateColumn( srcCol, aggType, associate, groupFlags, groups, strategy );
//     for ( int i = 0; i < groupCount; i++ )
//     {
//         ASSERT_EQ( res->GetDecimalAsString( i ), "49000000499995000.00" );
//     }

//     srcCol = GenerateSourceData( 15, 2, true, true, count );
//     res = AriesDecimalAlgorithm::GetInstance().AggregateColumn( srcCol, aggType, associate, groupFlags, groups, strategy );
//     for ( int i = 0; i < groupCount; i++ )
//     {
//         ASSERT_EQ( res->GetDecimalAsString( i ), "49000000499995000.00" );
//     }

//     // for SHARE_TEMP_SUM
//     count = 40000000;
//     groupCount = 1000;
//     aggType = AriesAggFunctionType::SUM;
//     strategy = SumStrategy::SHARE_TEMP_SUM;
//     srcCol = GenerateSourceData( 15, 2, false, false, count );
//     associate = GenerateAssociateArray( count );
//     groupFlags = GenerateIndicesArray( count, groupCount, groups );
//     res = AriesDecimalAlgorithm::GetInstance().AggregateColumn( srcCol, aggType, associate, groupFlags, groups, strategy );
//     for ( int i = 0; i < groupCount; i++ )
//     {
//         ASSERT_EQ( res->GetDecimalAsString( i ), "1960000019999800.00" );
//     }

//     // for cpu
//     count = 40000000;
//     groupCount = count - 1;
//     Decimal init( 15, 2, "8888888.88" );
//     srcCol = GenerateSourceData( 15, 2, false, false, count, &init );
//     aggType = AriesAggFunctionType::SUM;
//     strategy = SumStrategy::NO_TEMP_SUM;
//     groupFlags = GenerateIndicesArray( count, groupCount, groups );
//     res = AriesDecimalAlgorithm::GetInstance().AggregateColumn( srcCol, aggType, associate, groupFlags, groups, strategy );
//     for ( int i = 0; i < groupCount - 1; i++ )
//     {
//         ASSERT_EQ( res->GetDecimalAsString( i ), "8888888.88" );
//     }
//     ASSERT_EQ( res->GetDecimalAsString( groupCount - 1 ), "17777777.76" );
// }

// TEST( UT_TestAriesDecimalAlgorithm, count )
// {
//     AriesDecimalAlgorithm::GetInstance().SetGpuMemorySize( (size_t) 2 * 1024 * 1024 * 1024 );
//     int count = 100000000;
//     int groupCount = 1000;
//     AriesAggFunctionType aggType = AriesAggFunctionType::COUNT;
//     SumStrategy strategy = SumStrategy::SOLE_TEMP_SUM;
//     auto srcCol = GenerateSourceData( 15, 2, false, false, count );
//     auto associate = GenerateAssociateArray( count );
//     AriesInt32ArraySPtr groups;
//     auto groupFlags = GenerateIndicesArray( count, groupCount, groups );
//     auto res = AriesDecimalAlgorithm::GetInstance().AggregateColumn( srcCol, aggType, associate, groupFlags, groups, strategy );
//     for ( int i = 0; i < groupCount; i++ )
//     {
//         ASSERT_EQ( res->GetInt64( i ), count / groupCount );
//     }
//     strategy = SumStrategy::NO_TEMP_SUM;
//     res = AriesDecimalAlgorithm::GetInstance().AggregateColumn( srcCol, aggType, associate, groupFlags, groups, strategy );
//     for ( int i = 0; i < groupCount; i++ )
//     {
//         ASSERT_EQ( res->GetInt64( i ), count / groupCount );
//     }
//     strategy = SumStrategy::SHARE_TEMP_SUM;
//     res = AriesDecimalAlgorithm::GetInstance().AggregateColumn( srcCol, aggType, associate, groupFlags, groups, strategy );
//     for ( int i = 0; i < groupCount; i++ )
//     {
//         ASSERT_EQ( res->GetInt64( i ), count / groupCount );
//     }

//     srcCol = GenerateSourceData( 15, 2, true, false, count );
//     strategy = SumStrategy::SOLE_TEMP_SUM;
//     res = AriesDecimalAlgorithm::GetInstance().AggregateColumn( srcCol, aggType, associate, groupFlags, groups, strategy );
//     for ( int i = 0; i < groupCount; i++ )
//     {
//         ASSERT_EQ( res->GetInt64( i ), count / groupCount );
//     }

//     srcCol = GenerateSourceData( 15, 2, false, true, count );
//     res = AriesDecimalAlgorithm::GetInstance().AggregateColumn( srcCol, aggType, associate, groupFlags, groups, strategy );
//     for ( int i = 0; i < groupCount; i++ )
//     {
//         ASSERT_EQ( res->GetInt64( i ), count / groupCount );
//     }

//     srcCol = GenerateSourceData( 15, 2, true, true, count );
//     res = AriesDecimalAlgorithm::GetInstance().AggregateColumn( srcCol, aggType, associate, groupFlags, groups, strategy );
//     for ( int i = 0; i < groupCount; i++ )
//     {
//         ASSERT_EQ( res->GetInt64( i ), count / groupCount );
//     }
// }

// TEST( UT_TestAriesDecimalAlgorithm, maxmin )
// {
//     int count = 20;
//     int groupCount = 4;
//     string resMax[] = { "12000000021.21", "22000000036.26", "32000000051.31", "42000000066.36" };
//     string resMin[] = { "4000000009.17", "14000000024.22", "24000000039.27", "34000000054.32" };
//     AriesAggFunctionType aggType = AriesAggFunctionType::MAX;
//     SumStrategy strategy = SumStrategy::SOLE_TEMP_SUM;
//     auto srcCol = GenerateSourceData( 15, 2, false, false, count );
//     auto associate = GenerateAssociateArray( count );
//     AriesInt32ArraySPtr groups;
//     auto groupFlags = GenerateIndicesArray( count, groupCount, groups );
//     auto res = AriesDecimalAlgorithm::GetInstance().AggregateColumn( srcCol, aggType, associate, groupFlags, groups, strategy );
//     for ( int i = 0; i < groupCount; i++ )
//     {
//         ASSERT_EQ( res->GetDecimalAsString( i ), resMax[i] );
//     }

//     aggType = AriesAggFunctionType::MIN;
//     res = AriesDecimalAlgorithm::GetInstance().AggregateColumn( srcCol, aggType, associate, groupFlags, groups, strategy );
//     for ( int i = 0; i < groupCount; i++ )
//     {
//         ASSERT_EQ( res->GetDecimalAsString( i ), resMin[i] );
//     }

//     strategy = SumStrategy::NO_TEMP_SUM;
//     aggType = AriesAggFunctionType::MAX;
//     res = AriesDecimalAlgorithm::GetInstance().AggregateColumn( srcCol, aggType, associate, groupFlags, groups, strategy );
//     for ( int i = 0; i < groupCount; i++ )
//     {
//         ASSERT_EQ( res->GetDecimalAsString( i ), resMax[i] );
//     }

//     aggType = AriesAggFunctionType::MIN;
//     res = AriesDecimalAlgorithm::GetInstance().AggregateColumn( srcCol, aggType, associate, groupFlags, groups, strategy );
//     for ( int i = 0; i < groupCount; i++ )
//     {
//         ASSERT_EQ( res->GetDecimalAsString( i ), resMin[i] );
//     }
// }

// END_ARIES_ACC_NAMESPACE
