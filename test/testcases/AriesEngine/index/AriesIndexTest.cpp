/*
 * AriesIndexTest.cpp
 *
 *  Created on: Apr 27, 2020
 *      Author: lichi
 */

#include "gtest/gtest.h"
#include "AriesEngine/index/AriesIndex.h"
#include "aries_types.hxx"
#include <iostream>
#include <cstring>
#include <random>
#include <sys/time.h>
#include "CpuTimer.h"

using namespace aries_engine;
using namespace aries;
using namespace aries_acc;
using namespace std;
// namespace
// {
//     struct CPU_Timer
//     {
//         long start;
//         long stop;
//         void begin()
//         {
//             struct timeval tv;
//             gettimeofday( &tv, NULL );
//             start = tv.tv_sec * 1000 + tv.tv_usec / 1000;
//         }
//         long end()
//         {
//             struct timeval tv;
//             gettimeofday( &tv, NULL );
//             stop = tv.tv_sec * 1000 + tv.tv_usec / 1000;
//             long elapsed = stop - start;
//             printf( "cpu time: %ld\n", elapsed );
//             return elapsed;
//         }
//     };
// }

TEST(AriesIndex, intless)
{
    IAriesIndexSPtr indices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_unique );
    IAriesIndexSPtr dupIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_duplicate );
    int count = 10000;
    std::uniform_int_distribution< int > d( 0, 10000000 );
    std::mt19937 mt19937;
    cout << "insert " << count << " values:" << endl;
    aries::CPU_Timer t;
    t.begin();
    int val;
    for( int i = 0; i < count; ++i )
    {
        val = d( mt19937 ) % 100000;
        indices->Insert( &val, i );
        dupIndices->Insert( &val, i );
    }
    t.end();
    int key = 10000001;
    auto result = indices->FindLess( &key );
    string filename = "AriesIndexint.out";
    cout << "save:" << endl;
    t.begin();
    indices->Save( filename );
    t.end();
    cout << "load:" << endl;
    IAriesIndexSPtr indices2 = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_unique );
    t.begin();
    indices2->Load( filename );
    t.end();
    cout << "find less:" << endl;
    t.begin();
    auto result2 = indices2->FindLess( &key );
    auto result3 = dupIndices->FindLess( &key );
    t.end();
    ASSERT_EQ( result.size(), count );
    ASSERT_EQ( result.size(), result3.size() );
    for( int i = 0; i < result.size(); ++i )
        ASSERT_EQ( result[ i ], result3[ i ] );
}

TEST(AriesIndex, nullintless)
{
    IAriesIndexSPtr indices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, true, false },
            key_hint_t::most_unique );

    int count = 100;
    cout << "insert " << count << " values:" << endl;
    aries::CPU_Timer t;
    t.begin();
    nullable_type< int > data;
    for( int i = 0; i < count; ++i )
    {
        if( i % 10 )
            data.flag = 1;
        else
            data.flag = 0;

        indices->Insert( &data, i );
    }
    t.end();
    nullable_type< int > key = 10000001;
    auto result = indices->FindLess( &key );
    string filename = "AriesIndexint.out";
    cout << "save:" << endl;
    t.begin();
    indices->Save( filename );
    t.end();
    cout << "load:" << endl;
    IAriesIndexSPtr indices2 = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, true, false },
            key_hint_t::most_unique );
    t.begin();
    indices2->Load( filename );
    t.end();
    cout << "find less:" << endl;
    t.begin();
    auto result2 = indices2->FindLess( &key );
    t.end();
    ASSERT_EQ( result.size(), 90 );
    ASSERT_EQ( result.size(), result2.size() );
    for( int i = 0; i < result.size(); ++i )
        ASSERT_EQ( result[ i ], result2[ i ] );
}

TEST(AriesIndex, nullstringless)
{
    IAriesIndexSPtr uniqueIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::CHAR, 2 }, true, false },
            key_hint_t::most_unique );
    IAriesIndexSPtr dupIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::CHAR, 2 }, true, false },
            key_hint_t::most_duplicate );
    int count = 10;
    cout << "insert " << count << " values:" << endl;
    aries::CPU_Timer t;
    t.begin();
    unpacked_nullable_type< std::string > data;
    for( int i = 0; i < count; ++i )
    {
        if( i % 2 )
            data.flag = 1;
        else
            data.flag = 0;
        data.value = std::to_string( 10 + i );

        uniqueIndices->Insert( &data, i );
        dupIndices->Insert( &data, i );
    }
    t.end();
    unpacked_nullable_type< std::string > key = unpacked_nullable_type< std::string >( 1, "20" );
    auto result = uniqueIndices->FindLess( &key );
    string filename = "AriesIndexint.out";
    cout << "save:" << endl;
    t.begin();
    uniqueIndices->Save( filename );
    t.end();
    cout << "load:" << endl;
    IAriesIndexSPtr indices2 = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::CHAR, 2 }, true, false },
            key_hint_t::most_unique );
    t.begin();
    indices2->Load( filename );
    t.end();
    cout << "find less:" << endl;
    t.begin();
    auto result2 = indices2->FindLess( &key );
    t.end();
    auto result3 = dupIndices->FindLess( &key );
    ASSERT_EQ( result.size(), 5 );
    ASSERT_EQ( result.size(), result3.size() );
    for( int i = 0; i < result.size(); ++i )
        ASSERT_EQ( result[ i ], result3[ i ] );
    for( int i : result3 )
        ASSERT_EQ( i % 2, 1 );
}

TEST(AriesIndex, inserterase)
{
    IAriesIndexSPtr uniqueIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_unique );
    IAriesIndexSPtr dupIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_duplicate );

    int data[] = { 1, 2, 2, 3, 3, 4, 5 };
    for( int i = 0; i < sizeof( data ) / sizeof(int); ++i )
    {
        uniqueIndices->Insert( &data[ i ], i );
        dupIndices->Insert( &data[ i ], i );
    }

    //erase ( 1, 0 )
    int val = 1;
    uniqueIndices->Erase( &val, 0 );
    dupIndices->Erase( &val, 0 );

    //erase ( 2, 2 )
    val = 2;
    uniqueIndices->Erase( &val, 2 );
    dupIndices->Erase( &val, 2 );

    //erase ( 3, 3 )
    val = 3;
    uniqueIndices->Erase( &val, 3 );
    dupIndices->Erase( &val, 3 );

    //erase ( 3, 4 )
    val = 3;
    uniqueIndices->Erase( &val, 4 );
    dupIndices->Erase( &val, 4 );

    //erase ( 5, 6 )
    val = 5;
    uniqueIndices->Erase( &val, 6 );
    dupIndices->Erase( &val, 6 );

    int key = 5;
    auto resultUnique = uniqueIndices->FindLess( &key );
    auto resultDup = dupIndices->FindLess( &key );

    ASSERT_EQ( resultUnique.size(), 2 );
    ASSERT_EQ( resultUnique.size(), resultDup.size() );
    int res[] = { 1, 5 };
    for( int i = 0; i < resultUnique.size(); ++i )
    {
        ASSERT_EQ( resultUnique[ i ], res[ i ] );
        ASSERT_EQ( resultUnique[ i ], resultDup[ i ] );
    }
}

TEST(AriesIndex, inteq)
{
    IAriesIndexSPtr uniqueIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_unique );
    IAriesIndexSPtr dupIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_duplicate );

    int data[] = { 1, 2, 2, 3, 3, 4, 5 };
    for( int i = 0; i < sizeof( data ) / sizeof(int); ++i )
    {
        uniqueIndices->Insert( &data[ i ], i );
        dupIndices->Insert( &data[ i ], i );
    }

    int key = 2;
    auto resultUnique = uniqueIndices->FindEqual( &key );
    auto resultDup = dupIndices->FindEqual( &key );

    ASSERT_EQ( resultUnique.size(), 2 );
    ASSERT_EQ( resultUnique.size(), resultDup.size() );
    int res[] = { 1, 2 };
    for( int i = 0; i < resultUnique.size(); ++i )
    {
        ASSERT_EQ( resultUnique[ i ], res[ i ] );
        ASSERT_EQ( resultUnique[ i ], resultDup[ i ] );
    }

    key = 0;
    resultUnique = uniqueIndices->FindEqual( &key );
    resultDup = dupIndices->FindEqual( &key );

    ASSERT_EQ( resultUnique.size(), 0 );
    ASSERT_EQ( resultUnique.size(), resultDup.size() );
}

TEST(AriesIndex, intne)
{
    IAriesIndexSPtr uniqueIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_unique );
    IAriesIndexSPtr dupIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_duplicate );

    int data[] = { 1, 2, 2, 3, 3, 4, 5 };
    for( int i = 0; i < sizeof( data ) / sizeof(int); ++i )
    {
        uniqueIndices->Insert( &data[ i ], i );
        dupIndices->Insert( &data[ i ], i );
    }

    int key = 3;
    auto resultUnique = uniqueIndices->FindNotEqual( &key );
    auto resultDup = dupIndices->FindNotEqual( &key );

    ASSERT_EQ( resultUnique.size(), 5 );
    ASSERT_EQ( resultUnique.size(), resultDup.size() );
    int res[] = { 0, 1, 2, 5, 6 };
    for( int i = 0; i < resultUnique.size(); ++i )
    {
        ASSERT_EQ( resultUnique[ i ], res[ i ] );
        ASSERT_EQ( resultUnique[ i ], resultDup[ i ] );
    }
}

TEST(AriesIndex, intle)
{
    IAriesIndexSPtr uniqueIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_unique );
    IAriesIndexSPtr dupIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_duplicate );

    int data[] = { 1, 2, 2, 3, 3, 4, 5 };
    for( int i = 0; i < sizeof( data ) / sizeof(int); ++i )
    {
        uniqueIndices->Insert( &data[ i ], i );
        dupIndices->Insert( &data[ i ], i );
    }

    int key = 3;
    auto resultUnique = uniqueIndices->FindLessOrEqual( &key );
    auto resultDup = dupIndices->FindLessOrEqual( &key );

    ASSERT_EQ( resultUnique.size(), 5 );
    ASSERT_EQ( resultUnique.size(), resultDup.size() );
    int res[] = { 0, 1, 2, 3, 4 };
    for( int i = 0; i < resultUnique.size(); ++i )
    {
        ASSERT_EQ( resultUnique[ i ], res[ i ] );
        ASSERT_EQ( resultUnique[ i ], resultDup[ i ] );
    }
    key = 0;
    resultUnique = uniqueIndices->FindLessOrEqual( &key );
    resultDup = dupIndices->FindLessOrEqual( &key );

    ASSERT_EQ( resultUnique.size(), 0 );
}

TEST(AriesIndex, intgt)
{
    IAriesIndexSPtr uniqueIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_unique );
    IAriesIndexSPtr dupIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_duplicate );

    int data[] = { 1, 2, 2, 3, 3, 4, 5 };
    for( int i = 0; i < sizeof( data ) / sizeof(int); ++i )
    {
        uniqueIndices->Insert( &data[ i ], i );
        dupIndices->Insert( &data[ i ], i );
    }

    int key = 3;
    auto resultUnique = uniqueIndices->FindGreater( &key );
    auto resultDup = dupIndices->FindGreater( &key );

    ASSERT_EQ( resultUnique.size(), 2 );
    ASSERT_EQ( resultUnique.size(), resultDup.size() );
    int res[] = { 5, 6 };
    for( int i = 0; i < resultUnique.size(); ++i )
    {
        ASSERT_EQ( resultUnique[ i ], res[ i ] );
        ASSERT_EQ( resultUnique[ i ], resultDup[ i ] );
    }
}

TEST(AriesIndex, intge)
{
    IAriesIndexSPtr uniqueIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_unique );
    IAriesIndexSPtr dupIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_duplicate );

    int data[] = { 1, 2, 2, 3, 3, 4, 5 };
    for( int i = 0; i < sizeof( data ) / sizeof(int); ++i )
    {
        uniqueIndices->Insert( &data[ i ], i );
        dupIndices->Insert( &data[ i ], i );
    }

    int key = 3;
    auto resultUnique = uniqueIndices->FindGreaterOrEqual( &key );
    auto resultDup = dupIndices->FindGreaterOrEqual( &key );

    ASSERT_EQ( resultUnique.size(), 4 );
    ASSERT_EQ( resultUnique.size(), resultDup.size() );
    int res[] = { 3, 4, 5, 6 };
    for( int i = 0; i < resultUnique.size(); ++i )
    {
        ASSERT_EQ( resultUnique[ i ], res[ i ] );
        ASSERT_EQ( resultUnique[ i ], resultDup[ i ] );
    }
}

TEST(AriesIndex, intgtlt)
{
    IAriesIndexSPtr uniqueIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_unique );
    IAriesIndexSPtr dupIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_duplicate );

    int data[] = { 1, 2, 2, 3, 3, 4, 5 };
    for( int i = 0; i < sizeof( data ) / sizeof(int); ++i )
    {
        uniqueIndices->Insert( &data[ i ], i );
        dupIndices->Insert( &data[ i ], i );
    }

    int minKey = 2;
    int maxKey = 3;
    auto resultUnique = uniqueIndices->FindGTAndLT( &minKey, &maxKey );
    auto resultDup = dupIndices->FindGTAndLT( &minKey, &maxKey );

    ASSERT_EQ( resultUnique.size(), 0 );
    ASSERT_EQ( resultUnique.size(), resultDup.size() );

    minKey = 2;
    maxKey = 4;
    int res[] = { 3, 4 };
    resultUnique = uniqueIndices->FindGTAndLT( &minKey, &maxKey );
    resultDup = dupIndices->FindGTAndLT( &minKey, &maxKey );
    ASSERT_EQ( resultUnique.size(), 2 );
    ASSERT_EQ( resultUnique.size(), resultDup.size() );
    for( int i = 0; i < resultUnique.size(); ++i )
    {
        ASSERT_EQ( resultUnique[ i ], res[ i ] );
        ASSERT_EQ( resultUnique[ i ], resultDup[ i ] );
    }
}

TEST(AriesIndex, intgelt)
{
    IAriesIndexSPtr uniqueIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_unique );
    IAriesIndexSPtr dupIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_duplicate );

    int data[] = { 1, 2, 2, 3, 3, 4, 5 };
    for( int i = 0; i < sizeof( data ) / sizeof(int); ++i )
    {
        uniqueIndices->Insert( &data[ i ], i );
        dupIndices->Insert( &data[ i ], i );
    }

    int minKey = 2;
    int maxKey = 3;
    auto resultUnique = uniqueIndices->FindGEAndLT( &minKey, &maxKey );
    auto resultDup = dupIndices->FindGEAndLT( &minKey, &maxKey );

    ASSERT_EQ( resultUnique.size(), 2 );
    ASSERT_EQ( resultUnique.size(), resultDup.size() );
    int res[] = { 1, 2 };
    for( int i = 0; i < resultUnique.size(); ++i )
    {
        ASSERT_EQ( resultUnique[ i ], res[ i ] );
        ASSERT_EQ( resultUnique[ i ], resultDup[ i ] );
    }
}

TEST(AriesIndex, intgtle)
{
    IAriesIndexSPtr uniqueIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_unique );
    IAriesIndexSPtr dupIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_duplicate );

    int data[] = { 1, 2, 2, 3, 3, 4, 5 };
    for( int i = 0; i < sizeof( data ) / sizeof(int); ++i )
    {
        uniqueIndices->Insert( &data[ i ], i );
        dupIndices->Insert( &data[ i ], i );
    }

    int minKey = 2;
    int maxKey = 3;
    auto resultUnique = uniqueIndices->FindGTAndLE( &minKey, &maxKey );
    auto resultDup = dupIndices->FindGTAndLE( &minKey, &maxKey );

    ASSERT_EQ( resultUnique.size(), 2 );
    ASSERT_EQ( resultUnique.size(), resultDup.size() );
    int res[] = { 3, 4 };
    for( int i = 0; i < resultUnique.size(); ++i )
    {
        ASSERT_EQ( resultUnique[ i ], res[ i ] );
        ASSERT_EQ( resultUnique[ i ], resultDup[ i ] );
    }
}

TEST(AriesIndex, intgele)
{
    IAriesIndexSPtr uniqueIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_unique );
    IAriesIndexSPtr dupIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT32, 1 }, false, false },
            key_hint_t::most_duplicate );

    int data[] = { 1, 2, 2, 3, 3, 4, 5 };
    for( int i = 0; i < sizeof( data ) / sizeof(int); ++i )
    {
        uniqueIndices->Insert( &data[ i ], i );
        dupIndices->Insert( &data[ i ], i );
    }

    int minKey = 2;
    int maxKey = 3;
    auto resultUnique = uniqueIndices->FindGEAndLE( &minKey, &maxKey );
    auto resultDup = dupIndices->FindGEAndLE( &minKey, &maxKey );

    ASSERT_EQ( resultUnique.size(), 4 );
    ASSERT_EQ( resultUnique.size(), resultDup.size() );
    int res[] = { 1, 2, 3, 4 };
    for( int i = 0; i < resultUnique.size(); ++i )
    {
        ASSERT_EQ( resultUnique[ i ], res[ i ] );
        ASSERT_EQ( resultUnique[ i ], resultDup[ i ] );
    }
    int count = 100000000;
    vector< int > src;
    aries::CPU_Timer t;
    t.begin();
    for( int i = 0; i < count; ++i )
        src.push_back( i );
    t.end();
    vector< int > result;
    t.begin();
    result.insert( result.end(), src.begin(), src.end() );
    t.end();
    vector< int > src2;
    t.begin();
    for( int i = 0; i < count; ++i )
        src2.push_back( i );
    t.end();
    t.begin();
    int val;
    for( int i = 0; i < count; ++i )
    {
        val = i % 10;
        dupIndices->Insert( &val, i );
    }
    t.end();
}

using TestKeyType = char[14];

void InitTestKey( TestKeyType key, int val, const char* data )
{
    *( int* )key = val;
    char* p = &key[ sizeof(int) ];
    std::strncpy( p, data, min( strlen( data ), sizeof(TestKeyType) - sizeof(int) ) );
}

void InitAriesIndex( IAriesIndexSPtr& uniqueIndices, IAriesIndexSPtr& dupIndices )
{
    int count = 100;
    vector< string > data;
    for( int i = 0; i < count; ++i )
    {
        TestKeyType key;
        string date = "2020-11-1";
        date += std::to_string( i - i % 5 );
        //InitTestKey( key, i, date.c_str() );
        InitTestKey( key, i % 10, date.c_str() );
        string str;
        str.resize( sizeof(TestKeyType) );
        memcpy( ( char* )str.data(), key, sizeof(TestKeyType) );
        data.push_back( str );
    }

    vector< aries::AriesColumnType > types;
    types.push_back( aries::AriesColumnType { aries::AriesDataType { aries::AriesValueType::INT32, 1 }, false, false } );
    types.push_back( aries::AriesColumnType { aries::AriesDataType { aries::AriesValueType::CHAR, 10 }, false, false } );

    uniqueIndices = AriesIndexCreator::CreateAriesIndex( types, key_hint_t::most_unique );
    dupIndices = AriesIndexCreator::CreateAriesIndex( types, key_hint_t::most_duplicate );

    aries_engine::AriesCompositeKeyType key;
    for( int i = 0; i < count; ++i )
    {
        key = data[ i ];
        uniqueIndices->Insert( &key, i );
        dupIndices->Insert( &key, i );
    }
}

TEST(AriesIndex, CompositeKeyEQ)
{
    IAriesIndexSPtr uniqueIndices;
    IAriesIndexSPtr dupIndices;
    InitAriesIndex( uniqueIndices, dupIndices );
    int val = 2;
    aries_engine::AriesCompositeKeyType key;
    key.m_key.resize( sizeof(int) );
    memcpy( ( void* )key.m_key.data(), &val, sizeof(int) );

    auto resultUnique = uniqueIndices->FindEqual( &key );
    auto resultDup = dupIndices->FindEqual( &key );

    ASSERT_EQ( resultUnique.size(), 10 );
    ASSERT_EQ( resultUnique.size(), resultDup.size() );
    int res[] = { 2, 12, 22, 32, 42, 52, 62, 72, 82, 92 };
    for( int i = 0; i < resultUnique.size(); ++i )
    {
        ASSERT_EQ( resultUnique[ i ], res[ i ] );
        ASSERT_EQ( resultUnique[ i ], resultDup[ i ] );
    }
}

TEST(AriesIndex, CompositeKeyNE)
{
    IAriesIndexSPtr uniqueIndices;
    IAriesIndexSPtr dupIndices;
    InitAriesIndex( uniqueIndices, dupIndices );
    int val = 2;
    aries_engine::AriesCompositeKeyType key;
    key.m_key.resize( sizeof(int) );
    memcpy( ( void* )key.m_key.data(), &val, sizeof(int) );

    auto resultUnique = uniqueIndices->FindNotEqual( &key );
    auto resultDup = dupIndices->FindNotEqual( &key );

    ASSERT_EQ( resultUnique.size(), 90 );
    ASSERT_EQ( resultUnique.size(), resultDup.size() );

    set< int > res;
    for( int i = 0; i < 10; ++i )
    {
        for( int j = 0; j < 10; ++j )
        {
            if( ( j * 10 + i ) % 10 != 2 )
                res.insert( j * 10 + i );
        }
    }

    set< int > resultUniqueSet { resultUnique.begin(), resultUnique.end() };
    ASSERT_EQ( res, resultUniqueSet );
    for( int i = 0; i < resultUnique.size(); ++i )
        ASSERT_EQ( resultUnique[ i ], resultDup[ i ] );
}

TEST(AriesIndex, CompositeKeyLT)
{
    IAriesIndexSPtr uniqueIndices;
    IAriesIndexSPtr dupIndices;
    InitAriesIndex( uniqueIndices, dupIndices );
    int val = 2;
    aries_engine::AriesCompositeKeyType key;
    key.m_key.resize( sizeof(int) );
    memcpy( ( void* )key.m_key.data(), &val, sizeof(int) );

    auto resultUnique = uniqueIndices->FindLess( &key );
    auto resultDup = dupIndices->FindLess( &key );

    ASSERT_EQ( resultUnique.size(), 20 );
    ASSERT_EQ( resultUnique.size(), resultDup.size() );

    set< int > res = { 0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61, 70, 71, 80, 81, 90, 91 };
    set< int > resultUniqueSet { resultUnique.begin(), resultUnique.end() };
    ASSERT_EQ( res, resultUniqueSet );
    for( int i = 0; i < resultUnique.size(); ++i )
        ASSERT_EQ( resultUnique[ i ], resultDup[ i ] );
}

TEST(AriesIndex, CompositeKeyLE)
{
    IAriesIndexSPtr uniqueIndices;
    IAriesIndexSPtr dupIndices;
    InitAriesIndex( uniqueIndices, dupIndices );
    int val = 1;
    aries_engine::AriesCompositeKeyType key;
    key.m_key.resize( sizeof(int) );
    memcpy( ( void* )key.m_key.data(), &val, sizeof(int) );

    auto resultUnique = uniqueIndices->FindLessOrEqual( &key );
    auto resultDup = dupIndices->FindLessOrEqual( &key );

    ASSERT_EQ( resultUnique.size(), 20 );
    ASSERT_EQ( resultUnique.size(), resultDup.size() );

    set< int > res = { 0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61, 70, 71, 80, 81, 90, 91 };
    set< int > resultUniqueSet { resultUnique.begin(), resultUnique.end() };
    ASSERT_EQ( res, resultUniqueSet );
    for( int i = 0; i < resultUnique.size(); ++i )
        ASSERT_EQ( resultUnique[ i ], resultDup[ i ] );
}

TEST(AriesIndex, CompositeKeyGT)
{
    IAriesIndexSPtr uniqueIndices;
    IAriesIndexSPtr dupIndices;
    InitAriesIndex( uniqueIndices, dupIndices );
    int val = 7;
    aries_engine::AriesCompositeKeyType key;
    key.m_key.resize( sizeof(int) );
    memcpy( ( void* )key.m_key.data(), &val, sizeof(int) );

    auto resultUnique = uniqueIndices->FindGreater( &key );
    auto resultDup = dupIndices->FindGreater( &key );

    ASSERT_EQ( resultUnique.size(), 20 );
    ASSERT_EQ( resultUnique.size(), resultDup.size() );

    set< int > res = { 8, 9, 18, 19, 28, 29, 38, 39, 48, 49, 58, 59, 68, 69, 78, 79, 88, 89, 98, 99 };
    set< int > resultUniqueSet { resultUnique.begin(), resultUnique.end() };
    ASSERT_EQ( res, resultUniqueSet );
    for( int i = 0; i < resultUnique.size(); ++i )
        ASSERT_EQ( resultUnique[ i ], resultDup[ i ] );
}

TEST(AriesIndex, CompositeKeyGE)
{
    IAriesIndexSPtr uniqueIndices;
    IAriesIndexSPtr dupIndices;
    InitAriesIndex( uniqueIndices, dupIndices );
    int val = 8;
    aries_engine::AriesCompositeKeyType key;
    key.m_key.resize( sizeof(int) );
    memcpy( ( void* )key.m_key.data(), &val, sizeof(int) );

    auto resultUnique = uniqueIndices->FindGreaterOrEqual( &key );
    auto resultDup = dupIndices->FindGreaterOrEqual( &key );

    ASSERT_EQ( resultUnique.size(), 20 );
    ASSERT_EQ( resultUnique.size(), resultDup.size() );

    set< int > res = { 8, 9, 18, 19, 28, 29, 38, 39, 48, 49, 58, 59, 68, 69, 78, 79, 88, 89, 98, 99 };
    set< int > resultUniqueSet { resultUnique.begin(), resultUnique.end() };
    ASSERT_EQ( res, resultUniqueSet );
    for( int i = 0; i < resultUnique.size(); ++i )
        ASSERT_EQ( resultUnique[ i ], resultDup[ i ] );
}

TEST(AriesIndex, CompositeKeyGTLT)
{
    IAriesIndexSPtr uniqueIndices;
    IAriesIndexSPtr dupIndices;
    InitAriesIndex( uniqueIndices, dupIndices );
    int minVal = 1;
    aries_engine::AriesCompositeKeyType minKey;
    minKey.m_key.resize( sizeof(int) );
    memcpy( ( void* )minKey.m_key.data(), &minVal, sizeof(int) );

    int maxVal = 3;
    aries_engine::AriesCompositeKeyType maxKey;
    maxKey.m_key.resize( sizeof(int) );
    memcpy( ( void* )maxKey.m_key.data(), &maxVal, sizeof(int) );

    auto resultUnique = uniqueIndices->FindGTAndLT( &minKey, &maxKey );
    auto resultDup = dupIndices->FindGTAndLT( &minKey, &maxKey );

    ASSERT_EQ( resultUnique.size(), 10 );
    ASSERT_EQ( resultUnique.size(), resultDup.size() );

    set< int > res = { 2, 12, 22, 32, 42, 52, 62, 72, 82, 92 };
    set< int > resultUniqueSet { resultUnique.begin(), resultUnique.end() };
    ASSERT_EQ( res, resultUniqueSet );
    for( int i = 0; i < resultUnique.size(); ++i )
        ASSERT_EQ( resultUnique[ i ], resultDup[ i ] );
}

TEST(AriesIndex, CompositeKeyGELT)
{
    IAriesIndexSPtr uniqueIndices;
    IAriesIndexSPtr dupIndices;
    InitAriesIndex( uniqueIndices, dupIndices );
    int minVal = 2;
    aries_engine::AriesCompositeKeyType minKey;
    minKey.m_key.resize( sizeof(int) );
    memcpy( ( void* )minKey.m_key.data(), &minVal, sizeof(int) );

    int maxVal = 3;
    aries_engine::AriesCompositeKeyType maxKey;
    maxKey.m_key.resize( sizeof(int) );
    memcpy( ( void* )maxKey.m_key.data(), &maxVal, sizeof(int) );

    auto resultUnique = uniqueIndices->FindGEAndLT( &minKey, &maxKey );
    auto resultDup = dupIndices->FindGEAndLT( &minKey, &maxKey );

    ASSERT_EQ( resultUnique.size(), 10 );
    ASSERT_EQ( resultUnique.size(), resultDup.size() );

    set< int > res = { 2, 12, 22, 32, 42, 52, 62, 72, 82, 92 };
    set< int > resultUniqueSet { resultUnique.begin(), resultUnique.end() };
    ASSERT_EQ( res, resultUniqueSet );
    for( int i = 0; i < resultUnique.size(); ++i )
        ASSERT_EQ( resultUnique[ i ], resultDup[ i ] );
}

TEST(AriesIndex, CompositeKeyGTLE)
{
    IAriesIndexSPtr uniqueIndices;
    IAriesIndexSPtr dupIndices;
    InitAriesIndex( uniqueIndices, dupIndices );
    int minVal = 1;
    aries_engine::AriesCompositeKeyType minKey;
    minKey.m_key.resize( sizeof(int) );
    memcpy( ( void* )minKey.m_key.data(), &minVal, sizeof(int) );

    int maxVal = 2;
    aries_engine::AriesCompositeKeyType maxKey;
    maxKey.m_key.resize( sizeof(int) );
    memcpy( ( void* )maxKey.m_key.data(), &maxVal, sizeof(int) );

    auto resultUnique = uniqueIndices->FindGTAndLE( &minKey, &maxKey );
    auto resultDup = dupIndices->FindGTAndLE( &minKey, &maxKey );

    ASSERT_EQ( resultUnique.size(), 10 );
    ASSERT_EQ( resultUnique.size(), resultDup.size() );

    set< int > res = { 2, 12, 22, 32, 42, 52, 62, 72, 82, 92 };
    set< int > resultUniqueSet { resultUnique.begin(), resultUnique.end() };
    ASSERT_EQ( res, resultUniqueSet );
    for( int i = 0; i < resultUnique.size(); ++i )
        ASSERT_EQ( resultUnique[ i ], resultDup[ i ] );
}

TEST(AriesIndex, CompositeKeyGELE)
{
    IAriesIndexSPtr uniqueIndices;
    IAriesIndexSPtr dupIndices;
    InitAriesIndex( uniqueIndices, dupIndices );
    int minVal = 8;
    aries_engine::AriesCompositeKeyType minKey;
    minKey.m_key.resize( sizeof(int) );
    memcpy( ( void* )minKey.m_key.data(), &minVal, sizeof(int) );

    int maxVal = 9;
    aries_engine::AriesCompositeKeyType maxKey;
    maxKey.m_key.resize( sizeof(int) );
    memcpy( ( void* )maxKey.m_key.data(), &maxVal, sizeof(int) );

    auto resultUnique = uniqueIndices->FindGEAndLE( &minKey, &maxKey );
    auto resultDup = dupIndices->FindGEAndLE( &minKey, &maxKey );

    ASSERT_EQ( resultUnique.size(), 20 );
    ASSERT_EQ( resultUnique.size(), resultDup.size() );

    set< int > res = { 8, 9, 18, 19, 28, 29, 38, 39, 48, 49, 58, 59, 68, 69, 78, 79, 88, 89, 98, 99 };
    set< int > resultUniqueSet { resultUnique.begin(), resultUnique.end() };
    ASSERT_EQ( res, resultUniqueSet );
    for( int i = 0; i < resultUnique.size(); ++i )
        ASSERT_EQ( resultUnique[ i ], resultDup[ i ] );
}

TEST(AriesIndex, longsaveload)
{
    IAriesIndexSPtr indices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT64, 1 }, false, false },
            key_hint_t::most_unique );
    IAriesIndexSPtr dupIndices = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT64, 1 }, false, false },
            key_hint_t::most_duplicate );
    int count = 600000000;

    cout << "insert " << count << " values:" << endl;
    aries::CPU_Timer t;
    t.begin();
    int val;
    for( int i = 0; i < count; ++i )
    {
        val = i;
        indices->Insert( &val, i );
        dupIndices->Insert( &val, i );
    }
    t.end();
    int key = 10000001;
    auto result = indices->FindLess( &key );
    string filename = "AriesIndexlong_most_unique.out";
    string filename2 = "AriesIndexlong_most_duplicate.out";
    cout << "indices save:" << endl;
    t.begin();
    indices->Save( filename );
    t.end();
    cout << "dupIndices save:" << endl;
    t.begin();
    dupIndices->Save( filename2 );
    t.end();
    cout<<"most_unique indices status:"<<endl;
    indices->DumpStatus();
    cout<<"most_duplicate indices status:"<<endl;
    dupIndices->DumpStatus();
    indices = nullptr;
    dupIndices = nullptr;
    IAriesIndexSPtr indices2 = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT64, 1 }, false, false },
            key_hint_t::most_unique );
    IAriesIndexSPtr dupIndices2 = AriesIndexCreator::CreateAriesIndex( AriesColumnType { AriesDataType { AriesValueType::INT64, 1 }, false, false },
            key_hint_t::most_duplicate );
    cout << "indices load:" << endl;
    t.begin();
    indices2->Load( filename );
    t.end();
    cout << "dupIndices load:" << endl;
    t.begin();
    dupIndices2->Load( filename2 );
    t.end();
}

TEST(AriesIndex, compositesaveload)
{
    int count = 600000000;
    vector< aries::AriesColumnType > types;
    types.push_back( aries::AriesColumnType { aries::AriesDataType { aries::AriesValueType::INT32, 1 }, false, false } );
    types.push_back( aries::AriesColumnType { aries::AriesDataType { aries::AriesValueType::INT32, 1 }, false, false } );

    auto uniqueIndices = AriesIndexCreator::CreateAriesIndex( types, key_hint_t::most_unique );
    auto dupIndices = AriesIndexCreator::CreateAriesIndex( types, key_hint_t::most_duplicate );

    string data;
    data.resize( sizeof(int)*2 );
    int* pdata = (int*)data.data();
    aries_engine::AriesCompositeKeyType key;
    aries::CPU_Timer t;
    t.begin();
    for( int i = 0; i < count; ++i )
    {
        pdata[0] = i;
        pdata[1] = i;
        key = data;
		uniqueIndices->Insert( &key, i );
		dupIndices->Insert( &key, i );
    }
    t.end();
    cout<<"most_unique indices status:"<<endl;
    uniqueIndices->DumpStatus();
	cout<<"most_duplicate indices status:"<<endl;
	dupIndices->DumpStatus();
}
