/*
 * test_tablekeys.cu
 *
 *  Created on: Sep 10, 2020
 *      Author: lichi
 */

#include "test_common.h"
#include "AriesEngine/transaction/AriesTableKeys.h"
using namespace aries_engine;

TEST(tablekeys, primary)
{
    standard_context_t context;
    CPU_Timer t;
    aries_engine::AriesTableBlockUPtr table = ReadTable( "scale_1", "lineitem",
    { 1, 2, 3, 4 } );
    vector< aries_engine::AriesColumnSPtr > rawColumns;
    rawColumns.push_back( table->GetMaterilizedColumn( 1 ) );
    rawColumns.push_back( table->GetMaterilizedColumn( 4 ) );
    AriesTableKeys tableKeysPrimary;
    t.begin();
    tableKeysPrimary.Build( rawColumns, true );
    cout<<"tableKeysPrimary.build gpu time: "<<t.end()<<endl;

    rawColumns.clear();
    rawColumns.push_back( table->GetMaterilizedColumn( 2 ) );
    rawColumns.push_back( table->GetMaterilizedColumn( 3 ) );
    AriesTableKeys tableKeysForeign;
    t.begin();
    tableKeysForeign.Build( rawColumns, false );
    cout<<"tableKeysForeign.build gpu time: "<<t.end()<<endl;

    vector< string > lookupkeys;
    for( int i = 0; i < 10000000; ++i )
    {
        string a;
        a.insert( a.size(), ( const char* )&i, sizeof(int) );
        a.insert( a.size(), ( const char* )&i, sizeof(int) );
        lookupkeys.push_back( a );
    }

    t.begin();
    for( const auto& key : lookupkeys )
        tableKeysPrimary.FindKey( key );
    cout<<"FindKey gpu time: "<<t.end()<<endl;
}

TEST(tablekeys, merge)
{
    standard_context_t context;
    aries_engine::AriesTableBlockUPtr table = ReadTable( "scale_1", "lineitem",
    { 1, 2, 3, 4 } );
    vector< aries_engine::AriesColumnSPtr > rawColumns;
    auto col1 = table->GetMaterilizedColumn( 1 );
    auto col2 = table->GetMaterilizedColumn( 4 );
    rawColumns.push_back( col1 );
    rawColumns.push_back( col2 );
    AriesTableKeys tableKeysPrimary;
    tableKeysPrimary.Build( rawColumns, true );
    AriesTableKeysSPtr other = std::make_shared< AriesTableKeys >();
    other->Build( rawColumns, true );
    ASSERT_EQ( tableKeysPrimary.Merge( other ), false );
    {
        rawColumns.clear();
        size_t rowCount = 100000;
        auto subTable1 = table->GetSubTable( 0, rowCount, true );
        auto subTable2 = table->GetSubTable( rowCount, table->GetRowCount() - rowCount, true );

        auto col1 = subTable1->GetMaterilizedColumn( 1 );
        auto col2 = subTable1->GetMaterilizedColumn( 4 );
        rawColumns.push_back( col1 );
        rawColumns.push_back( col2 );
        AriesTableKeys tableKeysPrimary;
        tableKeysPrimary.Build( rawColumns, true );
        string key;
        key.resize( 8 );
        char* pKey = (char*)key.data();
        int val1 = 1;
        int val2 = 1;
        memcpy( pKey, &val1, 4 );
        memcpy( pKey + 4, &val2, 4 );
        ASSERT_EQ( tableKeysPrimary.InsertKey( key, 1 ), false );

        val1 = 7000000;
        val2 = 1;
        memcpy( pKey, &val1, 4 );
        memcpy( pKey + 4, &val2, 4 );
        ASSERT_EQ( tableKeysPrimary.InsertKey( key, 1 ), true );


        rawColumns.clear();
        col1 = subTable2->GetMaterilizedColumn( 1 );
        col2 = subTable2->GetMaterilizedColumn( 4 );
        rawColumns.push_back( col1 );
        rawColumns.push_back( col2 );

        AriesTableKeysSPtr other = std::make_shared< AriesTableKeys >();
        other->Build( rawColumns, true );
        ASSERT_EQ( tableKeysPrimary.Merge( other ), true );

        val1 = 6000000;
        val2 = 1;
        memcpy( pKey, &val1, 4 );
        memcpy( pKey + 4, &val2, 4 );
        ASSERT_EQ( tableKeysPrimary.InsertKey( key, 1 ), false );
    }
}
