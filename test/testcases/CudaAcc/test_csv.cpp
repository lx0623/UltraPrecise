#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <future>
#include <random>
#include <chrono>
#include <future>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include "DataConvertor.h"

//TEST( csv, import )
//{
//    shared_ptr< vector< CSVRow > > allRows = make_shared<vector< CSVRow >>();
//    auto start = std::chrono::high_resolution_clock::now();
//    vector< string > colNames;
//    for( int i = 0; i < 17; ++i )
//        colNames.push_back( std::to_string( i ) );
//    CSVReader reader( "/home/lichi/Downloads/tpc-h-tool/2.17.3/dbgen/lineitem.tbl", CSVFormat().delimiter( '|' ).column_names( colNames ) );
//    CSVRow row;
//    int BLOCK_SIZE = 5000000;
//    int count = BLOCK_SIZE;
//    bool bContinue = true;
//    shared_ptr< vector< shared_ptr< DataConverter > > > convertors = make_shared < vector< shared_ptr< DataConverter > > > ();
//    convertors->emplace_back( make_shared < SimpleConvertor<int32_t, Int32Converter > > ( 0, false, "/home/lichi/lineitem0", Int32Converter() ) );
//    convertors->emplace_back( make_shared < SimpleConvertor<int32_t, Int32Converter > > ( 1, false, "/home/lichi/lineitem1", Int32Converter() ) );
//    convertors->emplace_back( make_shared < SimpleConvertor<int32_t, Int32Converter > > ( 2, false, "/home/lichi/lineitem2", Int32Converter() ) );
//    convertors->emplace_back( make_shared < SimpleConvertor<int32_t, Int32Converter > > ( 3, false, "/home/lichi/lineitem3", Int32Converter() ) );
//    convertors->emplace_back( make_shared < DecimalConvertor > ( 4, false, 15, 2, "/home/lichi/lineitem4" ) );
//    convertors->emplace_back( make_shared < DecimalConvertor > ( 5, false, 15, 2, "/home/lichi/lineitem5" ) );
//    convertors->emplace_back( make_shared < DecimalConvertor > ( 6, false, 15, 2, "/home/lichi/lineitem6" ) );
//    convertors->emplace_back( make_shared < DecimalConvertor > ( 7, false, 15, 2, "/home/lichi/lineitem7" ) );
//    convertors->emplace_back( make_shared < StringConvertor > ( 8, false, "/home/lichi/lineitem8" ) );
//    convertors->emplace_back( make_shared < StringConvertor > ( 9, false, "/home/lichi/lineitem9" ) );
//    convertors->emplace_back( make_shared < SimpleConvertor<AriesDate, AriesDateConverter > > ( 10, false, "/home/lichi/lineitem10", AriesDateConverter() ) );
//    convertors->emplace_back( make_shared < SimpleConvertor<AriesDate, AriesDateConverter > > ( 11, false, "/home/lichi/lineitem11", AriesDateConverter() ) );
//    convertors->emplace_back( make_shared < SimpleConvertor<AriesDate, AriesDateConverter > > ( 12, false, "/home/lichi/lineitem12", AriesDateConverter() ) );
//    convertors->emplace_back( make_shared < StringConvertor > ( 13, false, "/home/lichi/lineitem13" ) );
//    convertors->emplace_back( make_shared < StringConvertor > ( 15, false, "/home/lichi/lineitem15" ) );
//    convertors->emplace_back( make_shared < StringConvertor > ( 14, false, "/home/lichi/lineitem14" ) );
//    int num = 0;
//    do
//    {
//        while( count-- && ( bContinue = reader.read_row( row ) ) )
//        {
//            ++num;
//            allRows->emplace_back( std::move( row ) );
//        }
//        vector< future< void > > workThreads;
//        for( auto& conv : *convertors )
//        {
//            workThreads.push_back( std::async( std::launch::async, [&]
//            {   conv->Convert( *allRows.get() );} ) );
//        }
//        for( auto& t : workThreads )
//            t.wait();
//        count = BLOCK_SIZE;
//        allRows->clear();
//    } while( bContinue );
//
//    vector< future< void > > workThreads;
//    for( auto& conv : *convertors )
//    {
//        workThreads.push_back( std::async( std::launch::async, [&]
//        {   conv->PostProcess();} ) );
//    }
//    for( auto& t : workThreads )
//        t.wait();
//
//    auto stop = std::chrono::high_resolution_clock::now();
//    std::cout << "reader.row_num:" << reader.correct_rows <<" tupleNum:"<<num<< std::endl;
//    auto duration = std::chrono::duration< double >( stop - start ).count();
//    std::cout << "time cost is :" << duration <<"s"<< endl;
//}
TEST( csv, import )
{
    shared_ptr< vector< CSVRow > > allRows = make_shared<vector< CSVRow >>();
    auto start = std::chrono::high_resolution_clock::now();
    vector< string > colNames;
    for( int i = 0; i < 3; ++i )
        colNames.push_back( std::to_string( i ) );
    CSVReader reader( "/home/lichi/region.tbl", CSVFormat().delimiter( '|' ).column_names( colNames ) );
    CSVRow row;
    int BLOCK_SIZE = 5000000;
    int count = BLOCK_SIZE;
    bool bContinue = true;
    shared_ptr< vector< shared_ptr< DataConverter > > > convertors = make_shared < vector< shared_ptr< DataConverter > > > ();
    convertors->emplace_back( make_shared < SimpleConvertor<int32_t, Int32Converter > > ( 0, false, "/home/lichi/region0", Int32Converter() ) );
    convertors->emplace_back( make_shared < StringConvertor > ( 1, false, "/home/lichi/region1" ) );
    convertors->emplace_back( make_shared < StringConvertor > ( 2, false, "/home/lichi/region2" ) );
    int num = 0;
    do
    {
        while( count-- && ( bContinue = reader.read_row( row ) ) )
        {
            ++num;
            allRows->emplace_back( std::move( row ) );
        }
        vector< future< void > > workThreads;
        for( auto& conv : *convertors )
        {
            workThreads.push_back( std::async( std::launch::async, [&]
            {   conv->Convert( *allRows.get() );} ) );
        }
        for( auto& t : workThreads )
            t.wait();
        count = BLOCK_SIZE;
        allRows->clear();
    } while( bContinue );

    vector< future< void > > workThreads;
    for( auto& conv : *convertors )
    {
        workThreads.push_back( std::async( std::launch::async, [&]
        {   conv->PostProcess();} ) );
    }
    for( auto& t : workThreads )
        t.wait();

    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "reader.row_num:" << reader.correct_rows <<" tupleNum:"<<num<< std::endl;
    auto duration = std::chrono::duration< double >( stop - start ).count();
    std::cout << "time cost is :" << duration <<"s"<< endl;
}
