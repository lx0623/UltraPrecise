// Copyright 2013 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gtest/gtest.h"
#include "AriesEngine/index/btree/btree_map.h"
#include "AriesEngine/index/btree/btree_set.h"
#include "AriesEngine/index/btree/AriesIndexBtree.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <random>
#include <unordered_map>
#include <sys/time.h>
#include "CpuTimer.h"

using namespace std;

namespace btree
{
    namespace
    {
        // struct CPU_Timer
        // {
        //     long start;
        //     long stop;
        //     void begin()
        //     {
        //         struct timeval tv;
        //         gettimeofday( &tv, NULL );
        //         start = tv.tv_sec * 1000 + tv.tv_usec / 1000;
        //     }
        //     long end()
        //     {
        //         struct timeval tv;
        //         gettimeofday( &tv, NULL );
        //         stop = tv.tv_sec * 1000 + tv.tv_usec / 1000;
        //         long elapsed = stop - start;
        //         printf( "cpu time: %ld\n", elapsed );
        //         return elapsed;
        //     }
        // };

        struct AriesKey
        {
            char* m_key;
        };

        struct AriesKeyEx
        {
            AriesKeyEx()
            {
            }
            AriesKeyEx( const AriesKeyEx& a )
            {
                m_key = a.m_key;
            }
            AriesKeyEx( AriesKeyEx && a )
            {
                m_key = std::move( a.m_key );
            }

            AriesKeyEx & operator=( const AriesKeyEx & a )
            {
                m_key = a.m_key;
                return *this;
            }
            AriesKeyEx & operator=( AriesKeyEx && a )
            {
                m_key = std::move( a.m_key );
                return *this;
            }

            string m_key;
        };
        struct IAriesCompareTo: public btree_key_compare_to_tag
        {
            virtual int CompareTo( const char* key1, const char* key2 ) const = 0;
            virtual ~IAriesCompareTo()
            {
            }
            int CompareFlag( char flag1, char flag2 ) const
            {
                assert( !( flag1 && flag2 ) );
                return flag1 ? 1 : flag2 ? -1 : 0;
            }
            bool m_bHasNull;
            int m_itemSizeInBytes;
        };

        template< typename type_t >
        struct AriesComparator: public IAriesCompareTo
        {
            AriesComparator( bool hasNull )
            {
                m_bHasNull = hasNull;
                m_itemSizeInBytes = sizeof(type_t) + hasNull;
            }
            virtual int CompareTo( const char* key1, const char* key2 ) const override final
            {
                if( m_bHasNull )
                {
                    if( !( *key1 && *key2 ) )
                        return CompareFlag( *key1, *key2 );
                    ++key1;
                    ++key2;
                }
                const type_t& k1 = *( const type_t* )key1;
                const type_t& k2 = *( const type_t* )key2;
                if( k1 < k2 )
                    return -1;
                else if( k1 == k2 )
                    return 0;
                else
                    return 1;
            }
        };

        struct AriesStringComparator: public IAriesCompareTo
        {
            AriesStringComparator( bool hasNull, int len )
            {
                m_bHasNull = hasNull;
                m_itemSizeInBytes = len;
            }
            virtual int CompareTo( const char* key1, const char* key2 ) const override final
            {
                int len = m_itemSizeInBytes;
                if( m_bHasNull )
                {
                    if( !( *key1 && *key2 ) )
                        return CompareFlag( *key1, *key2 );
                    ++key1;
                    ++key2;
                    --len;
                }
                return std::strncmp( key1, key2, len );
            }
        };

        struct AriesKeyCompartorV1: public btree_key_compare_to_tag
        {
            AriesKeyCompartorV1()
            {
            }
            void SetComparators( vector< shared_ptr< IAriesCompareTo > > && comparators )
            {
                m_comps = std::move( comparators );
            }
            int operator()( const AriesKey &a, const AriesKey &b ) const
            {
                const char* key1 = a.m_key;
                const char* key2 = b.m_key;
                int res;
                for( const auto& comp : m_comps )
                {
                    res = comp->CompareTo( key1, key2 );
                    if( res == 0 )
                    {
                        int offset = comp->m_itemSizeInBytes;
                        key1 += offset;
                        key2 += offset;
                    }
                    else
                        break;
                }
                return res;
            }
            vector< shared_ptr< IAriesCompareTo > > m_comps;
        };

        struct AriesKeyCompartorV2: public btree_key_compare_to_tag
        {
            AriesKeyCompartorV2()
            {
            }
            void SetComparators( shared_ptr< vector< shared_ptr< IAriesCompareTo > > > comparators )
            {
                m_comps = comparators;
            }
            int operator()( const AriesKey &a, const AriesKey &b ) const
            {
                const char* key1 = a.m_key;
                const char* key2 = b.m_key;
                int res;
                for( const auto& comp : *m_comps )
                {
                    res = comp->CompareTo( key1, key2 );
                    if( res == 0 )
                    {
                        int offset = comp->m_itemSizeInBytes;
                        key1 += offset;
                        key2 += offset;
                    }
                    else
                        break;
                }
                return res;
            }
            shared_ptr< vector< shared_ptr< IAriesCompareTo > > > m_comps;
        };

        struct AriesKeyCompartorV3: public btree_key_compare_to_tag
        {
            AriesKeyCompartorV3()
                    : m_comps( nullptr )
            {
            }
            void SetComparators( shared_ptr< vector< shared_ptr< IAriesCompareTo > > > comparators )
            {
                m_comps = comparators.get();
            }
            int operator()( const AriesKey &a, const AriesKey &b ) const
            {
                const char* key1 = a.m_key;
                const char* key2 = b.m_key;
                int res;
                for( const auto& comp : *m_comps )
                {
                    res = comp->CompareTo( key1, key2 );
                    if( res == 0 )
                    {
                        int offset = comp->m_itemSizeInBytes;
                        key1 += offset;
                        key2 += offset;
                    }
                    else
                        break;
                }
                return res;
            }
            vector< shared_ptr< IAriesCompareTo > >* m_comps;
        };

        struct AriesKeyCompartor: public btree_key_compare_to_tag
        {
            AriesKeyCompartor()
            {
            }
            void SetComparators( shared_ptr< vector< shared_ptr< IAriesCompareTo > > > comparators )
            {
                m_comps = comparators.get();
            }
            int operator()( const AriesKey &a, const AriesKey &b ) const
            {
                const char* key1 = a.m_key;
                const char* key2 = b.m_key;
                int res;
                for( const auto& comp : *m_comps )
                {
                    res = comp->CompareTo( key1, key2 );
                    if( res == 0 )
                    {
                        int offset = comp->m_itemSizeInBytes;
                        key1 += offset;
                        key2 += offset;
                    }
                    else
                        break;
                }
                return res;
            }
            vector< shared_ptr< IAriesCompareTo > >* m_comps;
        };

        struct AriesKeyCompartorEx: public btree_key_compare_to_tag
        {
            AriesKeyCompartorEx()
            {
            }
            void SetComparators( shared_ptr< vector< shared_ptr< IAriesCompareTo > > > comparators )
            {
                m_comps = comparators.get();
            }
            int operator()( const AriesKeyEx &a, const AriesKeyEx &b ) const
            {
                const char* key1 = a.m_key.data();
                const char* key2 = b.m_key.data();
                int res;
                for( const auto& comp : *m_comps )
                {
                    res = comp->CompareTo( key1, key2 );
                    if( res == 0 )
                    {
                        int offset = comp->m_itemSizeInBytes;
                        key1 += offset;
                        key2 += offset;
                    }
                    else
                        break;
                }
                return res;
            }
            vector< shared_ptr< IAriesCompareTo > >* m_comps;
        };

        struct AriesKeyCompartorString: public btree_key_compare_to_tag
        {
            AriesKeyCompartorString()
                    : m_comps( nullptr )
            {
            }
            void SetComparators( shared_ptr< vector< shared_ptr< IAriesCompareTo > > > comparators )
            {
                m_comps = comparators.get();
            }
            int operator()( const string &a, const string &b ) const
            {
                const char* key1 = a.data();
                const char* key2 = b.data();
                int res;
                for( const auto& comp : *m_comps )
                {
                    res = comp->CompareTo( key1, key2 );
                    if( res == 0 )
                    {
                        int offset = comp->m_itemSizeInBytes;
                        key1 += offset;
                        key2 += offset;
                    }
                    else
                        break;
                }
                return res;
            }

            AriesKeyCompartorString( const AriesKeyCompartorString& a )
            {
                m_comps = a.m_comps;
            }

            AriesKeyCompartorString & operator=( const AriesKeyCompartorString & a )
            {
                m_comps = a.m_comps;
                return *this;
            }
            AriesKeyCompartorString( AriesKeyCompartorString && a )
            {
                m_comps = std::move( a.m_comps );
            }

            AriesKeyCompartorString & operator=( AriesKeyCompartorString && a )
            {
                m_comps = std::move( a.m_comps );
                return *this;
            }
            vector< shared_ptr< IAriesCompareTo > >* m_comps;
            //shared_ptr<vector< shared_ptr< IAriesCompareTo > >> m_comps;
        };

        struct AriesKeyCompartorVChar: public btree_key_compare_to_tag
        {
            AriesKeyCompartorVChar()
            {
            }
            void SetComparators( shared_ptr< vector< shared_ptr< IAriesCompareTo > > > comparators )
            {
                m_comps = comparators.get();
            }
            int operator()( const vector< char > &a, const vector< char > &b ) const
            {
                const char* key1 = a.data();
                const char* key2 = b.data();
                int res;
                for( const auto& comp : *m_comps )
                {
                    res = comp->CompareTo( key1, key2 );
                    if( res == 0 )
                    {
                        int offset = comp->m_itemSizeInBytes;
                        key1 += offset;
                        key2 += offset;
                    }
                    else
                        break;
                }
                return res;
            }

            vector< shared_ptr< IAriesCompareTo > >* m_comps;
        };

        using TestKeyType = char[65];

        void InitTestKey( TestKeyType key, int val, const char* data )
        {
            *( int* )key = val;
            char* p = &key[ sizeof(int) ];
            std::strncpy( p, data, min( strlen( data ), sizeof(TestKeyType) - sizeof(int) ) );
        }

        TEST(Btree, AriesKeyCompartorV1)
        {
            vector< char > data;
            int count = 10000000;
            data.resize( count * sizeof(TestKeyType) );
            for( int i = 0; i < count; ++i )
            {
                TestKeyType key;
                InitTestKey( key, i % 5, "2020-11-11" );
                memcpy( data.data() + i * sizeof(TestKeyType), key, sizeof(TestKeyType) );
            }

            vector< shared_ptr< IAriesCompareTo > > comparators;
            comparators.push_back( std::make_shared< AriesComparator< int > >( false ) );
            comparators.push_back( std::make_shared< AriesStringComparator >( false, 10 ) );
            AriesKeyCompartorV1 keyComp;
            keyComp.SetComparators( std::move( comparators ) );
            btree_multiset< AriesKey, AriesKeyCompartorV1 > iset( keyComp );
            aries::CPU_Timer t;
            t.begin();
            for( int i = 0; i < count; ++i )
            {
                AriesKey key;
                key.m_key = ( char* )&data[ i * sizeof(TestKeyType) ];
                iset.insert( key );
            }
            t.end();
        }

        TEST(Btree, AriesKeyCompartorV2)
        {
            vector< char > data;
            int count = 10000000;
            data.resize( count * sizeof(TestKeyType) );
            for( int i = 0; i < count; ++i )
            {
                TestKeyType key;
                InitTestKey( key, i % 5, "2020-11-11" );
                memcpy( data.data() + i * sizeof(TestKeyType), key, sizeof(TestKeyType) );
            }

            shared_ptr< vector< shared_ptr< IAriesCompareTo > > > comparators = make_shared< vector< shared_ptr< IAriesCompareTo > > >();
            comparators->push_back( std::make_shared< AriesComparator< int > >( false ) );
            comparators->push_back( std::make_shared< AriesStringComparator >( false, 10 ) );
            AriesKeyCompartorV2 keyComp;
            keyComp.SetComparators( comparators );
            btree_multiset< AriesKey, AriesKeyCompartorV2 > iset( keyComp );
            aries::CPU_Timer t;
            t.begin();
            for( int i = 0; i < count; ++i )
            {
                AriesKey key;
                key.m_key = ( char* )&data[ i * sizeof(TestKeyType) ];
                iset.insert( key );
            }
            t.end();
        }

        TEST(Btree, AriesKeyCompartorV3)
        {
            vector< char > data;
            int count = 10000000;
            data.resize( count * sizeof(TestKeyType) );
            for( int i = 0; i < count; ++i )
            {
                TestKeyType key;
                InitTestKey( key, i % 5, "2020-11-11" );
                memcpy( data.data() + i * sizeof(TestKeyType), key, sizeof(TestKeyType) );
            }

            shared_ptr< vector< shared_ptr< IAriesCompareTo > > > comparators = make_shared< vector< shared_ptr< IAriesCompareTo > > >();
            comparators->push_back( std::make_shared< AriesComparator< int > >( false ) );
            comparators->push_back( std::make_shared< AriesStringComparator >( false, 10 ) );
            AriesKeyCompartorV3 keyComp;
            keyComp.SetComparators( comparators );
            btree_multiset< AriesKey, AriesKeyCompartorV3 > iset( keyComp );
            aries::CPU_Timer t;
            t.begin();
            for( int i = 0; i < count; ++i )
            {
                AriesKey key;
                key.m_key = ( char* )&data[ i * sizeof(TestKeyType) ];
                iset.insert( key );
            }
            t.end();
        }

        TEST(Btree, multiset)
        {
            vector< char > data;
            int count = 10000000;
            data.resize( count * sizeof(TestKeyType) );
            for( int i = 0; i < count; ++i )
            {
                TestKeyType key;
                InitTestKey( key, i % 5, "2020-11-11" );
                memcpy( data.data() + i * sizeof(TestKeyType), key, sizeof(TestKeyType) );
            }

            shared_ptr< vector< shared_ptr< IAriesCompareTo > > > comparators = make_shared< vector< shared_ptr< IAriesCompareTo > > >();
            comparators->push_back( std::make_shared< AriesComparator< int > >( false ) );
            comparators->push_back( std::make_shared< AriesStringComparator >( false, 10 ) );
            AriesKeyCompartor keyComp;
            keyComp.SetComparators( comparators );
            btree_multiset< AriesKey, AriesKeyCompartor > iset( keyComp );
            aries::CPU_Timer t;
            t.begin();
            for( int i = 0; i < count; ++i )
            {
                AriesKey key;
                key.m_key = ( char* )&data[ i * sizeof(TestKeyType) ];
                iset.insert( key );
            }
            t.end();
        }

        TEST(Btree, multisetEx)
        {
            int count = 10000000;
            vector< string > data;
            for( int i = 0; i < count; ++i )
            {
                TestKeyType key;
                InitTestKey( key, i % 5, "2020-11-11" );
                string str;
                str.resize( sizeof(TestKeyType) );
                memcpy( ( char* )str.data(), key, sizeof(TestKeyType) );
                data.push_back( str );
            }

            shared_ptr< vector< shared_ptr< IAriesCompareTo > > > comparators = make_shared< vector< shared_ptr< IAriesCompareTo > > >();
            comparators->push_back( std::make_shared< AriesComparator< int > >( false ) );
            comparators->push_back( std::make_shared< AriesStringComparator >( false, 10 ) );
            AriesKeyCompartorEx keyComp;
            keyComp.SetComparators( comparators );
            btree_multiset< AriesKeyEx, AriesKeyCompartorEx > iset( keyComp );
            aries::CPU_Timer t;
            t.begin();
            for( int i = 0; i < count; ++i )
            {
                AriesKeyEx key;
                key.m_key = data[ i ];
                iset.insert( std::move( key ) );
            }
            t.end();

        }
        TEST(Btree, multisetString)
        {
            int count = 10000000;
            vector< string > data;
            for( int i = 0; i < count; ++i )
            {
                TestKeyType key;
                string date = "2020-11-1";
                date += std::to_string( i - i % 5 );
                InitTestKey( key, i % 10, date.c_str() );
                string str;
                str.resize( sizeof(TestKeyType) );
                memcpy( ( char* )str.data(), key, sizeof(TestKeyType) );
                data.push_back( str );
            }

            shared_ptr< vector< shared_ptr< IAriesCompareTo > > > comparators = make_shared< vector< shared_ptr< IAriesCompareTo > > >();
            comparators->push_back( std::make_shared< AriesComparator< int > >( false ) );
            comparators->push_back( std::make_shared< AriesStringComparator >( false, 10 ) );
            AriesKeyCompartorString keyComp;
            keyComp.SetComparators( comparators );
            btree_multiset< string, AriesKeyCompartorString > iset( keyComp );
            aries::CPU_Timer t;
            t.begin();
            for( int i = 0; i < count - count / 10; ++i )
            {
                iset.insert( data[ i ] );
            }
            t.end();

            t.begin();
            for( int i = count - count / 10; i < count; ++i )
            {
                iset.insert( data[ i ] );
            }
            t.end();

            t.begin();
            for( int i = 0; i < count; ++i )
            {
                iset.erase( data[ i ] );
            }
            t.end();
            for( const auto& it : iset )
            {
                printf( "%d, %.10s\n", *( int* )it.data(), it.data() + sizeof(int) );
            }

        }

        TEST(Btree, multisetvchar)
        {
//            cout << sizeof(TestKeyType) << endl;
//            cout << sizeof(vector< char > ) << endl;
//            cout << sizeof(string) << endl;
            int count = 10000000;
            vector< vector< char > > data;
            for( int i = 0; i < count; ++i )
            {
                TestKeyType key;
                string date = "2020-11-1";
                date += std::to_string( i - i % 5 );
                InitTestKey( key, i % 10, date.c_str() );
                vector< char > str;
                str.resize( sizeof(TestKeyType) );
                memcpy( ( char* )str.data(), key, sizeof(TestKeyType) );
                data.push_back( str );
            }

            shared_ptr< vector< shared_ptr< IAriesCompareTo > > > comparators = make_shared< vector< shared_ptr< IAriesCompareTo > > >();
            comparators->push_back( std::make_shared< AriesComparator< int > >( false ) );
            comparators->push_back( std::make_shared< AriesStringComparator >( false, 10 ) );
            AriesKeyCompartorVChar keyComp;
            keyComp.SetComparators( comparators );
            btree_multiset< vector< char >, AriesKeyCompartorVChar > iset( keyComp );
            aries::CPU_Timer t;
            t.begin();
            for( int i = 0; i < count; ++i )
            {
                iset.insert( data[ i ] );
            }
            t.end();
//            for( const auto& it: iset )
//            {
//                printf("%d, %.10s\n", *(int*)it.data(), it.data() + sizeof( int ) );
//            }

        }

        TEST(Btree, multisetsimplestring)
        {
            btree_multiset< string, btree_key_compare_to_adapter< std::less< std::string > > > mset;
            int count = 100;
            vector< string > data;
            string aa = "aaaaaaaaaa";
            for( int i = 0; i < count; ++i )
            {
                data.push_back( aa + std::to_string( i % 5 ) );
            }
            aries::CPU_Timer t;
            t.begin();
            for( int i = 0; i < count; ++i )
            {
                mset.insert( data[ i ] );
            }
            t.end();
        }

        TEST(Btree, mapint)
        {
            aries_engine::AriesIndexBtree< int > indices( aries::AriesColumnType { aries::AriesDataType { aries::AriesValueType::INT32, 1 }, false, false } );
            int count = 100000000;
            aries::CPU_Timer t;
            t.begin();
            for( int i = 0; i < count; ++i )
            {
                indices.insert( { i, i } );
            }
            t.end();

            string filename = "btreeint.out";
            cout<<"save:"<<endl;
            t.begin();
            indices.Save( filename );
            t.end();
            cout<<"load:"<<endl;
            t.begin();
            indices.Load( filename );
            t.end();
//            for( const auto& it : indices )
//            {
//                printf( "%d, %d\n", it.first, it.second );
//            }
            vector< aries_engine::AriesTupleLocation > rows;
            t.begin();
            for( const auto& it : indices )
            {
                rows.push_back( it.second );
            }
            t.end();
        }

        TEST(Btree, mapintnull)
        {
            aries_engine::AriesIndexBtree< aries_acc::nullable_type< int > > indices( aries::AriesColumnType { aries::AriesDataType { aries::AriesValueType::INT32, 1 }, true, false } );
            int count = 100; //000000;

            aries::CPU_Timer t;
            t.begin();
            aries_acc::nullable_type< int > data;
            std::uniform_int_distribution< int > d( 0, 10000000 );
            std::mt19937 mt19937;
            for( int i = 0; i < count; ++i )
            {
                data.value = d( mt19937 );
                if( i % 10 )
                    data.flag = 1;
                else
                    data.flag = 0;

                indices.insert( { data, i } );
            }
            t.end();
            for( const auto& it : indices )
            {
                printf( "flag:%d, key:%d, value:%d\n", it.first.flag, it.first.value, it.second );
            }
            string filename = "btreeintnull.out";
            indices.Save( filename );
            indices.Load( filename );
            for( const auto& it : indices )
            {
                printf( "flag:%d, key:%d, value:%d\n", it.first.flag, it.first.value, it.second );
            }
            vector< aries_engine::AriesTupleLocation > rows;
            t.begin();
            for( const auto& it : indices )
            {
                rows.push_back( it.second );
            }
            t.end();
        }

        TEST(Btree, decimal)
        {
            aries_engine::AriesIndexBtree< aries_acc::Decimal > indices( aries::AriesColumnType { aries::AriesDataType { aries::AriesValueType::DECIMAL, 1 }, false, false } );
            int count = 100; //000000;
            aries::CPU_Timer t;
            t.begin();
            for( int i = 0; i < count; ++i )
            {
                indices.insert( { i, i } );
            }
            t.end();

            string filename = "btreedecimal.out";
            indices.Save( filename );
            indices.Load( filename );
            char dec[ 64 ];
            for( const auto& it : indices )
            {
                aries_acc::Decimal d = it.first;
                printf( "%s, %d\n", d.GetDecimal( dec ), it.second );
            }
            vector< aries_engine::AriesTupleLocation > rows;
            t.begin();
            for( const auto& it : indices )
            {
                rows.push_back( it.second );
            }
            t.end();
        }

        TEST(Btree, mapsaveload)
        {
            int count = 10000000;
            vector< string > data;
            for( int i = 0; i < count; ++i )
            {
                TestKeyType key;
                string date = "2020-11-1";
                date += std::to_string( i - i % 5 );
                InitTestKey( key, i, date.c_str() );
                //InitTestKey( key, i % 10, date.c_str() );
                string str;
                str.resize( sizeof(TestKeyType) );
                memcpy( ( char* )str.data(), key, sizeof(TestKeyType) );
                data.push_back( str );
            }

            vector< aries::AriesColumnType > types;
            types.push_back( aries::AriesColumnType { aries::AriesDataType { aries::AriesValueType::INT32, 1 }, false, false } );
            types.push_back( aries::AriesColumnType { aries::AriesDataType { aries::AriesValueType::CHAR, 10 }, false, false } );

            aries_engine::AriesIndexBtree< aries_engine::AriesCompositeKeyType, aries_engine::key_hint_t::most_unique > indices( types );
            aries_engine::AriesIndexBtree< aries_engine::AriesCompositeKeyType, aries_engine::key_hint_t::most_duplicate > indices2( types );
            std::multimap< string, int > imap;
            std::unordered_multimap< string, int > umap;
            aries::CPU_Timer t;
            cout << "AriesIndexBtree, multimap insert:" << endl;
            t.begin();
            for( int i = 0; i < count; ++i )
            {
                indices.insert( { aries_engine::AriesCompositeKeyType( data[ i ] ), i } );
            }
            t.end();

            cout << "AriesIndexBtree, map insert:" << endl;
            t.begin();
            for( int i = 0; i < count; ++i )
            {
                auto it = indices2.find( aries_engine::AriesCompositeKeyType( data[ i ] ) );
                if( it == indices2.end() )
                {
                    vector< aries_engine::AriesTupleLocation > v;
                    v.push_back( i );
                    indices2.insert( { aries_engine::AriesCompositeKeyType( data[ i ] ), v } );
                }
                else
                    it->second.push_back( i );
            }
            t.end();

            cout << "std::multimap" << endl;
            t.begin();
            for( int i = 0; i < count; ++i )
            {
                imap.insert( { data[ i ], i } );
            }
            t.end();

            cout << "std::unordered_map" << endl;
            t.begin();
            for( int i = 0; i < count; ++i )
            {
                umap.insert( { data[ i ], i } );
            }
            t.end();

            string filename = "btreemultimap.out";
            string filename2 = "btreemap.out";
            // write
            cout << "AriesIndexBtree, multimap save  :" << endl;
            t.begin();
            indices.Save( filename );
            t.end();
            cout << "AriesIndexBtree, map save:" << endl;
            t.begin();
            indices2.Save( filename2 );
            t.end();
            vector< aries_engine::AriesTupleLocation > rows;
            cout << "AriesIndexBtree, multimap interator:" << endl;
            t.begin();
            for( const auto& it : indices )
            {
                rows.push_back( it.second );
            }
            t.end();
            rows.clear();
            cout << "AriesIndexBtree, map interator:" << endl;
            t.begin();
            for( const auto& it : indices2 )
            {
                for( int d : it.second )
                    rows.push_back( d );
            }
            t.end();
            rows.clear();
            vector< aries_engine::AriesTupleLocation > rows2;
            cout << "std::multimap, interator:" << endl;
            t.begin();
            for( const auto& it : imap )
            {
                rows2.push_back( it.second );
            }
            t.end();

            rows2.clear();
            vector< aries_engine::AriesTupleLocation > rows3;
            cout << "std::unordered_multimap, interator:" << endl;
            t.begin();
            for( const auto& it : umap )
            {
                rows3.push_back( it.second );
            }
            t.end();
//            for( const auto& it : indices )
//            {
//                printf( "%d, %.10s, %d\n", *( int* )it.first.data(), it.first.data() + sizeof(int), it.second );
//            }
            cout << "-----------------------------------------------" << endl;
            cout << "AriesIndexBtree, multimap load:" << endl;
            t.begin();
            indices.Load( filename );
            t.end();
            cout << "AriesIndexBtree, map load:" << endl;
            t.begin();
            indices2.Load( filename2 );
            t.end();
//            for( const auto& it : indices2 )
//            {
//                for( int d : it.second )
//                    printf( "%d, %.10s, %d\n", *( int* )it.first.m_key.data(), it.first.m_key.data() + sizeof(int), d );
//            }
//            cout << "-----------------------------------------------" << endl;
//            for( const auto& it : indices )
//            {
//                printf( "%d, %.10s, %d\n", *( int* )it.first.m_key.data(), it.first.m_key.data() + sizeof(int), it.second );
//            }
        }

        TEST(Btree, vector)
        {
            int count = 10000000;
            vector< string > data;
            for( int i = 0; i < count; ++i )
            {
                TestKeyType key;
                InitTestKey( key, i, "2020-11-11" );
                string str;
                str.resize( sizeof(TestKeyType) );
                memcpy( ( char* )str.data(), key, sizeof(TestKeyType) );
                data.push_back( str );
            }

            shared_ptr< vector< shared_ptr< IAriesCompareTo > > > comparators = make_shared< vector< shared_ptr< IAriesCompareTo > > >();
            comparators->push_back( std::make_shared< AriesComparator< int > >( false ) );
            comparators->push_back( std::make_shared< AriesStringComparator >( false, 10 ) );
            AriesKeyCompartorEx keyComp;
            keyComp.SetComparators( comparators );
            btree_map< AriesKeyEx, vector< int >, AriesKeyCompartorEx > imap( keyComp );
            btree_multimap< AriesKeyEx, int, AriesKeyCompartorEx > mmap( keyComp );
            aries::CPU_Timer t;
            t.begin();
            for( int i = 0; i < count; ++i )
            {
                AriesKeyEx key;
                key.m_key = data[ i ];
                auto it = imap.find( std::move( key ) );
                if( it == imap.end() )
                {
                    vector< int > v;
                    v.push_back( i );
                    imap.insert( { key, v } );
                }
                else
                    it->second.push_back( i );
            }
            t.end();
            cout << "------------------------------------------------" << endl;
            t.begin();
            for( int i = 0; i < count; ++i )
            {
                AriesKeyEx key;
                key.m_key = data[ i ];
                mmap.insert( { key, i } );

            }
            t.end();
            cout << "=====================================================" << endl;
            vector< int > result;
            t.begin();
//            for( const auto& it : imap )
//            {
//                for( int d : it.second )
//                    printf( "%d, %.10s, %d\n", *( int* )it.first.m_key.data(), it.first.m_key.data() + sizeof(int), d );
//                    //printf( "%d\n", d );
//            }
            for( const auto& it : imap )
            {
                for( int d : it.second )
                    result.push_back( d );
            }
            t.end();
            result.clear();
            cout << "------------------------------------------------" << endl;
            t.begin();
            for( const auto& it : mmap )
            {
                result.push_back( it.second );
            }
//            for( const auto& it : mmap )
//            {
//                printf( "%d, %.10s, %d\n", *( int* )it.first.m_key.data(), it.first.m_key.data() + sizeof(int), it.second );
//            }

            t.end();
        }

        TEST(Btree, mapsearch)
        {
            btree_multimap< int, int > tree;
            int val[] = { 1, 2, 3, 3, 4, 5 };
            for( int i = 0; i < sizeof( val ) / sizeof( int ); ++i )
            {
                tree.insert( { val[i], i } );
            }
            auto itKey = tree.lower_bound( 1 );
            for( auto it = tree.begin(); it != itKey; ++it )
                std::cout<<it->first<<endl;
            cout<<"---------------------"<<endl;
            itKey = tree.upper_bound( 3 );
            for( auto it = tree.begin(); it != itKey; ++it )
                std::cout<<it->first<<endl;
        }
    }
// namespace
}// namespace btree
