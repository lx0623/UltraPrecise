/*
 * AriesTableKeys.cpp
 *
 *  Created on: Sep 9, 2020
 *      Author: lichi
 */

#include "AriesTableKeys.h"
#include "CudaAcc/AriesSqlOperator_helper.h"
#include "functions.hxx"
#include <future>

BEGIN_ARIES_ENGINE_NAMESPACE

    AriesTableKeys::AriesTableKeys()
            : m_initialKeyCount( 0 ), m_perKeySize( 0 )
    {
    }

    AriesTableKeys::~AriesTableKeys()
    {
    }

    bool AriesTableKeys::Build( const std::vector< aries_engine::AriesColumnSPtr >& columns, bool checkDuplicate, size_t reservedDeltaCount )
    {
        assert( !columns.empty() );
        bool bRet = false;
        m_perKeySize = 0;
        for( const auto& col : columns )
            m_perKeySize += col->GetColumnType().GetDataTypeSize();
        if( columns[0]->GetRowCount() > 0 )
        {
            AriesTableKeysDataSPtr result = GenerateTableKeys( columns, checkDuplicate );
            if( !result->KeysData.empty() )
            {
                m_initialKeys = std::move( result->KeysData );
                m_tupleLocations = std::move( result->TupleLocations );
                m_initialKeyCount = m_initialKeys.size() / m_perKeySize;
                assert( m_initialKeyCount == m_tupleLocations.size() );
                assert( m_initialKeys.size() % m_perKeySize == 0 );
                if( reservedDeltaCount > 0 )
                    m_deltaKeys.reserve( reservedDeltaCount );
                bRet = true;
            }
        }
        else
            bRet = true;
        return bRet;
    }

    bool AriesTableKeys::InsertKey( const string& key, RowPos rowPos, bool checkDuplicate, size_t existedLocationCount )
    {
        bool bRet = false;
        auto result = FindKeyInternal( key );
        bool canInsert = true;
        unique_lock< mutex > lock( m_mutex );
        if( checkDuplicate && result.first )
        {
            assert( result.second );
            //对于主键，在FindKey后，根据mvcc规则判断可以添加时，再调用InsertKey.但如果调用FindKey和InsetKey之间，有其他线程将新的数据添加，则返回失败．
            canInsert = ( result.second->size() == existedLocationCount );
        }

        if( canInsert )
        {
            if( result.first )
            {
                assert( result.second );
                result.second->push_back( rowPos );
            }
            else
            {
                AriesRowPosContainer tmp;
                tmp.push_back( rowPos );
                m_deltaKeys.emplace( key, std::move( tmp ) );
            }
            bRet = true;
        }

        return bRet;
    }

    pair< bool, AriesRowPosContainer > AriesTableKeys::FindKey( const string& key )
    {
        auto result = FindKeyInternal( key );
        AriesRowPosContainer data;

        if( result.first )
        {
            unique_lock< mutex > lock( m_mutex );
            assert( result.second );
            //返回拷贝，避免线程冲突
            data = std::move( result.second->clone() );
        }
        return
        {   result.first, std::move( data )};
    }

    bool AriesTableKeys::IsKeyInDeltaTable(const string &key)
    {
        unique_lock<mutex> lock(m_mutex);
        return m_deltaKeys.find(key) != m_deltaKeys.end();
    }

    pair< bool, AriesRowPosContainer* > AriesTableKeys::FindKeyInternal( const string& key )
    {
        bool bRet = false;
        AriesRowPosContainer* result = nullptr;

        int index = BinaryFind( m_initialKeys.data(), m_perKeySize, m_initialKeyCount, key.data() );
        if( index != -1 )
        {
            assert( index >= 0 && ( size_t )index < m_initialKeyCount );
            assert( !IsKeyInDeltaTable( key ) );
            result = &m_tupleLocations[index];
            bRet = true;
        }
        else
        {
            unique_lock< mutex > lock( m_mutex );
            auto it = m_deltaKeys.find( key );
            if( it != m_deltaKeys.end() )
            {
                result = &it->second;
                bRet = true;
            }
        }
        return
        {   bRet, result};
    }

    int AriesTableKeys::BinaryFind( const char* keys, size_t len, int count, const char* key )
    {
        int begin = 0;
        int end = count;
        while( begin < end )
        {
            int mid = ( begin + end ) / 2;
            const char* key2 = keys + mid * len;
            if( str_less_t( key2, key, len ) )
                begin = mid + 1;
            else
                end = mid;
        }
        if( begin < count && str_equal_to_t( key, keys + begin * len, len ) )
            return begin;
        else
            return -1;
    }

    bool AriesTableKeys::Merge( AriesTableKeysSPtr incr )
    {
        //断言一下key size是相同的
        assert( m_perKeySize == incr->m_perKeySize );
        //对于增量导入的incr, delta table一定是empty
        assert( incr->m_deltaKeys.empty() );

        bool bSuccess = false;
        if( m_initialKeyCount == 0 )
        {
            //把incr的intitial table keys数据拿过来就行了
            m_initialKeys = std::move( incr->m_initialKeys );
            m_initialKeyCount = incr->m_initialKeyCount;
            incr->m_initialKeyCount = 0;
            m_tupleLocations = std::move( incr->m_tupleLocations );
            bSuccess = true;
        }
        else 
        {
            if( !m_deltaKeys.empty() )
            {
                //检查incr intital keys和自己的delta table keys是否有冲突
                int64_t threadNum = thread::hardware_concurrency();
                int64_t blockSize = incr->m_initialKeyCount / threadNum;
                int64_t extraSize = incr->m_initialKeyCount % threadNum;
                vector< future< bool > > allThreads;
                int startIndex;
                const char* pIncrKeys = incr->m_initialKeys.data();
                for( int i = 0; i < threadNum; ++i )
                {
                    startIndex = blockSize * i;
                    if( i == threadNum - 1 )
                        allThreads.push_back( std::async( std::launch::async, [&]( int index )
                        {
                            size_t perKeySize = m_perKeySize;
                            string tmp;
                            for( int n = index; n < index + blockSize + extraSize; ++n )
                            {
                                tmp.clear();
                                tmp.insert( 0, pIncrKeys + n * perKeySize, perKeySize );
                                if( m_deltaKeys.find( tmp ) != m_deltaKeys.end() )
                                    return false;
                            }
                            return true;
                        }, startIndex ) );
                    else
                        allThreads.push_back( std::async( std::launch::async, [&]( int index )
                        {
                            size_t perKeySize = m_perKeySize;
                            string tmp;
                            for( int n = index; n < index + blockSize; ++n )
                            {
                                tmp.clear();
                                tmp.insert( 0, pIncrKeys + n * perKeySize, perKeySize );
                                if( m_deltaKeys.find( tmp ) != m_deltaKeys.end() )
                                    return false;
                            }
                            return true;
                        }, startIndex ) );
                }
                for( auto& t : allThreads )
                {
                    if( !t.get() )
                        return false;
                }
            }
            
            //检查incr intital keys和自己的init table keys是否有冲突
            size_t totalInitKeyCount = m_initialKeyCount + incr->m_initialKeyCount;
            AriesDataBufferSPtr totalInitKeys = std::make_shared< AriesDataBuffer >(
                AriesColumnType( AriesDataType( AriesValueType::CHAR, m_perKeySize ), false, false ), totalInitKeyCount );
            totalInitKeys->PrefetchToCpu();
            int8_t* pKeys = totalInitKeys->GetData();

            //拷贝自己的initial table keys
            memcpy( pKeys, m_initialKeys.data(), m_initialKeyCount * m_perKeySize );
            pKeys += m_initialKeyCount * m_perKeySize;

            //拷贝incr的intital table keys
            memcpy( pKeys, incr->m_initialKeys.data(), incr->m_initialKeyCount * m_perKeySize );
            totalInitKeys->PrefetchToGpu();
            auto mergedInitKeysResult = SortAndVerifyUniqueTableKeys( totalInitKeys );
            if( mergedInitKeysResult.first )
            {
                //initial keys OK
                totalInitKeys->PrefetchToCpu();
                m_initialKeys.clear();
                m_initialKeys.resize( totalInitKeys->GetTotalBytes() );
                memcpy( m_initialKeys.data(), totalInitKeys->GetData(), totalInitKeys->GetTotalBytes() );
                

                // merge m_tupleLocations
                vector< AriesRowPosContainer > totalTupleLocations;
                totalTupleLocations.resize( totalInitKeyCount );
                AriesRowPosContainer* pTupleLocations = totalTupleLocations.data();
                AriesRowPosContainer* pMyLocations = m_tupleLocations.data();
                AriesRowPosContainer* pIncrLocations = incr->m_tupleLocations.data();
                mergedInitKeysResult.second->PrefetchToCpu();
                int32_t* pIndices = mergedInitKeysResult.second->GetData();

                int64_t threadNum = thread::hardware_concurrency();
                int64_t blockSize = totalInitKeyCount / threadNum;
                int64_t extraSize = totalInitKeyCount % threadNum;
                vector< future< void > > allThreads;
                int startIndex;
                for( int i = 0; i < threadNum; ++i )
                {
                    startIndex = blockSize * i;
                    if( i == threadNum - 1 )
                        allThreads.push_back( std::async( std::launch::async, [&]( int index )
                        {
                            int64_t myInitCount = m_initialKeyCount;
                            for( int n = index; n < index + blockSize + extraSize; ++n )
                            {
                                int32_t pos = pIndices[ n ];
                                if( pos < myInitCount )
                                    pTupleLocations[ n ] = pMyLocations[ pos ];
                                else 
                                {
                                    pTupleLocations[ n ] = pIncrLocations[ pos - myInitCount ];
                                    pTupleLocations[ n ].add_offset( myInitCount );
                                }
                                    
                            }

                        }, startIndex ) );
                    else
                        allThreads.push_back( std::async( std::launch::async, [&]( int index )
                        {
                            int64_t myInitCount = m_initialKeyCount;
                            for( int n = index; n < index + blockSize; ++n )
                            {
                                int32_t pos = pIndices[ n ];
                                if( pos < myInitCount )
                                    pTupleLocations[ n ] = pMyLocations[ pos ];
                                else 
                                {
                                    pTupleLocations[ n ] = pIncrLocations[ pos - myInitCount ];
                                    pTupleLocations[ n ].add_offset( myInitCount );
                                }
                            }
                        }, startIndex ) );
                }
                for( auto& t : allThreads )
                    t.wait();
                m_initialKeyCount = totalInitKeyCount;
                std::swap( m_tupleLocations, totalTupleLocations );
                bSuccess = true;
            }
        }
        return bSuccess;
    }

END_ARIES_ENGINE_NAMESPACE
/* namespace aries_engine */