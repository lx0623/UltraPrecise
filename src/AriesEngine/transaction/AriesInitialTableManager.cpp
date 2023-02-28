//
// created by lidongyang on 2020.07.21
//
#include "schema/Schema.h"
#include "schema/SchemaManager.h"
#include "schema/DatabaseEntry.h"
#include "CpuTimer.h"
#include "AriesEngine/transaction/AriesInitialTable.h"
#include "AriesInitialTableManager.h"
#include "AriesEngine/AriesDataDef.h"
#include "Configuration.h"
#include "utils/string_util.h"
#include "AriesMvccTableManager.h"

using namespace std;
using namespace aries;
using namespace aries::schema;

BEGIN_ARIES_ENGINE_NAMESPACE
    void AriesInitialTableManager::cacheTables(const string &dbName){
        LOG( INFO ) << "Cache database " << dbName;
#ifdef ARIES_PROFILE
        CPU_Timer t;
        t.begin();
#endif
        auto database = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(dbName);
        if(!database)
        {
            throw AriesException(ER_BAD_DB_ERROR, "Unknown database '" + dbName + "'");
        }
        vector<string> tableNameList = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName(dbName)->GetNameListOfTables();
        for(const auto &tableName : tableNameList){
            cacheTable(dbName, tableName);
        }
#ifdef ARIES_PROFILE
        LOG( INFO ) << "Cache database " << dbName << " time: " << t.end();
#endif
    }

    void AriesInitialTableManager::allPrefetchToCpu()
    {
        for ( auto it = m_tableMap.begin(); it != m_tableMap.end(); ++it)
        {
            map< int32_t, AriesColumnSPtr > tableColumnsMap = it->second->GetAllColumns();
            for ( auto it = tableColumnsMap.begin(); it != tableColumnsMap.end(); ++it)
            {
                // std::cout << "pretch data to cpu: " << it->second << std::endl;
                it->second->PrefetchDataToCpu();
            }
        }
    }

    AriesInitialTableSPtr AriesInitialTableManager::cacheTable(const string &dbName, const string &tableName) {
        DatabaseEntrySPtr dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
        auto tableId = dbEntry->GetTableByName( tableName )->GetId();
        
        AriesInitialTableSPtr tableSPtr = m_tableMap[tableId];
        if (tableSPtr == nullptr) {
#ifdef ARIES_PROFILE
            CPU_Timer t;
            t.begin();
#endif
            LOG( INFO ) << "Load table " << dbName << "." << tableName;
            tableSPtr = m_tableMap[tableId] = make_shared< AriesInitialTable >( dbName, tableName );
            TableEntrySPtr tableEntry = dbEntry->GetTableByName(tableName);
            int columns_count = tableEntry->GetColumnsCount();
            vector<int> columns_ids;
            for(int i=0; i<columns_count; ++i){
                columns_ids.push_back(i+1);
            }
            if(tableName == "lineitem")
            {
                // lineitem last column is l_comment, which doesn't use in tpc-h test
                columns_ids.erase( columns_ids.end() - 1 );
            }
            tableSPtr->GetTable(columns_ids);
#ifdef ARIES_PROFILE
            LOG( INFO ) << "Load table " << dbName << "." << tableName << " time: " << t.end();
#endif
        }
        
        return tableSPtr;
    }

    AriesInitialTableSPtr AriesInitialTableManager::getTable(const string &dbName, const string &tableName) {
        auto tableId = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName )->GetTableByName( tableName )->GetId();

        std::lock_guard< std::mutex > lock( m_mutex4TableMap );

        AriesInitialTableSPtr tableSPtr;
        auto it = m_tableMap.find( tableId );
        if ( m_tableMap.end() == it )
        {
            tableSPtr = m_tableMap[tableId] = make_shared< AriesInitialTable >( dbName, tableName );
        }
        else
        {
            tableSPtr = it->second;
        }
        
        return tableSPtr;
    }

    void AriesInitialTableManager::clearAll()
    {
        std::lock_guard< std::mutex > lock( m_mutex4TableMap );
        m_tableMap.clear();
    }
    void AriesInitialTableManager::removeTable(const string &dbName, const string &tableName)
    {
        auto tableId = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName )->GetTableByName( tableName )->GetId();

        std::lock_guard< std::mutex > lock( m_mutex4TableMap );
        m_tableMap.erase( tableId );
    }

    void AriesInitialTableManager::DoPreCache()
    {
#ifndef NDEBUG
        std::cout << "start to pre-cache to cpu/gpu..." << std::endl;
#endif
        const auto& config = Configuartion::GetInstance().GetPreCacheConfiguration();
        auto schema = schema::SchemaManager::GetInstance()->GetSchema();
        for ( const auto& database_name : config.all_to_cpu )
        {
            auto database = schema->GetDatabaseByName( database_name );
            if ( !database )
            {
                LOG( ERROR  ) << "cannot find database with name: " << database_name;
                continue;
            }
#ifndef NDEBUG
            std::cout << "here start to cache database: " << database_name << std::endl;
#endif
            cacheTables( database_name );
#ifndef NDEBUG
            std::cout << "cache database: " << database_name << " DONE." << std::endl;
#endif
        }

        for ( const auto& item : config.all_to_gpu )
        {
            auto database = schema->GetDatabaseByName( item.first );
            if ( !database )
            {
                LOG( ERROR  ) << "cannot find database with name: " << item.first;
                continue;
            }

            auto target = item.second;
            auto& devices = target.devices_id;

            for ( const auto& table : database->GetTables() )
            {
                auto cached = cacheTable( item.first, table.first );

                if ( target.type == PreCacheTargetType::GPU_SLICE )
                {
                    std::map< int, std::vector< AriesDataBufferSPtr > > buffers_to_cache;
                    for ( const auto& pair : cached->GetAllColumns() )
                    {
                        auto buffers = pair.second->GetDataBuffers();
                        auto buffers_count = buffers.size();
                        auto avg = DIV_UP( buffers_count, devices.size() );
                        int index = 0;
                        for ( size_t i = 0; i < buffers_count; i++ )
                        {
                            if ( i >= ( index + 1 ) * avg )
                            {
                                index ++;
                            }
                            buffers_to_cache[ devices[ index ] ].emplace_back( buffers[ i ] );
                        }
                    }

                    for ( const auto& device_id : devices )
                    {
                        cudaSetDevice( device_id );
                        const auto& buffers = buffers_to_cache[ device_id ];
                        for ( const auto& buffer : buffers )
                        {
                            buffer->PrefetchToGpu( device_id );
                        }
                    }
                }
                else
                {
                    for ( const auto& pair : cached->GetAllColumns() )
                    {
                        auto buffers = pair.second->GetDataBuffers();
                        for ( const auto& id : devices )
                        {
                            cudaSetDevice( id );
                            for ( const auto& buffer : buffers )
                            {
                                buffer->PrefetchToGpu( id );
                            }
                        }
                    }
                }
            }
        }

        for ( const auto& item : config.to_cpu )
        {
            auto database_name = item.first;
            auto database = schema->GetDatabaseByName( database_name );
            if ( !database )
            {
                LOG( ERROR  ) << "cannot find database with name: " << item.first;
                continue;
            }
            for ( const auto& pair : item.second )
            {
                auto table_name = pair.first;
                if ( pair.second.all_columns )
                {
                    cacheTable( database_name, table_name );
                }
                else
                {
                    DatabaseEntrySPtr dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( database_name );
                    auto table_entry = dbEntry->GetTableByName( table_name );
                    if ( !table_entry )
                    {
                        LOG( ERROR  ) << "cannot find table with name: " << table_name;
                        continue;
                    }
                    auto tableId = table_entry->GetId();
                    auto columns = table_entry->GetColumns();

                    std::vector< int > columns_id_to_gpu;
                    
                    std::vector< int > columns_id_to_cache;
                    if ( m_tableMap.find( tableId ) == m_tableMap.cend() )
                    {
                        for ( const auto& col : pair.second.columns )
                        {
                            for ( size_t i = 0; i < columns.size(); i++ )
                            {
                                if ( aries_utils::convert_to_upper( col ) == columns[ i ]->GetName() )
                                {
                                    columns_id_to_gpu.emplace_back( i + 1 );
                                    columns_id_to_cache.emplace_back( i + 1 );
                                }
                            }
                        }

                        m_tableMap[ tableId ] = std::make_shared< AriesInitialTable >( database_name, table_name );
                    }
                    else
                    {
                        auto columns_cached = m_tableMap[ tableId ]->GetAllColumns();

                        for ( const auto& col : pair.second.columns )
                        {
                            for ( size_t i = 0; i < columns.size(); i++ )
                            {
                                if ( aries_utils::convert_to_upper( col ) == columns[ i ]->GetName() )
                                {
                                    columns_id_to_gpu.emplace_back( i + 1 );
                                    if ( columns_cached.find( i + 1 ) == columns_cached.cend() )
                                    {
                                        columns_id_to_cache.emplace_back( i + 1 );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for ( const auto& item : config.to_gpu )
        {
            auto database_name = item.first;
            auto database = schema->GetDatabaseByName( database_name );
            if ( !database )
            {
                LOG( ERROR  ) << "cannot find database with name: " << item.first;
                continue;
            }
            std::map< int, std::vector< AriesDataBufferSPtr > > buffers_to_gpu;
            for ( const auto& pair : item.second )
            {
                auto table_name = pair.first;
                auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( database_name );
                auto table_entry = dbEntry->GetTableByName( table_name );
                if ( !table_entry )
                {
                    LOG( ERROR  ) << "cannot find table with name: " << table_name;
                    continue;
                }
                auto tableId = table_entry->GetId();
                auto columns = table_entry->GetColumns();
                for ( const auto& gpu_config : pair.second )
                {
                    const auto& devices_id = gpu_config.devices_id;
                    std::vector< int > columns_id;
                    if ( gpu_config.all_columns )
                    {
                        columns_id.resize( columns.size() );
                        std::iota( columns_id.begin(), columns_id.end(), 1 );
                    }
                    else
                    {
                        for ( const auto& col : gpu_config.columns )
                        {
                            for ( size_t i = 0; i < columns.size(); i++ )
                            {
                                if ( aries_utils::convert_to_lower( col ) == columns[ i ]->GetName() )
                                {
                                    columns_id.emplace_back( i + 1 );
                                }
                            }
                        }
                    }

                    if ( m_tableMap.find( tableId ) == m_tableMap.cend() )
                    {
                        m_tableMap[ tableId ] = std::make_shared< AriesInitialTable >( database_name, table_name );
                    }

                    auto table_block = m_tableMap[ tableId ]->GetTable( columns_id );

                    if ( gpu_config.type == PreCacheTargetType::GPU_SLICE )
                    {
                        for ( size_t i = 0; i < devices_id.size(); i++ )
                        {
                            cudaSetDevice( devices_id[ i ] );
                            auto sub_table = table_block->GetSubTable2( devices_id.size(), i );
                            auto all_columns = sub_table->GetAllColumns();
                            for ( const auto& it : all_columns )
                            {
                                auto buffers = it.second->GetDataBuffers();
                                for ( const auto& buffer : buffers )
                                {
                                    buffer->PrefetchToGpu();
                                }
                            }
                        }
                    }
                    else
                    {
                        auto all_columns = table_block->GetAllColumns();
                        for ( const auto& column : all_columns )
                        {
                            auto buffers = column.second->GetDataBuffers();
                            for ( const auto& buffer : buffers )
                            {
                                for ( const auto& id : devices_id )
                                {
                                    buffers_to_gpu[ id ].emplace_back( buffer );
                                }
                            }
                        }
                    }
                }
            }

            for ( const auto& it : buffers_to_gpu )
            {
                auto id = it.first;
                cudaSetDevice( id );
                for ( const auto& buffer : it.second )
                {
                    buffer->PrefetchToGpu( id );
                }
            }
        }
#ifndef NDEBUG
        std::cout << "pre-cache to cpu/gpu done." << std::endl;
#endif
    }

    void AriesInitialTableManager::CreatePrimaryKeyIndex()
    {
        const auto& config = Configuartion::GetInstance().GetCreatePrimaryKeyIndexConfiguration();
        if ( config.empty() )
        {
            return;
        }

        for ( const auto& item : config )
        {
            const auto& database_name = item.first;
            auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( database_name );
            if ( !dbEntry )
            {
                LOG( ERROR  ) << "cannot find database with name: " << database_name;
                continue;
            }
            for ( const auto& table : item.second )
            {
#ifndef NDEBUG
                std::cout << "here create primary key index for " << database_name << "." << table << std::endl;
#endif
                auto tableEntry = dbEntry->GetTableByName( table );
                if ( !tableEntry )
                {
                    LOG( ERROR  ) << "cannot find table with name: " << table;
                    continue;
                }
                AriesMvccTableManager::GetInstance().getMvccTable( database_name, table )->CreatePrimaryKeyIndexIfNotExists();
            }
        }
    }

END_ARIES_ENGINE_NAMESPACE
