#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>

#include "Configuration.h"
#include "cuda_runtime.h"
#include "frontend/nlohmann/json.hpp"

namespace aries
{

using JSON = nlohmann::json;

Configuartion& Configuartion::GetInstance()
{
    static Configuartion instance;
    return instance;
}

static std::string DATA_DIR_NAME( "data" );
static std::string DICT_DATA_DIR_NAME( "dict" );
static std::string LOG_DIR_NAME( "log" );
static std::string XLOG_DIR_NAME( "xlog" );
static std::string XLOG_RECOVER_DIR_NAME( "xlog_recover" );
static std::string XLOG_RECOVER_DONE_FILE_NAME( "xlog_recover_done" );
static std::string DATA_BACKUP_DIR_NAME( "backup" );
static std::string DICT_BACKUP_DIR_NAME( "backup_dict" );

static std::string DEFAULT_BASE_DIR( "/var/rateup" );

Configuartion::Configuartion()
: exchange_enabled( false ),
  base_dir( DEFAULT_BASE_DIR )
{
    exchange_row_count_threshold = 600000000;

    SetDataDirectory( DEFAULT_BASE_DIR );
}

void Configuartion::InitCudaEnv()
{
    cudaDeviceProp prop;
    auto ret = cudaGetDeviceProperties( &prop, 0 );
    if( cudaSuccess == ret )
    {
        compute_version_major = prop.major;
        compute_version_minor = prop.minor;
    }
    else
    {
        compute_version_major = 7;
        compute_version_minor = 5;
    }

    ret = cudaGetDeviceCount( &cuda_device_count );
    if ( ret != cudaSuccess )
    {
#ifndef NDEBUG
        std::cout << "cannot get device count" << std::endl;
#endif
    }
}

template< typename type_t, typename Handler >
static void parse_array( JSON array, std::vector< type_t >& out, Handler handler )
{
    if ( !array.is_array() )
    {
        return;
    }

    for ( const auto& item : array.items() )
    {
        out.emplace_back( handler( item.value() ) );
    }
}

static bool parse_create_primary_key_index( const JSON& data, std::map< std::string, std::vector< std::string > >& configs )
{
    if ( !data.is_array() )
    {
        return false;
    }

    for ( const auto& item : data.items() )
    {
        const auto& item_data = item.value();
        const auto& database_name = item_data[ "database" ];
        
        const auto& tables = item_data[ "tables" ];
        if ( !tables.is_array() )
        {
            continue;
        }

        for ( const auto& table : tables.items() )
        {
            configs[ database_name ].emplace_back( table.value().get< std::string >() );
        }
    }

    return true;
}

static bool parse_pre_cache( const JSON& data, PreCacheConfigCollection& pre_cache_configurations )
{
    if ( !data.is_array() )
    {
        std::cerr << "precache data is not an array" << std::endl;
        return false;
    }

    for ( const auto& item : data.items() )
    {
        auto item_data = item.value();
        auto database_name = item_data.value< std::string >( "database", "" );

        const auto& to_cpu = item_data[ "to_cpu" ];
        const auto& to_gpu = item_data[ "to_gpu" ];
        auto& database_config = pre_cache_configurations.to_cpu[ database_name ];

        if ( !to_cpu.is_null() )
        {
            const auto& tables = to_cpu[ "tables" ];
            if ( tables.is_array() )
            {
                for ( const auto& table_item : tables.items() )
                {
                    auto table_name = table_item.value().value< std::string >( "table", "" );
                    const auto& columns = table_item.value()[ "columns" ];
                    if ( columns.is_string() )
                    {
                        database_config[ table_name ].all_columns = true;
                    }
                    else if ( columns.is_array() )
                    {
                        database_config[ table_name ].all_columns = false;
                        parse_array( columns, database_config[ table_name ].columns, []( const JSON& data )
                        {
                            return data.get< std::string >();
                        } );
                    }
                }
            }
            else if ( tables.is_string() )
            {
                auto tables_string = tables.get< std::string >();
                pre_cache_configurations.all_to_cpu.emplace_back( database_name );
            }
        }

        auto& to_gpu_configs = pre_cache_configurations.to_gpu[ database_name ];
        if ( to_gpu.is_array() )
        {
            for ( const auto& item : to_gpu.items() )
            {
                PreCacheConfigToGPU entry;
                const auto& devices = item.value()[ "devices" ];
                parse_array( devices, entry.devices_id, []( const JSON& data)
                {
                    return data.get< int >();
                } );

                entry.type = item.value().value( "type", "slice" ) == "slice" ? PreCacheTargetType::GPU_SLICE : PreCacheTargetType::GPU;
                auto tables = item.value()[ "tables" ];

                if ( tables.is_array() )
                {
                    for ( const auto& table_item : tables.items() )
                    {
                        auto table_name = table_item.value()[ "table" ].get< std::string >();
                        auto columns = table_item.value()[ "columns" ];
                        if ( columns.is_string() )
                        {
                            entry.all_columns = true;
                            to_gpu_configs[ table_name ].emplace_back( entry );
                        }
                        else if ( columns.is_array() )
                        {
                            parse_array( columns, entry.columns, []( const JSON& data )
                            {
                                return data.get< std::string >();
                            } );
                            to_gpu_configs[ table_name ].emplace_back( entry );
                        }
                    }
                }
                else if ( tables.is_string() )
                {
                    pre_cache_configurations.all_to_gpu[ database_name ].all_columns = true;
                    pre_cache_configurations.all_to_gpu[ database_name ].devices_id = entry.devices_id;
                    pre_cache_configurations.all_to_gpu[ database_name ].type = entry.type;
                }
            }
        }
    }

    return true;
}

bool Configuartion::LoadFromFile( const char* config_file )
{
    std::ifstream f( config_file );
    if ( !f.is_open() )
    {
#ifndef NDEBUG
        std::cout << "cannot open file: " << config_file << std::endl;
#endif
        return false;
    }
    std::string str((std::istreambuf_iterator<char>(f)),
                 std::istreambuf_iterator<char>());
    f.close();

    JSON json;
    try {
        json = JSON::parse( str );
    }
    catch ( ... )
    {
        std::cerr << "cannot parse config file, maybe not a valid json file" << std::endl;
        return false;
    }

    if ( json.contains( "precache") )
    {
        if ( !parse_pre_cache( json[ std::string( "precache" ) ], pre_cache_configurations ) )
        {
            std::cerr << "cannot parse precache" << std::endl;
        }
    }

    parse_create_primary_key_index( json[ "create_primary_key_index" ], tables_to_create_primary_key_index );

    exchange_enabled = json.value< bool >( "exchange_enabled", false );

    return true;
}

const std::map< std::string, std::vector< std::string > >& Configuartion::GetCreatePrimaryKeyIndexConfiguration() const
{
    return tables_to_create_primary_key_index;
}

int Configuartion::GetComputeVersionMajor() const
{
    return compute_version_major;
}

int Configuartion::GetComputeVersionMinor() const
{
    return compute_version_minor;
}

std::string Configuartion::GetDataDirectory() const
{
    return data_dir;
}
std::string Configuartion::GetColumnDataDirectory() const
{
    return column_data_dir;
}

std::string Configuartion::GetDataDirectory( const std::string& dbName ) const
{
    std::string dir = column_data_dir;
    dir.append( "/" ).append( dbName );
    return dir;
}

std::string Configuartion::GetDataDirectory(const std::string& dbName, const std::string& tableName ) const
{ 
    std::string dir = column_data_dir;
    dir.append( "/" ).append( dbName ).append( "/" ).append( tableName );
    return dir;
}

std::string Configuartion::GetDictDataDirectory() const
{
    return dict_data_dir;
}

void Configuartion::SetDataDirectory( std::string dir )
{
    data_dir = dir;
    column_data_dir = data_dir + "/" + DATA_DIR_NAME;
    dict_data_dir = data_dir + "/" + DICT_DATA_DIR_NAME;
    xlog_dir = data_dir + "/" + XLOG_DIR_NAME;
    xlog_recover_dir = data_dir + "/" + XLOG_RECOVER_DIR_NAME;
    data_xlog_recover_dir = xlog_recover_dir + "/" + DATA_DIR_NAME;
    dict_xlog_recover_dir = xlog_recover_dir + "/" + DICT_DATA_DIR_NAME;
    xlog_recover_done_file_path = xlog_recover_dir + "/" + XLOG_RECOVER_DONE_FILE_NAME;
    data_backup_dir = data_dir + "/" + DATA_BACKUP_DIR_NAME;
    // dict_backup_dir = data_dir + "/" + DICT_BACKUP_DIR_NAME;
    log_dir = data_dir + "/" + LOG_DIR_NAME;

    boost::filesystem::create_directories( column_data_dir );
    boost::filesystem::create_directories( dict_data_dir );
    // boost::filesystem::create_directories( dict_backup_dir );
}

std::string Configuartion::GetXLogRecoverDoneFilePath() const
{
    return xlog_recover_done_file_path;
}

std::string Configuartion::GetXLogDataDirectory() const
{
    return xlog_dir;
}

std::string Configuartion::GetDataXLogRecoverDirectory( const std::string& dbName, const std::string tableName ) const
{
    std::string dir( data_xlog_recover_dir );
    return dir.append( "/" ).append( dbName ).append( "/" ).append( tableName );
}

size_t Configuartion::GetExchangeRowCountThreshold() const
{
    return exchange_row_count_threshold;
}

int Configuartion::GetCudaDeviceCount() const
{
    return cuda_device_count;
}

const PreCacheConfigCollection& Configuartion::GetPreCacheConfiguration() const
{
    return pre_cache_configurations;
}

bool Configuartion::IsExchangeEnabled() const
{
    return exchange_enabled;
}

std::string Configuartion::GetTmpDirectory() const
{
    return data_dir + "/tmp";
}

}
