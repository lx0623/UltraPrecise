#pragma once

#include <string>
#include <vector>
#include <map>

namespace aries
{

enum class PreCacheTargetType : int32_t
{
    CPU,
    GPU,
    GPU_SLICE
};

const int PreCacheCPUTarget = -1;

struct PreCaccheConfigToCPU
{
    bool all_columns;
    std::vector< std::string > columns;
};

struct PreCacheConfigToGPU
{
    PreCacheConfigToGPU() : all_columns( false )
    {
    }

    bool all_columns;
    PreCacheTargetType type;
    std::vector< std::string > columns;
    std::vector< int > devices_id;
};

struct PreCacheConfigCollection
{
    std::vector< std::string > all_to_cpu;
    std::map< std::string, PreCacheConfigToGPU > all_to_gpu;
    std::map< std::string, std::map< std::string, PreCaccheConfigToCPU > > to_cpu;
    std::map< std::string, std::map< std::string, std::vector< PreCacheConfigToGPU > > > to_gpu;
};

class Configuartion
{

private:
    bool exchange_enabled;
    std::string base_dir;
    std::string data_dir;
    std::string column_data_dir; // store column data of tables
    std::string dict_data_dir; // store dict data
    std::string log_dir;
    std::string xlog_dir;
    std::string xlog_recover_dir;
    std::string data_xlog_recover_dir;
    std::string dict_xlog_recover_dir;
    std::string xlog_recover_done_file_path;
    std::string data_backup_dir;
    // std::string dict_backup_dir;

    int compute_version_major;
    int compute_version_minor;
    size_t exchange_row_count_threshold;
    int cuda_device_count;

    std::map< std::string, std::vector< std::string > > tables_to_create_primary_key_index;

    // {
    //     "database": {
    //         "table": [
    //             PreCacheConfig, PreCacheConfig, PreCacheConfig
    //         ]
    //     }
    // }
    PreCacheConfigCollection pre_cache_configurations;

    Configuartion();

public:
    static Configuartion& GetInstance();

    void InitCudaEnv();

    std::string GetBaseDirectory() const { return base_dir; }
    std::string GetDataDirectory() const;
    std::string GetColumnDataDirectory() const;
    std::string GetDataDirectory( const std::string& dbName ) const;
    std::string GetDataDirectory( const std::string& dbName, const std::string& tableName ) const;
    void SetDataDirectory( std::string dir );

    std::string GetDictDataDirectory() const;

    std::string GetLogDirectory() const { return log_dir; }

    std::string GetXLogRecoverDoneFilePath() const;
    std::string GetDataXLogRecoverDirectory( ) const { return data_xlog_recover_dir; }
    std::string GetDataXLogRecoverDirectory( const std::string& dbName, const std::string tableName ) const;
    std::string GetDictXLogRecoverDirectory() const { return dict_xlog_recover_dir; }

    std::string GetDataBackupDirectory() const { return data_backup_dir; }
    // std::string GetDictBackupDirectory() const { return dict_backup_dir; }

    std::string GetTmpDirectory() const;

    bool LoadFromFile( const char* config_file );

    std::string GetXLogDataDirectory() const;
    int GetComputeVersionMajor() const;
    int GetComputeVersionMinor() const;
    bool IsExchangeEnabled() const;

    size_t GetExchangeRowCountThreshold() const;

    int GetCudaDeviceCount() const;

    const PreCacheConfigCollection& GetPreCacheConfiguration() const;
    const std::map< std::string, std::vector< std::string > >& GetCreatePrimaryKeyIndexConfiguration() const;
};

}
