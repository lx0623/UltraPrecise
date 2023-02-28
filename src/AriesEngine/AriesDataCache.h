//
// Created by david.shen on 2019/8/29.
//

#pragma once

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include "AriesDefinition.h"
#include "AriesDataDef.h"
#include "CudaAcc/AriesEngineDef.h"
using namespace aries_acc;
using namespace std;

BEGIN_ARIES_ENGINE_NAMESPACE

    using MutexSPtr = shared_ptr<mutex>;


    struct BlockCacheInfo {
        MutexSPtr mutexSPtr;
        AriesDataBufferSPtr bufferSPtr;

        BlockCacheInfo(AriesDataBufferSPtr buffer);
        MutexSPtr& getMutex();
        AriesDataBufferSPtr& getAriesDataBuf();
    };

    using BlockCacheInfoSPtr = shared_ptr<BlockCacheInfo>;

    struct ColumnCacheInfo {
        std::mutex colMutex;
        string columnName;
        int64_t blockSize;
        vector<BlockCacheInfoSPtr> blocks;

        ColumnCacheInfo();
        ~ColumnCacheInfo();

        void setColumnName(const string &name);
        void setBlockSize(int64_t blockSize);
        void cacheData(int blockId, const AriesDataBufferSPtr & source);
        BlockCacheInfoSPtr getData(int blockId);
        void removeCache(int blockId);
        void removeAllCaches();
    };

    using ColumnCacheInfoSPtr = shared_ptr< ColumnCacheInfo >;

    struct TableCacheInfo {
        std::mutex tableMutex;
        string tableName;
        map<string, ColumnCacheInfoSPtr> columns;

        TableCacheInfo();
        ~TableCacheInfo();

        void setTableName(const string & name);
        ColumnCacheInfoSPtr getColumnCacheInfo(const string &colName);
        void removeCache(const string & colName);
        void removeAllCaches();
    };

    using TableCacheInfoSPtr = shared_ptr< TableCacheInfo >;

    struct DbCacheInfo {
        std::mutex dbMutex;
        string dbName;
        map<string, TableCacheInfoSPtr> tables;

        DbCacheInfo();
        ~DbCacheInfo();

        void setDbName(const string &name);
        TableCacheInfoSPtr getTableCacheInfo(const string &tableName);
        void removeCache(const string &tableName);
        void removeAllCaches();
    };

    using DbCacheInfoSPtr = shared_ptr< DbCacheInfo >;

    class AriesDataCache {
    public:
        static AriesDataCache& GetInstance();

        void setBlockSize(const string &dbName, const string &tableName, const string &columnName, int blockSize);

        AriesDataBufferSPtr getCacheData(const string &dbName, const string &tableName, const string &columnName, int blockId);
        MutexSPtr getCacheMutex(const string &dbName, const string &tableName, const string &columnName, int blockId);
        void cacheData(const string &dbName, const string &tableName, const string &columnName, int blockId, const AriesDataBufferSPtr & source);
        bool removeOldestUnusedOne();
        void removeCache(const string &dbName, const string& tableName = "");
        void removeAllCache();
    private:
        std::mutex dataMutex;
        vector<string> latestAccess;
        map<string, DbCacheInfoSPtr> dbs;

        string makeAccessFullColName(const string &dbName, const string &tableName, const string &columnName);
        void unmake2ColName(const string &fullColName, string &dbName, string &tableName, string &columnName);
        void save2LatestAccess(const string &fullColName);
        void discardCache(const string &dbName, const string &tableName, const string &colName);
        string popLatest();

        DbCacheInfoSPtr getDbCacheInfo(const string &dbName);

        AriesDataCache();
    };

END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */
