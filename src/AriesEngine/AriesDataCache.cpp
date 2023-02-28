//
// Created by david.shen on 2019/8/29.
//

#include "AriesDataCache.h"
#include "CudaAcc/AriesEngineException.h"

BEGIN_ARIES_ENGINE_NAMESPACE

    BlockCacheInfo::BlockCacheInfo(AriesDataBufferSPtr buffer): mutexSPtr(new mutex()), bufferSPtr(buffer) {
    }

    MutexSPtr& BlockCacheInfo::getMutex() {
        return mutexSPtr;
    }

    AriesDataBufferSPtr& BlockCacheInfo::getAriesDataBuf() {
        return bufferSPtr;
    }

    ColumnCacheInfo::ColumnCacheInfo() {
        blockSize = 0;
        blocks.clear();
    }

    ColumnCacheInfo::~ColumnCacheInfo() {
        blocks.clear();
    }

    void ColumnCacheInfo::setColumnName(const string &name) {
        columnName = name;
    }

    void ColumnCacheInfo::setBlockSize(int64_t bs) {
        //block size is changed, should remove all caches;
        std::lock_guard<std::mutex> guard(colMutex);
        if (blockSize != bs) {
            blocks.clear();
        }
        blockSize = bs;
    }
    void ColumnCacheInfo::cacheData(int blockId, const AriesDataBufferSPtr & source) {
        ARIES_ASSERT( blockId >= 0 && blocks.size() > std::size_t( blockId ), "cache Data called before getData");
        std::lock_guard<std::mutex> guard(colMutex);
        if (blocks[blockId]->getAriesDataBuf() == nullptr) {
            blocks[blockId]->bufferSPtr = source;
        }
    }

    BlockCacheInfoSPtr ColumnCacheInfo::getData(int blockId) {
        ARIES_ASSERT( blockId >= 0, "blockId should greater than -1");
        if ( blocks.size() <= size_t( blockId ) ) {
            std::lock_guard<std::mutex> guard(colMutex);
            for (size_t i = blocks.size(); i <= size_t( blockId ); i++) {
                blocks.push_back(make_shared<BlockCacheInfo>(nullptr));
            }
        }
        return blocks[blockId];
    }

    void ColumnCacheInfo::removeCache(int blockId) {
        ARIES_ASSERT( blockId >= 0, "blockId should greater than -1");
        if ( blocks.size() > std::size_t( blockId ) ) {
            blocks[blockId] = nullptr;
        }
    }

    void ColumnCacheInfo::removeAllCaches() {
        blocks.clear();
    }

    TableCacheInfo::TableCacheInfo() {
    }

    TableCacheInfo::~TableCacheInfo() {
        columns.clear();
    }

    void TableCacheInfo::setTableName(const string & name) {
        tableName = name;
    }

    ColumnCacheInfoSPtr TableCacheInfo::getColumnCacheInfo(const string &colName) {
        ColumnCacheInfoSPtr ptr = columns[colName];
        if (ptr == nullptr) {
            std::lock_guard<std::mutex> guard(tableMutex);
            if ((ptr = columns[colName]) == nullptr) {
                ptr = columns[colName] = make_shared<ColumnCacheInfo>();
            }
        }
        return ptr;
    }
    void TableCacheInfo::removeCache(const string &colName) {
        columns.erase(colName);
    }

    void TableCacheInfo::removeAllCaches() {
        columns.clear();
    }

    DbCacheInfo::DbCacheInfo() {
    }

    DbCacheInfo::~DbCacheInfo() {
        tables.clear();
    }

    void DbCacheInfo::setDbName(const string &name) {
        dbName = name;
    }

    TableCacheInfoSPtr DbCacheInfo::getTableCacheInfo(const string &tableName) {
        TableCacheInfoSPtr ptr = tables[tableName];
        if (ptr == nullptr) {
            std::lock_guard<std::mutex> guard(dbMutex);
            if ((ptr = tables[tableName]) == nullptr) {
                ptr = tables[tableName] = make_shared<TableCacheInfo>();
            }
        }
        return ptr;
    }

    void DbCacheInfo::removeCache(const string &tableName) {
        tables.erase(tableName);
    }

    void DbCacheInfo::removeAllCaches() {
        tables.clear();
    }

    AriesDataCache& AriesDataCache::GetInstance() {
        static AriesDataCache s_AriesDataCache;
        return s_AriesDataCache;
    }

    AriesDataCache::AriesDataCache() {}

    DbCacheInfoSPtr AriesDataCache::getDbCacheInfo(const string &dbName) {
        DbCacheInfoSPtr ptr = dbs[dbName];
        if (ptr == nullptr) {
            std::lock_guard<std::mutex> guard(dataMutex);
            if ((ptr = dbs[dbName]) == nullptr) {
                ptr = dbs[dbName] = make_shared<DbCacheInfo>();
            }
        }
        return ptr;
    }

    void AriesDataCache::setBlockSize(const string &dbName, const string &tableName, const string &columnName, int blockSize) {
        getDbCacheInfo(dbName)->getTableCacheInfo(tableName)->getColumnCacheInfo(columnName)->setBlockSize(blockSize);
    }

    AriesDataBufferSPtr AriesDataCache::getCacheData( const string &dbName, const string &tableName, const string &columnName, int blockId ) {
        return getDbCacheInfo(dbName)->getTableCacheInfo(tableName)->getColumnCacheInfo(columnName)->getData(blockId)->getAriesDataBuf();
    }

    MutexSPtr AriesDataCache::getCacheMutex(const string &dbName, const string &tableName, const string &columnName, int blockId) {
        return getDbCacheInfo(dbName)->getTableCacheInfo(tableName)->getColumnCacheInfo(columnName)->getData(blockId)->getMutex();
    }


    void AriesDataCache::cacheData(const string &dbName, const string &tableName, const string &columnName, int blockId,
                                   const AriesDataBufferSPtr &source) {
        getDbCacheInfo(dbName)->getTableCacheInfo(tableName)->getColumnCacheInfo(columnName)->cacheData(blockId, source);
        save2LatestAccess(makeAccessFullColName(dbName, tableName, columnName));
    }

    string AriesDataCache::makeAccessFullColName(const string &dbName, const string &tableName, const string &columnName) {
        return dbName + ":" + tableName + ":" + columnName;
    }

    void AriesDataCache::unmake2ColName(const string &fullColName, string &dbName, string &tableName, string &columnName) {
        //for dbName
        size_t start = 0;
        size_t end = fullColName.find(":");
        dbName = fullColName.substr(start, end - start);
        //for tableName
        start = end + 1;
        end = fullColName.find(":", start);
        tableName = fullColName.substr(start, end - start);
        //for tableName
        start = end + 1;
        tableName = fullColName.substr(start, fullColName.size() - start);
    }

    void AriesDataCache::save2LatestAccess(const string &fullColName) {
        std::lock_guard<std::mutex> guard(dataMutex);
        size_t  i = 0;
        for (; i < latestAccess.size(); i++) {
            if (latestAccess[i] == fullColName) {
                latestAccess.erase(latestAccess.begin() + i);
                break;
            }
        }
        latestAccess.insert(latestAccess.begin(), fullColName);
    }

    string AriesDataCache::popLatest() {
        std::lock_guard<std::mutex> guard(dataMutex);
        string fullColName = latestAccess.back();
        if (!fullColName.empty()) {
            latestAccess.pop_back();
        }
        return fullColName;
    }

    bool AriesDataCache::removeOldestUnusedOne() {
        string fullColName = popLatest();
        string dbName, tableName, colName;
        unmake2ColName(fullColName, dbName, tableName, colName);
        discardCache(dbName, tableName, colName);
        return true;
    }

    void AriesDataCache::discardCache(const string &dbName, const string &tableName, const string &colName) {
        getDbCacheInfo(dbName)->getTableCacheInfo(tableName)->removeCache(colName);
    }

    void AriesDataCache::removeCache(const string &dbName, const string& tableName ) {
        auto dbCache = getDbCacheInfo(dbName);
        if ( dbCache )
        {
            if ( tableName.empty() )
                dbCache->removeAllCaches();
            else
                dbCache->removeCache( tableName );
        }
    }

    void AriesDataCache::removeAllCache() {
        for (auto &it : dbs) {
            it.second->removeAllCaches();
        }
    }


END_ARIES_ENGINE_NAMESPACE

