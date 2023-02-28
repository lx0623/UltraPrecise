//
// created by lidongyang on 2020.07.21
//
#pragma once

#include <mutex>

#include "schema/Schema.h"
#include "schema/SchemaManager.h"
#include "schema/DatabaseEntry.h"

using namespace std;
using namespace aries;
using namespace aries::schema;

BEGIN_ARIES_ENGINE_NAMESPACE
    class AriesInitialTable;
    using AriesInitialTableSPtr = shared_ptr<AriesInitialTable>;
    class AriesInitialTableManager{
    public:
       static AriesInitialTableManager & GetInstance() {
            static AriesInitialTableManager instance;
            return instance;
       }
        /*
        启动时若指定了--cache-db <database name>，则调用此方法缓存指定数据库的所有表
        */
        void cacheTables(const string &dbName);

        void DoPreCache();

        void CreatePrimaryKeyIndex();

        void allPrefetchToCpu();

        AriesInitialTableSPtr cacheTable(const string &dbName, const string &tableName);

        AriesInitialTableSPtr getTable(const string &dbName, const string &tableName);

        void removeTable(const string &dbName, const string &tableName);

        void clearAll();

    private:
        map<TableId, AriesInitialTableSPtr> m_tableMap;
        std::mutex m_mutex4TableMap;
    };
END_ARIES_ENGINE_NAMESPACE
