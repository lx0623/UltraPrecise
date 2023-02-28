#pragma once
#include <string>
#include <memory>
#include <unordered_map>
#include <atomic>
#include <mutex>

#include "schema/ColumnEntry.h"

#include "AriesDict.h"

#include "AriesEngine/transaction/AriesDeltaTable.h"

using namespace aries_engine;

namespace aries {

const int32_t DICT_DELTA_ROW_DATA_PREFIX_SIZE = 4;
class AriesDictManager
{
private:
    AriesDictManager();
    std::unordered_map< std::string, AriesDictSPtr > m_dictNameMap;
    std::unordered_map< int64_t, AriesDictSPtr > m_dictIdMap;
    std::atomic_int64_t m_nextTableId;
    std::mutex m_dictLock;

    unordered_map< int64_t, AriesDeltaTableSPtr > m_dictDeltaTables;
    unordered_map< int64_t, shared_ptr< int8_t[] > > m_xlogRecoverDictBuffMap;
    unordered_map< int64_t, int32_t > m_xlogRecoverDictRowCount;

public:
    static AriesDictManager& GetInstance()
    {
        static AriesDictManager instance;
        return instance;
    }

    bool Init();

    int64_t GetDictId();

    AriesDictSPtr GetOrCreateDict( const std::string& name,
                                   schema::ColumnType indexDataType,
                                   bool nullable,
                                   int32_t charMaxLen );
    void AddDict( AriesDictSPtr dict );

    void DecDictRefCount( int64_t id );

    AriesDictSPtr GetDict( int64_t id );
    AriesDictSPtr GetDict( const string& name );
    AriesDictSPtr ReadDictData( int64_t id );
    AriesDictSPtr ReadDictData( AriesDictSPtr& dict );

    std::string GetDictFilePath( int64_t id, const std::string& name ) const;
    std::string GetDictFileName( int64_t id, const std::string& name ) const;

    bool AddDictTuple( AriesTransactionPtr transaction,
                       AriesDictSPtr dict,
                       aries_acc::AriesManagedIndicesArraySPtr dictIndices );
    bool AddDictTuple( AriesTransactionPtr transaction,
                       AriesDictSPtr dict,
                       const vector< int32_t> &newDictIndices );
    void XLogRecoverInsertDict( AriesDictSPtr& dict, const vector< int8_t* >& rowsData );
    void XLogRecoverDone();

private:
    void InitDictDeltaTable( AriesDictSPtr dict );
    void DeleteDict( AriesDictSPtr& dict );
    int8_t* GetDictBufferForXLogRecover( AriesDictSPtr& dict, bool createIfNotExists );
    void FlushXLogRecoverResult();
};

} // namespace aries