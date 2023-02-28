#pragma once

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>
#include <mutex>
#include <atomic>

#include <glog/logging.h>
#include "utils/murmur_hash.h"
#include "AriesAssert.h"
#include <server/mysql/include/my_sys.h>
#include <server/mysql/include/mysys_err.h>
// #include "schema/ColumnEntry.h"
#include "CudaAcc/AriesEngineDef.h"

using namespace std;

namespace aries {

class AriesDict;
using AriesDictSPtr = shared_ptr< AriesDict >;

class AriesDictManager;

class AriesDict
{
public:
    bool IsNullable() const { return m_nullable; }

    ~AriesDict()
    {
        if ( m_tmpBuff )
            delete[] m_tmpBuff;
    }

    int64_t GetId() const { return m_id; }
    std::string GetName() const { return m_name; }
    int64_t GetRefCount() const { return m_refCount; }
    int64_t IncRefCount() { return ++m_refCount; }
    int64_t DecRefCount()
    {
        assert( m_refCount > 0 );
        return --m_refCount;
    }

    uint32_t GetSchemaSize() const { return m_itemSchemaSize; }
    schema::ColumnType GetIndexDataType() const { return m_indexDataType; }

    size_t getDictCapacity() const
    {
        return m_dictCapacity;
    }
    uint32_t getDictIndexItemSize() const
    {
        return m_dictIndexItemSize;
    }

    uint32_t getDictItemStoreSize() const
    {
        return m_itemStoreSize;
    }
    const char* getDictData() const
    {
        return m_dictData;
    }

    const char* getDictItem( int32_t i ) const
    {
        return m_dictData + i * m_itemStoreSize;
    }

    aries_acc::AriesDataBufferSPtr getDictBuffer() const
    {
        return m_dictDataBuffer;
    }

    size_t getDictDataSize() const
    {
        return m_dictItemCnt * m_itemStoreSize;
    }
    size_t getDictItemCount()
    {
        return m_dictItemCnt;
    }

    // return: false is dict item already exists,
    //         true if new dict entry is added.
    bool addDict( const char* item,
                  size_t size,
                  size_t itemIndex,
                  int32_t* index,
                  int& errorCode,
                  string& errorMsg );

private:
    AriesDict( int64_t id,
               std::string name,
               schema::ColumnType indexDataType,
               int64_t refCount,
               bool nullable,
               int32_t char_max_len );

    aries_acc::AriesDataBufferSPtr ReadData();

    void ResetData();

    // 加载已有字典时，将字典全部读入后，创建字典hash
    void buildDictHash();

private:
    const int64_t m_id;
    const std::string m_name;
    const schema::ColumnType m_indexDataType;
    atomic_int64_t m_refCount;
    const bool m_nullable;
    const uint32_t m_itemSchemaSize; // not including nullable byte
    const uint32_t m_itemStoreSize;

    size_t m_dictCapacity; // 能容纳的字典条目数
    uint32_t m_dictIndexItemSize; // including nullable byte

    atomic_uint32_t m_dictItemCnt;

    mutex  m_dictLock;
    // 字典条目定长存储，目前只增不删
    char* m_dictData;
    // insert语句会涉及到新增字典条目， 使用AriesDataBuffer保存字典，
    // 方便Initial table输出。
    aries_acc::AriesDataBufferSPtr m_dictDataBuffer;
    unordered_map< string, int32_t > m_dictHash;

    char* m_tmpBuff;

    friend class AriesDictManager;
};

class ConvertDictColumnResult
{
public:
    ConvertDictColumnResult()
            : errorCode( 0 )
    {}
    int errorCode;
    string errorMsg;
};
using ConvertDictColumnResultSPtr = std::shared_ptr< ConvertDictColumnResult >;

aries_acc::AriesDataBufferSPtr
ConvertDictEncodedColumn( const string& colName,
                          const aries_acc::AriesDataBufferSPtr& column,
                          AriesDictSPtr& columnDict,
                          const AriesColumnType& indexDataType,
                          aries_acc::AriesManagedIndicesArraySPtr& dictIndices ); // 新增字典条目的索引

} // namespace aries