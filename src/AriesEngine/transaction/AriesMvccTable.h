//
// Created by david.shen on 2020/3/12.
//

#pragma once

#include "schema/SchemaManager.h"
#include "AriesTuple.h"
#include "AriesDeltaTable.h"
#include "AriesInitialTable.h"
#include "AriesTransManager.h"
#include "AriesXLog.h" // TableId
#include "AriesEngine/index/AriesIndex.h"
#include "AriesTableKeys.h"

using namespace std;

BEGIN_ARIES_ENGINE_NAMESPACE

/*!
 动态计算一个delta table block 包含的tuple数量。
 block中tuple数量预分配为initial table中tuple数量的1/50， 
 且介于 DELTA_TABLE_TUPLE_COUNT_MIN 和 DELTA_TABLE_TUPLE_COUNT_MAX 之间。
 */
const size_t DELTA_TABLE_TUPLE_COUNT_MIN = 10000; 
const size_t DELTA_TABLE_TUPLE_COUNT_MAX = 10000000;

    struct AriesPrimaryKeyInfo
    {
        vector< int > columnIds;
        AriesTableKeysSPtr index;
    };

#define PRIMARY_KEY_INDEX_IN_INDEXKEYS (-1)

    class AriesMvccTable
    {
    public:
        AriesMvccTable( const string &dbName, const string &tableName );

        AriesMvccTable( const string &dbName,
                        const string &tableName,
                        const string& tableDataDir );

        void ResetInitTable();

        //对某行加锁
        void Lock( RowPos pos );
        bool TryLock( RowPos pos );

        //对某行解锁
        void Unlock( RowPos pos );

        /*
         * AddTuple: 增加一条新的tuple记录
         * txId : transaction id
         * cid  : 本次操作在此txId中是第几条命令
         * dataBuffer : 添加新tuple数据所在Buffer
         * */
        bool AddTuple( AriesTransactionPtr transaction, TupleDataSPtr dataBuffer, bool checkKeys = true );

        /*
         * ModifyTuple: 修改(包括删除)tuple记录
         * txId  : transaction id
         * cid   : 本次操作在此txId中是第几条命令
         * oldPos: 老的tuple所在位置
         * dataBuffer  : 添加新tuple数据所在Buffer
         *       != nullptr : 增加一条更新数据的tuple记录
         *        = nullptr : 增加一条删除数据的tuple记录
         * dataIndex: 新tuple里的数据在dataBuffer里的位置, 如果dataBuffer为nullptr, 此参数无效
         * */
        bool ModifyTuple( AriesTransactionPtr transaction, RowPos oldPos, TupleDataSPtr dataBuffer = nullptr, int dataIndex = 0,
                bool checkKeys = true );

        bool DeleteTuple( AriesTransactionPtr transaction, const RowPos* tuplePos, size_t count );

        //设置对应数据行的t_xmax值
        void SetTxMax( RowPos pos, TxId value );

        //获取对应数据行的t_xmax值
        TxId GetTxMax( RowPos pos ) const;

        /*
         * GetTable: acquire raw data from colId according to SnapShot info
         * */
        AriesTableBlockUPtr GetTable(
            const AriesTransactionPtr &transaction,
            const vector< int > &colsId,
            const std::vector< AriesCommonExprUPtr >& partitionConditions = {} ) const;

        AriesInitialTableSPtr GetInitialTable() const
        {
            return m_InitialTableSPtr;
        }

        //对于txId对应的transaction而言，主键可能存在（tuple的xmin或xmax处于inprogress状态时获取的信息不一定准确，随着事务的提交或者终止，对应tuple的主键可能变成有效
        bool PrimaryKeyMightExists( TxId txId, const void *key, RowPos pos, const vector< int > &keyColumnIds ) const;

        //对于txId对应的transaction而言，主键肯定存在
        bool PrimaryKeyExists( TxId txId, const void *key, RowPos pos, const vector< int > &keyColumnIds ) const;

        static string MakeIndexKey( const vector< int > &colIds, const TupleDataSPtr &dataBuffer, const int &rowIndex );

        AriesPrimaryKeyInfo GetPrimaryKeyInfo( AriesTransactionPtr &transaction );
        AriesTableKeysSPtr GetPrimaryKey();

        //导入数据完成后，调用此函数检查是否有主键冲突，有主键冲突时抛出异常码ER_DUP_ENTRY
        void RebuildPrimaryKeyIndex();

        void CreatePrimaryKeyIndexIfNotExists();

        void CreateIncrementPrimaryKeyIndexIfNotExists(
                 const uint32_t startBlockIndex,
                 const uint32_t startBlockLineIndex );
        bool CreateKeyIndex( const vector< int > &keyColIds,
                             AriesTableKeysSPtr &index,
                             const uint32_t startBlockIndex,
                             const uint32_t startBlockLineIndex,
                             bool checkDuplicate );

    private:
        /*
         * AddTuple: 增加一条新的tuple记录
         * txId : transaction id
         * cid  : 本次操作在此txId中是第几条命令
         * dataBuffer : 添加新tuple数据所在Buffer
         * dataIndex: 新tuple里的数据在dataBuffer里的位置
         * */
        bool AddTuple( AriesTransactionPtr transaction, TupleDataSPtr dataBuffer, int dataIndex, bool checkKeys = true );

        AriesTableBlockUPtr MakeTable( const AriesTransactionPtr &transaction, const vector< int > &colsId, const std::vector< AriesCommonExprUPtr >& partitionConditions = {} ) const;

        const int8_t* GetFieldContent( RowPos pos, int columnId ) const;

        // pos是一个有效位置，并且对应key columns的值和参数key的值相同时返回true,否则返回false
        bool IsKeyValid( const void *key, RowPos pos, const vector< int > &keyColumnIds ) const;

        void ParseIndexKeys( TableEntrySPtr &tableEntry );

        string MakePrimaryKey( const TupleDataSPtr &dataBuffer, int rowIndex );

        string GetPrimaryKeyName();
        string GetPrimaryKeyValue( const int &rowIndex );
        void ThrowException4PrimaryKeyConflict( const TupleDataSPtr &dataBuffer, int rowIndex );
        void ThrowException4ForeignKeysNoReferenced( int errorKeyIndex );
        void ThrowException4PrimaryKeyDelete();

        bool CheckPrimaryKeyExist( AriesTransactionPtr &transaction, const string &key, size_t& existedPrimaryKeyLocationCount );

        bool CheckIfCanBeInserted( AriesTransactionPtr &transaction, const string& pkKey, int rowIndex, size_t& existedPrimaryKeyLocationCount );

        bool AddAllIndexKeys( const string &pkKey, RowPos rowPos, size_t existedPrimaryKeyLocationCount );
        bool CreateKeyIndex( const vector< int > &keyColIds, AriesTableKeysSPtr &index, bool checkDuplicate );
        bool HasPrimaryKey() const;

    private:
        string m_dbName;
        string m_tableName;
        schema::TableEntrySPtr m_tableEntry;
        TupleParserSPtr m_tupleParseSPtr;
        size_t m_perDeltaTableBlockCapacity;
        AriesDeltaTableSPtr m_deltaTableSPtr; //delta table: 行存储数据区
        AriesInitialTableSPtr m_InitialTableSPtr; //initial table: 列存储数据区
        TableId m_tableId;
        // lock the whole table, initial table and delta table,
        // used when merging delta table to initial table
        mutex m_tableLock;

        AriesPrimaryKeyInfo m_primaryKey;
        bool m_allIndexKeysChecked;
        atomic<bool> m_createKeyIndexInprogress;
        condition_variable m_statusCond;
        // m_indexKeyMap: colId <---> IndexKeyId
        // IndexKeyId:
        // primaryKey : PRIMARY_KEY_INDEX (-1)
        // foreignKeys: 0 ~ n
        map< int, int > m_colIdIndexKeyIdMap;
        mutex m_lock4CreateKeyIndex;
    };

    using AriesMvccTableSPtr = shared_ptr<AriesMvccTable>;

END_ARIES_ENGINE_NAMESPACE
