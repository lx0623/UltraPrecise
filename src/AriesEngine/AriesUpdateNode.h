#ifndef AIRES_ARIESUPDATENODE_H
#define AIRES_ARIESUPDATENODE_H

#include "AriesOpNode.h"
#include "transaction/AriesMvccTable.h"
#include "transaction/AriesTransaction.h"

BEGIN_ARIES_ENGINE_NAMESPACE

class AriesUpdateNode : public AriesOpNode
{
public:
    AriesUpdateNode( const AriesTransactionPtr& transaction, const string& dbName, const string& tableName );
    AriesUpdateNode( const AriesTransactionPtr& transaction) {}
    ~AriesUpdateNode();

    void SetUpdateColumnIds( const vector< int >& columnIds )
    {
        m_updateColumnIds.assign( columnIds.cbegin(), columnIds.cend() );
    }

    void SetColumnId4RowPos( const int colId )
    {
        m_ColumnId4RowPos = colId;
    }

    void SetCuModule( const vector< CUmoduleSPtr >& modules );
    string GetCudaKernelCode() const;

    bool Open() override final;
    AriesOpResult GetNext() override final;
    void Close() override final;
    JSON GetProfile() const override final;

private:
    bool UpdateTuple(const RowPos &oldPos, const TupleDataSPtr dataBuffer, int dataIndex);
    bool UpdateTupleFirstWriterWin(const RowPos &oldPos, const TupleDataSPtr dataBuffer, int dataIndex);
    TupleDataSPtr Convert2TupleDataBuffer(AriesTableBlockUPtr &tableBlock,
                                          unordered_map< int32_t, AriesManagedIndicesArraySPtr >& dictTupleData,
                                          vector< AriesDataBufferSPtr >& origDictStringColumns );

private:
    AriesTransactionPtr m_transaction;
    string m_targetDbName;
    string m_targetTableName;
    schema::DatabaseEntrySPtr m_dbEntry;
    schema::TableEntrySPtr m_tableEntry;
    /*columnIds should be updated*/
    vector< int > m_updateColumnIds;
    AriesMvccTableSPtr m_mvccTable;
    int m_ColumnId4RowPos;
    bool m_aborted;
};

using AriesUpdateNodeSPtr = shared_ptr< AriesUpdateNode >;

END_ARIES_ENGINE_NAMESPACE
#endif


