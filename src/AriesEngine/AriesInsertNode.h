#pragma once

#include "AriesOpNode.h"
#include "AriesCommonExpr.h"
#include "AriesCalcTreeGenerator.h"
#include "transaction/AriesTransaction.h"
#include "transaction/AriesMvccTable.h"

BEGIN_ARIES_ENGINE_NAMESPACE

struct ConcurrentInsertResult
{
    bool m_success = true;
    vector< RowPos > m_rowPoses;
};
using ConcurrentInsertResultSPtr = shared_ptr< ConcurrentInsertResult >;

class AriesInsertNode : public AriesOpNode {
public:
    AriesInsertNode( const AriesTransactionPtr& transaction, const std::string& dbName, const std::string& tableName );
    ~AriesInsertNode();

    void SetCuModule( const vector< CUmoduleSPtr >& modules )
    {
        m_dataSource->SetCuModule( modules );
    }

    std::string GetCudaKernelCode() const
    {
        return m_dataSource->GetCudaKernelCode();
    }

    void SetColumnIds( const std::vector< int >& ids );

    virtual bool Open() override final;
    virtual AriesOpResult GetNext() override final;
    virtual void Close() override final;
    JSON GetProfile() const override final;
private:
    AriesDataBufferSPtr GetColumnDefaultValueBuffer( int colIdx, size_t itemCount ) const;
    // ConcurrentInsertResultSPtr AddTuples( const TupleDataSPtr tupleData,
    //                                       size_t startRowIdx,
    //                                       size_t jobCount );

private:

    AriesTransactionPtr m_transaction;
    std::string m_dbName;
    std::string m_tableName;
    schema::DatabaseEntrySPtr m_dbEntry;
    schema::TableEntrySPtr m_tableEntry;
    std::vector< int > m_columnIds;
    AriesMvccTableSPtr m_mvccTable;    
};

END_ARIES_ENGINE_NAMESPACE
