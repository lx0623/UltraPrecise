#pragma once
#include "AriesOpNode.h"
#include "AriesEngine/transaction/AriesTransaction.h"
#include "AriesEngine/transaction/AriesMvccTable.h"

BEGIN_ARIES_ENGINE_NAMESPACE
class AriesDeleteNode : public AriesOpNode {
public:
    AriesDeleteNode( const AriesTransactionPtr& transaction,
                     const string& dbName,
                     const string& tableName );
    ~AriesDeleteNode();

    void SetCuModule( const vector< CUmoduleSPtr >& modules );

    string GetCudaKernelCode() const;

    void SetColumnId4RowPos( int columnId );

    bool Open() override final;
    AriesOpResult GetNext() override final;
    void Close() override final;
    JSON GetProfile() const override final;

private:
    bool internalDeleteFirstWriterWin( RowPos pos );

private:
    AriesTransactionPtr m_transaction;
    std::string m_dbName;
    std::string m_tableName;
    AriesMvccTableSPtr m_mvccTable;
    int m_ColumnId4RowPos;
};

using AriesDeleteNodeSPtr = shared_ptr< AriesDeleteNode >;
END_ARIES_ENGINE_NAMESPACE
