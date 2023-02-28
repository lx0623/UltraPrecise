#ifndef ARIES_EXECUTOR_PORTAL_H
#define ARIES_EXECUTOR_PORTAL_H

#include <string>
#include <memory>

#include "SQLTreeNode.h"

#include "AbstractQueryEngine.h"

namespace aries {


class ExecutorPortal {

private:

    ExecutorPortal(const ExecutorPortal &arg);

    ExecutorPortal &operator=(const ExecutorPortal &arg);

    AbstractQueryEnginePointer the_engine;

public:

    ExecutorPortal( AbstractQueryEnginePointer arg_engine );

    std::string ToString();

//	void ExecuteQuery(SQLTreeNodePointer arg_query_tree);
    AbstractMemTablePointer ExecuteQuery(const aries_engine::AriesTransactionPtr& tx, SQLTreeNodePointer arg_query_tree, const std::string& dbName);
    AbstractMemTablePointer ExecuteUpdate( const aries_engine::AriesTransactionPtr& tx,
                                           SQLTreeNodePointer arg_query_tree,
                                           const string& targetDbName,
                                           const std::string& defaultDbName );
    AbstractMemTablePointer ExecuteInsert( const aries_engine::AriesTransactionPtr& tx,
                                           const string& dbName,
                                           const string& tableName,
                                           vector< int >& insertColumnIds,
                                           VALUES_LIST& insertValuesList,
                                           vector< BiaodashiPointer >& optUpdateColumns,
                                           vector< BiaodashiPointer >& optUpdateValues,
                                           SQLTreeNodePointer queryPlanTree );
    AbstractMemTablePointer ExecuteDelete( const aries_engine::AriesTransactionPtr& tx,
                                           const string& dbName,
                                           const string& tableName,
                                           SQLTreeNodePointer arg_query_tree,
                                           const string& defaultDbName );
};
using ExecutorPortalSPtr = shared_ptr< ExecutorPortal >;

}//namespace




#endif
