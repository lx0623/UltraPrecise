#ifndef ARIES_ABSTARCT_QUERY_ENGINE
#define ARIES_ABSTARCT_QUERY_ENGINE

#include <string>

#include "SQLTreeNode.h"

#include "../AriesEngineWrapper/AbstractMemTable.h"

namespace aries_engine {
  class AriesTransaction;
  using AriesTransactionPtr = shared_ptr<AriesTransaction>;
}

namespace aries {
using VALUES = shared_ptr< vector< BiaodashiPointer > >;
using VALUES_LIST = vector< VALUES >;

class AbstractQueryEngine {
public:
    virtual ~AbstractQueryEngine() = default;

    virtual std::string ToString() = 0;

//	virtual void ExecuteQueryTree(SQLTreeNodePointer arg_query_tree) = 0;
    virtual AbstractMemTablePointer ExecuteQueryTree(const aries_engine::AriesTransactionPtr& tx, SQLTreeNodePointer arg_query_tree, const std::string& dbName) = 0;
    virtual AbstractMemTablePointer ExecuteUpdateTree(const aries_engine::AriesTransactionPtr& tx,
                                                      SQLTreeNodePointer arg_query_tree,
                                                      const string& targetDbName,
                                                      const std::string& dbName) = 0;
    virtual AbstractMemTablePointer ExecuteInsert( const aries_engine::AriesTransactionPtr& tx,
                                                   const string& dbName,
                                                   const string& tableName,
                                                   vector< int >& insertColumnIds,
                                                   VALUES_LIST& insertValuesList,
                                                   vector< BiaodashiPointer >& optUpdateColumnExprs,
                                                   vector< BiaodashiPointer >& optUpdateValueExprs,
                                                   SQLTreeNodePointer queryPlanTree ) = 0;
    virtual AbstractMemTablePointer ExecuteDeleteTree( const aries_engine::AriesTransactionPtr& tx,
                                                       const string& dbName,
                                                       const string& tableName,
                                                       SQLTreeNodePointer arg_query_tree,
                                                       const string& defaultDbName ) = 0;
};

typedef std::shared_ptr<AbstractQueryEngine> AbstractQueryEnginePointer;

}//namespace


#endif
