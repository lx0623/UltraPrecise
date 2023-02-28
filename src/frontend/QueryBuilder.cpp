#include "QueryBuilder.h"

#include "SelectStructure.h"

namespace aries {

QueryBuilder::QueryBuilder() {
}

void QueryBuilder::InitDatabaseSchema(DatabaseSchemaPointer arg_schema) {

    this->schema = arg_schema;

}


void QueryBuilder::BuildQuery(AbstractQueryPointer arg_query) {

    assert(arg_query != nullptr);


    /*1: setup agent*/
    SchemaAgentPointer schema_agent = nullptr;
    if (nullptr != schema) {
        schema_agent = std::make_shared<SchemaAgent>();
        schema_agent->SetDatabaseSchema(schema);
    }


    /*2: setup nodebuilder*/
    SQLTreeNodeBuilderPointer node_builder = std::make_shared<SQLTreeNodeBuilder>(arg_query);


    /*3: setup context*/
    QueryContextPointer query_context = std::make_shared<QueryContext>(QueryContextType::TheTopQuery,
                                                                       0,
                                                                       arg_query, //query
                                                                       nullptr, //parent
                                                                       nullptr //possible expr
    );


    SelectStructurePointer ssp = std::dynamic_pointer_cast<SelectStructure>(arg_query);
    ssp->CheckQueryGate(schema_agent, node_builder, query_context);

}


}
