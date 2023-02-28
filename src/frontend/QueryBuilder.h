#ifndef ARIES_H_QUERY_BUILDER
#define ARIES_H_QUERY_BUILDER


#include "AbstractQuery.h"
#include "DatabaseSchema.h"

namespace aries {

class QueryBuilder {


private:

    QueryBuilder(const QueryBuilder &arg);

    QueryBuilder &operator=(const QueryBuilder &arg);


    DatabaseSchemaPointer schema;

public:
    QueryBuilder();

    void InitDatabaseSchema(DatabaseSchemaPointer arg_schema);

    void BuildQuery(AbstractQueryPointer arg_query);

};

typedef std::shared_ptr<QueryBuilder> QueryBuilderPointer;

};


#endif
