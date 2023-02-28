#ifndef ARIES_SCHEMA_AGENT
#define ARIES_SCHEMA_AGENT

#include <iostream>
#include <fstream>
#include <vector>

#include "TroubleHandler.h"
#include "VariousEnum.h"

#include "ColumnStructure.h"
#include "RelationStructure.h"
#include "PhysicalTable.h"
#include "DatabaseSchema.h"

namespace aries {

class SchemaAgent {
public:
    DatabaseSchemaPointer schema;

    SchemaAgent();

    void SetDatabaseSchema(DatabaseSchemaPointer arg_schema);

    PhysicalTablePointer FindPhysicalTable(std::string arg_table_name);
};

typedef std::shared_ptr<SchemaAgent> SchemaAgentPointer;

}

#endif
