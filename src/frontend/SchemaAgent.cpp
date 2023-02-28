#include <glog/logging.h>

#include "SchemaAgent.h"

namespace aries {


SchemaAgent::SchemaAgent() {
}

void SchemaAgent::SetDatabaseSchema(DatabaseSchemaPointer arg_schema) {
    this->schema = arg_schema;
}

PhysicalTablePointer SchemaAgent::FindPhysicalTable(std::string arg_table_name) {
    LOG(INFO) << "SchemaAgent::FindPhysicalTable" << arg_table_name << "\n";

    assert(schema != nullptr);
    return schema->FindPhysicalTable(arg_table_name);
}

}


