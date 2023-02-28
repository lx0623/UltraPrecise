#ifndef ARIES_DATABASE_SCHEMA
#define ARIES_DATABASE_SCHEMA

#include <iostream>
#include <fstream>
#include <vector>

#include "TroubleHandler.h"
#include "VariousEnum.h"

#include "ColumnStructure.h"
#include "RelationStructure.h"
#include "PhysicalTable.h"


namespace aries {

class DatabaseSchema {

private:

    std::vector<PhysicalTablePointer> tables;
    std::map<std::string, PhysicalTablePointer> name_table_map;

    std::string schema_name;

    DatabaseSchema(const DatabaseSchema &arg);

    DatabaseSchema &operator=(const DatabaseSchema &arg);


public:

    DatabaseSchema(std::string schema_name);

    std::string GetName();

    bool AddPhysicalTable(PhysicalTablePointer arg_table);

    PhysicalTablePointer GetPhysicalTable(int arg_index);

    PhysicalTablePointer FindPhysicalTable(std::string arg_table_name);


};

//typedef std::unique_ptr<DatabaseSchema> DatabaseSchemaUniquePointer;
typedef std::shared_ptr<DatabaseSchema> DatabaseSchemaPointer;

}

#endif
