#ifndef ARIES_HEADER_2018_SCHEMA_BUILDER
#define ARIES_HEADER_2018_SCHEMA_BUILDER

#include <iostream>
#include <fstream>
#include <vector>


#include <boost/algorithm/string.hpp>

#include "TroubleHandler.h"
#include "VariousEnum.h"

#include "ColumnStructure.h"
#include "RelationStructure.h"
#include "PhysicalTable.h"
#include "DatabaseSchema.h"
#include "ColumnDescription.h"

#include "../schema/Schema.h"

namespace aries {

class SchemaBuilder {

private:
    SchemaBuilder();

    SchemaBuilder(const SchemaBuilder &arg);

    SchemaBuilder &operator=(const SchemaBuilder &arg);

    static DatabaseSchemaPointer convertDatabase(aries::schema::DatabaseEntry *database, bool needRowIdColumn);

    static PhysicalTablePointer convertTable(aries::schema::TableEntry *table);

public:
    static ColumnStructurePointer ConvertColumn(aries::schema::ColumnEntry *column);

public:

    static std::string ReadFileIntoString(std::string file_path);

    static void WriteStringIntoFile(std::string file_path, std::string content);

    static DatabaseSchemaPointer BuildFromDatabase(aries::schema::DatabaseEntry *database, bool needRowIdColumn = false);
};

}

#endif

