#include "SchemaBuilder.h"

#include <boost/algorithm/string.hpp>

using namespace aries::schema;

namespace aries {

DatabaseSchemaPointer SchemaBuilder::convertDatabase(aries::schema::DatabaseEntry *database, bool needRowIdColumn) {
    auto legacy_database = std::make_shared<DatabaseSchema>(database->GetName());

    auto name_of_tables = database->GetNameListOfTables();

    for (size_t i = 0; i < name_of_tables.size(); i++) {
        auto table_name = name_of_tables[i];
        auto table = database->GetTableByName(table_name);

        PhysicalTablePointer legacy_table = convertTable(table.get());

        legacy_database->AddPhysicalTable(legacy_table);

        int columns_count = table->GetColumnsCount();
        for (int id = 1; id <= columns_count; id++) {
            auto column = table->GetColumnById(id);
            auto legacy_column = ConvertColumn(column.get());

            legacy_table->AddColumn(legacy_column);
        }

        if ( needRowIdColumn )
        {
            auto legacy_column = std::make_shared<ColumnStructure>(DBEntry::ROWID_COLUMN_NAME, ColumnType::INT, 1, false, true);
            legacy_table->AddColumn(legacy_column);
        }

        legacy_table->SetConstraints( table->GetConstraints() );
    }

    return legacy_database;
}

PhysicalTablePointer SchemaBuilder::convertTable(aries::schema::TableEntry *table) {
    auto legacy_table = std::make_shared<PhysicalTable>(table->GetName());
    return legacy_table;
}

ColumnStructurePointer SchemaBuilder::ConvertColumn(aries::schema::ColumnEntry *column) {

    auto legacy_column = std::make_shared<ColumnStructure>(column->GetName(),
                                                           column->GetType(),
                                                           column->GetLength(),
                                                           column->IsAllowNull(),
                                                           column->IsPrimary());
    legacy_column->SetPresision(column->numeric_precision);
    legacy_column->SetScale(column->numeric_scale);
    if (column->IsForeignKey()) {
        legacy_column->SetIsFk(true);
        legacy_column->SetFkStr(column->GetReferenceDesc());
    }
    legacy_column->SetEncodeType( column->encode_type );
    legacy_column->SetEncodedIndexType( column->GetDictIndexDataType() );
    return legacy_column;
}

void SchemaBuilder::WriteStringIntoFile(std::string file_path, std::string content) {
    std::ofstream target_file;
    target_file.open(file_path);
    target_file << content;
    target_file.close();

}

DatabaseSchemaPointer SchemaBuilder::BuildFromDatabase(aries::schema::DatabaseEntry *database, bool needRowIdColumn) {
    return convertDatabase(database, needRowIdColumn);
}

}

