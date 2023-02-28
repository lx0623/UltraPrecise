#ifndef ARIES_PHYSICAL_TABLE
#define ARIES_PHYSICAL_TABLE

#include <iostream>
#include <fstream>
#include <vector>

#include "TroubleHandler.h"
#include "VariousEnum.h"

#include "ColumnStructure.h"
#include "RelationStructure.h"


namespace aries {

class PhysicalTable {
private:

    RelationStructurePointer relation_definition = nullptr;

    std::string table_name;

    PhysicalTable(const PhysicalTable &arg);

    PhysicalTable &operator=(const PhysicalTable &arg);


public:

    PhysicalTable(std::string arg_name);

    std::string GetName();

    RelationStructurePointer GetRelationStructure();

    bool AddColumn(ColumnStructurePointer arg_column_p);

//    bool AddColumnDirectly(std::string arg_name, ColumnValueType arg_type, int arg_length, bool arg_allow_null,
//                           bool arg_is_primary);
//
//    bool AddColumnDirectly_2(std::string arg_name, ColumnValueType arg_type, int arg_length, bool arg_allow_null,
//                             bool arg_is_primary, std::string arg_fk);

    void SetConstraints( const std::map< std::string, schema::TableConstraintSPtr >& constraints );
    // ColumnStructurePointer FindColumn(std::string arg_column_name);

    std::string ToString();


};

typedef std::shared_ptr<PhysicalTable> PhysicalTablePointer;


}

#endif
