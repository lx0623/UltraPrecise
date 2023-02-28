#include "PhysicalTable.h"

namespace aries {
PhysicalTable::PhysicalTable(std::string arg_name) {
    assert(arg_name.empty() == false);

    this->table_name = arg_name;
    this->relation_definition = std::make_shared<RelationStructure>();
    this->relation_definition->SetName(this->table_name);
}

std::string PhysicalTable::GetName() {
    return this->table_name;
}


RelationStructurePointer PhysicalTable::GetRelationStructure() {
    return this->relation_definition;
}

bool PhysicalTable::AddColumn(ColumnStructurePointer arg_column_p) {
    return this->relation_definition->AddColumn(arg_column_p);
}

//bool
//PhysicalTable::AddColumnDirectly(std::string arg_name, ColumnValueType arg_type, int arg_length, bool arg_allow_null,
//                                 bool arg_is_primary) {
//
//    ColumnStructurePointer csp = std::make_shared<ColumnStructure>(arg_name, arg_type, arg_length, arg_allow_null,
//                                                                   arg_is_primary);
//
//    return this->AddColumn(csp);
//
//}
//
//bool
//PhysicalTable::AddColumnDirectly_2(std::string arg_name, ColumnValueType arg_type, int arg_length, bool arg_allow_null,
//                                   bool arg_is_primary, std::string arg_fk) {
//
//    ColumnStructurePointer csp = std::make_shared<ColumnStructure>(arg_name, arg_type, arg_length, arg_allow_null,
//                                                                   arg_is_primary);
//
//
//    if (arg_fk.empty() == false) {
//        csp->SetIsFk(true);
//        csp->SetFkStr(arg_fk);
//    }
//
//    return this->AddColumn(csp);
//
//}


// ColumnStructurePointer PhysicalTable::FindColumn(std::string arg_column_name) {
//
//     return this->relation_definition->FindColumn(arg_column_name);
// }

std::string PhysicalTable::ToString() {

//	std::cout << boost::stacktrace::stacktrace();

    return this->table_name + ":\n" + this->relation_definition->ToString();
}

void PhysicalTable::SetConstraints( const std::map< std::string, schema::TableConstraintSPtr >& constraints )
{
    relation_definition->SetConstraints( constraints );
}


}
