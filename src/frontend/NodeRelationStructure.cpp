#include "NodeRelationStructure.h"

namespace aries {
NodeRelationStructure::NodeRelationStructure() {
}

void NodeRelationStructure::AddColumn(NodeColumnStructurePointer arg_column, std::string arg_table_name) {
    this->columns.push_back(arg_column);
    this->tables_where_column_belong.push_back(arg_table_name);

    this->column_count += 1;
}


size_t NodeRelationStructure::GetColumnCount() {
    return this->column_count;
}


NodeColumnStructurePointer NodeRelationStructure::GetColumnbyIndex(size_t arg_index) {
    assert(arg_index >= 0 && arg_index < this->columns.size());

    return this->columns[arg_index];
}


std::string NodeRelationStructure::GetColumnSourceTableNamebyIndex(size_t arg_index) {

//	std::cout << arg_index << "\tNodeRelationStructure::GetColumnSourceTableNamebyIndex: \n";

    assert(arg_index < this->tables_where_column_belong.size());


    return this->tables_where_column_belong[arg_index];
}


std::string NodeRelationStructure::ToString() {
    std::string ret = " [";

    for (size_t i = 0; i < this->GetColumnCount(); i++) {
        std::string tmp = " !";
        tmp += "(";
        tmp += std::to_string(i);
        tmp += ") ";
        tmp += this->GetColumnSourceTableNamebyIndex(i);
        tmp += " , ";
        tmp += this->GetColumnbyIndex(i)->GetName();
        tmp += "! ";

        ret += tmp;
        ret += ";";
    }

    ret += "] ";

    return ret;
}

}
