#include "NodeColumnStructure.h"

namespace aries {


NodeColumnStructure::NodeColumnStructure( std::string arg_name, ColumnValueType arg_type, int arg_length, bool arg_nullable ) 
{
    assert( !arg_name.empty());

    this->name = arg_name;
    this->type = arg_type;
    this->length = arg_length;
    this->nullable = arg_nullable;
}

std::string NodeColumnStructure::GetName() {

    return this->name;

}

ColumnValueType NodeColumnStructure::GetValueType() {
    return this->type;
}

bool NodeColumnStructure::IsNullable()
{
    return this->nullable;
}

std::string NodeColumnStructure::ToString() {

    return std::string("Node Column:(" + this->name + ", "
                       + std::to_string((int) this->type) + ", "
                       + std::to_string(this->length) + ")");

}


ColumnStructurePointer NodeColumnStructure::GetPossibleRoot() {
    return this->possible_root;

}

void NodeColumnStructure::SetPossibleRoot(ColumnStructurePointer arg_root) {
    this->possible_root = arg_root;
}

std::shared_ptr<NodeColumnStructure> NodeColumnStructure::CloneAsNullable()
{
    NodeColumnStructurePointer ret = std::make_shared<NodeColumnStructure>(
            this->name,
            this->type,
            this->length,
            true );
    ret->possible_root = this->possible_root;
    return ret;
}

std::shared_ptr<NodeColumnStructure>
NodeColumnStructure::__createInstanceFromColumnStructure(ColumnStructurePointer arg_column_structure) {

    NodeColumnStructurePointer ret = std::make_shared<NodeColumnStructure>(
            arg_column_structure->GetName(),
            arg_column_structure->GetValueType(),
            arg_column_structure->GetLength(),
            arg_column_structure->IsNullable());

    ret->SetPossibleRoot(arg_column_structure);

    return ret;

}
}
