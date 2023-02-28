#include "ColumnShell.h"
#include "BasicRel.h"
//using namespace aries;

namespace aries {
ColumnShell::ColumnShell(std::string arg_table_name, std::string arg_column_name) {
    this->table_name = arg_table_name;
    this->column_name = arg_column_name;
}

/*a str could be "table_name.column_name or just column_name"*/
// ColumnShell::ColumnShell(std::string arg_str) {
//
//     std::vector<std::string> av = ColumnShell::SplitString(arg_str, '.');
//     assert(av.size() == 1 || av.size() == 2);
//     if (av.size() == 1) {
//         this->column_name = av[0];
//     } else {
//         this->table_name = av[0];
//         this->column_name = av[1];
//     }
//
//     //std::cout << "created a columnshell: " + this->ToString() + "\n";
//
//
// }

void ColumnShell::SetInGroupList(bool arg_in_group_list) {
    this->in_group_list = arg_in_group_list;
}

bool ColumnShell::GetInGroupList() {
    return this->in_group_list;
}

int ColumnShell::GetAbsoluteLevel() {
    return this->absolute_level;
}

void ColumnShell::SetAbsoluteLevel(int arg_level) {
    this->absolute_level = arg_level;
}


BiaodashiValueType ColumnShell::GetValueType() {
    if (this->is_placeholder) {
        return this->my_own_value_type;
    }
    assert(this->column_structure != nullptr);
    return this->column_structure->GetValueType();
}

void ColumnShell::SetMyOwnValueType(BiaodashiValueType arg_value) {
    this->my_own_value_type = arg_value;
}

int ColumnShell::GetLength() {
    return this->table.lock()->GetRelationStructure()->FindColumn(this->column_name)->GetLength();
}

void ColumnShell::SetTableName(std::string arg_table_name) {
    this->table_name = arg_table_name;
}

void ColumnShell::SetTable(std::shared_ptr<BasicRel> arg_table) {
    this->table = arg_table;
}

std::shared_ptr<BasicRel> ColumnShell::GetTable() {
    return this->table.lock();
}

void ColumnShell::SetLocationInTable(int arg_location) {
    this->location_in_table = arg_location;
}

int ColumnShell::GetLocationInTable() const
{
    return this->location_in_table;
}

void ColumnShell::SetColumnStructure(ColumnStructurePointer arg_column_structure) {
    this->column_structure = arg_column_structure;
    if (arg_column_structure) {
        numeric_precision = arg_column_structure->GetPrecision();
        numeric_scale = arg_column_structure->GetScale();

        SetIsPrimaryKey( arg_column_structure->GetIsPrimary() );
        SetIsUnique( arg_column_structure->GetIsPrimary() );

        assert( IsPrimaryKey() == arg_column_structure->GetIsPrimary() );
        assert( IsUnique() == arg_column_structure->GetIsPrimary() );
    }
}

ColumnStructurePointer ColumnShell::GetColumnStructure() {
    return this->column_structure;
}

std::string ColumnShell::GetTableName() {
    return this->table_name;
}

std::string ColumnShell::GetColumnName() {
    return this->column_name;
}

// void ColumnShell::SetQueryLevel(int arg_level) {
//     this->query_level = arg_level;
// }

std::vector<std::string> ColumnShell::SplitString(std::string input_str, char delimitor) {

    std::istringstream input(input_str);
    std::string tokens;

    std::vector<std::string> vs;

    while (std::getline(input, tokens, delimitor)) {
        vs.push_back(tokens);
    }
    return vs;
}

std::string ColumnShell::ToString() {
    return this->table_name + "." + this->column_name;
}

void ColumnShell::SetAliasExprIndex(int arg_index) {
    this->alias_expr_index = arg_index;
}

int ColumnShell::GetAliasExprIndex() {
    return this->alias_expr_index;
}

void ColumnShell::SetExpr4Alias(BiaodashiPointer arg_bp) {
    this->expr_4_alias = arg_bp;
}

BiaodashiPointer ColumnShell::GetExpr4Alias() {
    return this->expr_4_alias;
}


//    int ColumnShell::GetIndexTableSource()
//    {
//	return this->index_table_source;
//    }
//    
//    void ColumnShell::SetIndexTableSource(int arg_value)
//    {
//	this->index_table_source = arg_value;
//	
//    }
//
//    int ColumnShell::GetIndexColumnSource()
//    {
//	return this->index_column_source;
//    }
//    
//    void ColumnShell::SetIndexColumnSource(int arg_value)
//    {
//	this->index_column_source = arg_value;
//    }


void ColumnShell::SetPositionInChildTables(int arg_value) {
    this->position_in_child_tables = arg_value;
}

int ColumnShell::GetPositionInChildTables() {
    return this->position_in_child_tables;
}


void ColumnShell::SetPlaceholderMark(bool arg_value) {
    this->is_placeholder = arg_value;
}

bool ColumnShell::GetPlaceholderMark() {
    return this->is_placeholder;
}

int ColumnShell::GetPrecision() {
    return numeric_precision;
}

int ColumnShell::GetScale() {
    return numeric_scale;
}

void ColumnShell::SetPresision(int precision) {
    numeric_precision = precision;
}

void ColumnShell::SetScale(int scale) {
    numeric_scale = scale;
}

void ColumnShell::SetIsPrimaryKey( bool is_primary )
{
    if ( is_primary )
    {
        key_flags |= PRIMARY_KEY_FLAG_MASK;
    }
    else
    {
        key_flags &= ~PRIMARY_KEY_FLAG_MASK;
    }
}

void ColumnShell::SetIsUnique( bool is_unique)
{
    if ( is_unique )
    {
        key_flags |= UNIQUE_FLAG_MASK;
    }
    else
    {
        key_flags &= ~UNIQUE_FLAG_MASK;
    }
}

bool ColumnShell::IsPrimaryKey() const
{
    return ( key_flags & PRIMARY_KEY_FLAG_MASK ) != 0;
}

bool ColumnShell::IsUnique() const
{
    return ( key_flags & UNIQUE_FLAG_MASK ) != 0;
}

}
