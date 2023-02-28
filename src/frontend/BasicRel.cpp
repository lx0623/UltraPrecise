#include "utils/string_util.h"
#include "BasicRel.h"
#include "CommonBiaodashi.h"
#include "SelectStructure.h"

namespace aries {
BasicRel::BasicRel(bool arg_issubquery, std::string arg_id, std::shared_ptr<std::string> arg_alias,
                   AbstractQueryPointer arg_subquery) {
    if (arg_issubquery) {
        assert(arg_alias != nullptr && arg_subquery != nullptr);
    }

    this->isSubquery = arg_issubquery;
    this->rel_id = arg_id;
    this->rel_alias_name = arg_alias;
    this->subquery = arg_subquery;
}

AbstractQueryPointer BasicRel::GetSubQuery() {
    return this->subquery;
}

void BasicRel::SetRelationStructure(RelationStructurePointer arg_rel) {
    this->relation_structure = arg_rel;
}

RelationStructurePointer BasicRel::GetRelationStructure() {
    return this->relation_structure;
}

void BasicRel::SetUnderlyingTable(PhysicalTablePointer arg_table) {
    this->underlying_table = arg_table;
}

PhysicalTablePointer BasicRel::GetPhysicalTable() {
    return this->underlying_table;
}

bool BasicRel::IsSubquery() {
    return this->isSubquery;
}

std::string BasicRel::GetID() {
    return this->rel_id;
}

std::shared_ptr<std::string> BasicRel::GetAliasNamePointer() {
    return this->rel_alias_name;

}

std::string BasicRel::ToString() {
    std::string result = "";
    if (this->isSubquery) {
        assert(this->subquery != nullptr);
        result += "(";
        //result += this->subquery->ToString();
        result += " %SUBQUERY% ";
        result += ")";
        result += " AS ";
        result += *(this->rel_alias_name);
    } else {
        result += this->rel_id;
        if (this->rel_alias_name != nullptr) {
            result += " AS ";
            result += *(this->rel_alias_name);
        }
    }

    return result;
}


std::vector<BiaodashiPointer> BasicRel::GetAllColumnsAsExpr() {
    std::vector<BiaodashiPointer> ret;

    std::string rel_name = this->relation_structure->GetName();
    for (size_t i = 0; i < this->relation_structure->GetColumnCount(); i++) {
        std::string colname = this->relation_structure->GetColumn(i)->GetName();
        // std::string expr_name = rel_name + "." + colname;
        //std::cout << "GetAllColumnsAsExpr: " <<expr_name << "\n";
        auto ident = std::make_shared<SQLIdent>("", rel_name, colname);
        auto expr = std::make_shared<CommonBiaodashi>(BiaodashiType::Biaoshifu, ident);
        expr->SetOrigName(colname, "");
        ret.push_back(expr);
    }

    return ret;
}


std::shared_ptr<SQLTreeNode> BasicRel::GetMyRelNode() {
    return this->rel_node;
}

void BasicRel::SetMyRelNode(std::shared_ptr<SQLTreeNode> arg_rel_node) {
    this->rel_node = arg_rel_node;
}

bool BasicRel::__compareTwoBasicRels(BasicRel *first, BasicRel *second) {
    assert(first != NULL && second != NULL);

    if (first->GetID() != second->GetID())
        return false;

    if (first->IsSubquery() != second->IsSubquery())
        return false;

    std::shared_ptr<std::string> first_alias = first->GetAliasNamePointer();
    std::shared_ptr<std::string> second_alias = second->GetAliasNamePointer();

    if (first_alias == nullptr && second_alias == nullptr)
        return true;

    if (first_alias == nullptr && second_alias != nullptr)
        return false;

    if (first_alias != nullptr && second_alias == nullptr)
        return false;

    if (*first_alias != *second_alias)
        return false;

    return true;

}


std::string BasicRel::GetMyOutputName() {
    std::string ret;

    if (this->rel_alias_name != nullptr) {
        ret = *(this->rel_alias_name);
    } else {
        ret = this->rel_id;
    }

    return ret;

}


void BasicRel::ResetAlias(std::string arg_alias) {
    this->rel_alias_name = std::make_shared<std::string>(aries_utils::to_lower(arg_alias));
}

void BasicRel::ResetToSubQuery( AbstractQueryPointer query )
{
    subquery = query;
    isSubquery = true;
}

void BasicRel::SetColumnsAlias( const std::vector< ColumnStructurePointer >& alias )
{
    columns_alias.assign( alias.cbegin(), alias.cend() );
}

const std::vector< ColumnStructurePointer >& BasicRel::GetColumnsAlias() const
{
    return columns_alias;
}

bool BasicRel::IsExpression()
{
    return isExpression;
}

void BasicRel::SetIsExpression( bool value )
{
    isExpression = value;
}

}
