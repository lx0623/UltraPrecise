#include "SQLTreeNode.h"
#include "SelectStructure.h"
#include "ShowUtility.h"
#include "schema/SchemaManager.h"
#include "BiaodashiAuxProcessor.h"
#include <algorithm>


#include "boost/algorithm/string.hpp"

namespace aries {

SQLTreeNode::SQLTreeNode(SQLTreeNodeType arg_type) {
    this->type = arg_type;
}

void SQLTreeNode::ClearChildren()
{
    children.clear();
    involved_table_list.clear();
}


SQLTreeNodeType SQLTreeNode::GetType() {
    return this->type;
}

void SQLTreeNode::SetMyQuery(AbstractQueryPointer arg_query) {
    this->query = arg_query;
}

AbstractQueryPointer SQLTreeNode::GetMyQuery() {
    return this->query.lock();
}

void SQLTreeNode::SetMySelectPartStructure( const SelectPartStructurePointer& structure )
{
    select_part_structure = structure;
}

SelectPartStructurePointer SQLTreeNode::GetMySelectPartStructure() {
    if ( !select_part_structure )
    {
        select_part_structure = ((SelectStructure *) ((this->query).lock().get()))->GetSelectPart();
    }
    return select_part_structure;
}

void SQLTreeNode::SetMyGroupbyStructure( const GroupbyStructurePointer& structure )
{
    groupby_structure = structure;
}

GroupbyStructurePointer SQLTreeNode::GetMyGroupbyStructure() {
    if ( !groupby_structure )
    {
        groupby_structure = ((SelectStructure *) ((this->query.lock()).get()))->GetGroupbyPart();
    }
    return groupby_structure;
}

OrderbyStructurePointer SQLTreeNode::GetMyOrderbyStructure() {
    if ( !orderby_structure )
    {
        orderby_structure = ((SelectStructure *) ((this->query.lock()).get()))->GetOrderbyPart();
    }
    return orderby_structure;
}

void SQLTreeNode::SetMyOrderbyStructure( const OrderbyStructurePointer& structure )
{
    orderby_structure = structure;
}

void SQLTreeNode::SetParent( const SQLTreeNodePointer& arg_parent ) {
    this->parent = arg_parent;
}

SQLTreeNodePointer SQLTreeNode::GetParent() {
    return parent.lock();
}

SQLTreeNodePointer SQLTreeNode::GetTheChild() {
    if (this->children.size() == 0) {
        return nullptr;
    }
    return this->children[0];
}

void SQLTreeNode::SetTheChild(SQLTreeNodePointer arg_child) {
    this->AddChild(arg_child);
}

/*this is only for predictpushdown! since we need to modify the involved_table_list*/
void SQLTreeNode::ResetTheChild(SQLTreeNodePointer arg_child) {
    assert(this->children.size() == 1);
    this->children[0] = arg_child;
    /*no need to modify the involved_table_list*/
}


void SQLTreeNode::ReCalculateInvolvedTableList() {
    //todo: no for table node.

    if (type == SQLTreeNodeType::Column_NODE) {
        return;
    }

    //std::cout << "entering recalculateinvolvedtablelist ---";
    this->involved_table_list.clear();

    for (size_t ci = 0; ci < this->children.size(); ci++) {
        this->involved_table_list.insert(
                this->involved_table_list.end(),
                this->children[ci]->involved_table_list.begin(),
                this->children[ci]->involved_table_list.end());
    }

    //std::cout << this->GetType() << "\n";
    if (this->test_limit++ <= 10) {
        if ( this->GetParent() && this->GetType() != SQLTreeNodeType::Column_NODE) {
            this->GetParent()->ReCalculateInvolvedTableList();
        }
    }
    //std::cout << "leaving recalculateinvolvedtablelist ---";
}

void SQLTreeNode::CompletelyResetAChild(std::shared_ptr<SQLTreeNode> arg_child_old,
                                        std::shared_ptr<SQLTreeNode> arg_child_new) {
    for (size_t ci = 0; ci < this->children.size(); ci++) {
        if (this->children[ci] == arg_child_old) {
            this->children[ci] = arg_child_new;
        }
    }
    to_remove.push_back( arg_child_old );
    this->ReCalculateInvolvedTableList();

}


void SQLTreeNode::CompletelyResetAChild_WithoutReCalculateInvolvedTableList(std::shared_ptr<SQLTreeNode> arg_child_old,
                                                                            std::shared_ptr<SQLTreeNode> arg_child_new) {
    for (size_t ci = 0; ci < this->children.size(); ci++) {
        if (this->children[ci] == arg_child_old) {
            this->children[ci] = arg_child_new;
        }
    }

    //this->ReCalculateInvolvedTableList();

}


SQLTreeNodePointer SQLTreeNode::GetLeftChild() {
    assert(this->children.size() == 2);
    return this->children[0];

}

SQLTreeNodePointer SQLTreeNode::GetRightChild() {
    assert(this->children.size() == 2);
    return this->children[1];

}

void SQLTreeNode::SetLeftChild(SQLTreeNodePointer arg_child) {
    this->AddChild(arg_child);
}

/*this is only for predictpushdown! since we need to modify the involved_table_list*/
void SQLTreeNode::ResetLeftChild(SQLTreeNodePointer arg_child) {
    /*no need to reset involved_table_list since unchanged!*/
    assert(this->children.size() == 2);
    this->children[0] = arg_child;
}

void SQLTreeNode::SetRightChild(SQLTreeNodePointer arg_child) {
    this->AddChild(arg_child);
}

/*this is only for predictpushdown! since we need to modify the involved_table_list*/
void SQLTreeNode::ResetRightChild(SQLTreeNodePointer arg_child) {
    /*no need to reset involved_table_list since unchanged!*/
    assert(this->children.size() == 2);
    this->children[1] = arg_child;
}


size_t SQLTreeNode::GetChildCount() {
    return this->children.size();
}


void SQLTreeNode::AddChild(SQLTreeNodePointer arg_child) {
    assert(arg_child != nullptr);
    this->children.push_back(arg_child);



    //TroubleHandler::showMessage("ok we add child: " + std::to_string(arg_child->involved_table_list.size()) + " \n");

    /*if child is a columnnode of a subquery, then we don't merge child's list*/

    //std::cout << "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n";

    size_t num = arg_child->involved_table_list.size();
    if (num == 0) {
        return;
    }

    if (arg_child->GetType() == SQLTreeNodeType::Column_NODE) {
        this->involved_table_list.push_back(arg_child->involved_table_list[num - 1]);
    } else {

        this->involved_table_list.insert(
                this->involved_table_list.end(),
                arg_child->involved_table_list.begin(),
                arg_child->involved_table_list.end());

    }

    //TroubleHandler::showMessage("done\n");
    //TroubleHandler::showMessage(this->ToString(0));

}

SQLTreeNodePointer SQLTreeNode::GetChildByIndex(size_t arg_index) {
    assert(arg_index >= 0 && arg_index < this->children.size());
    return this->children[arg_index];

}


std::string SQLTreeNode::getMyNodeInfo() {
    std::string ret;

    SelectStructurePointer ssp = std::dynamic_pointer_cast<SelectStructure>(this->query.lock());
    assert(ssp != nullptr);
//  std::cout << "getMyNodeInfo() ... begin \n" << (int)this->type << "\n";

    switch (this->type) {
        case SQLTreeNodeType::Table_NODE:
            assert( !this->my_basic_rel.expired() );
            ret = "TableNode: ";
            if (this->tree_formed == true) {
                for (size_t i = 0; i < this->referenced_column_array.size(); i++) {
                    ret += " ( ";
                    ret += this->referenced_column_array[i]->ToString() + " ";
                    ret += std::to_string(this->GetPositionForReferencedColumn(this->referenced_column_array[i]));
                    ret += " >> ";
                    ret += std::to_string(i) + " ";
                    ret += ") ";
                }
            }
            break;

        case SQLTreeNodeType::Filter_NODE:
            assert(this->my_filter_structure != nullptr);
            ret = "FilterNode:[" + this->my_filter_structure->ToString() + "]";
            break;

        case SQLTreeNodeType::BinaryJoin_NODE:
            ret = "BinaryJoinNode<" + std::to_string(this->primary_foreign_join_form) + ":" +
                  (this->primary_table_intact ? "TRUE" : "FALSE") + ">:[" +
                  ShowUtility::GetInstance()->GetTextFromJoinType(this->my_join_type) + " --- " +
                  (my_join_condition != nullptr ? my_join_condition->ToString() : "") + 
                  (my_join_other_condition != nullptr ? ((my_join_condition == nullptr ? "" : " && ") + my_join_other_condition->ToString()): "") + "]";
            break;

        case SQLTreeNodeType::Group_NODE:
            ret = "GroupbyNode: [" + ssp->GetGroupbyPart()->ToString() + "]";
            break;

        case SQLTreeNodeType::Sort_NODE:
            ret = "SortNode: [" + ssp->GetOrderbyPart()->ToString() + "]";
            break;

        case SQLTreeNodeType::Column_NODE:
            ret = "ColumnNode";
            if (this->IsColumnNodeRemovable() == true) {
                ret += "(R)";
            }

            ret += ": [" + ssp->GetSelectPart()->ToString() + "]";

            break;

        case SQLTreeNodeType::SetOp_NODE:
            ret = "SetOperationNode: [";
            ret += ShowUtility::GetInstance()->GetTextFromSetOperationType(this->my_setoperation_type);
            ret += "]";

            break;
        
        case SQLTreeNodeType::Limit_NODE:
            ret = "LimitNode";
            ret += "( " + std::to_string(limit_offset) + ", " + std::to_string(limit_size) + " )";
            ret += ": [" + ssp->GetSelectPart()->ToString() + "]";
            break;

        case SQLTreeNodeType::SELFJOIN_NODE:
        {
            ret = "SelfJoinNode:[ join on column index: " + std::to_string( m_selfJoinColumnId - 1 ) + ", filter condition: ";
            ret += m_selfJoinMainFilter ? m_selfJoinMainFilter->ToString() : "null";

            for( const auto& joinInfo : m_selfJoinInfo )
            {
                ret += "---";
                ret += joinInfo.ToString();
            }

            ret += "]";
            break;
        }
        case SQLTreeNodeType::StarJoin_NODE:
            ret = "StarJoinNode: [";
            ret += "]";
            break;
        case SQLTreeNodeType::InnerJoin_NODE:
            ret = "InnerJoinNode: [";
            ret += "]";
            break;
        case SQLTreeNodeType::Exchange_NODE:
            ret = "ExchangeNode: []";
            break;
        default:
        {
            ARIES_ASSERT( 0 , "unexpected node type: " +  std::to_string( (int) (this->type) ) );
        }
    }

//  std::cout << "swtich over\n";

    // if (this->tree_formed == true) {
//         /*noderelationstructure*/
//         // std::string ncs_string = this->node_relation_structure->ToString();
//         // std::string lowercase_version = boost::algorithm::to_lower_copy(ncs_string);
//         // ret += "\t\t";
//         // ret += lowercase_version;

//         /*refereenced columns*/

//         ret += "\t[ ";


//         for (int i = 0; i < this->referenced_column_array.size(); i++) {
// //      //std::cout << "\t\t\t" << i << "\n";
//             ret += " (";
//             ret += std::to_string(this->GetPositionForReferencedColumn(this->referenced_column_array[i]));
//             ret += " >> ";
//             ret += std::to_string(i) + " ";
//             ret += this->referenced_column_array[i]->ToString();
//             ret += ") ";
//         }


//         ret += "]";

    // }


    return ret;

}


std::string SQLTreeNode::ToString(int arg_level) {
    bool trouble = false;
    std::string init_str;
    if (this->GetParent() == NULL) {
        //i am the leading node
        SelectStructure *ss = (SelectStructure *) ((this->GetMyQuery()).get());

        assert(ss != NULL);
        if (ss->GetInitQueryCount() > 0) {
            trouble = true;
            init_str = "\n\n\n\t\t\tINIT QUERY PLAN\n------------------------------------------------------";

            auto init_query = ss->GetInitQueryByIndex(0);
            init_str += init_query->GetQueryPlanTree()->ToString(0);
            init_str += "\n-----------------------\n";

        }
    }


    std::string ret_str = "\n";

    std::string my_node_info = this->getMyNodeInfo() + " [ ";

    for (size_t i_0 = 0; i_0 < this->involved_table_list.size(); i_0++) {
        my_node_info += this->involved_table_list[i_0].lock()->ToString();
        my_node_info += " ";
    }

    my_node_info += "]";

    std::string myprefix = "\n";
    for (int i_1 = 0; i_1 < arg_level; i_1++) {
        myprefix += "----";
    }

    ret_str += (myprefix + " " + my_node_info);


    for (size_t i_2 = 0; i_2 < this->children.size(); i_2++) {
        ret_str += this->children[i_2]->ToString(arg_level + 1);
    }

    if (trouble) {
        ret_str += init_str;
    }

    return ret_str;
}


void SQLTreeNode::SetBasicRel(BasicRelPointer arg_basic_rel) {

    assert(arg_basic_rel != nullptr);

    this->my_basic_rel = arg_basic_rel;

    /*if this is a column_node, then it must reset its involved_table_list*/

    if (this->type == SQLTreeNodeType::Column_NODE) {
        // this->backup_involved_table_list = this->involved_table_list;
        this->involved_table_list.clear();
    }

    this->involved_table_list.push_back(arg_basic_rel);

    //TroubleHandler::showMessage("this involved_table_list len = " + std::to_string(this->involved_table_list.size()));

}

BasicRelPointer SQLTreeNode::GetBasicRel() {
    return this->my_basic_rel.lock();
}

void SQLTreeNode::ResetType( SQLTreeNodeType arg_type )
{
    type = arg_type;
}


void SQLTreeNode::SetFilterStructure(BiaodashiPointer arg_expr) {
    this->my_filter_structure = arg_expr;
}

BiaodashiPointer SQLTreeNode::GetFilterStructure() {
    return this->my_filter_structure;
}

void SQLTreeNode::SetIsNotInFlag( bool is_not_in ) {
    this->is_a_not_in = is_not_in;
}

bool SQLTreeNode::GetIsNotInFlag() {
    return this->is_a_not_in;
}

void SQLTreeNode::SetJoinType(JoinType arg_join_type) {
    this->my_join_type = arg_join_type;
}

JoinType SQLTreeNode::GetJoinType() {
    return this->my_join_type;
}

void SQLTreeNode::SetJoinCondition(BiaodashiPointer arg_expr) {
    this->my_join_condition = arg_expr;
    join_condition_constraint_type = CheckJoinConditionConstraintType( arg_expr );
}

BiaodashiPointer SQLTreeNode::GetJoinCondition() {
    return this->my_join_condition;
}

void SQLTreeNode::SetJoinConditionConstraintType( JoinConditionConstraintType type )
{
    join_condition_constraint_type = type;
}

JoinConditionConstraintType SQLTreeNode::GetJoinConditionConstraintType() const
{
    return join_condition_constraint_type;
}

void SQLTreeNode::SetJoinOtherCondition(BiaodashiPointer arg_expr) {
    this->my_join_other_condition = arg_expr;
}

BiaodashiPointer SQLTreeNode::GetJoinOtherCondition() {
    return this->my_join_other_condition;
}

void SQLTreeNode::SetSetOperationType(SetOperationType arg_type) {
    this->my_setoperation_type = arg_type;
}

SetOperationType SQLTreeNode::GetSetOperationType() {
    return this->my_setoperation_type;
}

CommonBiaodashiPtr SQLTreeNode::GetSelfJoinMainFilter()
{
    return this->m_selfJoinMainFilter;
}

int SQLTreeNode::GetSelfJoinColumnId()
{
    return this->m_selfJoinColumnId;
}

vector< HalfJoinInfo > SQLTreeNode::GetSelfJoinInfo()
{
    return this->m_selfJoinInfo;
}

SQLTreeNodePointer SQLTreeNode::makeTreeNode_Table(AbstractQueryPointer arg_query, BasicRelPointer arg_basic_rel) {
    SQLTreeNodePointer the_node = std::make_shared<SQLTreeNode>(SQLTreeNodeType::Table_NODE);

    the_node->SetMyQuery(arg_query);;

    the_node->SetBasicRel(arg_basic_rel);

    return the_node;
}

SQLTreeNodePointer SQLTreeNode::makeTreeNode_BinaryJoin(AbstractQueryPointer arg_query, JoinType arg_join_type,
                                                        BiaodashiPointer arg_expr, BiaodashiPointer arg_other_expr, bool is_not_in) {
    SQLTreeNodePointer the_node = std::make_shared<SQLTreeNode>(SQLTreeNodeType::BinaryJoin_NODE);

    the_node->SetMyQuery(arg_query);

    the_node->SetJoinType(arg_join_type);
    the_node->SetIsNotInFlag( is_not_in );
    the_node->SetJoinCondition(arg_expr);
    the_node->SetJoinOtherCondition( arg_other_expr );

    return the_node;

}

SQLTreeNodePointer SQLTreeNode::makeTreeNode_Filter(AbstractQueryPointer arg_query, BiaodashiPointer arg_expr) {
    SQLTreeNodePointer the_node = std::make_shared<SQLTreeNode>(SQLTreeNodeType::Filter_NODE);

    the_node->SetMyQuery(arg_query);;

    the_node->SetFilterStructure(arg_expr);

    return the_node;
}

SQLTreeNodePointer SQLTreeNode::makeTreeNode_Group(AbstractQueryPointer arg_query) {
    SQLTreeNodePointer the_node = std::make_shared<SQLTreeNode>(SQLTreeNodeType::Group_NODE);

    the_node->SetMyQuery(arg_query);;

    return the_node;

}

SQLTreeNodePointer SQLTreeNode::makeTreeNode_Sort(AbstractQueryPointer arg_query) {
    SQLTreeNodePointer the_node = std::make_shared<SQLTreeNode>(SQLTreeNodeType::Sort_NODE);

    the_node->SetMyQuery(arg_query);;

    return the_node;

}

SQLTreeNodePointer SQLTreeNode::makeTreeNode_Column(AbstractQueryPointer arg_query) {
    SQLTreeNodePointer the_node = std::make_shared<SQLTreeNode>(SQLTreeNodeType::Column_NODE);

    the_node->SetMyQuery(arg_query);;

    return the_node;

}

SQLTreeNodePointer SQLTreeNode::makeTreeNode_Limit(AbstractQueryPointer arg_query, int64_t offset, int64_t size) {
    SQLTreeNodePointer the_node = std::make_shared<SQLTreeNode>(SQLTreeNodeType::Limit_NODE);

    the_node->SetMyQuery(arg_query);;

    the_node->limit_offset = offset;
    the_node->limit_size = size;

    return the_node;

}

SQLTreeNodePointer SQLTreeNode::makeTreeNode_SelfJoin( AbstractQueryPointer arg_query, int joinColumnId, CommonBiaodashiPtr mainFilter,
        const vector< HalfJoinInfo >& joinInfo )
{
    SQLTreeNodePointer the_node = std::make_shared< SQLTreeNode >( SQLTreeNodeType::SELFJOIN_NODE );
    the_node->SetMyQuery( arg_query );
    the_node->m_selfJoinColumnId = joinColumnId;
    the_node->m_selfJoinMainFilter = mainFilter;
    the_node->m_selfJoinInfo = joinInfo;
    return the_node;
}

SQLTreeNodePointer SQLTreeNode::makeTreeNode_StarJoin( AbstractQueryPointer query )
{
    auto node = std::make_shared< SQLTreeNode >( SQLTreeNodeType::StarJoin_NODE );
    node->SetMyQuery( query );
    return node;
}

SQLTreeNodePointer SQLTreeNode::makeTreeNode_InnerJoin( AbstractQueryPointer query )
{
    auto node = std::make_shared< SQLTreeNode >( SQLTreeNodeType::InnerJoin_NODE );
    node->SetMyQuery( query );
    return node;
}

SQLTreeNodePointer SQLTreeNode::makeTreeNode_Exchange( AbstractQueryPointer query )
{
    auto node = std::make_shared< SQLTreeNode >( SQLTreeNodeType::Exchange_NODE );
    node->SetMyQuery( query );
    return node;
}

SQLTreeNodePointer
SQLTreeNode::makeTreeNode_SetOperation(AbstractQueryPointer arg_query, SetOperationType arg_set_type) {
    SQLTreeNodePointer the_node = std::make_shared<SQLTreeNode>(SQLTreeNodeType::SetOp_NODE);

    the_node->SetMyQuery(arg_query);;
    the_node->SetSetOperationType(arg_set_type);
    return the_node;
}

void SQLTreeNode::receive_pushed_expr_list(std::vector<BiaodashiPointer> arg_list) {
    this->pushed_list = arg_list;

}

std::vector<BiaodashiPointer> SQLTreeNode::GetPushedList() {
    return this->pushed_list;
}

std::vector<BasicRelPointer> SQLTreeNode::GetInvolvedTableList() {
    std::vector< BasicRelPointer > rels;
    for ( const auto& p : involved_table_list )
    {
        rels.emplace_back( p.lock() );
    }
    return rels;
}


NodeRelationStructurePointer SQLTreeNode::GetNodeRelationStructure() {
    return this->node_relation_structure;
}

void SQLTreeNode::SetNodeRelationStructure(NodeRelationStructurePointer arg_nrsp) {
    this->node_relation_structure = arg_nrsp;
}


bool SQLTreeNode::GetIsTopNodeOfAFromSubquery() {
    return this->IsTopNodeOfAFromSubquery;
}

void SQLTreeNode::SetIsTopNodeOfAFromSubquery(bool arg_value) {
    this->IsTopNodeOfAFromSubquery = arg_value; //must be true!
}


bool SQLTreeNode::GetForwardMode4ColumnNode() {
    return this->ForwardMode4ColumnNode;
}

void SQLTreeNode::SetForwardMode4ColumnNode(bool arg_value) {
    this->ForwardMode4ColumnNode = arg_value;
}

size_t SQLTreeNode::GetForwardedColumnCount() {
    return this->ForwardedColumnCount;
}

void SQLTreeNode::SetForwardedColumnCount(size_t arg_value) {
    this->ForwardedColumnCount = arg_value;
}


std::vector<ColumnShellPointer> SQLTreeNode::GetReferencedColumnArray() {
    return this->referenced_column_array;
}

void SQLTreeNode::SetReferencedColumnArray(std::vector<ColumnShellPointer> arg_array) {
    this->referenced_column_array = arg_array;
}

std::vector<ColumnShellPointer> SQLTreeNode::GetRequiredColumnArray() {
    return this->required_column_array;
}

void SQLTreeNode::SetRequiredColumnArray(std::vector<ColumnShellPointer> arg_array) {
    this->required_column_array = arg_array;
}

void SQLTreeNode::AddRequiredColumnArray(std::vector<ColumnShellPointer> arg_array) {
    if ( spool_id == -1 )
    {
        required_column_array = arg_array;
        return;
    }

    auto my_query = std::dynamic_pointer_cast< SelectStructure >( query.lock() );
    if ( my_query->GetQueryContext()->type != QueryContextType::TheTopQuery && my_query->GetQueryContext()->GetParent() )
    {
        my_query = std::dynamic_pointer_cast< SelectStructure >( my_query->GetQueryContext()->GetParent()->GetSelectStructure() );
    }

    const auto& alias_map = my_query->GetSpoolAlias();
    for ( const auto& column : arg_array )
    {
        bool found = false;
        auto table_name = column->GetTable()->GetMyOutputName();
        auto it = alias_map.find( table_name );
        std::string alias_name;
        if ( it != alias_map.cend() )
        {
            alias_name = it->second;
        }

        BasicRelPointer target_table;
        for ( const auto& ptr : involved_table_list )
        {
            auto table = ptr.lock();
            if ( table->GetMyOutputName() == table_name )
            {
                found = true;
                target_table = table;
                break;
            }
            else if ( !alias_name.empty() && table->GetMyOutputName() == alias_name )
            {
                target_table = table;
                break;
            }
            else if( column->GetTable()->GetID() == table->GetID() )// sub query may have different alias from outer query, we need use id to compare
            {
                target_table = table;
                break;
            }
        }

        auto target_table_name = target_table->GetMyOutputName();
        auto target_column_name = column->GetColumnName();

        bool already_in = false;
        for ( const auto& required_column : required_column_array )
        {
            if ( required_column->GetColumnName() == target_column_name
                 && required_column->GetTable()->GetMyOutputName() == target_table_name )
            {
                already_in = true;
                break;
            }
        }

        if ( !already_in )
        {
            if ( !found )
            {
                for ( size_t i = 0; i < target_table->GetRelationStructure()->GetColumnCount(); i++ )
                {
                    auto new_column_structure = target_table->GetRelationStructure()->GetColumn( i );
                    if ( new_column_structure->GetName() == target_column_name )
                    {
                        auto new_column = std::make_shared< ColumnShell >( target_table_name, column->GetColumnName() );
                        new_column->SetColumnStructure( new_column_structure );
                        new_column->SetTable( target_table );
                        required_column_array.emplace_back( new_column );
                        break;
                    }
                }
            }
            else
            {
                required_column_array.emplace_back( column );
            }
        }
    }
}

void SQLTreeNode::SetTreeFormedTag(bool arg_value) {
    this->tree_formed = arg_value;
}

bool SQLTreeNode::GetTreeFormedTag() {
    return this->tree_formed;
}


void SQLTreeNode::SetPositionForReferencedColumn(ColumnShellPointer arg_column, int arg_value) {
    assert(arg_column != nullptr);
    this->referenced_column_position_map[arg_column] = arg_value;

    std::string full_name = arg_column->GetTableName() + "." + arg_column->GetColumnName();

    this->column_pos_map_by_name[full_name] = arg_value;
}

int SQLTreeNode::GetPositionForReferencedColumn(ColumnShellPointer arg_column) const {
    assert(arg_column != nullptr);
    //return this->referenced_column_position_map.at(arg_column);

    std::string full_name = arg_column->GetTableName() + "." + arg_column->GetColumnName();
    auto it = this->column_pos_map_by_name.find( full_name );
    if ( it == column_pos_map_by_name.cend() )
    {
        if ( std::string::npos != full_name.find("!-!SELECT_ALIAS!-!") )
        {
            return 0;
        }
        else
        {
            ARIES_ASSERT( 0, "NO Referenced Column: " + full_name );
        }
    }

    return it->second;
}

void SQLTreeNode::SetSourceDevicesId( const std::vector< int >& ids )
{
    source_devices_id.assign( ids.cbegin(), ids.cend() );
}

const std::vector< int >& SQLTreeNode::GetSourceDevicesId() const
{
    return source_devices_id;
}

void SQLTreeNode::SetTargetDeviceId( int id )
{
    target_device_id = id;
}

int SQLTreeNode::GetTargetDeviceId() const

{
    return target_device_id;
}

void SQLTreeNode::SetSliceCount( int count )
{
    slice_count = count;
}

int SQLTreeNode::GetSliceCount() const
{
    return slice_count;
}

void SQLTreeNode::AddPartitionCondition( const BiaodashiPointer& condition )
{
    partition_conditions.emplace_back( condition );
}

const std::vector< BiaodashiPointer >& SQLTreeNode::GetPartitionCondition() const
{
    return partition_conditions;
}

void SQLTreeNode::SetSliceIndex( int index )
{
    slice_index = index;
}

int SQLTreeNode::GetSliceIndex() const
{
    return slice_index;
}

void SQLTreeNode::SetGroupForAggInSelect(bool arg_value) {
    this->group_for_agg_in_select = arg_value;
}

bool SQLTreeNode::GetGroupForAggInSelect() {
    return this->group_for_agg_in_select;
}


void SQLTreeNode::SetColumnOutputSequence(std::vector<int> arg_value) {
    this->column_output_sequence = arg_value;
}

std::vector<int> SQLTreeNode::GetColumnOutputSequence() {
    return this->column_output_sequence;
}


bool SQLTreeNode::IsColumnNodeRemovable() {
    return this->column_node_removable;
}

void SQLTreeNode::SetColumnNodeRemovable(bool arg_value) {
    this->column_node_removable = arg_value;
}

std::vector<BiaodashiPointer> SQLTreeNode::GetExprs4ColumnNode() {
    return this->exprs_for_column_node;
}

void SQLTreeNode::SetExprs4ColumnNode(std::vector<BiaodashiPointer> arg_value) {
    this->exprs_for_column_node = arg_value;
}


void SQLTreeNode::SetACreatedGroupNode(bool arg_value) {
    this->a_created_group_node = arg_value;
}

bool SQLTreeNode::GetACreatedGroupNode() {
    return this->a_created_group_node;
}

void SQLTreeNode::SetGroupHavingMode(bool arg_value) {
    this->group_having_mode = arg_value;
}

bool SQLTreeNode::GetGroupHavingMode() {
    return this->group_having_mode;
}

void SQLTreeNode::SetGroupOutputCount(int arg_value) {
    this->group_output_count = arg_value;
}

int SQLTreeNode::GetGroupOutputCount() {
    return this->group_output_count;
}


//int primary_foreign_join_form = 0; // 0: not, 1: left->primary_key, right->foreign_key; -1: left->foreign_key, right->primary_key
void SQLTreeNode::SetPrimaryForeignJoinForm(int arg_value) {
    this->primary_foreign_join_form = arg_value;
}

int SQLTreeNode::GetPrimaryForeignJoinForm() {
    return this->primary_foreign_join_form;
}

void SQLTreeNode::SetPrimaryTableIntact(bool arg_value) {
    this->primary_table_intact = arg_value;
}

bool SQLTreeNode::GetPrimaryTableIntact() {
    return this->primary_table_intact;
}

int64_t SQLTreeNode::GetLimitOffset() {
    return limit_offset;
}

void SQLTreeNode::SetLimitOffset(int64_t offset) {
    limit_offset = offset;
}

int64_t SQLTreeNode::GetLimitSize() {
    return limit_size;
}

void SQLTreeNode::SetLimitSize(int64_t size) {
    limit_size = size;
}

SQLTreeNodePointer SQLTreeNode::Clone() {
    SQLTreeNodePointer new_one = std::make_shared<SQLTreeNode>(type);

    new_one->query = query;
    
    new_one->limit_offset = limit_offset;
    new_one->limit_size = limit_size;

    new_one->group_output_count = group_output_count;
    new_one->a_created_group_node = a_created_group_node;
    new_one->column_pos_map_by_name = column_pos_map_by_name;
    new_one->exprs_for_column_node = exprs_for_column_node;
    new_one->my_join_condition = my_join_condition;
    new_one->my_join_other_condition = my_join_other_condition;
    new_one->my_join_type = my_join_type;
    new_one->inner_join_conditions = inner_join_conditions;
    new_one->group_for_agg_in_select =  group_for_agg_in_select;
    new_one->my_basic_rel = my_basic_rel;
    new_one->my_filter_structure = my_filter_structure;
    // new_one->involved_table_list = involved_table_list;
    new_one->column_output_sequence.assign( column_output_sequence.cbegin(), column_output_sequence.cend() );
    new_one->unique_columns.assign( unique_columns.cbegin(), unique_columns.cend() );
    new_one->spool_id = spool_id;

    if ( type == SQLTreeNodeType::Table_NODE )
    {
        new_one->involved_table_list = involved_table_list;
    }

    for (const auto& child : children) {
        auto new_child = child->Clone();
        new_one->AddChild( new_child );
        new_child->SetParent( new_one );
    }
    return new_one;
}

static JoinConditionConstraintType isForeignConstraintKey( CommonBiaodashiPtr left, CommonBiaodashiPtr right )
{
    if ( left->GetType() != BiaodashiType::Lie || right->GetType() != BiaodashiType::Lie )
    {
        return JoinConditionConstraintType::None;
    }

    auto left_column_shell = boost::get< ColumnShellPointer >( left->GetContent() );
    if ( left_column_shell->GetTable()->IsSubquery() )
    {
        return JoinConditionConstraintType::None;
    }

    auto left_db_name = left_column_shell->GetTable()->GetDb();
    auto left_table_name = left_column_shell->GetTable()->GetID();

    auto left_database = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( left_db_name );
    auto left_table = left_database->GetTableByName( left_table_name );
    auto left_column = left_table->GetColumnByName( left_column_shell->GetColumnName() );

    auto right_column_shell = boost::get< ColumnShellPointer >( right->GetContent() );
    if ( right_column_shell->GetTable()->IsSubquery() )
    {
        return JoinConditionConstraintType::None;
    }

    auto right_db_name = right_column_shell->GetTable()->GetDb();
    auto right_table_name = right_column_shell->GetTable()->GetID();

    auto right_database = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( right_db_name );
    auto right_table = right_database->GetTableByName( right_table_name );
    auto right_column = right_table->GetColumnByName( right_column_shell->GetColumnName() );

    schema::ColumnEntryPtr primary = nullptr;
    schema::ColumnEntryPtr foreign = nullptr;

    std::string primary_db_name, primary_table_name;
    std::string foreign_db_name, foreign_table_name;

    schema::TableEntrySPtr foreign_table = nullptr;

    JoinConditionConstraintType ret = JoinConditionConstraintType::None;

    if ( left_column->IsPrimary() )
    {
        primary = left_column;
        primary_db_name = left_db_name;
        primary_table_name = left_table_name;
        ret = JoinConditionConstraintType::RightReferencedLeft;
    }
    else if ( right_column->IsPrimary() )
    {
        primary = right_column;
        primary_db_name = right_db_name;
        primary_table_name = right_table_name;
        ret = JoinConditionConstraintType::LeftReferencedRight;
    }

    if ( left_column->IsForeignKey() )
    {
        foreign = left_column;
        foreign_table = left_table;
    }
    else if ( right_column->IsForeignKey() )
    {
        foreign = right_column;
        foreign_table = right_table;
    }

    if ( !primary || !foreign )
    {
        return JoinConditionConstraintType::None;
    }

    for ( const auto& foreign_key : foreign_table->GetForeignKeys() )
    {
        if ( foreign_key->referencedSchema != primary_db_name ||
             foreign_key->referencedTable != primary_table_name )
        {
            continue;
        }

        for ( size_t i = 0; i < foreign_key->keys.size(); i++ )
        {
            const auto& key = foreign_key->keys[ i ];

            if ( key == foreign->GetName() )
            {
                const auto& referenced_key = foreign_key->referencedKeys[ i ];
                if ( referenced_key == primary->GetName() )
                {
                    return ret;
                }

                return JoinConditionConstraintType::None;
            }
        }
    }

    return JoinConditionConstraintType::None;

}

JoinConditionConstraintType SQLTreeNode::CheckJoinConditionConstraintType( BiaodashiPointer arg_expr )
{
    auto parent = std::dynamic_pointer_cast< CommonBiaodashi >( arg_expr );
    if ( !parent || parent->GetType() != BiaodashiType::Bijiao )
    {
        return JoinConditionConstraintType::None;
    }

    auto left = std::dynamic_pointer_cast< CommonBiaodashi >( parent->GetChildByIndex( 0 ) );
    auto right = std::dynamic_pointer_cast< CommonBiaodashi >( parent->GetChildByIndex( 1 ) );

    return isForeignConstraintKey( left, right );
}

void SQLTreeNode::ReCheckJoinConditionConstraintType()
{
    join_condition_constraint_type = CheckJoinConditionConstraintType( my_join_condition );
}

void SQLTreeNode::AddUniqueKeys( const std::vector< int >& keys )
{
    if ( keys.size() > 0 )
        unique_columns.emplace_back( keys );
}

void SQLTreeNode::SetUniqueKeys( const std::vector< std::vector< int > >& unique_keys )
{
    unique_columns.clear();
    for ( const auto& keys : unique_keys )
    {
        if ( keys.size() > 0 )
            unique_columns.emplace_back( keys );
    }
}

void SQLTreeNode::ClearUniqueKeys()
{
    unique_columns.clear();
}

const std::vector< std::vector< int > >& SQLTreeNode::GetUniqueKeys() const
{
    return unique_columns;
}

static std::pair< int, int > get_column_ids_of_children(
    const SQLTreeNode* join_node,
    const JoinType joint_type,
    const CommonBiaodashiPtr& expression )
{
    auto* left = ( CommonBiaodashi* )( expression->GetChildByIndex( 0 ).get() );
    auto* right = ( CommonBiaodashi* )( expression->GetChildByIndex( 1 ).get() );

    auto result = std::make_pair( 0, 0 );
    if ( left->GetType() == BiaodashiType::Lie )
    {
        auto col = boost::get< ColumnShellPointer >( left->GetContent() );
        switch( col->GetValueType() )
        {
            case BiaodashiValueType::TINY_INT:
            case BiaodashiValueType::SMALL_INT:
            case BiaodashiValueType::INT:
            case BiaodashiValueType::LONG_INT:
            {
                result.first = join_node->GetPositionForReferencedColumn( col );
                break;
            }
            case BiaodashiValueType::TEXT:
            {
                if ( joint_type != JoinType::SemiJoin && joint_type != JoinType::AntiJoin )
                {
                    result.first = join_node->GetPositionForReferencedColumn( col );
                }
                break;
            }
            default:
                break;
        }
    }

    if ( right->GetType() == BiaodashiType::Lie )
    {
        auto col = boost::get< ColumnShellPointer >( right->GetContent() );
        switch( col->GetValueType() )
        {
            case BiaodashiValueType::TINY_INT:
            case BiaodashiValueType::SMALL_INT:
            case BiaodashiValueType::INT:
            case BiaodashiValueType::LONG_INT:
            {
                result.second = join_node->GetPositionForReferencedColumn( col );
                break;
            }
            case BiaodashiValueType::TEXT:
            {
                if ( joint_type != JoinType::SemiJoin && joint_type != JoinType::AntiJoin )
                {
                    result.second = join_node->GetPositionForReferencedColumn( col );
                }
                break;
            }
            default:
                break;
        }
    }
    return result;
}

static bool contains_all( const std::vector< BasicRelPointer >& tables, const std::vector< ColumnShellPointer >& columns )
{
    for ( const auto& column : columns )
    {
        auto column_table = column->GetTable();
        bool found = false;
        for ( const auto& table : tables )
        {
            if ( column_table == table )
            {
                found = true;
                break;
            }
        }

        if ( !found )
        {
            return false;
        }
    }

    return true;
}

static bool is_valid_equal_condition( SQLTreeNode* join_node, CommonBiaodashiPtr& expression )
{
    auto* left = ( CommonBiaodashi* )( expression->GetChildByIndex( 0 ).get() );
    auto* right = ( CommonBiaodashi* )( expression->GetChildByIndex( 1 ).get() );

    auto left_columns = left->GetAllReferencedColumns();
    auto right_columns = right->GetAllReferencedColumns();

    auto left_tables = join_node->GetLeftChild()->GetInvolvedTableList();
    auto right_tables = join_node->GetRightChild()->GetInvolvedTableList();


    if ( contains_all( left_tables, left_columns ) && contains_all( right_tables, right_columns ) )
    {
        return true;
    }

    if ( contains_all( left_tables, right_columns ) && contains_all( right_tables, left_columns ) )
    {
        expression->SwitchChild();
        return true;
    }

    return false;
}

bool SQLTreeNode::CanUseHashJoin( bool& left_as_hash, bool& right_as_hash )
{
    left_as_hash = false;
    right_as_hash = false;
    if ( type != SQLTreeNodeType::BinaryJoin_NODE )
    {
        return false;
    }

    // if ( my_join_type == JoinType::RightJoin || my_join_type == JoinType::FullJoin )
    // {
    //     return false;
    // }

    if ( !my_join_condition )
    {
        return false;
    }

    const auto& left = children[ 0 ];
    const auto& right = children[ 1 ];

    const auto& left_unique_keys = left->GetUniqueKeys();
    const auto& right_unique_keys = right->GetUniqueKeys();

    if ( left_unique_keys.empty() && right_unique_keys.empty() )
    {
        return false;
    }

    auto condition = std::dynamic_pointer_cast< CommonBiaodashi >( my_join_condition );
    if ( !condition->IsEqualCondition() )
    {
        return false;
    }

    std::vector< CommonBiaodashiPtr > equal_conditions;
    std::vector< CommonBiaodashiPtr > other_conditions;
    equal_conditions.emplace_back( condition );

    BiaodashiAuxProcessor processor;
    if ( my_join_other_condition )
    {
        auto list = processor.generate_and_list( my_join_other_condition );
        for ( const auto& item : list )
        {
            auto other_condition = std::dynamic_pointer_cast< CommonBiaodashi >( item );
            if ( other_condition->IsEqualCondition() && is_valid_equal_condition( this, other_condition ) )
            {
                equal_conditions.emplace_back( other_condition );
            }
            else
            {
                other_conditions.emplace_back( other_condition );
            }
        }
    }

    for ( const auto& keys : left_unique_keys )
    {
        if ( keys.size() > equal_conditions.size() )
        {
            continue;
        }

        bool unmatch = false;
        std::vector< CommonBiaodashiPtr > conditions;
        std::vector< BiaodashiPointer > left_other_conditions;
        left_other_conditions.insert( left_other_conditions.end(), other_conditions.cbegin(), other_conditions.cend() );
        for ( const auto& key : keys )
        {
            bool found = false;
            for ( const auto& condition : equal_conditions )
            {
                auto pair = get_column_ids_of_children( this, my_join_type, condition );
                if ( key == pair.first )
                {
                    found = true;
                    conditions.emplace_back( condition );
                    break;
                }
            }

            if ( !found )
            {
                unmatch = true;
                break;
            }
        }
        for ( const auto& condition : equal_conditions )
        {
            if ( std::find( conditions.cbegin(), conditions.cend(), condition ) == conditions.cend() )
            {
                left_other_conditions.emplace_back( condition );
            }
        }

        if ( !unmatch )
        {
            left_as_hash = true;
            left_hash_join_info.EqualConditions.assign( conditions.cbegin(), conditions.cend() );
            if ( !left_other_conditions.empty() )
            {
                left_hash_join_info.OtherCondition = std::dynamic_pointer_cast< CommonBiaodashi >( processor.make_biaodashi_from_and_list( left_other_conditions ) );
            }
            break;
        }
    }

    for ( const auto& keys : right_unique_keys )
    {
        if ( keys.size() > equal_conditions.size() )
        {
            continue;
        }

        bool unmatch = false;
        std::vector< CommonBiaodashiPtr > conditions;
        std::vector< BiaodashiPointer > right_other_conditions;
        right_other_conditions.insert( right_other_conditions.end(), other_conditions.cbegin(), other_conditions.cend() );
        for ( const auto& key : keys )
        {
            bool found = false;
            for ( const auto& condition : equal_conditions )
            {
                auto pair = get_column_ids_of_children( this, my_join_type, condition );
                if ( key == abs( pair.second ) )
                {
                    found = true;
                    conditions.emplace_back( condition );
                    break;
                }
            }

            if ( !found )
            {
                unmatch = true;
                break;
            }
        }

        for ( const auto& condition : equal_conditions )
        {
            if ( std::find( conditions.cbegin(), conditions.cend(), condition ) == conditions.cend() )
            {
                right_other_conditions.emplace_back( condition );
            }
        }

        if ( !unmatch )
        {
            right_as_hash = true;
            right_hash_join_info.EqualConditions.assign( conditions.cbegin(), conditions.cend() );
            if ( !right_other_conditions.empty() )
            {
                right_hash_join_info.OtherCondition = std::dynamic_pointer_cast< CommonBiaodashi >( processor.make_biaodashi_from_and_list( right_other_conditions ) );
            }
            else
            {
                right_hash_join_info.OtherCondition = nullptr;
            }
            break;
        }
    }

    auto IsKeyTypeValidForHashJoin = []( const std::vector< CommonBiaodashiPtr >& equalConditions )
    {
        assert( !equalConditions.empty() );
        set< BiaodashiValueType > validValueTypes{ BiaodashiValueType::SMALL_INT, BiaodashiValueType::TINY_INT, BiaodashiValueType::INT, BiaodashiValueType::LONG_INT, BiaodashiValueType::TEXT };
        for( auto condition : equalConditions )
        {
            assert( condition->GetChildrenCount() == 2 );
            auto leftValueType = std::dynamic_pointer_cast< CommonBiaodashi >( condition->GetChildByIndex( 0 ) )->GetValueType();
            auto rightValueType = std::dynamic_pointer_cast< CommonBiaodashi >( condition->GetChildByIndex( 1 ) )->GetValueType();
            if( validValueTypes.find( leftValueType ) == validValueTypes.end() || validValueTypes.find( rightValueType ) == validValueTypes.end() )
                return false;
        }
        return true;
    };

    if( left_as_hash )
        left_as_hash = IsKeyTypeValidForHashJoin( left_hash_join_info.EqualConditions );
    if( right_as_hash )
        right_as_hash = IsKeyTypeValidForHashJoin( right_hash_join_info.EqualConditions );


    return left_as_hash || right_as_hash;
}

const HashJoinInfo& SQLTreeNode::GetLeftHashJoinInfo() const 
{
    return left_hash_join_info;
}

const HashJoinInfo& SQLTreeNode::GetRightHashJoinInfo() const
{
    return right_hash_join_info;
}

BiaodashiPointer SQLTreeNode::GetLeftOfJoinCondition() const
{
    if ( type != SQLTreeNodeType::BinaryJoin_NODE || !my_join_condition )
    {
        return nullptr;
    }

    auto condition = std::dynamic_pointer_cast< CommonBiaodashi >( my_join_condition );
    if ( condition->GetType() != BiaodashiType::Bijiao )
    {
        return nullptr;
    }

    if ( boost::get< int >( condition->GetContent() ) != static_cast< int >( ComparisonType::DengYu ) )
    {
        return nullptr;
    }

    return condition->GetChildByIndex( 0 );
}

BiaodashiPointer SQLTreeNode::GetRightOfJoinCondition() const
{
    if ( type != SQLTreeNodeType::BinaryJoin_NODE || !my_join_condition )
    {
        return nullptr;
    }

    auto condition = std::dynamic_pointer_cast< CommonBiaodashi >( my_join_condition );
    if ( condition->GetType() != BiaodashiType::Bijiao )
    {
        return nullptr;
    }

    if ( boost::get< int >( condition->GetContent() ) != static_cast< int >( ComparisonType::DengYu ) )
    {
        return nullptr;
    }

    return condition->GetChildByIndex( 1 );
}

bool SQLTreeNode::IsInnerJoin() const
{
    return type == SQLTreeNodeType::BinaryJoin_NODE && my_join_type == JoinType::InnerJoin;
}

void SQLTreeNode::AddStarJoinCondition( const std::vector< CommonBiaodashiPtr >& conditions )
{
    star_join_conditions.emplace_back( conditions );
}

std::vector< std::vector< CommonBiaodashiPtr > > SQLTreeNode::GetStarJoinConditions() const
{
    return star_join_conditions;
}

void SQLTreeNode::SetConditionAsStarJoin( const BiaodashiPointer& condition )
{
    condition_as_star_join_center = condition;
}

BiaodashiPointer SQLTreeNode::GetConditionAsStarJoin() const
{
    return condition_as_star_join_center;
}


void SQLTreeNode::SetSpoolId( const int& id )
{
    spool_id = id;
}

int SQLTreeNode::GetSpoolId() const
{
    return spool_id;
}

std::vector< BiaodashiPointer >& SQLTreeNode::GetInnerJoinConditions()
{
    return inner_join_conditions;
}

void SQLTreeNode::SetSameNode( const SQLTreeNodePointer& node)
{
    same_node = node;
}

SQLTreeNodePointer SQLTreeNode::GetSameNode() const{
    return same_node.lock();
}

void SQLTreeNode::SetSpoolAlias( const std::string left, const std::string right )
{
    spool_alias_map[ left ] = right;
}

const std::map< std::string, std::string >& SQLTreeNode::GetSpoolAlias() const
{
    return spool_alias_map;
}

}
