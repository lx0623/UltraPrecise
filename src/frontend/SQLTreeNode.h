#ifndef ARIES_SQL_TREE_NODE
#define ARIES_SQL_TREE_NODE


#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

#include "TroubleHandler.h"
#include "VariousEnum.h"
#include "ShowUtility.h"

#include "AbstractQuery.h"
#include "AbstractBiaodashi.h"

#include "QueryContext.h"
#include "ExprContext.h"


#include "BasicRel.h"

#include "NodeRelationStructure.h"

#include "SelectPartStructure.h"
#include "GroupbyStructure.h"
#include "OrderbyStructure.h"


namespace aries {

struct HalfJoinInfo
{
    JoinType HalfJoinType;
    CommonBiaodashiPtr JoinConditionExpr;

    string ToString() const
    {
        string result;
        result += HalfJoinType == JoinType::SemiJoin ? "join type: semi join" : "join type: anti join";
        result += ", join condition: ";
        result += JoinConditionExpr ? JoinConditionExpr->ToString() : "null";
        return result;
    }
};

enum class JoinConditionConstraintType : uint8_t
{
    None,
    LeftReferencedRight, // 左-外键，右-主键
    RightReferencedLeft // 左-主键，右-外键
};

struct HashJoinInfo
{
    std::vector< CommonBiaodashiPtr > EqualConditions;
    CommonBiaodashiPtr OtherCondition;
};

class SQLTreeNode {
private:

    SQLTreeNode(const SQLTreeNode &arg);

    SQLTreeNode &operator=(const SQLTreeNode &arg);


    bool optimized = false;
    NodeRelationStructurePointer node_relation_structure; /*what is my output columns?*/


    SQLTreeNodeType type;
    std::weak_ptr< AbstractQuery > query;  /*where do i come from*/
    // AbstractQuery *query_p;

    // SQLTreeNode *parent = NULL; //really useful?
    std::weak_ptr< SQLTreeNode > parent;
    std::vector<std::shared_ptr<SQLTreeNode>> children;
    std::vector<std::shared_ptr<SQLTreeNode>> to_remove;


    std::vector< std::weak_ptr< BasicRel > > involved_table_list;

    /*actually useless*/
    // std::vector<BasicRelPointer> backup_involved_table_list;

    /*table*/
    std::weak_ptr< BasicRel > my_basic_rel;

    /*filter*/
    BiaodashiPointer my_filter_structure;

    /*binary join*/
    JoinType my_join_type;
    // not exists and not in is different in anti join if it has null value.
    bool is_a_not_in = false;
    // the equal join condition
    BiaodashiPointer my_join_condition;
    // other join conditions
    BiaodashiPointer my_join_other_condition;

    std::vector< BiaodashiPointer > partition_conditions;

    std::vector< std::vector< CommonBiaodashiPtr > > star_join_conditions;
    HashJoinInfo left_hash_join_info;
    HashJoinInfo right_hash_join_info;

    std::vector< BiaodashiPointer > inner_join_conditions;

    /**
     * 当一个节点作为 star join 的中心节点，它与其他节点（非 dimension node） join 的条件
     */
    BiaodashiPointer condition_as_star_join_center;

    JoinConditionConstraintType join_condition_constraint_type = JoinConditionConstraintType::None;

    std::vector<BiaodashiPointer> pushed_list;
    int primary_foreign_join_form = 0; // 0: not, 1: left->primary_key, right->foreign_key; -1: left->foreign_key, right->primary_key
    bool primary_table_intact = false;
    /*Column*/
    //SelectParttructurePointer my_selectpart_structure;

    /*groupby*/
    //GroupbyStructurePointer my_groupby_structure;
    bool group_for_agg_in_select = false;
    bool a_created_group_node = false;

    /*sort*/
    //OrderbyStructurePointer my_orderby_structure;


    /*set*/
    SetOperationType my_setoperation_type;


    bool IsTopNodeOfAFromSubquery = false;


    /*what columns do I have to use?*/
    std::vector<ColumnShellPointer> referenced_column_array;
    /*This is for HavingToFilter: what exprs do I have to use?*/
    // std::vector<BiaodashiPointer> referenced_expr_array;


    /*what columns do I have to generate -- passed from the parent node!*/
    std::vector<ColumnShellPointer> required_column_array;

    /*each column's position in child node's output relation. start from 1. -1 means the right child node for join!*/
    std::map<ColumnShellPointer, int> referenced_column_position_map;
    std::map<std::string, int> column_pos_map_by_name;
    /*This is for HavingToFilter*/
    // std::map<BiaodashiPointer, int> expr_pos_map;


    bool ForwardMode4ColumnNode = false;
    size_t ForwardedColumnCount = 0;

    bool tree_formed = false;

    std::vector<int> column_output_sequence;

    std::vector< std::vector< int > > unique_columns;

    bool column_node_removable = false;

    std::vector<BiaodashiPointer> exprs_for_column_node;


    //i am a filter node, but i work as the having expr for groupby!
    bool group_having_mode = false;
    int group_output_count = -1;

    int test_limit = 1;

    /**
     * For Limit node
     */
    int64_t limit_offset = 0;
    int64_t limit_size = 0;

    //for self join node
    int m_selfJoinColumnId = 0;
    CommonBiaodashiPtr m_selfJoinMainFilter;
    vector< HalfJoinInfo > m_selfJoinInfo;

    int spool_id = -1;

    std::weak_ptr< SQLTreeNode > same_node;
    std::map< std::string, std::string > spool_alias_map;

    GroupbyStructurePointer groupby_structure;
    SelectPartStructurePointer select_part_structure;
    OrderbyStructurePointer orderby_structure;

    int target_device_id;
    std::vector< int > source_devices_id;

    int slice_count = 0;
    int slice_index = -1;

public:
    SQLTreeNode(SQLTreeNodeType arg_type);

    void SetOptimized( bool b ) { optimized = b; }
    bool IsOptimized() const { return optimized; }
    void ResetType( SQLTreeNodeType arg_type );

    void ClearChildren();

    SQLTreeNodeType GetType();

    void SetMyQuery(AbstractQueryPointer arg_query);

    AbstractQueryPointer GetMyQuery();

    void SetMySelectPartStructure( const SelectPartStructurePointer& structure );

    SelectPartStructurePointer GetMySelectPartStructure();

    void SetMyGroupbyStructure( const GroupbyStructurePointer& structure );

    GroupbyStructurePointer GetMyGroupbyStructure();

    void SetMyOrderbyStructure( const OrderbyStructurePointer& structure );

    OrderbyStructurePointer GetMyOrderbyStructure();


    void SetParent( const std::shared_ptr< SQLTreeNode >& arg_parent );

    std::shared_ptr< SQLTreeNode > GetParent();


    void AddChild(std::shared_ptr<SQLTreeNode> arg_child);

    size_t GetChildCount();

    std::shared_ptr<SQLTreeNode> GetTheChild();

    void SetTheChild(std::shared_ptr<SQLTreeNode> arg_child);

    std::shared_ptr<SQLTreeNode> GetLeftChild();

    std::shared_ptr<SQLTreeNode> GetRightChild();

    void SetLeftChild(std::shared_ptr<SQLTreeNode> arg_child);

    void SetRightChild(std::shared_ptr<SQLTreeNode> arg_child);

    std::shared_ptr<SQLTreeNode> GetChildByIndex(size_t arg_index);

    std::vector<BasicRelPointer> GetInvolvedTableList();

    std::string getMyNodeInfo();

    std::string ToString(int arg_level);

    void SetBasicRel(BasicRelPointer arg_basic_rel);

    BasicRelPointer GetBasicRel();

    void SetFilterStructure(BiaodashiPointer arg_expr);

    BiaodashiPointer GetFilterStructure();

    void SetIsNotInFlag( bool is_not_in );

    bool GetIsNotInFlag();

    void SetJoinType(JoinType arg_join_type);

    JoinType GetJoinType();

    void SetJoinCondition(BiaodashiPointer arg_expr);

    BiaodashiPointer GetJoinCondition();

    void SetJoinOtherCondition(BiaodashiPointer arg_expr);

    BiaodashiPointer GetJoinOtherCondition();

    void SetJoinConditionConstraintType( JoinConditionConstraintType type );
    JoinConditionConstraintType GetJoinConditionConstraintType() const;
    JoinConditionConstraintType CheckJoinConditionConstraintType( BiaodashiPointer arg_expr );
    void ReCheckJoinConditionConstraintType();

    void SetSetOperationType(SetOperationType arg_type);

    SetOperationType GetSetOperationType();

    int GetSelfJoinColumnId();

    CommonBiaodashiPtr GetSelfJoinMainFilter();

    vector< HalfJoinInfo > GetSelfJoinInfo();

    static std::shared_ptr<SQLTreeNode>
    makeTreeNode_Table(AbstractQueryPointer arg_query, BasicRelPointer arg_basic_rel);

    static std::shared_ptr<SQLTreeNode>
    makeTreeNode_BinaryJoin(AbstractQueryPointer arg_query, JoinType arg_join_type, BiaodashiPointer arg_expr, BiaodashiPointer arg_other_expr = nullptr, bool is_not_in = false );

    static std::shared_ptr<SQLTreeNode> makeTreeNode_Filter(AbstractQueryPointer arg_query, BiaodashiPointer arg_expr);

    static std::shared_ptr<SQLTreeNode> makeTreeNode_Group(AbstractQueryPointer arg_query);

    static std::shared_ptr<SQLTreeNode> makeTreeNode_Sort(AbstractQueryPointer arg_query);

    static std::shared_ptr<SQLTreeNode> makeTreeNode_Column(AbstractQueryPointer arg_query);

    static std::shared_ptr<SQLTreeNode>
    makeTreeNode_SetOperation(AbstractQueryPointer arg_query, SetOperationType arg_set_type);

    static std::shared_ptr<SQLTreeNode> makeTreeNode_Limit(AbstractQueryPointer arg_query, int64_t offset = 0, int64_t size = 0);

    static std::shared_ptr< SQLTreeNode > makeTreeNode_SelfJoin( AbstractQueryPointer arg_query, int joinColumnId, CommonBiaodashiPtr mainFilter,
            const vector< HalfJoinInfo >& joinInfo );
    static std::shared_ptr< SQLTreeNode > makeTreeNode_StarJoin( AbstractQueryPointer arg_query );


    static std::shared_ptr< SQLTreeNode > makeTreeNode_InnerJoin( AbstractQueryPointer arg_query );

    static std::shared_ptr< SQLTreeNode > makeTreeNode_Exchange( AbstractQueryPointer query );

    void receive_pushed_expr_list(std::vector<BiaodashiPointer> arg_list);

    std::vector<BiaodashiPointer> GetPushedList();


    /*The following three functions are used to reset correspoinding child when doing predicate_pushdown. They directly replace the child node in the children vector, without re-calculating involved_table_list. These functions are ugly!! Because the SetChild verions are UGLY! todo*/
    void ResetTheChild(std::shared_ptr<SQLTreeNode> arg_child);

    void ResetLeftChild(std::shared_ptr<SQLTreeNode> arg_child);

    void ResetRightChild(std::shared_ptr<SQLTreeNode> arg_child);


    //reset the child and re-calculating involved_table_list
    void ReCalculateInvolvedTableList();

    void CompletelyResetAChild(std::shared_ptr<SQLTreeNode> arg_child_old, std::shared_ptr<SQLTreeNode> arg_child_new);

    void CompletelyResetAChild_WithoutReCalculateInvolvedTableList(std::shared_ptr<SQLTreeNode> arg_child_old,
                                                                   std::shared_ptr<SQLTreeNode> arg_child_new);

    NodeRelationStructurePointer GetNodeRelationStructure();

    void SetNodeRelationStructure(NodeRelationStructurePointer arg_nrsp);

    bool GetIsTopNodeOfAFromSubquery();

    void SetIsTopNodeOfAFromSubquery(bool arg_value);

    bool GetForwardMode4ColumnNode();

    void SetForwardMode4ColumnNode(bool arg_value);

    size_t GetForwardedColumnCount();

    void SetForwardedColumnCount(size_t arg_value);


    std::vector<ColumnShellPointer> GetReferencedColumnArray();

    void SetReferencedColumnArray(std::vector<ColumnShellPointer> arg_array);

    std::vector<ColumnShellPointer> GetRequiredColumnArray();

    void SetRequiredColumnArray(std::vector<ColumnShellPointer> arg_array);

    void AddRequiredColumnArray(std::vector<ColumnShellPointer> arg_array);

    std::shared_ptr< SQLTreeNode > GetSameNode() const;

    void SetSameNode( const std::shared_ptr< SQLTreeNode >& node);

    void SetTreeFormedTag(bool arg_value);

    bool GetTreeFormedTag();

    void SetPositionForReferencedColumn(ColumnShellPointer arg_column, int arg_value);

    int GetPositionForReferencedColumn(ColumnShellPointer arg_column) const;


    void SetGroupForAggInSelect(bool arg_value);

    bool GetGroupForAggInSelect();

    void SetColumnOutputSequence(std::vector<int> arg_value);

    std::vector<int> GetColumnOutputSequence();

    bool IsColumnNodeRemovable();

    void SetColumnNodeRemovable(bool arg_value);


    std::vector<BiaodashiPointer> GetExprs4ColumnNode();

    void SetExprs4ColumnNode(std::vector<BiaodashiPointer> arg_value);

    void SetACreatedGroupNode(bool arg_value);

    bool GetACreatedGroupNode();


    void SetGroupHavingMode(bool arg_value);

    bool GetGroupHavingMode();

    void SetGroupOutputCount(int arg_value);

    int GetGroupOutputCount();

    //int primary_foreign_join_form = 0; // 0: not, 1: left->primary_key, right->foreign_key; -1: left->foreign_key, right->primary_key
    void SetPrimaryForeignJoinForm(int arg_value);

    int GetPrimaryForeignJoinForm();

    void SetPrimaryTableIntact(bool arg_value);

    bool GetPrimaryTableIntact();

    int64_t GetLimitOffset();
    void SetLimitOffset(int64_t offset);

    int64_t GetLimitSize();
    void SetLimitSize(int64_t size);

    std::shared_ptr<SQLTreeNode> Clone();

    void AddUniqueKeys( const std::vector< int >& keys );
    void SetUniqueKeys( const std::vector< std::vector< int > >& unique_keys );
    void ClearUniqueKeys();
    const std::vector< std::vector< int > >& GetUniqueKeys() const;

    bool CanUseHashJoin( bool& left_as_hash, bool& right_as_hash );
    bool IsInnerJoin() const;
    BiaodashiPointer GetLeftOfJoinCondition() const;
    BiaodashiPointer GetRightOfJoinCondition() const;

    const HashJoinInfo& GetLeftHashJoinInfo() const;
    const HashJoinInfo& GetRightHashJoinInfo()  const;

    void AddStarJoinCondition( const std::vector< CommonBiaodashiPtr >& conditions );
    std::vector< std::vector< CommonBiaodashiPtr > > GetStarJoinConditions() const;

    void SetConditionAsStarJoin( const BiaodashiPointer& condition );
    BiaodashiPointer GetConditionAsStarJoin() const;

    void SetSpoolId( const int& id );
    int GetSpoolId() const;

    void SetSpoolAlias( const std::string left, const std::string right );
    const std::map< std::string, std::string >& GetSpoolAlias() const;

    std::vector< BiaodashiPointer >& GetInnerJoinConditions();

    /**
     * for exchange node
     */
    const std::vector< int >& GetSourceDevicesId() const;
    int GetTargetDeviceId() const;

    void SetSourceDevicesId( const std::vector< int >& ids );
    void SetTargetDeviceId( int id );

    void SetSliceCount( int count );
    void SetSliceIndex( int index );

    int GetSliceCount() const;
    int GetSliceIndex() const;

    void AddPartitionCondition( const BiaodashiPointer& condition );
    const std::vector< BiaodashiPointer >& GetPartitionCondition() const;

    friend int compare( const SQLTreeNode& left, const SQLTreeNode& right );

    inline static void AddTreeNodeChild( const std::shared_ptr<SQLTreeNode>& parent, const std::shared_ptr<SQLTreeNode>& child )
    {
        parent->AddChild( child );
        child->SetParent( parent );
    }

    inline static void SetTreeNodeChild( const std::shared_ptr<SQLTreeNode>& parent, const std::shared_ptr<SQLTreeNode>& child )
    {
        parent->AddChild( child );
        child->SetParent( parent );
    }
};

typedef std::shared_ptr<SQLTreeNode> SQLTreeNodePointer;

}//namespace
#endif
