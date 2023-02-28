#ifndef ARIES_BASIC_REL
#define ARIES_BASIC_REL

#include <cassert>
#include <memory>
#include <vector>

#include "TroubleHandler.h"
#include "VariousEnum.h"

#include "AbstractQuery.h"
#include "AbstractBiaodashi.h"

#include "PhysicalTable.h"


namespace aries {

/***************************************************************************
 *
 *                          BasicRel
 *
 **************************************************************************/

class SQLTreeNode;

class BasicRel {

    /*A basicrel can be either a simple table or a subquery*/
private:

    BasicRel(const BasicRel &arg);

    BasicRel &operator=(const BasicRel &arg);


    bool isSubquery;
    bool isExpression;
    std::string rel_id;
    std::string db_name; // current db if it's empty

    std::shared_ptr<std::string> rel_alias_name; /*the alias name!*/

    AbstractQueryPointer subquery;

    PhysicalTablePointer underlying_table; //the real table
    RelationStructurePointer relation_structure; // what do i look like?


    std::shared_ptr< SQLTreeNode > rel_node;
    std::vector< ColumnStructurePointer > columns_alias;

public:


    BasicRel(bool arg_issubquery, std::string arg_id, std::shared_ptr<std::string> arg_alias,
             AbstractQueryPointer arg_subquery);

    void SetDb(const std::string& arg_db) { db_name = arg_db; }
    std::string GetDb() const { return db_name; }

    AbstractQueryPointer GetSubQuery();

    void SetRelationStructure(RelationStructurePointer arg_rel);

    RelationStructurePointer GetRelationStructure();

    void SetUnderlyingTable(PhysicalTablePointer arg_table);

    PhysicalTablePointer GetPhysicalTable();

    bool IsSubquery();

    bool IsExpression();
    void SetIsExpression( bool value );

    std::string GetID();

    std::shared_ptr<std::string> GetAliasNamePointer();

    std::string ToString();

    std::vector<BiaodashiPointer> GetAllColumnsAsExpr();

    std::shared_ptr<SQLTreeNode> GetMyRelNode();

    void SetMyRelNode(std::shared_ptr<SQLTreeNode> arg_rel_node);

    static bool __compareTwoBasicRels(BasicRel *first, BasicRel *second);

    std::string GetMyOutputName();

    /*if no alias, then we set this! Otherwise, we change to this new one*/
    void ResetAlias(std::string arg_alias);

    void ResetToSubQuery( AbstractQueryPointer query );

    void SetColumnsAlias( const std::vector< ColumnStructurePointer >& alias );

    const std::vector< ColumnStructurePointer >& GetColumnsAlias() const;

    bool operator==(const BasicRel& target) const{
        if (db_name != target.db_name) {
            return false;
        }

        if (rel_id != target.rel_id) {
            return false;
        }

        if(rel_alias_name && target.rel_alias_name)
        {
            if (*rel_alias_name != *(target.rel_alias_name)) {
                return false;
            }
        }
        else if (!rel_alias_name && !target.rel_alias_name)
        {
            return true;
        }
        else
        {
            return false;
        }

        return true;
    }

    bool operator!=( const BasicRel& target ) const
    {
        return !( *this == target );
    }
};

typedef std::shared_ptr<BasicRel> BasicRelPointer;

}//namespace
#endif
