#ifndef ARIES_COLUMN_SHELL
#define ARIES_COLUMN_SHELL

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <vector>


#include "TroubleHandler.h"
#include "VariousEnum.h"


#include "ColumnStructure.h"
#include "AbstractBiaodashi.h"

namespace aries {
class BasicRel;

#define PRIMARY_KEY_FLAG_MASK 1
#define UNIQUE_FLAG_MASK 2

class ColumnShell {

private:

    std::string table_name;
    std::string column_name;


    ColumnShell(const ColumnShell &arg);

    ColumnShell &operator=(const ColumnShell &arg);

    /*This is unnecessay. The absolute_level works! todo*/
    // int query_level = -1; /*where am i from. current query is 0, then 1, 2...*/

    int absolute_level = -1; /*The top 0, then increases until to my level*/

    bool in_group_list = false;
    int index = -1; /*what is it? I just copy it from Javascript*/

    std::weak_ptr< BasicRel > table; // the table where I am in


    int location_in_table;
    ColumnStructurePointer column_structure = nullptr;


    int alias_expr_index = -1; //am I just an select alias used by orderby or groupby? If so, which expr is me? -1 means no!
    BiaodashiPointer expr_4_alias = nullptr;


    /*when used in a node of the query plan tree!*/
    //int index_table_source = 0;
    //int index_column_source = -1;

    int position_in_child_tables = 0;

    //this is for placeholder
    bool is_placeholder = false;
    BiaodashiValueType my_own_value_type;

    int numeric_precision;
    int numeric_scale;

    uint8_t key_flags;

public:

    ColumnShell(std::string arg_table_name, std::string arg_column_name);

    /*a str could be "table_name.column_name or just column_name"*/
    // ColumnShell(std::string arg_str);

    void SetInGroupList(bool arg_in_group_list);

    bool GetInGroupList();

    int GetAbsoluteLevel();

    void SetAbsoluteLevel(int arg_level);

    BiaodashiValueType GetValueType();

    void SetMyOwnValueType(BiaodashiValueType arg_value);

    void SetTableName(std::string arg_table_name);

    void SetTable(std::shared_ptr<BasicRel> arg_table);

    std::shared_ptr<BasicRel> GetTable();

    void SetLocationInTable(int arg_location);
    int GetLocationInTable() const;

    void SetColumnStructure(ColumnStructurePointer arg_column_structure);

    ColumnStructurePointer GetColumnStructure();

    std::string GetTableName();

    std::string GetColumnName();

    int GetLength();

    void SetQueryLevel(int arg_level);

    static std::vector<std::string> SplitString(std::string input_str, char delimitor);

    std::string ToString();

    void SetAliasExprIndex(int arg_index);

    int GetAliasExprIndex();

    void SetExpr4Alias(BiaodashiPointer arg_bp);

    BiaodashiPointer GetExpr4Alias();

    //int index_table_source = 0;
    //int index_column_source = 0;

    //int GetIndexTableSource();
    //void SetIndexTableSource(int arg_value);

    //int GetIndexColumnSource();
    //void SetIndexColumnSource(int arg_value);

    void SetPositionInChildTables(int arg_value);

    int GetPositionInChildTables();

    void SetPlaceholderMark(bool arg_value);

    bool GetPlaceholderMark();

    int GetPrecision();
    int GetScale();
    void SetPresision(int precision);
    void SetScale(int scale);

    void SetIsPrimaryKey( bool is_primary );
    void SetIsUnique( bool is_unique);
    bool IsPrimaryKey() const;
    bool IsUnique() const;
};


typedef std::shared_ptr<ColumnShell> ColumnShellPointer;

} //namespace



#endif
