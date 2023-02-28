#ifndef ARIES_COMMON_BIAODASHI
#define ARIES_COMMON_BIAODASHI

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

//#include <folly/dynamic.h>
#include <boost/variant.hpp>
#include <server/mysql/include/binary_log_types.h>

#include "ShowUtility.h"
#include "TroubleHandler.h"
#include "VariousEnum.h"

#include "AbstractBiaodashi.h"
#include "AbstractQuery.h"

#include "ExprContext.h"
#include "QueryContext.h"

#include "BasicRel.h"
#include "ColumnShell.h"
#include "SQLFunction.h"
#include "CudaAcc/AriesEngineDef.h"
#include "SQLIdent.h"

class sys_var;
class user_var_entry;

namespace aries
{

/*In order to avoid unnecessary OOP, we try to use a single class
 *to represent all possibilities of Biaodashi. To achieve this,
 *folly::dynamic is easier to use, but it cannot hold a shared_ptr.
 *if use folly, you have to handle (BiaodashiType::Query) separately.
 *Two bad things for variant: (1) it cannot be nullptr; (2) you have to
 *explicitly use std::string(char *) instead of directly char* for init;
 *Another choice is to use antlrcpp::Any, but I want to limit antrl just
 *inside the Parser, and no more further. Lestat-06/29/2018.
 */

//    typedef  folly::dynamic BiaodashiContent;
typedef boost::variant<bool, int, double, std::string, AbstractQueryPointer,
                       ColumnShellPointer, SQLFunctionPointer, sys_var*, SQLIdentPtr, int64_t>
    BiaodashiContent;

std::string get_name_of_expr_type(BiaodashiType type);
class CommonBiaodashi;
using CommonBiaodashiPtr = std::shared_ptr<CommonBiaodashi>;
class CommonBiaodashi : public AbstractBiaodashi
{
private:
    /*A Biaodashi has three components:*/
    BiaodashiType type;
    std::string origName; // original expression string, used as column name returned to client
    BiaodashiContent content;

    // for prepared statements long data value
    bool is_param_set = false;
    bool is_long_data = false;
    string long_data;
    enum enum_field_types param_type;
    // end for prepared statements long data value

    std::vector<BiaodashiPointer> children;

    ExprContextPointer expr_context;

    CommonBiaodashi(const CommonBiaodashi &arg);

    CommonBiaodashi &operator=(const CommonBiaodashi &arg);

    BiaodashiValueType value_type = BiaodashiValueType::UNKNOWN;

    int column_starting_position = -1;
    int column_ending_position = -1;

    // std::vector<std::string> involved_table_list;
    std::vector<BasicRelPointer> involved_table_list;

    bool contain_outer = false;
    bool contain_subquery = false;
    bool nullable = false;
    bool is_calculable = true;
    std::int32_t length = 1;
    int associated_length = -1;

    int numeric_type_value = 0;

    bool is_visible_in_result = true;
    bool need_show_alias = true;

    aries_acc::AriesDataBufferSPtr buffer_ptr;
    bool expect_buffer = false;

public:
    static int NumericTypeOffsetInt8;
    static int NumericTypeOffsetInt16;
    static int NumericTypeOffsetInt32;
    static int NumericTypeOffsetInt64;
    static int NumericTypeOffsetDecimal;
    static int NumericTypeOffsetFloat;
    static int NumericTypeOffsetDouble;

private:
    int getStringLength();
    void handleNullableFunction(aries_acc::AriesSqlFunctionType type);
    void SetChild( int childIndex, const BiaodashiPointer child )
    {
        if ( child )
            child->SetParent( this );
        children[ childIndex ] = child;
    }

    void SetChildren( const std::vector<BiaodashiPointer>& arg_children )
    {
        ClearChildren();
        for ( auto& child : arg_children )
        {
            if ( child )
                child->SetParent( this );
            AddChild( child );
        }
    }

    void CheckCastFunctionExpr();

public:
    CommonBiaodashi(BiaodashiType arg_type, BiaodashiContent arg_content);

    bool IsLiteral() const;

    void SetOrigName(std::string origExpString, std::string alias);

    std::string GetOrigName() { return origName; }

    int GetColumnStartingPosition();

    void SetColumnStartingPosition(int arg);

    int GetColumnEndingPosition();

    void SetColumnEndingPosition(int arg);

    BiaodashiValueType GetValueType();

    void SetValueType(BiaodashiValueType arg_value);

    BiaodashiType GetType();
    void SetType(BiaodashiType arg_type) { type = arg_type; }

    BiaodashiContent GetContent();
    void SetContent(BiaodashiContent arg_content) { content = arg_content; }

    // for prepared statements long data value
    bool IsParamSet() { return is_param_set; }
    string GetLongData() { return long_data; }
    void SetPreparedStmtNullParam();
    void SetPreparedStmtParam(enum enum_field_types arg_param_type, unsigned char **arg_data,
                              unsigned char *arg_data_end);
    void SetPreparedStmtParam(const std::shared_ptr<user_var_entry>& userVarEntryPtr);
    enum enum_field_types GetParamType() { return param_type; }
    void SetPreparedStmtLongData(const char *arg_str, ulong arg_length);
    void SetPreparedStmtLongParam(enum enum_field_types arg_param_type);
    void ClearPreparedStmtParam();
    bool IsLongDataValue() { return is_long_data; }
    // end for prepared statements long data value

    size_t GetChildrenCount();

    int GetNumericTypeValue();

    BiaodashiPointer GetChildByIndex(size_t arg_index);

    void SetExprContext(ExprContextPointer arg);

    ExprContextPointer GetExprContext();

    void AddChild(BiaodashiPointer arg_child);

    /*This function is used to judge whether the whole xpr or one part is in groupby
     * list -- then even a column is naked, it is safe*/
    bool helpNakedColumnInExpr(int arg_column_index,
                               ExprContextPointer arg_expr_context);

    void CheckExprPostWork(ExprContextPointer arg_expr_context);

    bool CheckExprTypeMulti(std::vector<BiaodashiValueType> arg_type_array);

    bool CheckExprType(BiaodashiValueType arg_type);

    void CheckExpr4Children(ExprContextPointer arg_expr_context, bool expectOnlyConst);

    static bool __compareTwoExpr4Types(CommonBiaodashi *left, CommonBiaodashi *right);

    void CheckStarExpr(ExprContextPointer arg_expr_context);

    void CheckFunctionExpr(ExprContextPointer arg_expr_context, bool expectOnlyConst);

    std::string CheckFunctionName();

    void CheckQueryExpr(ExprContextPointer arg_expr_context);

    /*touch work!*/
    void CheckExpr(ExprContextPointer arg_expr_context, bool expectOnlyConst = false);

    /* convert biaoshifu's content from string to ColumnShell */
    void convertBiaoshifu();

    bool findColumnInTableArray(ColumnShellPointer arg_column,
                                const std::vector<BasicRelPointer>& arg_table_array
                                /* int arg_level */ );

    static bool __compareTwoExprs(CommonBiaodashi *left, CommonBiaodashi *right);

    bool CompareMyselfToAnotherExpr(BiaodashiPointer arg_expr);

    bool checkExprinGroupbyList(QueryContextPointer arg_query_context);

    bool checkColumninGroupbyList(QueryContextPointer arg_query_context);

    bool findColumnInQueryContext(
        QueryContextPointer arg_query_context,   /* where to search?*/
        ExprContextPointer arg_expr_context,     /* whose context*/
        // int arg_level,                           /* how deep we have been so far*/
        QueryContextPointer arg_subquery_context /* who gives me this job?*/
    );

    bool findColumnInLoop(ExprContextPointer arg_expr_context);

    /*arg_biaoshifu could be either just a column_name or table_name.column_name*/
    void checkBiaoshifu(ExprContextPointer arg_expr_context, bool expectOnlyConst);

    std::string ContentToString();

    std::string ChildrenToString();

    std::string ChildrenToString_Skip0();

    std::string CaseToString();

    std::string ToString();

    /*return column name of this expr. "a.b" should return b*/
    /*these methods are only for ColumnShell!*/
    std::string GetName();

    std::string GetTableName();

    void SetAggStatus(int arg_starting_pos, int arg_ending_pos,
                      ExprContextPointer arg_expr_context);

    LogicType GetLogicType4AndOr();

    bool GetMyBoolValue();

    void ObtainReferenceTableInfo();
    void ObtainReferenceTableAndOtherInfo( vector< AbstractQueryPointer >& subqueries, vector< ColumnShellPointer >& columns );
    void ResetReferencedColumnsInfo( const string& oldTableName, BasicRelPointer& newTable, int absoluteLevel );
    void ReplaceReferencedColumns( const ColumnShellPointer& oldCol, const ColumnShellPointer& newCol );

    // std::vector<std::string> involved_table_list;
    // bool contain_outer = false;
    // bool contain_subquery = false;

    void SetContainOuter(bool arg_value);

    void SetContainSubquery(bool arg_value);

    void ClearInvolvedTableList();

    bool GetContainOuter();

    bool GetContainSubquery();

    std::vector<BasicRelPointer> GetInvolvedTableList();

    bool IsSelectAlias();

    BiaodashiPointer GetRealExprIfAlias();

    std::vector<ColumnShellPointer> GetAllReferencedColumns();

    std::vector<ColumnShellPointer> GetAllReferencedColumns_NoQuery();

    void ResetChildByIndex(int arg_index, BiaodashiPointer arg_bp);

    // void ConvertSelfToFloat(double arg_value);

    void ConvertSelfToDecimal( const std::string& arg_value );
    void ConvertSelfToString( const std::string& arg_value );
    void ConvertSelfToNull();

    void ConvertSelfToBuffer(const aries_acc::AriesDataBufferSPtr& buffer);

    bool IsNullable();

    void SetIsNullable(bool is_nullable);

    std::int32_t GetLength();
    int GetAssociatedLength() const;

    void SetLength(std::int32_t);
    void SetAssociatedLength(int);

    BiaodashiValueType GetPromotedValueType();

    bool IsCalculable();

    std::shared_ptr<CommonBiaodashi> Clone();

    std::shared_ptr<CommonBiaodashi> CloneUsingNewExprContext( ExprContextPointer context );

    void Clone(const std::shared_ptr<CommonBiaodashi>& other);

    void ClearChildren();

    bool IsVisibleInResult();

    void SetIsVisibleInResult(bool value);

    void SetNeedShowAlias(bool value);

    bool NeedShowAlias();

    bool IsSameAs(CommonBiaodashi* other);

    void SetExpectBuffer(bool value);

    bool IsExpectBuffer();

    bool IsAggFunction() const;

    bool ContainsAggFunction() const;

    bool CouldBeCompactDecimal() const;

    bool IsEqualCondition() const;

    bool IsTrueConstant() const;

    void SwitchChild();

    aries_acc::AriesDataBufferSPtr GetBuffer();

    std::shared_ptr<CommonBiaodashi> Normalized();

    /*!
     @brief 尝试用exprsToCheck中的表达式替换类型为Biaoshifu的子节点.
     @param [in] exprsToCheck 用来查找和替换的表达式
     @return 
     @note 当子节点类型为Biaoshifu而且GetOrigName和exprsToCheck中的某个表达式GetOrigName的值一样时,进行替换
    */
    void ReplaceBiaoshifu( const vector< CommonBiaodashi* >& exprsToCheck );

    friend int compare( const CommonBiaodashi& l, const CommonBiaodashi& r );
    friend bool operator>( const CommonBiaodashi& l, const CommonBiaodashi& r );
    friend bool operator<( const CommonBiaodashi& l, const CommonBiaodashi& r );
    friend bool operator==( const CommonBiaodashi& l, const CommonBiaodashi& r );
    friend bool operator!=( const CommonBiaodashi& l, const CommonBiaodashi& r );

    friend bool kv_compare( const CommonBiaodashi& l, const CommonBiaodashi& r );
};

std::string get_name_of_value_type(BiaodashiValueType type);

struct BiaodashiComparator
{
    bool operator()( const BiaodashiPointer& a, const BiaodashiPointer& b ) const
    {
        return kv_compare( *( ( CommonBiaodashi* ) a.get() ), *( ( CommonBiaodashi* ) b.get() ) );
    }
};

} // namespace aries
#endif
