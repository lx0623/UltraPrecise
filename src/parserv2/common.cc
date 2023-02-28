#include <glog/logging.h>
#include <boost/variant.hpp>
#include "version.h"
#include <utils/string_util.h>
#include "server/mysql/include/sys_vars.h"
#include <server/mysql/include/mysqld_error.h>
#include <server/mysql/include/derror.h>
#include <AriesAssert.h>
#include <server/mysql/include/mysqld.h>
#include "common.h"
#include "driver.h"
#include "utils/datatypes.h"

using namespace aries;

namespace aries_parser {
// copy from datatypes/decimal.hxx
#define SUPPORTED_MAX_PRECISION (NUM_TOTAL_DIG*DIG_PER_INT32)
#define SUPPORTED_MAX_SCALE (NUM_TOTAL_DIG*DIG_PER_INT32)

int decode_field_len( const string& name, const string& lenStr )
{
    int len = 0;
    try
    {
        len = std::stoi( lenStr.data() );
    }
    catch(const std::out_of_range& e)
    {
        ARIES_EXCEPTION( ER_TOO_BIG_DISPLAYWIDTH, name.data(), INT32_MAX );
    }
    catch(const std::exception& e)
    {
        ARIES_ASSERT( 0, "invalid length str: " + lenStr );
    }
    return len;
}

/*
  Return an unescaped text literal without quotes
  Fix sometimes to do only one scan of the string
*/

string get_text(const char* str, size_t len, const u_char sep)
{
    string ret;
    const CHARSET_INFO *cs= default_charset_info;

    const char* end = str + len;
    char* start;
    if (!(start= static_cast<char *>(malloc((uint) (end-str)+1))))
        ARIES_EXCEPTION( ER_OUTOFMEMORY, (end -  str) + 1 );

    char *to;

    for (to=start ; str != end ; str++)
    {
        int l;
        if (use_mb(cs) &&
            (l = my_ismbchar(cs, str, end))) {
            while (l--)
                *to++ = *str++;
            str--;
            continue;
        }
        // if (!(lip->m_thd->variables.sql_mode & MODE_NO_BACKSLASH_ESCAPES) &&
        if ( *str == '\\' && str+1 != end )
        {
            switch(*++str) {
                case 'n':
                    *to++='\n';
                    break;
                case 't':
                    *to++= '\t';
                    break;
                case 'r':
                    *to++ = '\r';
                    break;
                case 'b':
                    *to++ = '\b';
                    break;
                case '0':
                    *to++= 0;			// Ascii null
                    break;
                case 'Z':			// ^Z must be escaped on Win32
                    *to++='\032';
                    break;
                case '_':
                case '%':
                    *to++= '\\';		// remember prefix for wildcard
                    /* Fall through */
                default:
                    *to++= *str;
                    break;
            }
        }
        else if (*str == sep)
            *to++= *str++;		// Two ' or "
        else
            *to++ = *str;
    }
    *to=0;
    ret.assign( start, to - start );
    free( start );
    return ret;
}

Expression CreateBetweenExpr( const Expression &expr1, const Expression &expr2, const Expression &expr3 )
{
    LogicType logicType = LogicType::AND;
    auto expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Andor, static_cast<int>( logicType ) );

    auto leftExpr = std::make_shared<CommonBiaodashi>(BiaodashiType::Bijiao, static_cast<int>(ComparisonType::DaYuDengYu) );
    leftExpr->AddChild( expr1 );
    leftExpr->AddChild( expr2 );

    auto rightExpr = std::make_shared<CommonBiaodashi>(BiaodashiType::Bijiao, static_cast<int>(ComparisonType::XiaoYuDengYu) );
    auto expr1Clone = std::dynamic_pointer_cast< CommonBiaodashi >( expr1 )->Clone();
    rightExpr->AddChild( expr1Clone );
    rightExpr->AddChild( expr3 );

    expression->AddChild( leftExpr );
    expression->AddChild( rightExpr );
    return expression;
}

Expression CreateRowIdExpr()
{
    auto ident = make_shared< SQLIdent >( "", "", schema::DBEntry::ROWID_COLUMN_NAME );
    auto rowIdExpr = make_shared< CommonBiaodashi >( BiaodashiType::Biaoshifu, ident );
    rowIdExpr->SetOrigName( schema::DBEntry::ROWID_COLUMN_NAME, "" );
    return rowIdExpr;
}

Expression LiteralToExpression( const Literal& literal )
{
    Expression expr;
    switch (literal.type) {
        case LiteralType::INT:
        case LiteralType::LONG_INT:
        case LiteralType::ULONGLONG_INT:
            expr = CreateIntegerExpression( literal );
            break;
        case LiteralType::DECIMAL:
            expr = CreateDecimalExpression(literal.str);
            break;
        case LiteralType::STRING:
            expr = CreateStringExpression(literal.str);
            break;
        case LiteralType::NULL_LITERAL:
            expr = CreateNullExpression();
            break;
        case LiteralType::BOOL_FALSE:
            expr = CreateBoolExpression(false);
            break;
        case LiteralType::BOOL_TRUE:
            expr = CreateBoolExpression(true);
            break;
        default:
            ThrowNotSupportedException( "Literal type " + std::to_string( int( literal.type ) ) );
    }
    return expr;
}

string NormalizeIdent(string ident) {
    return aries_utils::strip_quotes(aries_utils::to_lower(ident));
}
void SetExprOrigName(Expression& expr, const std::string& origExpString, const std::string& alias) {
    auto commonExpr = std::dynamic_pointer_cast<CommonBiaodashi>(expr);
    if (!commonExpr->GetOrigName().empty() && alias.empty()) {
        return;
    }
    auto exprStrNoQuotes = aries_utils::strip_quotes(origExpString);
    auto aliasNoQuotes = aries_utils::strip_quotes(alias);
    commonExpr->SetOrigName(exprStrNoQuotes, aliasNoQuotes);
}

JoinStructurePointer CreateDerivedTableJoinStructure(const AbstractQueryPointer& arg_subquery,
                                                     std::string& arg_alias,
                                                     const std::vector< ColumnDescriptionPointer >& columns ) {
    /*
      The alias is actually not optional at all, but being MySQL we
      are friendly and give an informative error message instead of
      just 'syntax error'.
    */
    std::shared_ptr<std::string> alias_ptr = nullptr;
    LOG(INFO) << "sub_QUERY: " << arg_subquery->ToString();
    if (!arg_alias.empty()) {
        std::string alias = aries_utils::to_lower(arg_alias);
        alias_ptr = std::make_shared<std::string>(alias);
    } else {
        ARIES_EXCEPTION_SIMPLE(ER_SYNTAX_ERROR, "derived table need alias");
    }

    auto rel = CreateTableIdent(true, std::string(), std::string(), alias_ptr, arg_subquery);

    if ( !columns.empty() )
    {
        std::vector< ColumnStructurePointer > columns_alias;
        for ( const auto& column : columns )
        {
            columns_alias.emplace_back( std::make_shared< ColumnStructure >(column->column_name, ColumnValueType::INT, 1, false, false ) );
        }
        rel->SetColumnsAlias( columns_alias );
    }

    auto join_structure = std::make_shared<JoinStructure>();
    join_structure->SetLeadingRel(rel);
    return join_structure;
}

BasicRelPointer CreateTableIdent(bool arg_issubquery,
                                 const std::string& arg_db,
                                 const std::string& arg_id,
                                 std::shared_ptr<std::string> arg_alias,
                                 const AbstractQueryPointer& arg_subquery) {
    auto basic_rel = std::make_shared<BasicRel>(arg_issubquery, NormalizeIdent(arg_id), arg_alias, arg_subquery);
    basic_rel->SetDb(NormalizeIdent(arg_db));
    return basic_rel;
}
static std::shared_ptr<CommonBiaodashi> new_expression(const BiaodashiType& type, const BiaodashiContent& content) {
    auto expression = std::make_shared<CommonBiaodashi>(type, content);
    return expression;
}


Expression CreatePreparedStmtParamExpression() {
    auto p = std::make_shared<CommonBiaodashi>(BiaodashiType::QuestionMark, "?");
    current_thd->stmt_params.emplace_back(p);
    return p;
}
Expression CreateCaseWhenExpression(const Expression& target,
                                    const std::vector<std::tuple<Expression, Expression>>& when_list,
                                    const Expression& elseExpr) {
    auto expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Case, 0);

    expression->AddChild( nullptr );

    for (const auto &item : when_list) {
        auto whenExpr = std::get<0>(item);
        auto thenExpr = std::get<1>(item);
        if ( target )
        {
            auto newWhenExpr = std::make_shared< CommonBiaodashi >( BiaodashiType::Bijiao, static_cast<int>(ComparisonType::DengYu) );
            newWhenExpr->AddChild( target );
            newWhenExpr->AddChild( whenExpr );
            whenExpr = newWhenExpr;
        }
        expression->AddChild( whenExpr );
        expression->AddChild( thenExpr );
    }

    expression->AddChild(elseExpr);

    return expression;

}
// Expression CheckCaseWhenExpression(const Expression& caseExpr, Driver* driver, location l) {
//     static string s1 = R"(case when table_type = 'base table' then 'table' else 'view' end)";
//     static string s2 = R"(case when table_type = 'base table' then 1 else 2 end)";
//     string exprStr = driver->get_string_at_location(l);
//     exprStr = aries_utils::to_lower(exprStr);
//     if (exprStr == s1) {
//         auto ident = std::make_shared<SQLIdent>("", "", "table_type");
//         return CreateIdentExpression(ident);
//     }
//     return caseExpr;
// }
Expression CreateComparationExpression(Expression left, const BiaodashiContent& content, Expression right) {
    auto expression = new_expression(BiaodashiType::Bijiao, content);

    expression->AddChild(left);
    expression->AddChild(right);

    return expression;
}

Expression CreateLogicExpression(LogicType type, Expression left, Expression right) {
    auto expression = new_expression(BiaodashiType::Andor, static_cast<int>(type));
    expression->AddChild(left);
    expression->AddChild(right);

    return expression;
}

Expression CreateFunctionExpression(const std::string& name, Expression arg) {
    auto expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Hanshu, name);
    expression->AddChild(arg);

    return expression;
}

Expression CreateExprListExpression()
{
    auto expression = std::make_shared<CommonBiaodashi>(BiaodashiType::ExprList, 0);
    return expression;
}

Expression CreateFunctionExpression(const std::string& name, std::vector<Expression>& args) {
    Expression expression;
    string lowerName = name;
    aries_utils::to_lower( lowerName );
    if ( "isnull" == lowerName )
    {
        expression = std::make_shared<CommonBiaodashi>(BiaodashiType::IsNull, 0);
        if ( 1 != args.size() )
        {
            ARIES_EXCEPTION( ER_WRONG_PARAMCOUNT_TO_NATIVE_FCT, name.data() ) ;
        }
        expression->AddChild( args[0] );
    }
    else
    {
        expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Hanshu, name);

        for (const auto& arg : args) {
            expression->AddChild(arg);
        }
    }

    return expression;
}

Expression CreateCalcExpression(Expression left, const BiaodashiContent& content, Expression right) {
    auto expression = new_expression(BiaodashiType::Yunsuan, content);

    expression->AddChild(left);
    expression->AddChild(right);

    return expression;
}

Expression CreateStringExpression(const std::string& content) {
    auto expression = new_expression(BiaodashiType::Zifuchuan, content);
    expression->SetValueType( BiaodashiValueType::TEXT );

    return expression;
}
// TODO: support GROUP_CONCAT
Expression CreateGroupConcatExpression() {
    std::string emtpy = "";
    auto expression = new_expression(BiaodashiType::Zifuchuan, emtpy);
    return expression;
}

Expression CreateIntegerExpression(const std::string& content) {
    int64_t value;
    const auto& data = content.data();
    if ( data[ 0 ] == '0' && ( data[ 1 ] == 'x' || data[ 1 ] == 'X' ) )
    {
        value = std::stoll( content, 0, 16 );
    }
    else
    {
        value = std::atoll(content.c_str());
    }
    auto expression = new_expression(BiaodashiType::Zhengshu, value);
    if (value > INT_MAX32 || value < INT_MIN32 ) {
        expression->SetValueType(BiaodashiValueType::LONG_INT);
    } else {
        expression->SetValueType(BiaodashiValueType::INT);
        expression->SetContent((int32_t)value);
    }
    return expression;
}

Expression CreateULLExpression(const std::string& content) {
    /*
    char *tail;
    errno  = 0;
    int64_t value = std::strtoull(content.c_str(), &tail, 10);
    if (errno == ERANGE )
    {
        ARIES_EXCEPTION_SIMPLE( ER_WARN_DATA_OUT_OF_RANGE, "Out of range value" );
    }
    if ((ulonglong) value >= (ulonglong) LLONG_MIN &&
        ((ulonglong) value != (ulonglong) LLONG_MIN
        || args[0]->type() != INT_ITEM ))
    {
    */

      // Ensure that result is converted to DECIMAL, as longlong can't hold
      // the negated number
      // hybrid_type= DECIMAL_RESULT;
      return CreateDecimalExpression( content );
    /*
    }
    else
    {
        auto expression = new_expression(BiaodashiType::Zhengshu, value);
        if (value > INT_MAX32 || value < INT_MIN32 ) {
            expression->SetValueType(BiaodashiValueType::LONG_INT);
        } else {
            expression->SetValueType(BiaodashiValueType::INT);
            expression->SetContent((int32_t)value);
        }
        return expression;
    }
    */
}

Expression CreateIntegerExpression( const Literal& literal )
{
    Expression expression;
    switch ( literal.type )
    {
        case LiteralType::INT:
        case LiteralType::LONG_INT:
        {
            expression = CreateIntegerExpression( literal.str );
            break;
        }
        case LiteralType::ULONGLONG_INT:
        {
            expression = CreateULLExpression( literal.str );
            break;
        }
        default:
        {
            string msg( "unexpected integer literal type: " + std::to_string( (int)literal.type ) );
            ARIES_ASSERT( 0, msg );
        }
    }
    return expression;
}

Expression CreateDecimalExpression(const std::string& content) {
    return new_expression(BiaodashiType::Decimal, content);
}

Expression NegateDecimalExpresstion( const BiaodashiContent& content )
{
    auto value = boost::get<string>( content );
    aries_acc::Decimal decimal(value.data());
    decimal = 0 - decimal;
    char buf[64] = {0};
    value = decimal.GetDecimal(buf);
    return std::make_shared<CommonBiaodashi>(BiaodashiType::Decimal, value);
}

Expression CreateIdentExpression(const SQLIdentPtr& ident) {
    string origName = ident->id;
    aries_utils::to_lower(ident->db);
    aries_utils::to_lower(ident->table);
    aries_utils::to_lower(ident->id);

    auto expression = new_expression(BiaodashiType::Biaoshifu, ident);
    expression->SetOrigName(origName, "");

    return expression;
}

Expression CreateNullExpression() {
    return new_expression(BiaodashiType::Null, 0);
}

Expression CreateBoolExpression(bool value) {
    return new_expression(BiaodashiType::Zhenjia, value);
}


Expression GenerateNotExpression(Expression origin) {
    BiaodashiPointer expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Qiufan, 0);
    expression->AddChild(origin);
    return expression;
}

Expression CreateDistinctFunctionExpression(const std::string& name, Expression arg) {
    auto expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Hanshu, name);
    expression->AddChild(std::make_shared<CommonBiaodashi>(BiaodashiType::Distinct, true));
    expression->AddChild(arg);

    return expression;
}

Expression CreateDistinctFunctionExpression(const std::string& name, std::vector<Expression>& args) {
    auto expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Hanshu, name);
    expression->AddChild(std::make_shared<CommonBiaodashi>(BiaodashiType::Distinct, true));
    for (const auto& arg : args) {
        expression->AddChild(arg);
    }
    return expression;
}
/**
 * https://dev.mysql.com/doc/refman/5.7/en/set-variable.html
 * A reference to a system variable in an expression as @@var_name (with @@ rather than @@GLOBAL. or @@SESSION.)
 * returns the session value if it exists and the global value otherwise.
 * This differs from SET @@var_name = expr, which always refers to the session value.
 * @param variableStructurePtr
 * @return
 */
Expression CreateExpressionFromVariable(const VariableStructurePtr& variableStructurePtr) {
    CommonBiaodashiPtr expression = nullptr;
    switch (variableStructurePtr->varType) {
        case VAR_TYPE::SYS_VAR: {
            SysVarStructurePtr sysVarStructurePtr = std::dynamic_pointer_cast<SysVarStructure>(variableStructurePtr);
            sys_var* sysVar = find_sys_var(sysVarStructurePtr->varName.data());
            /*
             * Different behavious for sys var and user var:
mysql> select @@adbd;
ERROR 1193 (HY000): Unknown system variable 'adbd'
mysql> select @adbd;
+-------+
| @adbd |
+-------+
| NULL  |
+-------+
             * */
            if (!sysVar) {
                // ERROR 1193 (HY000): Unknown system variable 'time_zoneaaa'
                ARIES_EXCEPTION(ER_UNKNOWN_SYSTEM_VARIABLE, sysVarStructurePtr->varName.data());
            }
            if ( !sysVar->check_scope( sysVarStructurePtr->varScope ) )
            {
                if ( OPT_DEFAULT != sysVarStructurePtr->varScope )
                {
                    string errMsg = ( sysVarStructurePtr->varScope == OPT_GLOBAL ? "SESSION" : "GLOBAL" );
                    ARIES_EXCEPTION( ER_INCORRECT_GLOBAL_LOCAL_VAR, sysVar->name.data(), errMsg.data() );
                }
                sysVarStructurePtr->varScope = OPT_GLOBAL;
            }
            switch (sysVar->show_type()) {
                case SHOW_MY_BOOL: {
                    my_bool value = get_sys_var_value<my_bool >(sysVar, sysVarStructurePtr->varScope);
                    expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Zhenjia, (bool)value);
                    expression->SetValueType( BiaodashiValueType::BOOL );
                    break;
                }
                case SHOW_BOOL: {
                    bool value = get_sys_var_value<bool>(sysVar, sysVarStructurePtr->varScope);
                    expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Zhenjia, value);
                    expression->SetValueType( BiaodashiValueType::BOOL );
                    break;
                }
                case SHOW_INT:
                {
                    int value = get_sys_var_value<int>(sysVar, sysVarStructurePtr->varScope);
                    expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Zhengshu, value);
                    expression->SetValueType( BiaodashiValueType::INT );
                    break;
                }
                case SHOW_LONG:
                case SHOW_HA_ROWS:
                case SHOW_SIGNED_LONG:
                case SHOW_LONGLONG: {
                    long value = get_sys_var_value<long>(sysVar, sysVarStructurePtr->varScope);
                    expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Zhengshu, value);
                    expression->SetValueType( BiaodashiValueType::LONG_INT );
                    break;
                }
                case SHOW_DOUBLE: {
                    double value = get_sys_var_value<double >(sysVar, sysVarStructurePtr->varScope);
                    expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Decimal, std::to_string(value));
                    expression->SetValueType( BiaodashiValueType::DECIMAL );
                    break;
                }
                case SHOW_CHAR: {
                    string value((char*)get_sys_var_value<uchar*>(sysVar, sysVarStructurePtr->varScope));
                    expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Zifuchuan, value);
                    expression->SetValueType( BiaodashiValueType::TEXT );
                    break;
                }
                case SHOW_CHAR_PTR: {
                    string value((char*)get_sys_var_value<uchar**, uchar*>(sysVar, sysVarStructurePtr->varScope));
                    expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Zifuchuan, value);
                    expression->SetValueType( BiaodashiValueType::TEXT );
                    break;
                }
                default:
                    ARIES_EXCEPTION(ER_UNKNOWN_SYSTEM_VARIABLE, sysVarStructurePtr->varName.data());
            }

            string exprName;
            switch (sysVarStructurePtr->varScope) {
                case OPT_DEFAULT:
                    exprName = "@@";
                    break;
                case OPT_SESSION:
                    exprName = "@@session.";
                    break;
                case OPT_GLOBAL:
                    exprName = "@@global.";
                    break;
            }
            exprName.append(sysVarStructurePtr->varName);
            expression->SetName(exprName);
            break;
        }
        case VAR_TYPE::USER_VAR: {
            UserVarStructurePtr userVarStructurePtr = std::dynamic_pointer_cast<UserVarStructure>(variableStructurePtr);
            user_var_entry_ptr userVarEntryPtr = current_thd->get_user_var(userVarStructurePtr->varName);
            if (!userVarEntryPtr) {
                expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Null, 0);
            } else {
                expression = std::dynamic_pointer_cast<CommonBiaodashi>( userVarEntryPtr->ToBiaodashi() );
            }
            expression->SetName(userVarStructurePtr->origVarName);
            break;
        }
    }
    return expression;
}

VariableStructurePtr CreateUserVariableStructure(const std::string& name) {
    auto userVarStructurePtr = std::make_shared<UserVarStructure>();
    userVarStructurePtr->varType = VAR_TYPE::USER_VAR;
    /**
     * https://dev.mysql.com/doc/refman/5.7/en/user-variables.html
     * User variable names are not case-sensitive. Names have a maximum length of 64 characters.
     * mysql-5.7.26 benavious:
     * 64 chars:
mysql> set @a123456789a123456789a123456789a123456789a123456789a1234567891234 = 1;
Query OK, 0 rows affected (0.00 sec)

     65 chars
mysql> set @a123456789a123456789a123456789a123456789a123456789a12345678912345 = 1;
ERROR 3061 (42000): User variable name 'a123456789a123456789a123456789a123456789a123456789a12345678912345' is illegal
mysql> select @a123456789a123456789a123456789a123456789a123456789a1234567891234;
+-------------------------------------------------------------------+
| @a123456789a123456789a123456789a123456789a123456789a1234567891234 |
+-------------------------------------------------------------------+
|                                                                 1 |
+-------------------------------------------------------------------+
1 row in set (0.00 sec)

mysql> select @a123456789a123456789a123456789a123456789a123456789a12345678912345;
+--------------------------------------------------------------------+
| @a123456789a123456789a123456789a123456789a123456789a12345678912345 |
+--------------------------------------------------------------------+
| NULL                                                               |
+--------------------------------------------------------------------+
     */
    userVarStructurePtr->origVarName = aries_utils::strip_quotes(name);
    userVarStructurePtr->varName = aries_utils::to_lower(userVarStructurePtr->origVarName);
    return userVarStructurePtr;
}

VariableStructurePtr CreateSysVariableStructure(enum_var_type varScope, const std::string& name) {
    auto sysVarStructurePtr = std::make_shared<SysVarStructure>();
    sysVarStructurePtr->varType = VAR_TYPE ::SYS_VAR;
    sysVarStructurePtr->varScope = varScope;
    sysVarStructurePtr->varName = aries_utils::strip_quotes(name);
    return sysVarStructurePtr;
}

PrepareSrcPtr CreatePrepareSrc(bool isVarRef, const string& stmtCode) {
    auto prepareSrcPtr = std::make_shared<PrepareSrc>();
    prepareSrcPtr->isVarRef = isVarRef;
    prepareSrcPtr->stmtCode = aries_utils::strip_quotes(stmtCode);
    return prepareSrcPtr;
}
PreparedStmtStructurePtr CreatePrepareStmt(const string& stmtName, const PrepareSrcPtr& prepareSrcPtr) {
    auto preparedStmtPtr = std::make_shared<PreparedStmtStructure>();
    preparedStmtPtr->stmtCmd = PREPARED_STMT_CMD::PREPARE;
    preparedStmtPtr->stmtName = aries_utils::strip_quotes(stmtName);
    preparedStmtPtr->prepareSrcPtr = prepareSrcPtr;
    return preparedStmtPtr;
}
PreparedStmtStructurePtr CreateDeallocateStmt(const string& stmtName) {
    auto preparedStmtPtr = std::make_shared<PreparedStmtStructure>();
    preparedStmtPtr->stmtCmd = PREPARED_STMT_CMD::DEALLOCATE;
    preparedStmtPtr->stmtName = aries_utils::strip_quotes(stmtName);
    return preparedStmtPtr;
}
PreparedStmtStructurePtr CreateExecuteStmt(const string& stmtName, const std::vector<string>& varNames) {
    auto preparedStmtPtr = std::make_shared<PreparedStmtStructure>();
    preparedStmtPtr->stmtCmd = PREPARED_STMT_CMD::EXECUTE;
    preparedStmtPtr->stmtName = aries_utils::strip_quotes(stmtName);
    preparedStmtPtr->executeVars = varNames;
    return preparedStmtPtr;

}

SetStructurePtr CreateSetUserVarStructure(const string& name, const Expression& expression) {
    auto newVarName = aries_utils::strip_quotes(name);
    size_t size = newVarName.size();
    if (size > 64) {
        ARIES_EXCEPTION(ER_ILLEGAL_USER_VAR, newVarName.data());
    }

    auto userVarStructurePtr = std::make_shared<UserVarStructure>();
    userVarStructurePtr->origVarName = newVarName;
    userVarStructurePtr->varName = aries_utils::to_lower(newVarName);

    auto setStructurePtr = std::make_shared< SetUserVarStructure >( userVarStructurePtr, expression );
    return setStructurePtr;
}

SetStructurePtr CreateSetSysVarStructure(bool global, const string& name, const string& fullName, const Expression& expression) {
    string varName = NormalizeIdent(name);
    auto sysVarStructurePtr = std::make_shared< SysVarStructure >();
    if (global) {
        sysVarStructurePtr->varScope = OPT_GLOBAL;
    } else {
        sysVarStructurePtr->varScope = OPT_SESSION;
    }
    sysVarStructurePtr->varName = varName;
    sysVarStructurePtr->varFullName = fullName;

    auto setStructurePtr = std::make_shared< SetSysVarStructure >( sysVarStructurePtr, expression );

    return setStructurePtr;
}
SetStructurePtr CreateSetSysVarStructure(const enum_var_type varType, const string& name, const Expression& expression) {
    string varName = NormalizeIdent(name);
    auto sysVarStructurePtr = std::make_shared<SysVarStructure>();
    sysVarStructurePtr->varName = varName;
    sysVarStructurePtr->varFullName.append("@@");
    switch (varType) {
        case OPT_GLOBAL:
            sysVarStructurePtr->varScope = OPT_GLOBAL;
            sysVarStructurePtr->varFullName.append("global.");
            break;
        case OPT_SESSION:
            sysVarStructurePtr->varScope = OPT_SESSION;
            sysVarStructurePtr->varFullName.append("session.");
            break;
        case OPT_DEFAULT:
            sysVarStructurePtr->varScope = OPT_DEFAULT;
            break;
    }
    sysVarStructurePtr->varFullName.append(sysVarStructurePtr->varName);

    auto setStructurePtr = std::make_shared< SetSysVarStructure >( sysVarStructurePtr, expression );
    return setStructurePtr;
}

SetStructurePtr CreateSetSysVarStructure(const enum_var_type varScope, const PT_option_value_following_option_type_ptr& optionValuePtr) {
    string varName = NormalizeIdent(optionValuePtr->varName);

    auto sysVarStructurePtr = std::make_shared<SysVarStructure>();
    sysVarStructurePtr->varScope = varScope;
    sysVarStructurePtr->varName = varName;

    auto setStructurePtr = std::make_shared< SetSysVarStructure >( sysVarStructurePtr, optionValuePtr->expression );
    return setStructurePtr;
}

void ReviseSetStructuresHead(enum_var_type headVarScope, const std::shared_ptr<std::vector<SetStructurePtr>>& setStructures) {
    if (setStructures->size() > 0 && SET_CMD::SET_SYS_VAR == setStructures->at(0)->m_setCmd) {
         auto setVarStructurePtr = std::dynamic_pointer_cast<SetSysVarStructure>(setStructures->at(0));
         setVarStructurePtr->m_sysVarStructurePtr->varScope = headVarScope;
    }
}

std::shared_ptr<std::vector<SetStructurePtr>> AppendOptionSetStructures(const SetStructurePtr& item, const std::shared_ptr<std::vector<SetStructurePtr>>& optionListPtr) {
    std::shared_ptr<std::vector<SetStructurePtr>> setStructures = optionListPtr;
    if (!setStructures) {
        setStructures = std::make_shared<std::vector<SetStructurePtr>>();
    }
    setStructures->emplace_back(item);
    return setStructures;
}
CommandStructurePointer CreateDropDatabaseStructure(bool ifExists, const string &name) {
    auto command = std::make_shared<CommandStructure>();
    string lowerName = name;
    command->SetCommandType(CommandType::DropDatabase);
    command->ifExists = ifExists;
    command->SetDatabaseName(NormalizeIdent(name));
    return command;
}

CommandStructurePointer CreateCreateDbStructure(bool ifNotExists, const string& name) {
    auto command = std::make_shared<CommandStructure>();
    command->SetCommandType(CommandType::CreateDatabase);
    command->ifNotExists = ifNotExists;

    command->SetDatabaseName(NormalizeIdent(name));
    return command;
}

SetStructurePtr CreateSetPasswordStructure( std::string& user, const string& password )
{
    auto setStructurePtr = std::make_shared< SetPasswordStructure >();
    setStructurePtr->m_setCmd = SET_CMD::SET_PASSWORD;
    if ( user.empty() )
    {
       user = current_thd->get_user_name();
    }

    setStructurePtr->user = user;
    setStructurePtr->password = password;
    return setStructurePtr;
}

AccountMgmtStructureSPtr CreateAccountMgmtStructure( CommandType cmdType,
                                                     bool ifNotExists,
                                                     shared_ptr< vector< AccountSPtr > >& userList )
{
    auto command = std::make_shared< AccountMgmtStructure >( cmdType );
    if ( CommandType::CreateUser == cmdType )
        command->ifNotExists = ifNotExists;
    else if ( CommandType::DropUser == cmdType )
        command->ifExists = ifNotExists;
    command->SetAccounts( *userList );

    return command;
}

PT_column_attr_base_ptr CreateDefaultValueAttr(const Expression& expr)
{
    return std::make_shared<PT_default_column_attr>( expr );
}

Field_def_ptr CreateFieldDef(const PT_ColumnType_ptr& column_type, const ColAttrList& col_attr_list) {
    auto field_def = std::make_shared<Field_def>();
    field_def->column_type = column_type;
    field_def->col_attr_list = col_attr_list;
    return field_def;
}
ColumnDescriptionPointer CreateColumnDef(const string& ident, const Field_def_ptr& fieldDefPtr) {
    return CreateColumnDef( ident, fieldDefPtr, nullptr, std::vector< std::string >() );
}

/*
mysql> create table t(f char(-1));
ERROR 1064 (42000): You have an error in your SQL syntax; check the manual that corresponds to your Rateup server version for the right syntax to use near '-' at line 1
mysql> create table t(f int(-1));
ERROR 1064 (42000): You have an error in your SQL syntax; check the manual that corresponds to your Rateup server version for the right syntax to use near '-' at line 1
mysql> create table t(f decimal(-1));
ERROR 1064 (42000): You have an error in your SQL syntax; check the manual that corresponds to your Rateup server version for the right syntax to use near '-' at line 1

mysql> create table t(f int(99999999999999999999999999999999));
ERROR 1439 (42000): Display width out of range for column 'f' (max = 4294967295)

mysql> create table t(f char(99999999999999999999999999999999));
ERROR 1439 (42000): Display width out of range for column 'f' (max = 4294967295)
mysql> create table t(f char(99999999));
ERROR 1074 (42000): Column length too big for column 'f' (max = 255); use BLOB or TEXT instead

mysql> create table t(f decimal(99999999));
ERROR 1426 (42000): Too-big precision 99999999 specified for 'f'. Maximum is 65.
mysql> create table t(f decimal(9999999999999999999999999999999));
ERROR 1439 (42000): Display width out of range for column 'f' (max = 4294967295)

mysql> create table t(f datetime(9999999999));
ERROR 1064 (42000): You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use near '9999999999))' at line 1
mysql> create table t(f datetime(99999999));
ERROR 1426 (42000): Too-big precision 99999999 specified for 'f'. Maximum is 6.

mysql> create table t_dec(f decimal(0) );
Query OK, 0 rows affected (0.02 sec)

mysql> insert into t_dec values( 1.1 );
Query OK, 1 row affected, 1 warning (0.01 sec)

mysql> show warnings;
+-------+------+----------------------------------------+
| Level | Code | Message                                |
+-------+------+----------------------------------------+
| Note  | 1265 | Data truncated for column 'f' at row 1 |
+-------+------+----------------------------------------+
1 row in set (0.00 sec)

mysql> select * from t_dec;
+------+
| f    |
+------+
|    1 |
+------+

mysql> create table t_int(f int(0));
Query OK, 0 rows affected (0.03 sec)

mysql> insert into t_int values(1111);
Query OK, 1 row affected (0.00 sec)

mysql> select * from t_int;
+------+
| f    |
+------+
| 1111 |
+------+
1 row in set (0.00 sec)

mysql> create table t_char(f char(0));
Query OK, 0 rows affected (0.03 sec)

mysql> insert into t_char values("abc");
ERROR 1406 (22001): Data too long for column 'f' at row 1
mysql> insert into t_char values("");
Query OK, 1 row affected (0.01 sec)

mysql> insert into t_char values("a");
ERROR 1406 (22001): Data too long for column 'f' at row 1
mysql> select * from t_char;
+------+
| f    |
+------+
|      |
+------+

// TODO: handle float and double ??
mysql> create table t_dec(f decimal(10, 11));
ERROR 1427 (42000): For float(M,D), double(M,D) or decimal(M,D), M must be >= D (column 'f').

*/
// TODO: 处理major len 为0的情况
ColumnDescriptionPointer CreateColumnDef( const string& ident,
                                          const Field_def_ptr& fieldDefPtr,
                                          const BasicRelPointer& referencedTable,
                                          const std::vector< std::string >& referencedColumns
                                        )
{
    // parser ensure that type->majorLen and type->minorLen will always be positive integer
    PT_ColumnType_ptr type = fieldDefPtr->column_type;
    ColAttrList col_attr_list = fieldDefPtr->col_attr_list;

    auto desc = std::make_shared<ColumnDescription>();
    desc->column_name = NormalizeIdent(ident);
    desc->column_type = aries_utils::to_lower(type->name);
    if (!type->majorLen.empty()) {
        desc->column_major_len = decode_field_len( ident, type->majorLen );
    }
    if (!type->minorLen.empty()) {
        desc->column_minor_len = decode_field_len( ident, type->minorLen );
    }

    if ( "datetime" == desc->column_type ||
         "timestamp" == desc->column_type ||
         "time" == desc->column_type )
    {
        if (!type->majorLen.empty() && desc->column_major_len > 6 )
        {
            ARIES_EXCEPTION( ER_TOO_BIG_PRECISION, desc->column_major_len, desc->column_name.data(), 6);
        }
    }
    else if ( "decimal" == desc->column_type )
    {
        if ( 0 == desc->column_major_len )
        {
            string msg( "Not support for decimal(0)");
            ARIES_EXCEPTION_SIMPLE(ER_NOT_SUPPORTED_YET, msg.data());
        }
        if (!type->majorLen.empty() && desc->column_major_len > SUPPORTED_MAX_PRECISION )
        {
            ARIES_EXCEPTION( ER_TOO_BIG_PRECISION,
                             desc->column_major_len, desc->column_name.data(),
                             SUPPORTED_MAX_PRECISION );
        }
        if (!type->minorLen.empty() && desc->column_minor_len > SUPPORTED_MAX_SCALE )
        {
            ARIES_EXCEPTION( ER_TOO_BIG_SCALE,
                             desc->column_minor_len,
                             desc->column_name.data(),
                             SUPPORTED_MAX_SCALE );
        }
        if ( desc->column_major_len < desc->column_minor_len )
        {
            ARIES_EXCEPTION( ER_M_BIGGER_THAN_D, ident.data() );
        }
    }
    else if("char" == desc->column_type)
    {
        if(!type->majorLen.empty() && desc->column_major_len == 0)
        {
            string msg( "Not support for char(0)");
            ARIES_EXCEPTION_SIMPLE(ER_NOT_SUPPORTED_YET, msg.data());
        }
        if(!type->majorLen.empty() && (desc->column_major_len < 0 || desc->column_major_len > ARIES_MAX_CHAR_WIDTH))
        {
            string msg( "Column length too big for column '");
            msg.append( desc->column_name ).append( "' (max = ").append( std::to_string( ARIES_MAX_CHAR_WIDTH ) ).append( ")" );
            ARIES_EXCEPTION_SIMPLE( ER_TOO_BIG_FIELDLENGTH, msg.data() );
        }
    }

    desc->type_flags = type->typeFlags;
    desc->InitColumnAttr(col_attr_list);

    /**
     * mysql 在创建外键约束时，如果失败会静默处理，不会报错
     */
    if ( referencedTable && referencedColumns.size() == 1 )
    {
        desc->foreign_key = true;
        desc->fk_table_name = referencedTable->GetID();
        desc->fk_column_name = referencedColumns[ 0 ];
    }
    else
    {
        desc->foreign_key = false;
    }

    return desc;
}

PT_ColumnType_ptr CreateColumnType(const string& typeName, Field_option options) {
    auto type = std::make_shared<PT_ColumnType>();
    type->name = typeName;
    type->typeFlags = static_cast<ulong>(options);
    return type;
}
PT_ColumnType_ptr CreateColumnType(const string& typeName, const string& len, Field_option options) {
    auto type = std::make_shared<PT_ColumnType>();
    type->name = typeName;
    type->majorLen = len;
    type->typeFlags = static_cast<ulong>(options);
    return type;
}
PT_ColumnType_ptr CreateColumnType(const string& typeName, const string& len, const string& decimal, Field_option options) {
    auto type = std::make_shared<PT_ColumnType>();
    type->name = typeName;
    type->majorLen = len;
    type->minorLen = decimal;
    type->typeFlags = static_cast<ulong>(options);
    return type;
}
PT_ColumnType_ptr CreateColumnType(const string& typeName, const std::tuple<std::string, string>& precision, Field_option options) {
    auto type = std::make_shared<PT_ColumnType>();
    type->name = typeName;
    type->majorLen = std::get<0>(precision);
    type->minorLen = std::get<1>(precision);
    type->typeFlags = static_cast<ulong>(options);
    return type;
}
CommandStructurePointer CreateDropTablesStructure(bool dropTemporary, bool dropIfExists, const TABLE_LIST& tableList) {
    auto csp = std::make_shared<CommandStructure>();
    csp->SetCommandType(CommandType::DropTable);
    csp->ifExists = dropIfExists;
    csp->SetTableList(tableList);
    return csp;
}

CommandStructurePointer CreateDropViewsStructure(bool dropTemporary, bool dropIfExists, const TABLE_LIST& tableList) {
    auto csp = std::make_shared<CommandStructure>();
    csp->SetCommandType(CommandType::DropView);
    csp->ifExists = dropIfExists;
    csp->SetTableList(tableList);
    return csp;
}

CommandStructurePointer CreateCreateTableStructure(
    bool temp,
    bool ifNotExists,
    const std::shared_ptr<BasicRel>& table_ident,
    const std::shared_ptr<std::vector<TableElementDescriptionPtr>>& table_element_list,
    const CreateTableOptions& options )
{
    auto csp = std::make_shared<CommandStructure>();
    csp->SetCommandType(CommandType::CreateTable);
    csp->ifNotExists = ifNotExists;
    csp->SetDatabaseName(NormalizeIdent(table_ident->GetDb()));
    csp->SetTableName(NormalizeIdent(table_ident->GetID()));
    csp->SetColumns(table_element_list);
    csp->SetCreateTableOptions( options );
    return csp;
}

CommandStructurePointer CreateCreateViewStructure(const std::shared_ptr<BasicRel>& table_ident,
                                                  const std::shared_ptr<std::vector<TableElementDescriptionPtr>>& table_element_list,
                                                  const AbstractQueryPointer arg_query ) {
    auto csp = std::make_shared<CommandStructure>();
    csp->SetCommandType(CommandType::CreateView);
    csp->SetDatabaseName(NormalizeIdent(table_ident->GetDb()));
    csp->SetTableName(NormalizeIdent(table_ident->GetID()));
    csp->SetColumns(table_element_list);
    csp->SetQuery( arg_query );
    return csp;
}

ShowStructurePtr CreateShowCreateDbStructure(const string& dbName) {
    auto showStructurePointer = std::make_shared<ShowSchemaInfoStructure>();
    showStructurePointer->showCmd = SHOW_CMD::SHOW_CREATE_DB;
    showStructurePointer->id = NormalizeIdent(dbName);
    return showStructurePointer;
}
ShowStructurePtr CreateShowCreateTableStructure(const string& dbName, const string& tableName) {
    auto showStructurePointer = std::make_shared<ShowSchemaInfoStructure>();
    showStructurePointer->showCmd = SHOW_CMD::SHOW_CREATE_TABLE;
    showStructurePointer->tableNameStructureSPtr->dbName = NormalizeIdent(dbName);
    showStructurePointer->tableNameStructureSPtr->tableName = NormalizeIdent(tableName);
    return showStructurePointer;
}
ShowStructurePtr CreateShowFunctionStatusStructure(const string& wild, const Expression& where) {
    auto showStructurePointer = std::make_shared<ShowStructure>();
    showStructurePointer->showCmd = SHOW_CMD::SHOW_FUNC_STATUS;
    showStructurePointer->wild = wild;
    showStructurePointer->where = where;
    return showStructurePointer;
}
// ShowStructurePtr CreateShowFunctionCodeStructure(const std::shared_ptr<BasicRel>& spname ) {
//     auto showStructurePointer = std::make_shared<ShowStructure>();
//     showStructurePointer->showCmd = SHOW_CMD::SHOW_FUNC_CODE;
//     return showStructurePointer;
// }
// ShowStructurePtr CreateShowProcedureCodeStructure(const std::shared_ptr<BasicRel>& spname ) {
//     auto showStructurePointer = std::make_shared<ShowStructure>();
//     showStructurePointer->showCmd = SHOW_CMD::SHOW_PROCEDURE_CODE;
//     return showStructurePointer;
// }

ShowStructurePtr CreateShowProcedureStatusStructure(const string& wild, const Expression& where) {
    auto showStructurePointer = std::make_shared<ShowStructure>();
    showStructurePointer->showCmd = SHOW_CMD::SHOW_PROCEDURE_STATUS;
    showStructurePointer->wild = wild;
    showStructurePointer->where = where;
    return showStructurePointer;
}
ShowStructurePtr CreateShowProcessListStructure(bool full) {
    auto showStructurePointer = std::make_shared<ShowStructure>();
    showStructurePointer->showCmd = SHOW_CMD::SHOW_PROCESS_LIST;
    showStructurePointer->full = full;
    return showStructurePointer;
}

ShowStructurePtr CreateShowIndexStructure(bool extended, const string& db, const std::shared_ptr<BasicRel>& table, const Expression& where) {
    auto showStructurePointer = std::make_shared<ShowSchemaInfoStructure>();
    showStructurePointer->showCmd = SHOW_CMD::SHOW_INDEX;
    showStructurePointer->full = extended;
    string dbName = NormalizeIdent(db);
    showStructurePointer->tableNameStructureSPtr->dbName = dbName;
    if (dbName.empty()) {
        showStructurePointer->tableNameStructureSPtr->dbName = NormalizeIdent(table->GetDb());
    }
    showStructurePointer->tableNameStructureSPtr->tableName = NormalizeIdent(table->GetID());
    showStructurePointer->where = where;
    return showStructurePointer;
}
ShowStructurePtr CreateShowEngineStructure(string& engineName, SHOW_CMD cmd) {
    auto showStructurePointer = std::make_shared<ShowSchemaInfoStructure>();
    showStructurePointer->showCmd = cmd;
    showStructurePointer->id = NormalizeIdent(engineName);
    return showStructurePointer;
}
ShowStructurePtr CreateShowErrorsStructure(LimitStructurePointer limitExpr) {
    auto showStructurePointer = std::make_shared<ShowStructure>();
    showStructurePointer->showCmd = SHOW_CMD::SHOW_ERRORS;
    showStructurePointer->limitExpr = limitExpr;
    return showStructurePointer;
}
ShowStructurePtr CreateShowWarningsStructure(LimitStructurePointer limitExpr) {
    auto showStructurePointer = std::make_shared<ShowStructure>();
    showStructurePointer->showCmd = SHOW_CMD::SHOW_WARNINGS;
    showStructurePointer->limitExpr = limitExpr;
    return showStructurePointer;
}
ShowStructurePtr CreateShowOpenTablesStructure(string& optDb,
                                               const std::string& wild, const Expression& where) {
    auto showStructurePointer = std::make_shared<ShowSchemaInfoStructure>();
    showStructurePointer->showCmd = SHOW_CMD::SHOW_OPEN_TABLES;
    showStructurePointer->tableNameStructureSPtr->dbName = NormalizeIdent(optDb);
    showStructurePointer->wild = wild;
    showStructurePointer->where = where;
    return showStructurePointer;
}
ShowStructurePtr CreateShowEventsStructure(string& optDb,
                                           const std::string& wild, const Expression& where) {
    auto showStructurePointer = std::make_shared<ShowSchemaInfoStructure>();
    showStructurePointer->showCmd = SHOW_CMD::SHOW_EVENTS;
    showStructurePointer->tableNameStructureSPtr->dbName = NormalizeIdent(optDb);
    showStructurePointer->wild = wild;
    showStructurePointer->where = where;
    return showStructurePointer;
}
ShowStructurePtr CreateShowTriggersStructure(bool full, string& optDb,
                                             const std::string& wild, const Expression& where,
                                             const string& wildOrWhereStr) {
    auto showStructurePointer = std::make_shared<ShowSchemaInfoStructure>();
    showStructurePointer->showCmd = SHOW_CMD::SHOW_TRIGGERS;
    showStructurePointer->full = full;
    showStructurePointer->id = NormalizeIdent(optDb);
    showStructurePointer->wild = wild;
    showStructurePointer->wildOrWhereStr = wildOrWhereStr;
    showStructurePointer->where = where;
    return showStructurePointer;
}
ShowStructurePtr CreateShowDatabasesStructure(const std::string& wild,
                                              const Expression& where) {
    auto showStructurePointer = std::make_shared<ShowStructure>();
    showStructurePointer->showCmd = SHOW_CMD::SHOW_DATABASES;
    showStructurePointer->wild = wild;
    showStructurePointer->where = where;
    return showStructurePointer;
}
ShowStructurePtr CreateShowTableStatusStructure(string& optDb,
                                                const std::string& wild, const Expression& where) {
    auto showStructurePointer = std::make_shared<ShowSchemaInfoStructure>();
    showStructurePointer->showCmd = SHOW_CMD::SHOW_TABLE_STATUS;
    showStructurePointer->tableNameStructureSPtr->dbName = NormalizeIdent(optDb);
    showStructurePointer->wild = wild;
    showStructurePointer->where = where;
    return showStructurePointer;
}
ShowStructurePtr CreateShowTablesStructure(Show_cmd_type showCmdType, string& optDb,
                                           const std::string& wild, const Expression& where) {
    auto showStructurePointer = std::make_shared<ShowSchemaInfoStructure>();
    if (Show_cmd_type::FULL_SHOW == showCmdType) {
        showStructurePointer->full = true;
    }
    showStructurePointer->showCmd = SHOW_CMD::SHOW_TABLES;
    showStructurePointer->tableNameStructureSPtr->dbName = NormalizeIdent(optDb);
    showStructurePointer->wild = wild;
    showStructurePointer->where = where;
    return showStructurePointer;
}

ShowStructurePtr CreateShowColumnsStructure(Show_cmd_type showCmdType,
        std::shared_ptr<BasicRel>& rel, const string& db,
        const std::string& wild, const Expression& where) {
    auto showColumnsStructure = std::make_shared<ShowColumnsStructure>();
    if (Show_cmd_type::FULL_SHOW == showCmdType) {
        showColumnsStructure->full = true;
    }
    string dbName = rel->GetDb();
    if (dbName.empty()) {
        dbName = db;
    }
    showColumnsStructure->tableNameStructurePtr->dbName = NormalizeIdent(dbName);
    showColumnsStructure->tableNameStructurePtr->tableName = NormalizeIdent(rel->GetID());
    showColumnsStructure->wild = wild;
    showColumnsStructure->where = where;
    return showColumnsStructure;
}

CommandStructurePointer CreateChangeDbStructure(const string& dbName) {
    auto csp = std::make_shared<CommandStructure>();
    csp->SetCommandType(CommandType::ChangeDatabase);
    csp->SetDatabaseName(NormalizeIdent(dbName));
    return csp;
}
Expression CreateCurrentUserExpression() {
    std::string userName = current_thd->get_user_name();
    auto expr = std::make_shared<CommonBiaodashi>(BiaodashiType::Zifuchuan, userName);
    expr->SetName("current_user()");
    return expr;
}
Expression CreateCurrentDbExpression() {
    std::string dbName = current_thd->db();
    auto expr = std::make_shared<CommonBiaodashi>(BiaodashiType::Zifuchuan, dbName);
    expr->SetName("database()");
    return expr;
}
Expression CreateServerVersionExpression() {
    std::string version = VERSION_INFO_STRING;
    auto expr = std::make_shared<CommonBiaodashi>(BiaodashiType::Zifuchuan, version);
    expr->SetName("version()");
    return expr;
}
Expression CreateConnectionIdExpression() {
    auto expr = std::make_shared<CommonBiaodashi>(BiaodashiType::Zhengshu, (int)current_thd->m_connection_id);
    expr->SetValueType( BiaodashiValueType::INT );
    expr->SetName("connection_id()");
    return expr;
}
ShowStructurePtr CreateShowCharsetStructure(const string& wild, const Expression& where, const string& wildOrWhereStr) {
    auto showStructurePointer = std::make_shared<ShowStructure>();
    showStructurePointer->showCmd = SHOW_CMD::SHOW_CHAR_SET;
    showStructurePointer->wild = wild;
    showStructurePointer->wildOrWhereStr = wildOrWhereStr;
    showStructurePointer->where = where;
    return showStructurePointer;
}
ShowStructurePtr CreateShowCollationStructure(const string& wild, const Expression& where, const string& str) {
    auto showStructurePointer = std::make_shared<ShowStructure>();
    showStructurePointer->showCmd = SHOW_CMD::SHOW_COLLATION;
    showStructurePointer->wild = wild;
    showStructurePointer->wildOrWhereStr = str;
    showStructurePointer->where = where;
    return showStructurePointer;
}
ShowStructurePtr CreateShowVariableStructure(enum enum_var_type varType, const string& wild, const Expression& where, const string& str) {
    auto statusVariableStructure = std::make_shared<ShowStatusVariableStructure>();
    statusVariableStructure->showCmd = SHOW_CMD::SHOW_VARIABLES;
    if (OPT_GLOBAL == varType) {
        statusVariableStructure->global = true;
    } else {
        statusVariableStructure->global = false;
    }
    statusVariableStructure->wild = wild;
    statusVariableStructure->wildOrWhereStr = str;
    statusVariableStructure->where = where;
    return statusVariableStructure;
}
KillStructurePtr CreateKillStructure(int killOpt, const Expression& procId) {
    auto killStructurePtr = std::make_shared<KillStructure>();
    killStructurePtr->killOpt = killOpt;
    killStructurePtr->procIdExpr = procId;
    return killStructurePtr;
}
AdminStmtStructurePtr CreateShutdownStructure()
{
    return std::make_shared<AdminStmtStructure>(ADMIN_STMT::SHUTDOWN);
}
LoadDataStructurePtr CreateLoadDataStructure(enum_filetype loadFileType,
                                             thr_lock_type loadDataLock,
                                             bool optLocal,
                                             const string& file,
                                             On_duplicate optOnDuplicate,
                                             const std::shared_ptr<BasicRel>& tableIdent,
                                             const string& charset,
                                             const Field_separators& fieldSeparators,
                                             const Line_separators& lineSeparators,
                                             ulong ignoreLines)
{
    return std::make_shared<LoadDataStructure>(loadFileType,
                                               loadDataLock,
                                               optLocal,
                                               file,
                                               optOnDuplicate,
                                               tableIdent,
                                               charset,
                                               fieldSeparators,
                                               lineSeparators,
                                               ignoreLines);
}
TransactionStructurePtr CreateEndTxStructure( TX_CMD cmd,
                                           enum_yes_no_unknown chain,
                                           enum_yes_no_unknown release )
{
    /* Don't allow AND CHAIN RELEASE. */
    if ( chain != TVL_YES || release != TVL_YES)
    {
      auto txStructure = make_shared< TransactionStructure >();
      txStructure->txCmd = cmd;
      txStructure->txChain = chain;
      txStructure->txRelease = release;
      return txStructure;
    }
    else
    {
        ARIES_EXCEPTION_SIMPLE(ER_SYNTAX_ERROR, "end transaction with options chain release not allowed");
    }

}
InsertStructurePtr CreateInsertStructure( const BasicRelPointer& tableIdent,
                                          const vector< BiaodashiPointer >& insertColumns,
                                          VALUES_LIST&& insertValuesList,
                                          const vector< BiaodashiPointer >& updateColumns,
                                          const vector< BiaodashiPointer >& updateValues )
{
    vector< BiaodashiPointer > tmp;
    if( !insertValuesList.empty() )
        tmp.reserve( insertValuesList.size() * insertValuesList[0]->size() );

    auto select_part = std::make_shared< SelectPartStructure >();
    for( auto& lineExprs : insertValuesList )
        std::copy( lineExprs->begin(), lineExprs->end(), back_inserter( tmp ) );
    select_part->ResetSelectExprsAndClearAlias( std::move( tmp ) );

    auto select_structure = std::make_shared< SelectStructure >();
    select_structure->init_simple_query( select_part, nullptr, nullptr, nullptr, nullptr );

    auto insertStructure = make_shared< InsertStructure >( tableIdent );
    insertStructure->SetInsertColumns( insertColumns );
    insertStructure->SetInsertColumnValues( std::move( insertValuesList ) );
    insertStructure->SetInsertRowsSelectStructure( select_structure );
    insertStructure->SetOptUpdateColumnValues( updateColumns, updateValues );
    return insertStructure;
}

InsertStructurePtr CreateInsertStructure( const BasicRelPointer& tableIdent,
                                          const vector< BiaodashiPointer >& insertColumns,
                                          const vector<BiaodashiPointer>& insertValues,
                                          const vector< BiaodashiPointer >& updateColumns,
                                          const vector< BiaodashiPointer >& updateValues )
{
    VALUES_LIST insertValuesList = vector< VALUES > ();
    VALUES values = make_shared< vector< BiaodashiPointer > >();
    *values = insertValues;
    insertValuesList.push_back( values );
    return CreateInsertStructure( tableIdent, insertColumns, std::move( insertValuesList ),
                                  updateColumns, updateValues );

}
InsertStructurePtr CreateInsertStructure( const BasicRelPointer& tableIdent,
                                          const vector< BiaodashiPointer >& insertColumns,
                                          const AbstractQueryPointer& selectStructure,
                                          const vector< BiaodashiPointer >& updateColumns,
                                          const vector< BiaodashiPointer >& updateValues )
{
    auto insertStructure = make_shared< InsertStructure >( tableIdent );
    insertStructure->SetInsertColumns( insertColumns );
    insertStructure->SetSelectStructure( selectStructure );
    insertStructure->SetOptUpdateColumnValues( updateColumns, updateValues );
    return insertStructure;
}
UpdateStructurePtr CreateUpdateStructure( const vector< JoinStructurePointer > refTables,
                                          const EXPR_LIST updateColumns,
                                          const EXPR_LIST updateValues,
                                          const BiaodashiPointer& whereExpr,
                                          const OrderbyStructurePointer& orderbyPart,
                                          const LimitStructurePointer& limitPart )
{
    if ( refTables.size() > 1 )
        ThrowNotSupportedException( "update multi tables" );

    auto refTable = refTables[ 0 ];
    if ( refTable->GetRelCount() > 1 )
        ThrowNotSupportedException( "update multi tables" );

    auto from_part = std::make_shared<FromPartStructure>();
    from_part->AddFromItem( refTable );

    auto select_part = std::make_shared< SelectPartStructure >();
    auto rowIdExpr = CreateRowIdExpr();
    select_part->AddSelectExpr( rowIdExpr, nullptr );

    for ( auto& expr : updateColumns )
    {
      select_part->AddSelectExpr( expr, nullptr );
    }

    for ( auto& expr : updateValues )
    {
      select_part->AddSelectExpr( expr, nullptr );
    }

    auto select_structure = std::make_shared<SelectStructure>();
    select_structure->init_simple_query( select_part, from_part, whereExpr, nullptr, nullptr );
    select_structure->SetLimitStructure( limitPart );
    if ( orderbyPart ) {
      select_structure->SetOrderbyPart( orderbyPart );
    }

    auto updateStructure = make_shared< UpdateStructure >();
    updateStructure->SetSelectStructure( select_structure );
    updateStructure->SetTarget( refTable->GetLeadingRel()->GetDb(), refTable->GetLeadingRel()->GetID() );
    return updateStructure;
}

DeleteStructurePtr CreateDeleteStructure( const BasicRelPointer targetTale,
                                          const string& tableAlias,
                                          const BiaodashiPointer& whereExpr,
                                          const OrderbyStructurePointer& orderbyPart,
                                          const LimitStructurePointer& limitPart )
{
    auto deleteStructure = make_shared< DeleteStructure >();
    if ( !tableAlias.empty() )
        targetTale->ResetAlias( tableAlias );
    deleteStructure->AddTargetTable( targetTale );

    auto select_part = std::make_shared< SelectPartStructure >();
    auto rowIdExpr = CreateRowIdExpr();
    select_part->AddSelectExpr( rowIdExpr, nullptr );

    auto from_part = std::make_shared<FromPartStructure>();
    auto join_structure = std::make_shared<JoinStructure>();
    join_structure->SetLeadingRel( targetTale );
    from_part->AddFromItem( join_structure );

    auto select_structure = std::make_shared<SelectStructure>();
    select_structure->init_simple_query( select_part, from_part, whereExpr, nullptr, nullptr );
    select_structure->SetLimitStructure( limitPart );
    if ( orderbyPart ) {
        select_structure->SetOrderbyPart( orderbyPart );
    }
    deleteStructure->SetSelectStructure( select_structure );
    return deleteStructure;
}

DeleteStructurePtr CreateDeleteStructure( const vector< BasicRelPointer >& targetTables,
                                          const vector< JoinStructurePointer > refTables,
                                          const BiaodashiPointer& whereExpr )
{
    if ( refTables.size() > 1 )
        ThrowNotSupportedException( "update multi tables" );

    auto refTable = refTables[ 0 ];
    if ( refTable->GetRelCount() > 1 )
        ThrowNotSupportedException( "update multi tables" );

    auto deleteStructure = make_shared< DeleteStructure >();
    deleteStructure->SetTargetTables( targetTables );

    auto rowIdExpr = CreateRowIdExpr();
    auto select_part = std::make_shared< SelectPartStructure >();
    select_part->AddSelectExpr( rowIdExpr, nullptr );

    auto from_part = std::make_shared<FromPartStructure>();
    from_part->AddFromItem( refTable );

    auto select_structure = std::make_shared<SelectStructure>();
    select_structure->init_simple_query( select_part, from_part, whereExpr, nullptr, nullptr );
    deleteStructure->SetSelectStructure( select_structure );
    return deleteStructure;
}
Expression CreateCastFunctionExpr( Driver& driver, const location& l, Expression& expr, CastType& castType )
{
    int len = 0;
    int associated_length = 0;
    switch ( castType.value_type )
    {
        // case BiaodashiValueType::TEXT:
        // {
        //     if ( !castType.length.empty() )
        //     {
        //         len = decode_field_len( castType.length );
        //         if ( ( errno != 0 ) || ( len > ARIES_MAX_CHAR_WIDTH ) )
        //         {
        //             ARIES_EXCEPTION( ER_TOO_BIG_DISPLAYWIDTH, "cast as char", ARIES_MAX_CHAR_WIDTH );
        //         }
        //         if ( 0 == len )
        //             return CreateStringExpression( "" );
        //     }

        //     break;
        // }
        case BiaodashiValueType::TEXT:
        {
            ThrowNotSupportedException( "cast as char" );
            break;
        }
        case BiaodashiValueType::INT:
        case BiaodashiValueType::LONG_INT:
        case BiaodashiValueType::DATE:
        case BiaodashiValueType::DATE_TIME:
            len = 1;
            break;

        // NOT supported for now
        // case BiaodashiValueType::DOUBLE:
        // case BiaodashiValueType::FLOAT:
        // {
        //     if ( !castType.length.empty() )
        //     {
        //         len = decode_field_len( castType.length );
        //         if ( errno != 0 )
        //         {
        //             string exprStr = driver.get_string_at_location( l );
        //             ARIES_EXCEPTION( ER_TOO_BIG_PRECISION, INT_MAX, exprStr.data(),
        //                              static_cast<ulong>( SUPPORTED_MAX_PRECISION ) );
        //         }
        //         /*
        //         MySQL:

        //         FLOAT[(P)]
        //         If the precision P is not specified, produces a result of type FLOAT. If P is provided and 0 <= < P <=
        //         24, the result is of type FLOAT. If 25 <= P <= 53, the result is of type REAL. If P < 0 or P > 53, an
        //         error is returned. Added in MySQL 8.0.17.
        //         */
        //         if ( len > 24 )
        //         {
        //             castType.value_type = BiaodashiValueType::DOUBLE;
        //         }
        //     }
        //     len = 1;
        //     break;
        // }

        case BiaodashiValueType::DECIMAL:
        {
            if ( !castType.length.empty() )
            {
                string exprStr = driver.get_string_at_location( l );
                len = decode_field_len( exprStr, castType.length );
            }
            if ( !castType.associated_length.empty() )
            {
                string exprStr = driver.get_string_at_location( l );
                associated_length = decode_field_len( exprStr, castType.associated_length );
            }

            // parser already ensured len and associated_length >= 0
            if ( 0 == len )
                len = 10;
            if ( len < associated_length )
            {
                ARIES_EXCEPTION( ER_M_BIGGER_THAN_D, "" );
            }

            if ( len > SUPPORTED_MAX_PRECISION )
            {
                string exprStr = driver.get_string_at_location( l );
                ARIES_EXCEPTION( ER_TOO_BIG_PRECISION, len, exprStr.data(),
                                 static_cast<ulong>( SUPPORTED_MAX_PRECISION ) );
            }
            if ( associated_length > SUPPORTED_MAX_SCALE )
            {
                string exprStr = driver.get_string_at_location( l );
                ARIES_EXCEPTION( ER_TOO_BIG_SCALE, associated_length, exprStr.data(),
                                 static_cast<ulong>( SUPPORTED_MAX_SCALE ) );
            }
            break;
        }
        case BiaodashiValueType::TIME:
        {
            ThrowNotSupportedException( "cast as time" );
            break;
        }

        default:
        {
            string msg( "cast as " + get_name_of_value_type( castType.value_type ) );
            ThrowNotSupportedException( msg.data() );
        }
    }

    auto expression = CreateFunctionExpression( "CAST", expr );
    auto commExpr = (CommonBiaodashi*)(expression.get());
    commExpr->SetValueType( castType.value_type );
    commExpr->SetLength( len );
    commExpr->SetAssociatedLength( associated_length );
    return expression;
}

void CheckInExprFirstArg( Expression& expr )
{
    auto commExpr = (CommonBiaodashi*)(expr.get());
    if ( aries::BiaodashiType::ExprList == commExpr->GetType() )
        ThrowNotSupportedException( "multiple columns for 'IN' and 'NOT IN'" );
}

void ThrowNotSupportedException(const string& msg) {
    ARIES_EXCEPTION(ER_NOT_SUPPORTED_YET, msg.data());
}
} // namespace aries_parser
