#pragma once

#include <string>
#include <memory>
#include <exception>

#include "datatypes/decimal.hxx"
#include "server/mysql/include/mysqld.h"
#include "server/mysql/include/sql_class.h"
#include "frontend/SQLIdent.h"
#include "frontend/CommonBiaodashi.h"
#include "frontend/SelectStructure.h"
#include "frontend/LimitStructure.h"
#include "frontend/VariableStructure.h"
#include "frontend/ShowColumnsStructure.h"
#include "frontend/ShowGlobalInfoStructure.h"
#include "frontend/ShowStatusVariableStructure.h"
#include "frontend/SetStructure.h"
#include "frontend/PreparedStmtStructure.h"
#include "frontend/CommandStructure.h"
#include "frontend/ColumnDescription.h"
#include "frontend/AdminStmtStructure.h"
#include "frontend/LoadDataStructure.h"
#include "frontend/TransactionStructure.h"
#include "frontend/UpdateStructure.h"
#include "frontend/InsertStructure.h"
#include "frontend/DeleteStructure.h"
#include "frontend/AccountMgmtStructure.h"
#include "frontend/PartitionStructure.h"
#include "AriesException.h"
#include "schema/ColumnEntry.h"
#include "location.h"

#ifdef NDEBUG
#   define debug_line()
#else
#   define debug_line()                                    \
do {                                                       \
    printf("here debug line: %s:%d\n", __FILE__, __LINE__);\
} while(0)

#endif

//enable this line if you want see how bison match rules.
//#define YYDEBUG 1

#define YYSTYPE_IS_DECLARED 1

using namespace aries;

namespace aries_parser
{
    int decode_field_len( const string& name, const string& lenStr );

    union NUMERIC_VALUE
    {
        int intValue;
        long long longlongValue;
        aries_acc::Decimal decimal;

        NUMERIC_VALUE()
        {
            intValue = 0;
        }

        NUMERIC_VALUE( int value )
        {
            intValue = value;
        }

        NUMERIC_VALUE( long long value )
        {
            longlongValue = value;
        }
    };

    enum class Field_option
        : ulong
        {
            NONE = 0, UNSIGNED = UNSIGNED_FLAG, ZEROFILL_UNSIGNED = UNSIGNED_FLAG | ZEROFILL_FLAG
    };

    enum class Int_type
        : ulong
        {
            INT = MYSQL_TYPE_LONG, TINYINT = MYSQL_TYPE_TINY, SMALLINT = MYSQL_TYPE_SHORT, MEDIUMINT = MYSQL_TYPE_INT24, BIGINT = MYSQL_TYPE_LONGLONG,
    };

    enum class Numeric_type
        : ulong
        {
            DECIMAL = MYSQL_TYPE_NEWDECIMAL, FLOAT = MYSQL_TYPE_FLOAT, DOUBLE = MYSQL_TYPE_DOUBLE,
    };

    using Expression = aries::BiaodashiPointer;

    struct OrderItem
    {
        Expression order_expr;
        OrderbyDirection direction;
    };

    /*
     STRAIGHT_JOIN
     | HIGH_PRIORITY
     | DISTINCT
     | SQL_SMALL_RESULT
     | SQL_BIG_RESULT
     | SQL_BUFFER_RESULT
     | SQL_CALC_FOUND_ROWS
     | ALL
     */

    enum SelectOption
    {
        UNSET = 0,
        STRAIGHT_JOIN = 1 << 1,
        HIGH_PRIORITY = 1 << 2,
        DISTINCT = 1 << 3,
        SQL_SMALL_RESULT = 1 << 4,
        SQL_BIG_RESULT = 1 << 5,
        SQL_BUFFER_RESULT = 1 << 6,
        SQL_CALC_FOUND_ROWS = 1 << 7,
        SQL_NO_CACHE_SYM = 1 << 8,
        ALL = 1 << 9
    };

    enum class LiteralType
    {
        STRING, INT, LONG_INT, ULONGLONG_INT, DECIMAL, NULL_LITERAL, BOOL_TRUE, BOOL_FALSE
    };

    struct Literal
    {
        LiteralType type;
        std::string str;
    };

    enum class Show_cmd_type
    {
        STANDARD, FULL_SHOW, EXTENDED_SHOW, EXTENDED_FULL_SHOW
    };

    struct PT_option_value_following_option_type
    {
        string varName;
        Expression expression;
    };

    using PT_option_value_following_option_type_ptr = std::shared_ptr<PT_option_value_following_option_type>;

    struct PT_ColumnType
    {
        string name;
        string majorLen;
        string minorLen;
        ulong typeFlags = 0;
    };

    using PT_ColumnType_ptr = std::shared_ptr<PT_ColumnType>;

    struct Field_def
    {
        PT_ColumnType_ptr column_type;
        ColAttrList col_attr_list;
    };

    using Field_def_ptr = std::shared_ptr<Field_def>;

    using TABLE_LIST = std::shared_ptr<vector<std::shared_ptr<BasicRel>>>;

    using EXPR_LIST = vector< BiaodashiPointer >;

    struct WildOrWhere
    {
        WildOrWhere( const std::string& _wild, const Expression& _where, const std::string _str )
                : wild( _wild ), where( _where ), str( _str )
        {

        }
        std::string wild;
        Expression where;
        std::string str; // the original like string or expression content of where
    };
    using WildOrWhere_ptr = std::shared_ptr<WildOrWhere>;

    struct CastType
    {
        BiaodashiValueType value_type;
        bool has_null;
        string length;
        string associated_length;
    };

    using Precision_ptr = std::shared_ptr< pair<string,string> >;
    using Field_separators_ptr = std::shared_ptr< Field_separators >;

    class Driver;
    string get_text( const char* str, size_t len, const u_char sep );
    string NormalizeIdent( string ident );
    void SetExprOrigName( Expression& expr, const std::string& origExpString, const std::string& alias );
    JoinStructurePointer CreateDerivedTableJoinStructure( const AbstractQueryPointer& arg_subquery, std::string& arg_alias, const std::vector< ColumnDescriptionPointer >& columns  );
    Expression LiteralToExpression( const Literal& literal );
    BasicRelPointer CreateTableIdent( bool arg_issubquery, const std::string& arg_db, const std::string& arg_id,
            std::shared_ptr< std::string > arg_alias, const AbstractQueryPointer& arg_subquery );

    Expression CreateBetweenExpr( const Expression &expr1, const Expression &expr2, const Expression &expr3 );
    Expression CreatePreparedStmtParamExpression();
    Expression CreateCaseWhenExpression( const Expression& target, const std::vector< std::tuple< Expression, Expression > >& when_list,
            const Expression& elseExpr );
// Expression CheckCaseWhenExpression(const Expression& caseExpr, Driver* driver, location l);
    Expression CreateComparationExpression( Expression left, const BiaodashiContent& content, Expression right );

    Expression CreateCalcExpression( Expression left, const BiaodashiContent& content, Expression right );

    Expression CreateStringExpression( const std::string& content );

    Expression CreateIntegerExpression( const std::string& content );

    Expression CreateIntegerExpression( const Literal& literal );

    Expression CreateULLExpression( const std::string& content );

    Expression CreateDecimalExpression( const std::string& content );

    Expression NegateDecimalExpresstion( const BiaodashiContent& content );

    Expression CreateLogicExpression( LogicType type, Expression left, Expression right );

    Expression CreateFunctionExpression( const std::string& name, Expression arg );
    Expression CreateFunctionExpression( const std::string& name, std::vector< Expression >& args );

// `a.b`

//Expression CreateNumericExpression(const BiaodashiContent& content);

    Expression CreateIdentExpression( const SQLIdentPtr & ident );

    Expression CreateNullExpression();

    Expression CreateBoolExpression( bool value );

    Expression GenerateNotExpression( Expression origin );

    Expression CreateDistinctFunctionExpression( const std::string& name, Expression arg );
    Expression CreateDistinctFunctionExpression( const std::string& name, std::vector< Expression >& args );

    Expression CreateExpressionFromVariable( const VariableStructurePtr& variableStructurePtr );
    Expression CreateGroupConcatExpression();

    VariableStructurePtr CreateUserVariableStructure( const std::string& name );
    VariableStructurePtr CreateSysVariableStructure( enum_var_type varScope, const std::string& name );

    PrepareSrcPtr CreatePrepareSrc( bool isVarRef, const string& stmtCode );
    PreparedStmtStructurePtr CreatePrepareStmt( const string& stmtName, const PrepareSrcPtr& prepareSrcPtr );
    PreparedStmtStructurePtr CreateDeallocateStmt( const string& stmtName );
    PreparedStmtStructurePtr CreateExecuteStmt( const string& stmtName, const std::vector< string >& varNames );

    SetStructurePtr CreateSetUserVarStructure( const string& name, const Expression& expression );
    SetStructurePtr CreateSetSysVarStructure( bool global, const string& name, const string& fullName, const Expression& expression );
    SetStructurePtr CreateSetSysVarStructure( const enum_var_type varType, const string& name, const Expression& expression );
    SetStructurePtr CreateSetSysVarStructure( const enum_var_type varScope, const PT_option_value_following_option_type_ptr& optionValuePtr );
    void ReviseSetStructuresHead( enum_var_type headVarScope, const std::shared_ptr< std::vector< SetStructurePtr > >& setStructures );
    std::shared_ptr< std::vector< SetStructurePtr > > AppendOptionSetStructures( const SetStructurePtr& item,
            const std::shared_ptr< std::vector< SetStructurePtr > >& optionListPtr );

    CommandStructurePointer CreateDropDatabaseStructure( bool ifExists, const string &name );
    CommandStructurePointer CreateCreateDbStructure( bool ifNotExists, const string& name );
    CommandStructurePointer CreateDropTablesStructure( bool dropTemporary, bool dropIfExists, const TABLE_LIST& tableList );
    CommandStructurePointer CreateDropViewsStructure( bool dropTemporary, bool dropIfExists, const TABLE_LIST& tableList );
    CommandStructurePointer CreateCreateTableStructure( bool temp,
                                                       bool ifNotExists,
                                                       const std::shared_ptr< BasicRel >& table_ident,
                                                       const std::shared_ptr< std::vector< TableElementDescriptionPtr > >& table_element_list,
                                                       const CreateTableOptions& options );

    CommandStructurePointer CreateCreateViewStructure( const std::shared_ptr< BasicRel >& table_ident,
        const std::shared_ptr< std::vector< TableElementDescriptionPtr > >& table_element_list,
        const AbstractQueryPointer arg_query );
    CommandStructurePointer CreateChangeDbStructure( const string& dbName );
    AccountMgmtStructureSPtr CreateAccountMgmtStructure( CommandType cmdType,
                                                         bool ifNotExists,
                                                         shared_ptr< vector< AccountSPtr > >& userList );
    SetStructurePtr CreateSetPasswordStructure( std::string& user, const string& password );

    PT_column_attr_base_ptr CreateDefaultValueAttr( const Expression& expr );
    Field_def_ptr CreateFieldDef( const PT_ColumnType_ptr& column_type, const ColAttrList& col_attr_list );
    ColumnDescriptionPointer CreateColumnDef( const string& ident, const Field_def_ptr& fieldDefPtr );

    ColumnDescriptionPointer CreateColumnDef( const string& ident, const Field_def_ptr& fieldDefPtr, const BasicRelPointer& referencedTable,
            const std::vector< std::string >& referencedColumns );

    PT_ColumnType_ptr CreateColumnType( const string& typeName, Field_option options = Field_option::NONE );
    PT_ColumnType_ptr CreateColumnType( const string& typeName, const string& len, Field_option options = Field_option::NONE );
    PT_ColumnType_ptr CreateColumnType( const string& typeName, const string& len, const string& decimal, Field_option options = Field_option::NONE );
    PT_ColumnType_ptr CreateColumnType( const string& typeName, const std::tuple< std::string, string >& precision, Field_option options =
            Field_option::NONE );

    ShowStructurePtr CreateShowCreateDbStructure( const string& dbName );
    ShowStructurePtr CreateShowCreateTableStructure( const string& dbName, const string& tableName );
// ShowStructurePtr CreateShowFunctionCodeStructure(const std::shared_ptr<BasicRel>& spname );
// ShowStructurePtr CreateShowProcedureCodeStructure(const std::shared_ptr<BasicRel>& spname );
    ShowStructurePtr CreateShowFunctionStatusStructure( const string& wild, const Expression& where );
    ShowStructurePtr CreateShowProcedureStatusStructure( const string& wild, const Expression& where );
    ShowStructurePtr CreateShowProcessListStructure( bool full );
    ShowStructurePtr CreateShowIndexStructure( bool extended, const string& db, const std::shared_ptr< BasicRel >& table, const Expression& where );

    ShowStructurePtr CreateShowErrorsStructure( LimitStructurePointer limitExpr );
    ShowStructurePtr CreateShowWarningsStructure( LimitStructurePointer limitExpr );
    ShowStructurePtr CreateShowEngineStructure( string& engineName, SHOW_CMD cmd );
    ShowStructurePtr CreateShowOpenTablesStructure( string& optDb, const std::string& wild, const Expression& where );
    ShowStructurePtr CreateShowEventsStructure( string& optDb, const std::string& wild, const Expression& where );
    ShowStructurePtr CreateShowTriggersStructure( bool full, string& optDb, const std::string& wild, const Expression& where,
            const string& wildOrWhereStr );
    ShowStructurePtr CreateShowTableStatusStructure( string& optDb, const std::string& wild, const Expression& where );
    ShowStructurePtr CreateShowDatabasesStructure( const std::string& wild, const Expression& where );
    ShowStructurePtr CreateShowTablesStructure( Show_cmd_type showCmdType, string& optDb, const std::string& wild, const Expression& where );
    ShowStructurePtr CreateShowColumnsStructure( Show_cmd_type showCmdType, std::shared_ptr< BasicRel >& rel, const string& db,
            const std::string& wild, const Expression& where );
    Expression CreateCurrentUserExpression();
    Expression CreateCurrentDbExpression();
    Expression CreateServerVersionExpression();
    Expression CreateConnectionIdExpression();
    ShowStructurePtr CreateShowCharsetStructure( const string& wild, const Expression& where, const string& str );
    ShowStructurePtr CreateShowCollationStructure( const string& wild, const Expression& where, const string& str );
    ShowStructurePtr CreateShowVariableStructure( enum enum_var_type varType, const string& wild, const Expression& where, const string& str );

    KillStructurePtr CreateKillStructure( int killOpt, const Expression& procId );
    AdminStmtStructurePtr CreateShutdownStructure();
    LoadDataStructurePtr CreateLoadDataStructure( enum_filetype loadFileType, thr_lock_type loadDataLock, bool optLocal, const string& file,
            On_duplicate optOnDuplicate, const std::shared_ptr< BasicRel >& tableIdent, const string& charset,
            const Field_separators& fieldSeparators, const Line_separators& lineSeparators, ulong ignoreLines );
    TransactionStructurePtr CreateEndTxStructure( TX_CMD cmd, enum_yes_no_unknown chain, enum_yes_no_unknown release );

    InsertStructurePtr CreateInsertStructure( const BasicRelPointer& tableIdent, const vector< BiaodashiPointer >& insertColumns,
            VALUES_LIST&& insertValuesList, const vector< BiaodashiPointer >& updateColumns, const vector< BiaodashiPointer >& updateValues );
    InsertStructurePtr CreateInsertStructure( const BasicRelPointer& tableIdent, const vector< BiaodashiPointer >& insertColumns,
            const vector< BiaodashiPointer >& insertValues, const vector< BiaodashiPointer >& updateColumns,
            const vector< BiaodashiPointer >& updateValues );
    InsertStructurePtr CreateInsertStructure( const BasicRelPointer& tableIdent, const vector< BiaodashiPointer >& insertColumns,
            const AbstractQueryPointer& selectStructure, const vector< BiaodashiPointer >& updateColumns,
            const vector< BiaodashiPointer >& updateValues );
    UpdateStructurePtr CreateUpdateStructure( const vector< JoinStructurePointer > refTables, const EXPR_LIST updateColumns,
            const EXPR_LIST updateValues, const BiaodashiPointer& whereExpr, const OrderbyStructurePointer& orderbyPart,
            const LimitStructurePointer& limitPart );

    DeleteStructurePtr CreateDeleteStructure( const BasicRelPointer targetTale, const string& tableAlias, const BiaodashiPointer& whereExpr,
            const OrderbyStructurePointer& orderbyPart, const LimitStructurePointer& limitPart );
    DeleteStructurePtr CreateDeleteStructure( const vector< BasicRelPointer >& targetTables, const vector< JoinStructurePointer > refTables,
            const BiaodashiPointer& whereExpr );
    void CheckUpdateColumns( const vector< BiaodashiPointer > cols );

    Expression CreateExprListExpression();
    Expression CreateCastFunctionExpr( Driver& driver, const location& l, Expression& expr, CastType& castType );
    void CheckInExprFirstArg( Expression& expr );

    PartitionStructureSPtr CreatePartitionStructure(
        const PartTypeDef &partTypeDef,
        uint16_t partCount );

} // namespace aries_parser
