#include "CommandExecutor.h"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <server/mysql/include/mysqld.h>

#include "SQLExecutor.h"
#include "SchemaBuilder.h"
#include "AriesEngineWrapper/AriesMemTable.h"
//#include "../AriesEngineV2/AriesCommandExecutor.h"
#include "schema/SchemaManager.h"
#include "Compression/dict/AriesDictManager.h"
#include "frontend/ViewManager.h"
#include "server/mysql/include/sql_class.h"

#include "server/mysql/include/my_config.h"
#include "server/mysql/include/derror.h"
#include "datatypes/AriesDatetimeTrans.h"
#include "AriesEngine/transaction/AriesMvccTableManager.h"
#include "AriesEngine/transaction/AriesInitialTableManager.h"
#include "server/Configuration.h"
#include "frontend/AccountMgmtStructure.h"

#include "server/mysql/include/password.h"
#include "server/mysql/include/sql_const.h"
#include "server/mysql/include/sql_authentication.h"
#include "server/mysql/include/sql_auth_cache.h"
#include "utils/string_util.h"
#include "schema/TablePartition.h"
#include "AriesEngineWrapper/AriesExprBridge.h"

bool mysql_change_db(THD *thd, const string &new_db_name, bool force_switch);
namespace aries {

CommandExecutor::CommandExecutor() {
}

/**
  Auxiliary function for constructing a  user list string.
  This function is used for error reporting and logging.
 
  @param thd     Thread context
  @param str     A String to store the user list.
  @param user    A LEX_USER which will be appended into user list.
  @param comma   If TRUE, append a ',' before the the user.
  @param ident   If TRUE, append ' IDENTIFIED BY/WITH...' after the user,
                 if the given user has credentials set with 'IDENTIFIED BY/WITH'
 */
void append_user(THD *thd, std::string *str, Account *user, bool comma= true,
                 bool ident= false)
{
  // String from_user(user->user.str, user->user.length, system_charset_info);
  // String from_plugin(user->plugin.str, user->plugin.length, system_charset_info);
  // String from_auth(user->auth.str, user->auth.length, system_charset_info);
  // String from_host(user->host.str, user->host.length, system_charset_info);

  // if (comma)
  //   str->append(',');
  // append_query_string(thd, system_charset_info, &from_user, str);
  // str->append(STRING_WITH_LEN("@"));
  // append_query_string(thd, system_charset_info, &from_host, str);

  // if (ident)
  // {
  //   if (user->plugin.str && (user->plugin.length > 0) &&
  //       memcmp(user->plugin.str, native_password_plugin_name.str,
  //              user->plugin.length))
  //   {
  //     /** 
  //         The plugin identifier is allowed to be specified,
  //         both with and without quote marks. We log it with
  //         quotes always.
  //       */
  //     str->append(STRING_WITH_LEN(" IDENTIFIED WITH "));
  //     append_query_string(thd, system_charset_info, &from_plugin, str);

  //     if (user->auth.str && (user->auth.length > 0))
  //     {
  //       str->append(STRING_WITH_LEN(" AS "));
  //       append_query_string(thd, system_charset_info, &from_auth, str);
  //     }
  //   }
  //   else if (user->auth.str)
  //   {
  //     str->append(STRING_WITH_LEN(" IDENTIFIED BY PASSWORD '"));
  //     if (user->uses_identified_by_password_clause ||
  //         user->uses_authentication_string_clause)
  //     {
  //       str->append(user->auth.str, user->auth.length);
  //       str->append("'");
  //     }
  //     else
  //     {
  //       /*
  //         Password algorithm is chosen based on old_passwords variable or
  //         TODO the new password_algorithm variable.
  //         It is assumed that the variable hasn't changed since parsing.
  //       */
  //       if (thd->variables.old_passwords == 0)
  //       {
  //         /*
  //           my_make_scrambled_password_sha1() requires a target buffer size of
  //           SCRAMBLED_PASSWORD_CHAR_LENGTH + 1.
  //           The extra character is for the probably originate from either '\0'
  //           or the initial '*' character.
  //         */
  //         char tmp[SCRAMBLED_PASSWORD_CHAR_LENGTH + 1];
  //         my_make_scrambled_password_sha1(tmp, user->auth.str,
  //                                         user->auth.length);
  //         str->append(tmp);
  //       }
  //       else
  //       {
  //         /*
  //           With old_passwords == 2 the scrambled password will be binary.
  //         */
  //         DBUG_ASSERT(thd->variables.old_passwords = 2);
  //         str->append("<secret>");
  //       }
  //       str->append("'");
  //     }
  //   }
  // }
}

void CommandExecutor::ExecuteCreateUser( AccountMgmtStructure *arg_command_p )
{
    std::string sql = "insert into mysql.user( Host, User, authentication_string ) values ";
    int n = 0;
    for ( const auto& acct : arg_command_p->GetAccounts() )
    {
        if ( FindAclUser( acct->m_user, acct->m_host ) )
        {
            string errMsg = format_mysql_err_msg( ER_USER_ALREADY_EXISTS, acct->m_user.data() );
            if ( !arg_command_p->ifNotExists )
                ARIES_EXCEPTION_SIMPLE( ER_USER_ALREADY_EXISTS, errMsg.data() );
            else
            {
                LOG( WARNING ) << errMsg;
                continue;
            }
        }
        ++n;

        char outbuf[MAX_FIELD_WIDTH]= {0};
        unsigned int buflen= MAX_FIELD_WIDTH;

        if ( generate_native_password( outbuf, &buflen, acct->m_authStr.data(), acct->m_authStr.length() ) )
            ARIES_EXCEPTION( ER_CANNOT_USER, "CREATE USER", acct->m_user.data()  ) ;

        acct->m_authStr = outbuf;

        sql.append( "( " );
        sql.append( "'" ).append( acct->m_host ).append( "', " );
        sql.append( "'" ).append( acct->m_user ).append( "', " );
        sql.append( "'" ).append( outbuf ).append( "' " );
        sql.append( ")," );
    }

    if ( 0 == n )
        return;

    sql = aries_utils::rtrim( sql, "," );

    auto sqlResult = SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    if ( !sqlResult->IsSuccess() )
    {
        const auto& errMsg = sqlResult->GetErrorMessage();
        ARIES_EXCEPTION_SIMPLE( sqlResult->GetErrorCode(), errMsg.data() );
    }

    for ( const auto& acct : arg_command_p->GetAccounts() )
        AddAclUser( acct->m_user, acct->m_host, acct->m_authStr );
}

void CommandExecutor::ExecuteSetPassword( std::string& user, std::string& host, std::string password )
{
    if ( !FindAclUser( user, host ) )
    {
        ARIES_EXCEPTION( ER_USER_DOES_NOT_EXIST, user.data() );
    }
    char outbuf[MAX_FIELD_WIDTH]= {0};
    unsigned int buflen= MAX_FIELD_WIDTH;

    if ( generate_native_password( outbuf, &buflen, password.data(), password.length() ) )
        ARIES_EXCEPTION( ER_CANNOT_USER, "SET PASSWORD", user.data()  ) ;

    std::string sql = "update mysql.user set authentication_string = '";
    sql.append( outbuf ).append( "' where User = '" ).append( user ).append( "'" );

    THD* thd = current_thd;
    bool oldIsDDL = false;
    if ( thd->m_tx )
    {
        oldIsDDL = thd->m_tx->isDDL();
        if ( !oldIsDDL )
            thd->m_tx->SetIsDDL( true ); 
    }

    auto sqlResult = SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    if ( thd->m_tx )
    {
        if ( !oldIsDDL )
            thd->m_tx->SetIsDDL( oldIsDDL ); 
    }

    if ( !sqlResult->IsSuccess() )
    {
        const auto& errMsg = sqlResult->GetErrorMessage();
        ARIES_EXCEPTION_SIMPLE( sqlResult->GetErrorCode(), errMsg.data() );
    }
    UpdateAclUser( user, host, outbuf );
}
void CommandExecutor::ExecuteDropUser(AccountMgmtStructure *arg_command_p)
{
    int n = 0;
    std::string sql = "delete from mysql.user where User in ( ";
    for ( const auto& acct : arg_command_p->GetAccounts() )
    {
        if ( "rateup" == acct->m_user )
        {
            ARIES_EXCEPTION( ER_CANNOT_USER, "DROP USER", "rateup" );
        }

        if ( !FindAclUser( acct->m_user, acct->m_host ) )
        {
            string errMsg = format_mysql_err_msg( ER_USER_ALREADY_EXISTS, acct->m_user.data() );
            if ( !arg_command_p->ifExists )
                ARIES_EXCEPTION_SIMPLE( ER_USER_DOES_NOT_EXIST, errMsg.data() );
            else
            {
                LOG( WARNING ) << errMsg;
                continue;
            }
        }

        ++n;

        sql.append( "'" ).append( acct->m_user ).append( "'," );
    }

    if ( 0 == n )
        return;

    sql = aries_utils::rtrim( sql, "," );
    sql.append( " )" );

    auto sqlResult = SQLExecutor::GetInstance()->ExecuteSQL( sql, "" );
    if ( !sqlResult->IsSuccess() )
    {
        const auto& errMsg = sqlResult->GetErrorMessage();
        ARIES_EXCEPTION_SIMPLE( sqlResult->GetErrorCode(), errMsg.data() );
    }

    for ( const auto& acct : arg_command_p->GetAccounts() )
        RemoveAclUser( acct->m_user, acct->m_host );
}

std::string CommandExecutor::ExecuteCreateDatabase(CommandStructure *arg_command_p) {
    auto database_name = arg_command_p->GetDatabaseName();
    std::string db_path = Configuartion::GetInstance().GetDataDirectory( database_name );

    auto schema = schema::SchemaManager::GetInstance()->GetSchema();
    if (schema == nullptr) {
        ARIES_EXCEPTION_SIMPLE(ER_UNKNOWN_ERROR, "Create Database ERROR: schema not found");
        return std::to_string(-1) + "Create Database ERROR: schema not found";
    }

    auto database = schema->GetDatabaseByName(database_name);
    if (database != nullptr) {
        if (!arg_command_p->ifNotExists) {
            ARIES_EXCEPTION( ER_DB_CREATE_EXISTS, database_name.data() );
        }
        LOG(INFO) << "Create Database: DB [" + database_name + "] already exists!";
        return std::to_string(-1) + "Create Database ERROR: DB [" + database_name + "] already exists!";
    }

    boost::filesystem::path p1(db_path);
    if (boost::filesystem::exists(p1)) {
        string errMsg( "Can't create database '" );
        errMsg.append( database_name ).append("'; directory already exsits");
        ARIES_EXCEPTION_SIMPLE( ER_DB_CREATE_EXISTS, errMsg );
    }

    database = std::make_shared<schema::DatabaseEntry>(database_name);
    database->SetCreateString(arg_command_p->GetCommandString());

    if (aries::schema::INFORMATION_SCHEMA == database_name) {
        // schema->InitInfoSchemaBaseTables();
    } else {
        schema->InsertDbSchema(database_name);
    }

    schema->AddDatabase(database);
    boost::filesystem::create_directory(p1);

    return "";
}


std::string CommandExecutor::ExecuteDropDatabase(CommandStructure *arg_command_p, const string& currentDbName) {
    auto schema = schema::SchemaManager::GetInstance()->GetSchema();
    std::shared_ptr<schema::DatabaseEntry> database = nullptr;

    string dbName = arg_command_p->GetDatabaseName();
    if (aries::schema::IsSysDb(dbName)) {
        string msg("Access denied to database ");
        msg.append(dbName);
        ARIES_EXCEPTION_SIMPLE(ER_DBACCESS_DENIED_ERROR, msg);
        return "";
    }
    if ( !schema ) {
        ARIES_EXCEPTION_SIMPLE(ER_UNKNOWN_ERROR, "Drop Database ERROR: schema  NOT exists!");
        return std::to_string(-1) + "Drop Database ERROR: schema  NOT exists!";
    } else {
        database = schema->GetDatabaseByName(dbName);
    }

    if ( !database ) {
        if (!arg_command_p->ifExists) {
            ARIES_EXCEPTION( ER_DB_DROP_EXISTS, dbName.data() );
        }
        string msg = "Drop Database: database " + dbName +
                     " NOT exists!";
        LOG(INFO) << msg;
        return msg;
    }

    for (auto entry : database->GetTables()) {
        entry.second->OnDrop( dbName );
    }
    database->ClearTable();
    schema->RemoveDatabase(database);
    if (dbName == currentDbName) {
        if (current_thd) {
            string empty = "";
            current_thd->set_db(empty);
        }
    }
    schema->DeleteDbSchema(dbName);

    std::string db_path = Configuartion::GetInstance().GetDataDirectory( dbName );
    boost::filesystem::path p(db_path);
    boost::filesystem::remove_all(p);
    return "Drop database success";
}

/*
MySQL 5.7 benavior

unique key constraint name:

mysql> create table t1(f1 int, f2 int, unique key uk1(f1), unique key uk1(f1));
ERROR 1061 (42000): Duplicate key name 'uk1'

mysql> create table t1(f1 int, f2 int, constraint uk1 unique key (f1), constraint uk1 unique key (f1));
ERROR 1061 (42000): Duplicate key name 'uk1'

mysql> create table t5(f1 int, f2 int, constraint uk1 unique key (f1), unique key uk1(f2));
ERROR 1061 (42000): Duplicate key name 'uk1'

mysql> create table t1(f1 int, f2 int, constraint uk1 unique key (f1), constraint uk2 unique key (f1));
Query OK, 0 rows affected, 1 warning (0.02 sec)

mysql> create table t2(f1 int, f2 int,  unique key (f1), unique key (f1));
Query OK, 0 rows affected, 1 warning (0.03 sec)

mysql> create table t3(f1 int, f2 int,  unique key uk1 (f1), unique key uk2 (f1));
Query OK, 0 rows affected, 1 warning (0.03 sec)

mysql> create table t4(f1 int, f2 int, constraint uk1 unique key uk1(f1));
Query OK, 0 rows affected (0.02 sec)

mysql> select * from information_schema.table_constraints where TABLE_SCHEMA='test';
+--------------------+-------------------+-----------------+--------------+------------+-----------------+
| CONSTRAINT_CATALOG | CONSTRAINT_SCHEMA | CONSTRAINT_NAME | TABLE_SCHEMA | TABLE_NAME | CONSTRAINT_TYPE |
+--------------------+-------------------+-----------------+--------------+------------+-----------------+
| def                | test              | uk1             | test         | t1         | UNIQUE          |
| def                | test              | uk2             | test         | t1         | UNIQUE          |
| def                | test              | f1              | test         | t2         | UNIQUE          |
| def                | test              | f1_2            | test         | t2         | UNIQUE          |
| def                | test              | uk1             | test         | t3         | UNIQUE          |
| def                | test              | uk2             | test         | t3         | UNIQUE          |
| def                | test              | uk1             | test         | t4         | UNIQUE          |
+--------------------+-------------------+-----------------+--------------+------------+-----------------+

foreign key:

mysql> create table t8( f1 int primary key, f2 int, f3 int, constraint c1 unique key (f2), constraint c1 foreign key (f3) references t8(f1));
ERROR 1061 (42000): Duplicate key name 'c1'

mysql> create table t8( f1 int primary key, f2 int, f3 int, constraint c1 unique key (f2), foreign key c1 (f3) references t8(f1));
ERROR 1061 (42000): Duplicate key name 'c1'

*/
TableEntrySPtr CommandToTableEntry( CommandStructure *arg_command_p,
                                    const schema::DatabaseEntrySPtr& database )
{
    std::string tableName = arg_command_p->GetTableName();
    auto table_to_create = std::make_shared<schema::TableEntry>(database,
                                                                tableName,
                                                                schema::ENGINE_RATEUP);
    table_to_create->SetCreateString(arg_command_p->GetCommandString());

    auto columns = arg_command_p->GetColumns();
    bool see_primary_key = false;
    int colIdx = 0;
    std::unordered_map<string, bool> columnMap;
    TableConstraintSPtr primary_constraint = nullptr;
    std::vector< TableConstraintSPtr > foreigns;
    std::vector< TableConstraintSPtr > uniqkeys;

    int foreign_constraint_index = 1;
    int uniq_constraint_index = 1;
    for ( auto& colDesc : *columns ) {
        if (!colDesc->IsColumnDesc()) {
            auto constraint = std::dynamic_pointer_cast< PT_table_key_constraint_def >( colDesc );
            if ( constraint )
            {
                auto constraint_name = constraint->GetName();
                auto index_name = constraint->GetIndexName();
                if ( constraint_name.empty() )
                    constraint_name = index_name;

                TableConstraintSPtr foreign = nullptr;
                TableConstraintSPtr uniqkey = nullptr;

                if ( !constraint_name.empty() )
                {
                    for ( const auto& key : foreigns )
                    {
                        if ( constraint_name == key->name )
                        {
                             ARIES_EXCEPTION( ER_FK_DUP_NAME, constraint_name.c_str() );
                        }
                    }
                    for ( const auto& key : uniqkeys )
                    {
                        if ( constraint_name == key->name )
                        {
                            ARIES_EXCEPTION( ER_DUP_KEYNAME, constraint_name.c_str() );
                        }
                    }
                }
                switch ( constraint->GetType() )
                {
                    case KEYTYPE_FOREIGN:
                    {
                        foreign = std::make_shared< TableConstraint >();
                        if ( constraint_name.empty() )
                        {
                            constraint_name = tableName + "_ibfk_" + std::to_string( foreign_constraint_index );
                            foreign_constraint_index ++;
                        }
                        foreign->name = constraint_name;
                        foreign->type = TableConstraintType::ForeignKey;
                        break;
                    }
                    case KEYTYPE_PRIMARY:
                        if ( !see_primary_key )
                        {
                            see_primary_key = true;
                            primary_constraint = std::make_shared< TableConstraint >();
                            primary_constraint->name = "PRIMARY";
                            primary_constraint->type = TableConstraintType::PrimaryKey;
                        }
                        else
                        {
                            ARIES_EXCEPTION( ER_MULTIPLE_PRI_KEY );
                        }
                        break;
                    case KEYTYPE_UNIQUE:
                    {
                        uniqkey = std::make_shared< TableConstraint >();
                        if ( constraint_name.empty() )
                        {
                            constraint_name = tableName + "_ibuk_" + std::to_string( uniq_constraint_index++ );
                        }
                        uniqkey->name = constraint_name;
                        uniqkey->type = TableConstraintType::UniqueKey;
                        break;
                    }
                    default:
                        ARIES_EXCEPTION_SIMPLE( ER_NOT_SUPPORTED_YET, "key type " + std::to_string( ( int )constraint->GetType() ) );
                }

                for ( const auto& key : constraint->GetKeys() )
                {
                    switch ( constraint->GetType() )
                    {
                        case KEYTYPE_PRIMARY:
                            if ( std::find( primary_constraint->keys.cbegin(), primary_constraint->keys.cend(), key ) != primary_constraint->keys.cend() )
                            {
                                ARIES_EXCEPTION( ER_DUP_FIELDNAME, key.c_str() );
                            }
                            primary_constraint->keys.emplace_back( key );
                            break;
                    
                        case KEYTYPE_FOREIGN:
                            if ( std::find( foreign->keys.cbegin(), foreign->keys.cend(), key ) != foreign->keys.cend() )
                            {
                                ARIES_EXCEPTION( ER_DUP_FIELDNAME, key.c_str() );
                            }
                            foreign->keys.emplace_back( key );

                            break;
                        case KEYTYPE_UNIQUE:
                            if ( std::find( uniqkey->keys.cbegin(), uniqkey->keys.cend(), key ) != uniqkey->keys.cend() )
                            {
                                ARIES_EXCEPTION( ER_DUP_FIELDNAME, key.c_str() );
                            }
                            uniqkey->keys.emplace_back( key );
                            break;
                        default:
                            break;
                    }
                }

                switch ( constraint->GetType() )
                {
                    case KEYTYPE_FOREIGN:
                    {
                        const auto& foreign_keys = constraint->GetForeignKeys();
                        if ( foreign_keys.empty() )
                        {
                            ARIES_EXCEPTION( ER_CANNOT_ADD_FOREIGN );
                        }

                        foreign->referencedTable = constraint->GetReferencedTable()->GetID();
                        foreign->referencedSchema = constraint->GetReferencedTable()->GetDb();
                        if ( foreign->referencedSchema.empty() )
                        {
                            foreign->referencedSchema = database->GetName();
                        }

                        for ( const auto& key :foreign_keys )
                        {
                            foreign->referencedKeys.emplace_back( key );
                        }

                        foreigns.emplace_back( foreign );

                        break;
                    }

                    case KEYTYPE_UNIQUE:
                    {
                        uniqkeys.emplace_back( uniqkey );
                        break;
                    }
                
                    default:
                        break;
                }
            }
            continue;
        }

        ColumnDescriptionPointer a_col = std::dynamic_pointer_cast<ColumnDescription>(colDesc);
        if ( columnMap.end() != columnMap.find( a_col->column_name ) )
            ARIES_EXCEPTION( ER_DUP_FIELDNAME, a_col->column_name.data() );
        columnMap[ a_col->column_name ] = true;

        //check type
        std::string col_type_str = boost::algorithm::to_lower_copy(a_col->column_type);
        schema::ColumnType column_type = aries::schema::ToColumnType( col_type_str );

        if ( EncodeType::DICT == a_col->encode_type )
        {
            if ( !ColumnEntry::IsStringType( column_type ) )
            {
                ThrowNotSupportedException("dict encoding for non-char data type");
            }
        }

        //check primary key setting
        if (a_col->primary_key) {
            if ( a_col->explicit_nullable || a_col->explicit_default_null )
                ARIES_EXCEPTION_SIMPLE( ER_PRIMARY_CANT_HAVE_NULL, "All parts of a PRIMARY KEY must be NOT NULL" );
            if (!see_primary_key) {
                see_primary_key = true;
            } else {
                ARIES_EXCEPTION(ER_MULTIPLE_PRI_KEY);
            }

            primary_constraint = std::make_shared< TableConstraint >();
            primary_constraint->name = "PRIMARY";
            primary_constraint->type = TableConstraintType::PrimaryKey;
            primary_constraint->keys.emplace_back( a_col->column_name );
        }
        else if ( a_col->unique_key )
        {
            auto uniqkey = std::make_shared< TableConstraint >();
            uniqkey->name = tableName + "_ibuk_" + std::to_string( uniq_constraint_index++ );
            uniqkey->type = TableConstraintType::UniqueKey;
            uniqkey->keys.emplace_back( a_col->column_name );

            uniqkeys.emplace_back( uniqkey );
        }

        auto column = schema::ColumnEntry::MakeColumnEntry(
                a_col->column_name, column_type, colIdx,
                a_col->primary_key, a_col->unique_key, a_col->multi_key, a_col->foreign_key,
                a_col->is_unsigned, !a_col->not_null, a_col->has_default, a_col->default_value,
                a_col->column_major_len, a_col->column_major_len,
                a_col->column_major_len, a_col->column_minor_len, a_col->column_major_len,
                aries::schema::DEFAULT_CHARSET_NAME,
                aries::schema::DEFAULT_UTF8_COLLATION,
                a_col->explicit_nullable,
                a_col->explicit_default_null );

        if ( EncodeType::DICT == a_col->encode_type )
        {
            schema::ColumnType indexDataType = aries::schema::ToColumnType( a_col->encode_index_data_type );
            auto dict = AriesDictManager::GetInstance().GetOrCreateDict( a_col->dict_name,
                                                                         indexDataType,
                                                                         column->IsAllowNull(),
                                                                         column->char_max_len );
            column->SetDict( dict );
        }

        //check foreign key setting
        // mysql 忽略了在 column 定义中声明外键的方法，所以这里我们也不做处理
        #if 0
        if (a_col->foreign_key) {
            auto reference_table = database->GetTableByName(a_col->fk_table_name);
            std::shared_ptr<schema::ColumnEntry> reference_column = nullptr;

            string errMsg;
            if (reference_table == nullptr) {
                errMsg = format_mysql_err_msg( ER_CANNOT_ADD_FOREIGN ) +
                         ", ERROR: FK Table [" + a_col->fk_table_name + "] NOT exists!";
                ARIES_EXCEPTION_SIMPLE( ER_CANNOT_ADD_FOREIGN, errMsg );
            } else {
                reference_column = reference_table->GetColumnByName(a_col->fk_column_name);
                if (reference_column == nullptr) {
                    errMsg = format_mysql_err_msg( ER_CANNOT_ADD_FOREIGN ) +
                             ", ERROR: FK Column [" + a_col->fk_column_name +
                             "] NOT exists!";
                    ARIES_EXCEPTION_SIMPLE( ER_CANNOT_ADD_FOREIGN, errMsg );
                } else if (!reference_column->IsPrimary()) {
                    errMsg = format_mysql_err_msg( ER_CANNOT_ADD_FOREIGN ) +
                             ", ERROR: FK Column [" + a_col->fk_column_name +
                             "] NOT a primary key!";
                    ARIES_EXCEPTION_SIMPLE( ER_CANNOT_ADD_FOREIGN, errMsg );
                }
            }
            //todo: we also need to check type matching between primary-foreign keys
        }
        #endif

        table_to_create->AddColumn(column);
        ++colIdx;
    }

    if ( primary_constraint )
    {
        for ( const auto& column_name : primary_constraint->keys )
        {
            auto column = table_to_create->GetColumnByName( column_name );
            if ( !column )
            {
                ARIES_EXCEPTION( ER_KEY_COLUMN_DOES_NOT_EXITS, column_name.c_str() );
            }
            if ( column->IsExplicitNullable() || column->IsExplicitDefaultNull() )
                ARIES_EXCEPTION_SIMPLE( ER_PRIMARY_CANT_HAVE_NULL, "All parts of a PRIMARY KEY must be NOT NULL" );
            if ( EncodeType::DICT == column->encode_type )
                ARIES_EXCEPTION( ER_NOT_SUPPORTED_YET, "dict encoded column as primary key" );
            column->is_primary = true;
            column->allow_null = false;
        }

        table_to_create->AddConstraint( primary_constraint );
    }

    std::map< std::string, bool > exist_constraints_name;
    if ( !foreigns.empty() )
    {
        auto result = SQLExecutor::GetInstance()->ExecuteSQL(
                "select constraint_name from table_constraints where table_schema='" + database->GetName() + "' and CONSTRAINT_TYPE='FOREIGN KEY'",
                "information_schema" );
        if ( result->IsSuccess() )
        {
            auto table_block = std::dynamic_pointer_cast< aries_engine::AriesMemTable >( result->GetResults()[ 0 ] )->GetContent();
            auto column_buffer = table_block->GetColumnBuffer( 1 );
            for ( int i = 0; i < table_block->GetRowCount(); i++ )
            {
                exist_constraints_name[ column_buffer->GetString( i ) ] = true;
            }
        }
    }

    for ( const auto& key : foreigns )
    {
        auto target_table = database->GetTableByName( key->referencedTable );
        if ( !target_table )
        {
            ARIES_EXCEPTION( ER_CANNOT_ADD_FOREIGN );
        }

        if ( exist_constraints_name[ key->name ] )
        {
            ARIES_EXCEPTION( ER_FK_DUP_NAME, key->name.c_str() );
        }

        exist_constraints_name[ key->name ] = true;

        const auto& primary_key = target_table->GetPrimaryKey();

        for ( size_t i = 0; i < key->keys.size(); i++ )
        {
            const auto& self = key->keys[ i ];
            if ( i >= key->referencedKeys.size() )
            {
                ARIES_EXCEPTION( ER_WRONG_FK_DEF, self.c_str(), "Key reference and table reference don't match" );
            }

            const auto& fk = key->referencedKeys[ i ];

            if ( fk != primary_key[ i ] )
            {
                ARIES_EXCEPTION( ER_CANNOT_ADD_FOREIGN );
            }

            auto column = table_to_create->GetColumnByName( self );
            if ( !column )
            {
                ARIES_EXCEPTION( ER_KEY_COLUMN_DOES_NOT_EXITS, self.c_str() );
            }

            auto fk_column = target_table->GetColumnByName( fk );
            if ( !fk_column )
            {
                ARIES_EXCEPTION( ER_CANNOT_ADD_FOREIGN );
            }

            if ( column->GetType() != fk_column->GetType() )
            {
                ARIES_EXCEPTION( ER_WRONG_FK_DEF, self.c_str(), "Key reference and table reference don't match" );
            }
            column->is_foreign_key = true;
        }

        table_to_create->AddConstraint( key );
    }
    for ( const auto& key : uniqkeys )
    {
        for ( size_t i = 0; i < key->keys.size(); i++ )
        {
            const auto& keyCol = key->keys[ i ];

            auto column = table_to_create->GetColumnByName( keyCol );
            if ( !column )
            {
                ARIES_EXCEPTION( ER_KEY_COLUMN_DOES_NOT_EXITS, keyCol.c_str() );
            }
        }

        table_to_create->AddConstraint( key );
    }

    return table_to_create;
}

void CommandExecutor::CheckTablePartitionOptions(
    TableEntrySPtr &tableEntry,
    const CreateTableOptions &options,
    const schema::DatabaseEntrySPtr& database )
{
    // table partitions
    if ( options.m_partitionStructure )
    {
        if ( "RANGE" == options.m_partitionStructure->m_partMethod )
        {
            auto &partExpr = options.m_partitionStructure->m_partitionExprs[ 0 ];
            CommonBiaodashi *commonExpr = ( CommonBiaodashi* )partExpr.get();
            switch ( commonExpr->GetType() )
            {
                case BiaodashiType::Biaoshifu:
                {
                    auto exprContent = commonExpr->GetContent();
                    SQLIdentPtr ident = boost::get<SQLIdentPtr>( exprContent );
                    if ( !ident->db.empty() && ident->db != database->GetName() )
                    {
                        ARIES_EXCEPTION( ER_BAD_FIELD_ERROR,
                                         ident->ToString().data(),
                                         "partition function" );
                    }
                    if ( !ident->table.empty() && ident->table != tableEntry->GetName() )
                    {
                        ARIES_EXCEPTION( ER_BAD_FIELD_ERROR,
                                         ident->ToString().data(),
                                         "partition function" );
                    }
                    auto colName = ident->id;
                    auto colEntry = tableEntry->GetColumnByName( colName );
                    if ( !colEntry )
                    {
                        ARIES_EXCEPTION( ER_BAD_FIELD_ERROR,
                                         ident->ToString().data(),
                                         "partition function" );

                    }
                    auto colValueType = colEntry->GetType();
                    if ( ColumnEntry::IsIntegerType( colValueType ) )
                    {
                        ThrowNotSupportedException( "partition by integer column" );
                    }

                    if ( // !ColumnEntry::IsIntegerType( colValueType ) &&
                         ColumnType::DATE != colValueType &&
                         ColumnType::DATE_TIME != colValueType )
                    {
                        ARIES_EXCEPTION( ER_FIELD_TYPE_NOT_ALLOWED_AS_PARTITION_FIELD,
                                         DataTypeString( colEntry->GetType() ).data() );
                    }
                    // ER_PARTITION_FUNCTION_IS_NOT_ALLOWED 
                    vector< PartDef > partDefs;
                    if ( options.m_partitionStructure->m_partitionDefList )
                    {
                        partDefs = *( options.m_partitionStructure->m_partitionDefList );
                    }
                    if ( 0 == partDefs.size() )
                    {
                        ARIES_EXCEPTION( ER_PARTITIONS_MUST_BE_DEFINED_ERROR, "RANGE" );
                    }

                    DatabaseSchemaPointer schema_p = SchemaBuilder::BuildFromDatabase( database.get() );
                    SchemaAgentPointer schema_agent = std::make_shared<SchemaAgent>();
                    schema_agent->SetDatabaseSchema(schema_p);

                    auto dummy_query = std::make_shared< SelectStructure >();
                    dummy_query->SetDefaultSchemaAgent( schema_agent );

                    QueryContextPointer query_context =
                    std::make_shared<QueryContext>( QueryContextType::TheTopQuery,
                                                    0,
                                                    dummy_query, //query
                                                    nullptr, //parent
                                                    nullptr //possible expr
                                                    );

                    ExprContextPointer exprContext =
                        std::make_shared< ExprContext >(
                            ExprContextType::VirtualExpr,
                            nullptr,
                            query_context,
                            nullptr,
                            0 );

                    uint16_t partPos = 1;

                    // PARTITION p4 VALUES LESS THAN (18446744073709551616):
                    // ERROR 1697 (HY000): VALUES value for partition 'p4' must have type INT

                    // mysql> CREATE TABLE test.t2 ( date_col date, some_data INT ) PARTITION BY RANGE (year(date_col) + some_data) (PARTITION p0 VALUES LESS THAN (1991), PARTITION p4 VALUES LESS THAN (18446744073709551615) );
                    // ERROR 1493 (HY000): VALUES LESS THAN value must be strictly increasing for each partition
                    // mysql> CREATE TABLE test.t2 ( date_col date, some_data INT ) PARTITION BY RANGE (year(date_col) + some_data) (PARTITION p0 VALUES LESS THAN (1991), PARTITION p4 VALUES LESS THAN (MAXVALUE) );
                    // Query OK, 0 rows affected (0.03 sec)
                    // PARTITION p4 VALUES LESS THAN (18446744073709551615), PARTITION p5 VALUES LESS THAN MAXVALUE  );
                    // ERROR 1493 (HY000): VALUES LESS THAN value must be strictly increasing for each partition
                    // 
                    vector< int64_t > rangeValues;
                    std::unordered_set< string > partNames;
                    for ( size_t partIndex = 0; partIndex < partDefs.size(); ++partIndex )
                    {
                        auto partDef = partDefs[ partIndex ];
                        auto part = std::make_shared< schema::TablePartition >();
                        if ( partNames.end() != partNames.find( partDef.m_partitionName ) )
                        {
                            ARIES_EXCEPTION( ER_SAME_NAME_PARTITION, partDef.m_partitionName.data() );
                        }
                        partNames.emplace( partDef.m_partitionName );

                        part->m_partitionName = partDef.m_partitionName;
                        part->m_partOrdPos = partPos++;

                        if ( partDef.m_partValues->m_isMaxValue )
                        {
                            if ( partIndex != partDefs.size() - 1 )
                            {
                                ARIES_EXCEPTION( ER_PARTITION_MAXVALUE_ERROR );
                            }
                            part->m_partDesc = "MAXVALUE";
                            part->m_value = INT64_MAX;
                            part->m_isMaxValue = true;
                            tableEntry->AddPartition( part );
                            break;
                        }
                        else
                        {
                            PartValueItemsSPtr partValueItems =
                                partDef.m_partValues->m_valueItemsList[ 0 ];
                            if ( partValueItems->m_isMaxValue )
                            {
                                if ( partIndex != partDefs.size() - 1 )
                                {
                                    ARIES_EXCEPTION( ER_PARTITION_MAXVALUE_ERROR );
                                }
                                part->m_partDesc = "MAXVALUE";
                                part->m_isMaxValue = true;
                                part->m_value = INT64_MAX;
                                tableEntry->AddPartition( part );
                                break;
                            }
                            PartValueItem valueItem = partValueItems->m_valueItems[ 0 ];
                            assert( !valueItem.m_isMaxValue );

                            auto *commonBiaodashi = ( CommonBiaodashi* )valueItem.m_expr.get();
                            commonBiaodashi->CheckExpr( exprContext, true );
                            switch( commonBiaodashi->GetType() )
                            {
                                case BiaodashiType::Zifuchuan:
                                {
                                    if ( ColumnEntry::IsIntegerType( colValueType ) )
                                    {
                                        ARIES_EXCEPTION( ER_VALUES_IS_NOT_INT_TYPE_ERROR,
                                                         partDef.m_partitionName.data() );
                                    }
                                    break;
                                }
                                case BiaodashiType::Zhengshu:
                                    if ( !ColumnEntry::IsIntegerType( colValueType ) )
                                    {
                                        ARIES_EXCEPTION( ER_FIELD_TYPE_NOT_ALLOWED_AS_PARTITION_FIELD,
                                                         ShowUtility::GetInstance()->GetTextFromBiaodashiType( commonBiaodashi->GetType() ).data() );
                                    }
                                    break;
                                case BiaodashiType::Null:
                                    ARIES_EXCEPTION( ER_NULL_IN_VALUES_LESS_THAN );
                                    break;
                                default:
                                    ARIES_EXCEPTION( ER_FIELD_TYPE_NOT_ALLOWED_AS_PARTITION_FIELD,
                                                     ShowUtility::GetInstance()->GetTextFromBiaodashiType( commonBiaodashi->GetType() ).data() );
                            }
                            AriesExprBridge bridge;
                            AriesCommonExprUPtr commonExpr = bridge.Bridge( valueItem.m_expr );

                            auto valueType = commonExpr->GetValueType().DataType.ValueType;
                            int64_t rangeValue;
                            switch (valueType)
                            {
                                case aries::AriesValueType::INT8: {
                                    int8_t value = boost::get<int8_t>( commonExpr->GetContent() );
                                    rangeValue = value;
                                    part->m_partDesc = std::to_string( value );
                                    break;
                                }
                                case aries::AriesValueType::INT16: {
                                    int16_t value = boost::get<int16_t>( commonExpr->GetContent() );
                                    rangeValue = value;
                                    part->m_partDesc = std::to_string( value );
                                    break;
                                }
                                case aries::AriesValueType::INT32: {
                                    int32_t value = boost::get<int32_t>( commonExpr->GetContent() );
                                    rangeValue = value;
                                    part->m_partDesc = std::to_string( value );
                                    break;
                                }
                                case aries::AriesValueType::INT64: {
                                    int64_t value = boost::get<int64_t>(commonExpr->GetContent());
                                    rangeValue = value;
                                    part->m_partDesc = std::to_string( value );
                                    break;
                                }
                                case aries::AriesValueType::UINT8: {
                                    uint8_t value = boost::get<uint8_t>(commonExpr->GetContent());
                                    rangeValue = value;
                                    part->m_partDesc = std::to_string( value );
                                    break;
                                }
                                case aries::AriesValueType::UINT16: {
                                    uint16_t value = boost::get<uint16_t>(commonExpr->GetContent());
                                    rangeValue = value;
                                    part->m_partDesc = std::to_string( value );
                                    break;
                                }
                                case aries::AriesValueType::UINT32: {
                                    uint32_t value = boost::get<uint32_t>(commonExpr->GetContent());
                                    rangeValue = value;
                                    part->m_partDesc = std::to_string( value );
                                    break;
                                }
                                case aries::AriesValueType::DATE:
                                case aries::AriesValueType::DATETIME:
                                {
                                    auto transfer = aries_acc::AriesDatetimeTrans::GetInstance();
                                    string value = boost::get<string>(commonExpr->GetContent());
                                    rangeValue = transfer.ToAriesDate( value ).toTimestamp();
                                    part->m_partDesc = value;
                                    break;
                                }
                                case aries::AriesValueType::CHAR: {
                                    auto transfer = aries_acc::AriesDatetimeTrans::GetInstance();
                                    string value = boost::get<string>(commonExpr->GetContent());
                                    string valueTypeStr;
                                    try
                                    {
                                        switch ( colValueType )
                                        {
                                            case ColumnType::DATE:
                                                valueTypeStr = "date";
                                                rangeValue = transfer.ToAriesDate( value, ARIES_DATE_STRICT_MODE ).toTimestamp();
                                                break;
                                            case ColumnType::DATE_TIME:
                                                valueTypeStr = "datetime";
                                                rangeValue = transfer.ToAriesDatetime( value, ARIES_DATE_STRICT_MODE ).toTimestamp();
                                                break;
                                            default:
                                                assert( 0 );
                                                break;
                                        }
                                    }
                                    catch ( ... )
                                    {
                                        string msg( "Incorrect " + valueTypeStr + " value: '" + value + "'" );
                                        ARIES_EXCEPTION_SIMPLE( ER_TRUNCATED_WRONG_VALUE_FOR_FIELD,
                                                                msg );
                                    }
                                    part->m_partDesc = value;
                                    break;
                                }
                                default:
                                {
                                    ARIES_EXCEPTION( ER_VALUES_IS_NOT_INT_TYPE_ERROR,
                                                     partDef.m_partitionName.data() );
                                    break;
                                }
                            }
                            part->m_value = rangeValue;
                            if ( rangeValues.size() > 0 )
                            {
                                if ( rangeValue <= rangeValues[ rangeValues.size() - 1 ] )
                                {
                                    ARIES_EXCEPTION( ER_RANGE_NOT_INCREASING_ERROR );
                                }
                            }
                            rangeValues.push_back( rangeValue );
                        }

                        tableEntry->AddPartition( part );
                    }

                    tableEntry->SetPartitionTypeDefInfo(
                        options.m_partitionStructure->m_partMethod,
                        colEntry->GetColumnIndex(),
                        options.m_partitionStructure->m_partitionExprStr );
                    
                    break;
                }
                default:
                {
                    ThrowNotSupportedException( "Non column partition expression" );
                    break;
                }
            }
        }
        else
        {
            ThrowNotSupportedException(
                "partition method '" + options.m_partitionStructure->m_partMethod + "'" );
        }
    }
}

std::string CommandExecutor::ExecuteCreateTable(CommandStructure *arg_command_p, const string& currentDbName) {
    auto schema = schema::SchemaManager::GetInstance()->GetSchema();
    std::shared_ptr<schema::DatabaseEntry> database = nullptr;
    if (schema == nullptr) {
        ARIES_EXCEPTION_SIMPLE(ER_UNKNOWN_ERROR, "Create Table ERROR: schema not exists!");
        return "";
    }

    string dbName = arg_command_p->GetDatabaseName();
    if (dbName.empty()) {
        if (currentDbName.empty()) {
            ARIES_EXCEPTION( ER_NO_DB_ERROR );
        }
        dbName = currentDbName;
    }
    database = schema->GetDatabaseByName(dbName);
    if (database == nullptr) {
        ARIES_EXCEPTION( ER_BAD_DB_ERROR, dbName.data() );
    }

    std::string tableName = arg_command_p->GetTableName();
    if (database->GetTableByName(tableName) != nullptr) {
        string errMsg = format_mysql_err_msg(ER_TABLE_EXISTS_ERROR, tableName.data());
        if (!arg_command_p->ifNotExists) {
            ARIES_EXCEPTION_SIMPLE( ER_TABLE_EXISTS_ERROR, errMsg );
        }
        LOG(INFO) << errMsg;
        return std::to_string(-1) + "Create Table ERROR: Table [" + tableName + "] exists!";
    }

    auto table_to_create = CommandToTableEntry( arg_command_p, database );
    CheckTablePartitionOptions( table_to_create,
                                arg_command_p->GetCreateTableOptions(),
                                database );
    table_to_create->SetId( aries::schema::get_auto_increment_table_id() );

    for ( auto const& key : table_to_create->GetForeignKeys() )
    {
        auto referenced_schema = schema->GetDatabaseByName( key->referencedSchema );
        auto referenced_table = referenced_schema->GetTableByName( key->referencedTable );

        for ( size_t i = 0; i < key->keys.size(); i++ )
        {
            const auto& keyName = key->keys[ i ];
            const auto& referencedKeyName = key->referencedKeys[ i ];
            referenced_table->AddReferencing( referencedKeyName, keyName, table_to_create->GetName(), database->GetName(), key->name, table_to_create->GetId() );
        }
    }

    bool insertSchema = true;
    if (dbName == aries::schema::INFORMATION_SCHEMA) {
        // handled by Schema::InitInfoSchemaBaseTables()
        if (schema::SCHEMATA == tableName || "tables" == tableName || "columns" == tableName) {
            insertSchema = false;
        }
    }
    if (insertSchema) {
        schema->InsertTableSchema(dbName, table_to_create);
    }
    string tableDataDir = Configuartion::GetInstance().GetDataDirectory( dbName, tableName );
    try
    {
        boost::filesystem::create_directory( tableDataDir );
    }
    catch (...)
    {
        ThrowNotSupportedException("table name: "+tableName);
    }

    database->PutTable(table_to_create);

    aries_engine::AriesInitialTable::InitFiles( dbName, tableName );

    return "";
}

/*
mysql:
mysql> drop database test;
ERROR 1217 (23000): Cannot delete or update a parent row: a foreign key constraint fails
mysql> drop table test.t_refed;
ERROR 1217 (23000): Cannot delete or update a parent row: a foreign key constraint fails
*/
std::string CommandExecutor::ExecuteDropTable(CommandStructure *arg_command_p, const string& currentDbName) {
    auto schema = schema::SchemaManager::GetInstance()->GetSchema();
    std::shared_ptr<schema::DatabaseEntry> currentDatabase = nullptr;
    if (schema == nullptr) {
        ARIES_EXCEPTION_SIMPLE(ER_UNKNOWN_ERROR, "Drop Table ERROR: schema  NOT exists!");
        return std::to_string(-1) + "Drop Table ERROR: schema  NOT exists!";
    } else {
        currentDatabase = schema->GetDatabaseByName(currentDbName);
    }

    auto tableList = arg_command_p->GetTableList();
    string table_name;
    if (currentDbName.empty()) {
        for (std::shared_ptr<BasicRel> tableIdent : *tableList) {
            if (tableIdent->GetDb().empty()) {
                ARIES_EXCEPTION( ER_NO_DB_ERROR );
            }
        }
    }

    string ret = "drop success";
    /**
     * mysql> drop tables db1.t1, db1.t2;
Query OK, 0 rows affected (0.01 sec)
     */
    string dbName;
    std::shared_ptr<schema::DatabaseEntry> database = nullptr;
    for (std::shared_ptr<BasicRel> tableIdent : *tableList) {
        dbName = tableIdent->GetDb();
        if (dbName.empty()) {
            dbName = currentDbName;
            database = currentDatabase;
        } else {
            database = schema->GetDatabaseByName(dbName);
        }

        if ( aries::schema::IsSysDb(dbName) && !schema->IsInit() ) {
            string msg("Access denied to database ");
            msg.append(dbName);
            ARIES_EXCEPTION_SIMPLE(ER_DBACCESS_DENIED_ERROR, msg);
        }

        table_name = tableIdent->GetID();
        string fullName = dbName;
        fullName.append(".").append(table_name);
        if ( !database ) {
            if (!arg_command_p->ifExists)
            {
                /**
                 * mysql 5.7.26 behaviour:
                 * mysql> drop table ad.d;
    ERROR 1051 (42S02): Unknown table 'ad.d'
    mysql> drop table db1.ta;
    ERROR 1051 (42S02): Unknown table 'db1.ta'
    mysql> drop table db1.t1;
    Query OK, 0 rows affected (0.01 sec)
                 */
                ARIES_EXCEPTION( ER_BAD_TABLE_ERROR, fullName.data() );
            }
            else
            {
                continue;
            }
        }

        auto table = database->GetTableByName(table_name);

        if ( !table ) {
            if ( !arg_command_p->ifExists )
            {
                ARIES_EXCEPTION( ER_BAD_TABLE_ERROR, fullName.data() );
            }
            else
            {
                continue;
            }
        }

        table->OnDrop( dbName );
        database->RemoveTable( table );
    }

    return ret;
}

// char CommandExecutor::GetSepCh(const std::string &param) {
//     char ch = ',';
//     std::regex r(".*?\\(\\s*?(\\S)\\s*?\\)");
//     std::smatch m;
//     if (std::regex_match(param, m, r)) {
//         ch = m[1].str()[0];
//     }
//     return ch;
// }

std::string CommandExecutor::ExecuteCopyTable(CommandStructure *arg_command_p, AriesServerSession *arg_session_p, const string& dbName) {

    std::string table_name = arg_command_p->GetTableName();
    assert(arg_session_p->session_schema);
    if (arg_session_p->session_schema->FindPhysicalTable(table_name) == nullptr) {
        ARIES_EXCEPTION( ER_BAD_TABLE_ERROR, table_name.data() );
    }
//    int direction = arg_command_p->GetCopyDirection();
//    char sep = GetSepCh(arg_command_p->GetCopyFormatReq());
//    switch (sep) {
//        case '|': {
//            switch (direction) {
//                case 1:
//                    AriesEngineV2::AriesCommandExecutor::CopyTableFromFile<'|'>(arg_session_p, table_name.c_str(),
//                                                                                arg_command_p->GetCopyFileLocation().c_str());
//                    break;
//                case -1:
//                    AriesEngineV2::AriesCommandExecutor::CopyTableToFile<'|'>(arg_session_p, table_name.c_str(),
//                                                                              arg_command_p->GetCopyFileLocation().c_str());
//                    break;
//                default:
//                    assert(0);
//                    break;
//            }
//            break;
//        }
//        case ',': {
//            switch (direction) {
//                case 1:
//                    AriesEngineV2::AriesCommandExecutor::CopyTableFromFile<','>(arg_session_p, table_name.c_str(),
//                                                                                arg_command_p->GetCopyFileLocation().c_str());
//                    break;
//                case -1:
//                    AriesEngineV2::AriesCommandExecutor::CopyTableToFile<','>(arg_session_p, table_name.c_str(),
//                                                                              arg_command_p->GetCopyFileLocation().c_str());
//                    break;
//                default:
//                    assert(0);
//                    break;
//            }
//            break;
//        }
//        case ';': {
//            switch (direction) {
//                case 1:
//                    AriesEngineV2::AriesCommandExecutor::CopyTableFromFile<';'>(arg_session_p, table_name.c_str(),
//                                                                                arg_command_p->GetCopyFileLocation().c_str());
//                    break;
//                case -1:
//                    AriesEngineV2::AriesCommandExecutor::CopyTableToFile<';'>(arg_session_p, table_name.c_str(),
//                                                                              arg_command_p->GetCopyFileLocation().c_str());
//                    break;
//                default:
//                    assert(0);
//                    break;
//            }
//            break;
//        }
//        case '\t': {
//            switch (direction) {
//                case 1:
//                    AriesEngineV2::AriesCommandExecutor::CopyTableFromFile<'\t'>(arg_session_p, table_name.c_str(),
//                                                                                 arg_command_p->GetCopyFileLocation().c_str());
//                    break;
//                case -1:
//                    AriesEngineV2::AriesCommandExecutor::CopyTableToFile<'\t'>(arg_session_p, table_name.c_str(),
//                                                                               arg_command_p->GetCopyFileLocation().c_str());
//                    break;
//                default:
//                    assert(0);
//                    break;
//            }
//            break;
//        }
//        default:
//            assert(0);
//            //FIXME support more seps.
//    }

    return std::to_string(0) + "Copy Table Successfully! [" + arg_command_p->ToString() + "]";
}

std::string CommandExecutor::ExecuteInsertQuery(CommandStructure *arg_command_p) {
    std::string ret;
    return ret;

}

std::string CommandExecutor::ExecuteCommand(AbstractCommand *arg_abstr_command_p, AriesServerSession *arg_session_p, const string& dbName) {
    assert(arg_abstr_command_p);

    LOG(INFO) << "CommandStructure::Execute()--->[" + arg_abstr_command_p->ToString() + "]";

    std::string ret;
    CommandStructure* command_p;

    switch (arg_abstr_command_p->GetCommandType()) {
        case CommandType::ChangeDatabase: {
            command_p = dynamic_cast< CommandStructure* >( arg_abstr_command_p );
            if (!mysql_change_db(current_thd, command_p->GetDatabaseName(), false)) {
                my_ok(current_thd);
            }
            break;
        }
        case CommandType::CreateDatabase:
            command_p = dynamic_cast< CommandStructure* >( arg_abstr_command_p );
            ret = ExecuteCreateDatabase(command_p);
            break;

        case CommandType::DropDatabase:
            command_p = dynamic_cast< CommandStructure* >( arg_abstr_command_p );
            ret = ExecuteDropDatabase(command_p, dbName);
            break;

        case CommandType::CreateTable:
            command_p = dynamic_cast< CommandStructure* >( arg_abstr_command_p );
            ret = ExecuteCreateTable(command_p, dbName);
            break;


        case CommandType::DropTable:
            command_p = dynamic_cast< CommandStructure* >( arg_abstr_command_p );
            ret = ExecuteDropTable(command_p, dbName);
            break;

        case CommandType::CopyTable:
            command_p = dynamic_cast< CommandStructure* >( arg_abstr_command_p );
            ret = ExecuteCopyTable(command_p, arg_session_p, dbName);
            break;

        case CommandType::InsertQuery:
            command_p = dynamic_cast< CommandStructure* >( arg_abstr_command_p );
            ret = ExecuteInsertQuery(command_p);
            break;

        case CommandType::CreateView:
            command_p = dynamic_cast< CommandStructure* >( arg_abstr_command_p );
            ret = ExecuteCreateView(command_p, dbName);
            break;

        case CommandType::DropView:
            command_p = dynamic_cast< CommandStructure* >( arg_abstr_command_p );
            ret = ExecuteDropView(command_p, dbName);
            break;

        case CommandType::CreateUser:
        {
            AccountMgmtStructure* createUserCommand = dynamic_cast< AccountMgmtStructure*  >( arg_abstr_command_p );
            ExecuteCreateUser( createUserCommand );
            break;
        }

        case CommandType::DropUser:
        {
            AccountMgmtStructure* createUserCommand = dynamic_cast< AccountMgmtStructure*  >( arg_abstr_command_p );
            ExecuteDropUser( createUserCommand );
            break;
        }

        default:
            ARIES_ASSERT(0, "wrong commandtype:" + std::to_string((int) arg_abstr_command_p->GetCommandType()));
    }


    return ret;

}

std::string CommandExecutor::ExecuteCreateView(CommandStructure *arg_command_p, const string& currentDbName) {
    auto schema = schema::SchemaManager::GetInstance()->GetSchema();
    std::shared_ptr<schema::DatabaseEntry> database = nullptr;
    if (schema == nullptr) {
        ARIES_EXCEPTION_SIMPLE(ER_UNKNOWN_ERROR, "Create View ERROR: schema not exists!");
        return "";
    }

    string dbName = arg_command_p->GetDatabaseName();
    if (dbName.empty()) {
        if (currentDbName.empty()) {
            ARIES_EXCEPTION( ER_NO_DB_ERROR );
        }
        dbName = currentDbName;
    }
    database = schema->GetDatabaseByName(dbName);
    if (database == nullptr) {
        ARIES_EXCEPTION( ER_BAD_DB_ERROR, dbName.data() );
    }

    std::string tableName = arg_command_p->GetTableName();
    if (database->GetTableByName(tableName) != nullptr) {
        string errMsg = format_mysql_err_msg(ER_TABLE_EXISTS_ERROR, tableName.data());
        if (!arg_command_p->ifNotExists) {
            ARIES_EXCEPTION_SIMPLE( ER_TABLE_EXISTS_ERROR, errMsg );
        }
        LOG(INFO) << errMsg;
        return std::to_string(-1) + "Create View ERROR: Table [" + tableName + "] exists!";
    }

    auto create_string = arg_command_p->GetCommandString();

    auto view_node = ViewManager::BuildFromCommand(arg_command_p, dbName);

    if (!view_node) {
        return std::to_string(-1) + "Create View ERROR";
    }

    if (!ViewManager::GetInstance().RegisterViewNode(view_node, tableName, dbName)) {
        ARIES_EXCEPTION(ER_TABLE_EXISTS_ERROR, tableName.data());
    }

    auto query = std::dynamic_pointer_cast<SelectStructure>(view_node->GetMyQuery());
    auto relation = query->GetRelationStructure();

    std::vector<schema::ColumnEntryPtr> columns;
    for (size_t i = 0; i < relation->GetColumnCount(); i++) {
        auto column = relation->GetColumn(i);
        auto column_entry = ColumnEntry::MakeColumnEntry(
                column->GetName(), column->GetValueType(), i,
                false, false, false, false,
                false,
                column->IsNullable(), false, nullptr,
                column->GetLength(), column->GetLength(),
                -1, -1, -1,
                std::string(),
                std::string());

        columns.emplace_back(column_entry);
    }

    schema->InsertViewSchema(dbName, tableName, create_string, columns);

    return "";
}

std::string CommandExecutor::ExecuteDropView(CommandStructure *arg_command_p, const string& currentDbName) {
    auto schema = schema::SchemaManager::GetInstance()->GetSchema();
    std::shared_ptr<schema::DatabaseEntry> currentDatabase = nullptr;
    if (schema == nullptr) {
        ARIES_EXCEPTION_SIMPLE(ER_UNKNOWN_ERROR, "Drop View ERROR: schema  NOT exists!");
        return std::to_string(-1) + "Drop View ERROR: schema  NOT exists!";
    } else {
        currentDatabase = schema->GetDatabaseByName(currentDbName);
    }

    auto tableList = arg_command_p->GetTableList();
    string table_name;

    for (const auto& table : *tableList) {
        const std::string& db_name = table->GetDb().empty() ? currentDbName : table->GetDb();
        table_name = table->GetID();
        
        if (db_name.empty()) {
            ARIES_EXCEPTION( ER_NO_DB_ERROR );
        }

        auto view_node = ViewManager::GetInstance().GetViewNode(table_name, db_name);
        if (!view_node) {
            if ( !arg_command_p->ifExists )
            {
                string full_name = db_name;
                full_name.append(".").append(table_name);
                ARIES_EXCEPTION( ER_BAD_TABLE_ERROR, full_name.data() );
            }
            else
            {
                continue;
            }
        }
        schema->RemoveView( db_name, table_name );
        ViewManager::GetInstance().UnregisterViewNode(table_name, db_name);
    }
    return std::string();
}


}//ns
