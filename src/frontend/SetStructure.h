//
// Created by tengjp on 19-9-4.
//

#ifndef AIRES_SETSTRUCTURE_H
#define AIRES_SETSTRUCTURE_H

#include <string>
#include <memory>
#include "AriesDefinition.h"
#include "VariableStructure.h"
#include "CommonBiaodashi.h"
#include "AriesEngine/AriesCommonExpr.h"
using aries_engine::AriesCommonExprUPtr;

using std::string;
class sys_var;

NAMESPACE_ARIES_START

// https://dev.mysql.com/doc/refman/5.7/en/set-statement.html
// set variable error handing
/**
 * Use of SET (any variant) to set a read-only variable:

mysql> SET GLOBAL version = 'abc';
ERROR 1238 (HY000): Variable 'version' is a read only variable
Use of GLOBAL to set a variable that has only a session value:

mysql> SET GLOBAL sql_log_bin = ON;
ERROR 1231 (42000): Variable 'sql_log_bin' can't be
set to the value of 'ON'
Use of SESSION to set a variable that has only a global value:

mysql> SET SESSION max_connections = 1000;
ERROR 1229 (HY000): Variable 'max_connections' is a
GLOBAL variable and should be set with SET GLOBAL
Omission of GLOBAL to set a variable that has only a global value:


mysql> SET max_connections = 1000;
ERROR 1229 (HY000): Variable 'max_connections' is a
GLOBAL variable and should be set with SET GLOBAL

 The @@GLOBAL., @@SESSION., and @@ modifiers apply only to system variables.
 An error occurs for attempts to apply them to user-defined variables,
 stored procedure or function parameters, or stored program local variables.

Not all system variables can be set to DEFAULT. In such cases, assigning DEFAULT results in an error.

An error occurs for attempts to assign DEFAULT to user-defined variables,
 stored procedure or function parameters, or stored program local variables.


 *
 * User variables can be assigned a value from a limited
 * set of data types: integer, decimal, floating-point,
 * binary or nonbinary string, or NULL value.
 * Assignment of decimal and real values does not
 * preserve the precision or scale of the value.
 * A value of a type other than one of the permissible types
 * is converted to a permissible type. For example,
 * a value having a temporal or spatial data type is converted
 * to a binary string. A value having the JSON data type
 * is converted to a string with a character set of utf8mb4 and a collation of utf8mb4_bin.

 * If a user variable is assigned a nonbinary (character) string value,
 * it has the same character set and collation as the string.
 * The coercibility of user variables is implicit. (This is the same coercibility as for table column values.)

 * Hexadecimal or bit values assigned to user variables are treated as binary strings.
 * To assign a hexadecimal or bit value as a number to a user variable,
 * use it in numeric context. For example, add 0 or use CAST(... AS UNSIGNED):
 *
 * set @a= (select n_nationkey from NATION);
 * ERROR 1242 (21000): Subquery returns more than 1 row
 *
 * set @d = @a + @b + @c;
 *
 */
enum class SET_CMD : std::uint8_t {
    // https://dev.mysql.com/doc/refman/5.7/en/set-variable.html
    SET_USER_VAR,
    SET_SYS_VAR,
    // https://dev.mysql.com/doc/refman/5.7/en/set-character-set.html
    //  SET CHARACTER SET sets three session system variables:
    //  character_set_client and character_set_results are set to the given character set,
    //  and character_set_connection to the value of character_set_database.
    // SET {CHARACTER SET | CHARSET}
    //    {'charset_name' | DEFAULT}
    SET_CHAR_SET,
    // https://dev.mysql.com/doc/refman/5.7/en/set-names.html
    // SET NAMES {'charset_name'
    //    [COLLATE 'collation_name'] | DEFAULT}
    SET_NAMES,
    // https://dev.mysql.com/doc/refman/5.7/en/set-password.html
    // SET PASSWORD [FOR user] = password_option
    //
    //password_option: {
    //    'auth_string'
    //  | PASSWORD('auth_string')
    //}
    SET_PASSWORD,
    SET_TX
};

class SetStructure {
public:
    SetStructure() = default;
    virtual ~SetStructure() = default;
    virtual void Check( THD* thd ){}
    virtual void Update( THD* thd ){}

    SET_CMD m_setCmd;
};
using SetStructurePtr = std::shared_ptr<SetStructure>;

class SetSysVarStructure : public SetStructure {
public:
    SetSysVarStructure( const SysVarStructurePtr& sysVarStructurePtr, const BiaodashiPointer& expr )
        : m_sysVarStructurePtr( sysVarStructurePtr )
    {
        m_setCmd = SET_CMD::SET_SYS_VAR;
        m_valueExpr = std::dynamic_pointer_cast< CommonBiaodashi >( expr );
    }

    void Check( THD* thd );
    void Update( THD* thd );

    SysVarStructurePtr m_sysVarStructurePtr;
    CommonBiaodashiPtr m_valueExpr;
    AriesCommonExprUPtr m_valueCommonExpr;
    sys_var* m_sysVar;

    std::string m_strValue;
    ulonglong m_ulonglongValue;
    double m_doubleValue;
};
using SetSysVarStructurePtr = std::shared_ptr< SetSysVarStructure >;

class SetUserVarStructure : public SetStructure {
public:
    SetUserVarStructure( const UserVarStructurePtr& userVarStructurePtr, const BiaodashiPointer& expr )
        : m_userVarStructurePtr( userVarStructurePtr )
    {
        m_setCmd = SET_CMD::SET_USER_VAR;
        m_valueExpr = std::dynamic_pointer_cast< CommonBiaodashi >( expr );
    }
    void Check( THD* thd );
    UserVarStructurePtr m_userVarStructurePtr;
    CommonBiaodashiPtr m_valueExpr;
};
using SetUserVarStructurePtr = std::shared_ptr<SetUserVarStructure>;

class SetPasswordStructure : public SetStructure
{
public:
    std::string user;
    std::string host;
    std::string password;

    void Check( THD* thd );
};
using SetPasswordStructurePtr = std::shared_ptr< SetPasswordStructure >;

NAMESPACE_ARIES_END // namespace aries


#endif //AIRES_SETSTRUCTURE_H
