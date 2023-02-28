//
// Created by tengjp on 19-8-13.
//

#ifndef AIRES_SET_VAR_H
#define AIRES_SET_VAR_H

#include <string>
#include <vector>

#include "server/mysql/include/my_getopt.h"
#include "server/mysql/include/sys_vars_shared.h"
#include "frontend/VariousEnum.h"
#include "frontend/CommonBiaodashi.h"
#include "frontend/SetStructure.h"
#include "mysql_com.h"
#include "m_ctype.h"
#include "protocol.h"

using std::string;
using aries::SetSysVarStructurePtr;

sys_var *find_sys_var(const char *str);
/**
  Declarations for SHOW STATUS support in plugins
*/
enum enum_mysql_show_type
{
    SHOW_UNDEF, SHOW_BOOL,
    SHOW_INT,        ///< shown as _unsigned_ int
    SHOW_LONG,       ///< shown as _unsigned_ long
    SHOW_LONGLONG,   ///< shown as _unsigned_ longlong
    SHOW_CHAR, SHOW_CHAR_PTR,
    SHOW_ARRAY, SHOW_FUNC, SHOW_DOUBLE
// #ifdef MYSQL_SERVER
    /*
    This include defines server-only values of the enum.
    Using them in plugins is not supported.
  */
//  #include "sql_plugin_enum.h"
    , SHOW_KEY_CACHE_LONG,
    SHOW_KEY_CACHE_LONGLONG,
    SHOW_LONG_STATUS,
    SHOW_DOUBLE_STATUS,
    SHOW_HAVE,
    SHOW_MY_BOOL,
    SHOW_HA_ROWS,
    SHOW_SYS,
    SHOW_LONG_NOFLUSH,
    SHOW_LONGLONG_STATUS,
    SHOW_SIGNED_LONG
//#endif
};

typedef enum enum_mysql_show_type SHOW_TYPE;

struct sys_var_chain
{
    sys_var *first;
    sys_var *last;
};

/**
  A class representing one system variable - that is something
  that can be accessed as @@global.variable_name or @@session.variable_name,
  visible in SHOW xxx VARIABLES and in INFORMATION_SCHEMA.xxx_VARIABLES,
  optionally it can be assigned to, optionally it can have a command-line
  counterpart with the same name.
*/
class sys_var
{
public:
    sys_var *next;
    string name;
    enum flag_enum
    {
        GLOBAL=       0x0001,
        SESSION=      0x0002,
        ONLY_SESSION= 0x0004,
        SCOPE_MASK=   0x03FF, // 1023
        READONLY=     0x0400, // 1024
        ALLOCATED=    0x0800, // 2048
        INVISIBLE=    0x1000, // 4096
        TRI_LEVEL=    0x2000  // 8192 - default is neither GLOBAL nor SESSION
    };
    static const int PARSE_EARLY= 1;
    static const int PARSE_NORMAL= 2;
    /**
      Enumeration type to indicate for a system variable whether
      it will be written to the binlog or not.
    */
    enum binlog_status_enum { VARIABLE_NOT_IN_BINLOG,
        SESSION_VARIABLE_IN_BINLOG } binlog_status;

protected:
    typedef bool (*on_check_function)(sys_var *self, THD *thd, const SetSysVarStructure* var);
    typedef bool (*on_update_function)(sys_var *self, THD *thd, enum_var_type type);

    int flags;            ///< or'ed flag_enum values
    int m_parse_flag;     ///< either PARSE_EARLY or PARSE_NORMAL.
    const SHOW_TYPE show_val_type; ///< what value_ptr() returns for sql_show.cc
    my_option option;     ///< min, max, default values are stored here
    PolyLock *guard;      ///< *second* lock that protects the variable
    ptrdiff_t offset;     ///< offset to the value from global_system_variables
    on_check_function on_check;
    on_update_function on_update;
    const char *const deprecation_substitute;
    bool is_os_charset; ///< true if the value is in character_set_filesystem

public:
    sys_var(sys_var_chain *chain, const char *name_arg, const char *comment,
            int flag_args, ptrdiff_t off, // int getopt_id,
            // enum get_opt_arg_type getopt_arg_type,
                    SHOW_TYPE show_val_type_arg,
            longlong def_val, PolyLock *lock, enum binlog_status_enum binlog_status_arg,
            on_check_function on_check_func, on_update_function on_update_func,
            const char *substitute, int parse_flag);

    virtual ~sys_var() {}

    /**
      All the cleanup procedures should be performed here
    */
    virtual void cleanup() {}

    void check(THD *thd, SetSysVarStructure* var);
    uchar *value_ptr(THD *running_thd, THD *target_thd, enum_var_type type);
    uchar *value_ptr(THD *thd, enum_var_type type);
    virtual void update_default(longlong new_def_value)
    { option.def_value= new_def_value; }

    /**
       Update the system variable with the default value from either
       session or global scope.  The default value is stored in the
       'var' argument. Return false when successful.
    */
    void set_default(THD *thd, SetSysVarStructure* var);
    bool update(THD *thd, const SetSysVarStructure* var);

    SHOW_TYPE show_type() { return show_val_type; }
    int scope() const { return flags & SCOPE_MASK; }
    const CHARSET_INFO *charset(THD *thd);
    bool is_readonly() const { return flags & READONLY; }
    bool not_visible() const { return flags & INVISIBLE; }
    bool is_trilevel() const { return flags & TRI_LEVEL; }
    bool is_session_var() const { return flags & SESSION; }
    bool is_session_only_var() const { return flags & ONLY_SESSION; }
    bool is_global_var() const { return flags & GLOBAL; }
    /**
      the following is only true for keycache variables,
      that support the syntax @@keycache_name.variable_name
    */
    bool is_struct() { return option.var_type & GET_ASK_ADDR; }
    virtual bool check_update_type( aries::BiaodashiValueType value_type ) = 0;

    /**
      Return TRUE for success if:
        Global query and variable scope is GLOBAL or SESSION, or
        Session query and variable scope is SESSION or ONLY_SESSION.
    */
    bool check_scope(enum_var_type query_type)
    {
        switch (query_type)
        {
            case OPT_GLOBAL:  return scope() & (GLOBAL | SESSION);
            case OPT_SESSION: return scope() & (SESSION | ONLY_SESSION);
            case OPT_DEFAULT: return scope() & (SESSION | ONLY_SESSION);
        }
        return false;
    }

    bool register_option(std::vector<my_option> *array, int parse_flags)
    {
        return (option.id != -1) && (m_parse_flag & parse_flags) &&
               (array->push_back(option), false);
    }

    std::string to_string(enum_var_type scope);

private:
    virtual void do_check(THD *thd, SetSysVarStructure* var) = 0;
    /**
      save the session default value of the variable in var
    */
    virtual void session_save_default(THD *thd, SetSysVarStructure* var) = 0;
    /**
      save the global default value of the variable in var
    */
    virtual void global_save_default(THD *thd, SetSysVarStructure* var) = 0;
    virtual bool session_update(THD *thd, const SetSysVarStructure* var) = 0;
    virtual bool global_update(THD *thd, const SetSysVarStructure* var) = 0;
protected:
    /**
      A pointer to a value of the variable for SHOW.
      It must be of show_val_type type (bool for SHOW_BOOL, int for SHOW_INT,
      longlong for SHOW_LONGLONG, etc).
    */
    virtual uchar *session_value_ptr(THD *running_thd, THD *target_thd);
    virtual uchar *global_value_ptr(THD *thd);

    /**
      A pointer to a storage area of the variable, to the raw data.
      Typically it's the same as session_value_ptr(), but it's different,
      for example, for ENUM, that is printed as a string, but stored as a number.
    */
    uchar *session_var_ptr(THD *thd);

    uchar *global_var_ptr();
};

int sys_var_init();
int sys_var_add_options(std::vector<my_option> *long_options, int parse_flags);
void sys_var_end(void);
#endif //AIRES_SET_VAR_H
